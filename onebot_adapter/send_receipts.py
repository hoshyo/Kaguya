"""Durable at-most-once part receipts, no message bodies or credentials.

A pending record after a crash means outcome unknown: never blindly resend.
This intentionally prefers a missing bubble to duplicate notifications.
"""
from __future__ import annotations

import json
import sqlite3
import time


class SendReceipts:
    def __init__(self, path):
        self.db = sqlite3.connect(path, timeout=2)
        self.db.execute("CREATE TABLE IF NOT EXISTS parts (id TEXT PRIMARY KEY, fingerprint TEXT, "
                        "created REAL, result TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS handled_wakeups (group_id TEXT, message_id TEXT, "
                        "handled_at REAL, PRIMARY KEY(group_id,message_id))")
        self.db.commit()

    def lookup(self, key, fingerprint):
        row = self.db.execute("SELECT fingerprint, result FROM parts WHERE id=?", (key,)).fetchone()
        if row is None:
            return None
        if row[0] != fingerprint:
            return {"success": False, "error": "part content changed", "retryable": False}
        return json.loads(row[1]) if row[1] else {
            "success": False, "error": "send outcome unknown; automatic resend suppressed", "retryable": False,
        }

    def claim(self, key, fingerprint):
        self.db.execute("DELETE FROM parts WHERE created < ?", (time.time() - 86400,))
        if self.db.execute("SELECT COUNT(*) FROM parts").fetchone()[0] >= 10000:
            raise RuntimeError("send receipt capacity reached")
        self.db.execute("INSERT INTO parts VALUES (?, ?, ?, NULL)", (key, fingerprint, time.time()))
        self.db.commit()  # durable BEFORE the external side effect

    def complete(self, key, result):
        safe = {k: result[k] for k in ("success", "message_id", "retryable") if k in result}
        if (result.get("data") or {}).get("suppressed"):
            safe["data"] = {"suppressed": True}
        self.db.execute("UPDATE parts SET result=? WHERE id=?", (json.dumps(safe), key))
        self.db.commit()

    def close(self):
        self.db.close()

    def wakeup_handled(self, group, message_id):
        row = self.db.execute("SELECT handled_at FROM handled_wakeups WHERE group_id=? AND message_id=?",
                              (str(group), str(message_id))).fetchone()
        return bool(row and row[0] >= time.time() - 86400)

    def mark_wakeups_handled(self, group, message_ids):
        """Processing/send receipts, deliberately separate from history read progress."""
        now = time.time()
        with self.db:
            self.db.execute("DELETE FROM handled_wakeups WHERE handled_at<?", (now - 86400,))
            self.db.executemany("INSERT INTO handled_wakeups VALUES(?,?,?) ON CONFLICT(group_id,message_id) "
                                "DO UPDATE SET handled_at=excluded.handled_at",
                                [(str(group), str(mid), now) for mid in set(message_ids) if mid])
