"""Small persistent, same-group/member-only participation preferences."""
from __future__ import annotations

import os
import re
import sqlite3
import time


class MemberPreferences:
    def __init__(self, path=None):
        self._db = sqlite3.connect(str(path) if path is not None else ":memory:", timeout=5)
        if path is not None:
            os.chmod(path, 0o600)
        self._db.execute("CREATE TABLE IF NOT EXISTS preferences (group_id TEXT, user_id TEXT, mode TEXT, "
                         "updated REAL, PRIMARY KEY(group_id,user_id))")
        self._db.commit()
        self._values = {(g, u): mode for g, u, mode in self._db.execute(
            "SELECT group_id,user_id,mode FROM preferences")}

    def mention_only(self, group, user):
        return self._values.get((str(group), str(user))) == "mention_only"

    def set(self, group, user, mode):
        group, user = str(group), str(user)
        if mode not in {"normal", "mention_only"} or any(
            re.fullmatch(r"[1-9][0-9]{0,19}", item) is None for item in (group, user)
        ):
            raise ValueError("invalid same-group member preference")
        if (group, user) not in self._values and len(self._values) >= 4096:
            raise ValueError("member preference capacity reached")
        self._values[(group, user)] = mode
        with self._db:
            self._db.execute("INSERT INTO preferences VALUES(?,?,?,?) ON CONFLICT(group_id,user_id) "
                             "DO UPDATE SET mode=excluded.mode,updated=excluded.updated",
                             (group, user, mode, time.time()))

    def close(self):
        self._db.close()
