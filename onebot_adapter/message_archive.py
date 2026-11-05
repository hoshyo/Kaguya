"""Private, group-scoped received index for time reads and explicit context packets."""
from __future__ import annotations

import json
import os
import sqlite3
import time

from onebot_adapter.history_search import documents, forward_id, normalized, order_key


class MessageArchive:
    def __init__(self, path=None):
        self.db = sqlite3.connect(str(path) if path else ":memory:")
        if path:
            os.chmod(path, 0o600)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS group_messages (
                group_id TEXT, message_id TEXT, user_id TEXT, timestamp REAL,
                text TEXT, payload TEXT, PRIMARY KEY(group_id,message_id));
            CREATE INDEX IF NOT EXISTS group_message_time ON group_messages(group_id,timestamp);
            CREATE INDEX IF NOT EXISTS group_member_time ON group_messages(group_id,user_id,timestamp);
            CREATE TABLE IF NOT EXISTS read_progress (group_id TEXT PRIMARY KEY, timestamp REAL);
            CREATE TABLE IF NOT EXISTS group_search_documents (
                group_id TEXT, message_id TEXT, part TEXT, body TEXT, normalized_body TEXT,
                PRIMARY KEY(group_id,message_id,part));
        """)
        # Additive migration: leave original messages/receipts untouched, index existing payloads once.
        with self.db:
            missing = self.db.execute(
                "SELECT g.group_id,g.message_id,g.payload FROM group_messages g WHERE NOT EXISTS "
                "(SELECT 1 FROM group_search_documents d WHERE d.group_id=g.group_id "
                "AND d.message_id=g.message_id AND d.part='[]')").fetchall()
            for group, mid, payload in missing:
                self._index(group, mid, json.loads(payload))
        self.db.commit()

    def _index(self, group, mid, raw):
        self.db.execute("DELETE FROM group_search_documents WHERE group_id=? AND message_id=?", (group, mid))
        self.db.executemany("INSERT INTO group_search_documents VALUES(?,?,?,?,?)",
                            [(group, mid, json.dumps(path), body, normalized(body)) for path, body in documents(raw)])

    @staticmethod
    def _preserve_forwards(raw, previous):
        """Native text-only pages must not erase already recovered forward bodies."""
        import copy

        def forwards(tree):
            pending, seen = [(tree, ())], set()
            while pending:
                value, path = pending.pop()
                if not isinstance(value, (dict, list)) or id(value) in seen:
                    continue
                seen.add(id(value))
                if isinstance(value, list):
                    pending.extend((item, (*path, index)) for index, item in enumerate(value))
                    continue
                data = value.get("data")
                if value.get("type") == "forward" and isinstance(data, dict):
                    resource = str(data.get("id") or data.get("resid") or "")
                    if resource:
                        yield path, data, resource
                pending.extend((item, (*path, key)) for key, item in value.items())

        known, resources = {}, {}
        for path, data, resource in forwards(previous):
            content = data.get("content")
            resources.setdefault(resource, []).append(content)
            if isinstance(content, list) and content:
                known[path, resource] = content
        if not known:
            return raw
        raw = copy.deepcopy(raw)
        # Collect BEFORE filling: never recursively expand newly copied cyclic resource references.
        for path, data, resource in list(forwards(raw)):
            if data.get("content"):
                continue
            content = known.get((path, resource))
            choices = resources.get(resource, [])
            if content is None and len(choices) == 1:
                content = choices[0]
            if isinstance(content, list) and content:
                data["content"] = copy.deepcopy(content)
        return raw

    def record(self, data):
        if not isinstance(data, dict) or data.get("message_type") != "group":
            return
        group, mid = str(data.get("group_id", "")), str(data.get("message_id", ""))
        if not group.isdecimal() or not mid:
            return
        previous = self.get(group, mid)
        if previous:
            data = self._preserve_forwards(data, previous)
        payload = json.dumps(data, ensure_ascii=False)
        if previous is not None and payload == json.dumps(previous, ensure_ascii=False):
            return
        text = "".join(str(s.get("data", {}).get("text", "")) for s in data.get("message", [])
                       if isinstance(s, dict) and s.get("type") == "text")
        stamp = float(data.get("time") or time.time())
        with self.db:
            self.db.execute("INSERT INTO group_messages VALUES(?,?,?,?,?,?) "
                            "ON CONFLICT(group_id,message_id) DO UPDATE SET payload=excluded.payload, "
                            "text=excluded.text, timestamp=excluded.timestamp, user_id=excluded.user_id",
                            (group, mid, str(data.get("user_id", "")), stamp, text,
                             payload))
            self._index(group, mid, data)

    def search(self, group, *, start=None, end=None, users=None, before=None, matcher):
        """Search message fragments with bound SQL parameters, before applying any result limit."""
        group = str(group)
        clauses, args = ["g.group_id=?"], [group]
        for op, stamp in ((">=", start), ("<=", end)):
            if stamp is not None:
                clauses.append("g.timestamp" + op + "?")
                args.append(stamp)
        if users:
            clauses.append("g.user_id IN (" + ",".join("?" for _ in users) + ")")
            args.extend(users)
        boundary = self.get(group, before) if before else None
        if before and boundary is None:
            raise ValueError("before不属于当前群已索引的消息，请使用本群返回的next_before")
        if boundary:
            clauses.append("g.timestamp<=?")
            args.append(float(boundary["time"]))
        condition, values = matcher.sql()
        clauses.append("EXISTS (SELECT 1 FROM group_search_documents d WHERE d.group_id=g.group_id "
                       "AND d.message_id=g.message_id AND " + condition + ")")
        matcher.begin()
        self.db.create_function("history_regex", 2, matcher.regex_match)
        try:
            rows = self.db.execute("SELECT g.payload FROM group_messages g WHERE " + " AND ".join(clauses),
                                   args + values).fetchall()
            result = []
            for (payload,) in rows:
                raw = json.loads(payload)
                if boundary and order_key(raw) >= order_key(boundary):
                    continue
                hits = self.db.execute(
                    "SELECT d.part FROM group_search_documents d WHERE d.group_id=? AND d.message_id=? AND "
                    + condition, [group, str(raw["message_id"]), *values]).fetchall() if matcher.terms else []
                matches = [{"kind": "forward" if part != "[]" else "message",
                            "message_id": forward_id(raw["message_id"], tuple(json.loads(part)))
                            if part != "[]" else str(raw["message_id"])} for (part,) in hits]
                result.append((raw, matches))
            return sorted(result, key=lambda item: order_key(item[0]))
        except sqlite3.OperationalError:
            if matcher.failure:
                raise ValueError(matcher.failure) from None
            raise
        finally:
            self.db.create_function("history_regex", 2, None)

    def coverage(self, group, *, allowed=None):
        rows = self.db.execute("SELECT timestamp,user_id FROM group_messages WHERE group_id=?", (str(group),))
        stamps = [stamp for stamp, user in rows if allowed is None or allowed(str(user))]
        return {"earliest_time": min(stamps) if stamps else None, "latest_time": max(stamps) if stamps else None,
                "message_count": len(stamps),
                "continuous_history_guaranteed": False}

    def query(self, group, *, start=None, end=None, count=None, users=None, keyword=None, before=None):
        clauses, args = ["group_id=?"], [str(group)]
        for op, stamp in ((">=", start), ("<=", end)):
            if stamp is not None:
                clauses.append("timestamp" + op + "?")
                args.append(stamp)
        if users:
            clauses.append("user_id IN (" + ",".join("?" for _ in users) + ")")
            args.extend(users)
        if keyword:
            clauses.append("instr(text,?)>0")
            args.append(keyword)
        if before:
            found = self.db.execute("SELECT timestamp FROM group_messages WHERE group_id=? AND message_id=?",
                                    (str(group), before)).fetchone()
            if found:
                clauses.append("(timestamp<? OR (timestamp=? AND message_id<?))")
                args.extend((found[0], found[0], before))
        sql = "SELECT payload FROM group_messages WHERE " + " AND ".join(clauses)
        sql += " ORDER BY timestamp DESC,message_id DESC"
        if count is not None:
            sql += " LIMIT ?"
            args.append(count)
        return list(reversed([json.loads(r[0]) for r in self.db.execute(sql, args)]))

    def progress(self, group):
        row = self.db.execute("SELECT timestamp FROM read_progress WHERE group_id=?", (str(group),)).fetchone()
        return row[0] if row else None

    def get(self, group, message_id):
        row = self.db.execute("SELECT payload FROM group_messages WHERE group_id=? AND message_id=?",
                              (str(group), str(message_id))).fetchone()
        return json.loads(row[0]) if row else None

    def mark_read(self, group, stamp):
        with self.db:
            self.db.execute("INSERT INTO read_progress VALUES(?,?) ON CONFLICT(group_id) "
                            "DO UPDATE SET timestamp=max(timestamp,excluded.timestamp)", (str(group), stamp))

    def close(self):
        self.db.close()
