"""Bounded text + separate @ invitations; no timers, API calls or persistence."""
from __future__ import annotations

import math
import time
from collections import OrderedDict
from dataclasses import dataclass


@dataclass(frozen=True)
class PairedInput:
    data: dict
    message_ids: list[str]
    order: str


class GroupInvitations:
    WINDOW = 30.0
    MAXIMUM = 1024
    MAX_TEXT = 2000

    def __init__(self, *, clock=time.monotonic, wall_clock=time.time):
        self.clock, self.wall_clock = clock, wall_clock
        self.recent, self.pending, self.seen = OrderedDict(), OrderedDict(), OrderedDict()

    def clear(self):
        for entries in (self.recent, self.pending, self.seen):
            entries.clear()

    def forget(self, group, user):
        key = (str(group), str(user))
        self.recent.pop(key, None)
        self.pending.pop(key, None)

    def _prune(self):
        now = self.clock()
        for entries in (self.recent, self.pending, self.seen):
            for key, value in list(entries.items()):
                if now - value[0] >= self.WINDOW:
                    entries.pop(key, None)

    def _put(self, entries, key, *values):
        entries[key] = (self.clock(), *values)
        entries.move_to_end(key)
        while len(entries) > self.MAXIMUM:
            entries.popitem(last=False)

    def consider(self, data, *, self_id, reason, observed):
        """Called only after the normal parser, for empty/trigger drops.

        An empty @ has already passed group/user/blacklist admission. Ordinary
        text must also have passed the social observer's same-group filters.
        Paired data is reparsed through ALL normal filters before dispatch.
        """
        self._prune()
        if data.get("post_type") != "message" or data.get("message_type") != "group":
            return None
        group, user, mid = (str(data.get(key, "")) for key in ("group_id", "user_id", "message_id"))
        if not all(value and len(value) <= 80 for value in (group, user, mid)) or user == str(self_id):
            return None
        key = (group, user)
        segments = data.get("message", []) or []
        if not isinstance(segments, list) or not all(
            isinstance(s, dict) and isinstance(s.get("data", {}), dict) for s in segments
        ):
            return None
        ats = [str(s.get("data", {}).get("qq", "")) for s in segments if s.get("type") == "at"]
        text = "".join(str(s.get("data", {}).get("text", ""))
                       for s in segments if s.get("type") == "text").strip()
        kinds = {s.get("type") for s in segments}
        if any(at != str(self_id) for at in ats) or "reply" in kinds or text.startswith("/"):
            self.forget(group, user)
            return None
        try:
            age = self.wall_clock() - float(data.get("time") or self.wall_clock())
        except (ValueError, TypeError, OverflowError):
            return None
        if not math.isfinite(age) or age >= self.WINDOW or age < -60:
            return None
        if reason == "empty" and ats and not text and kinds <= {"at", "text"}:
            identity = (*key, mid)
            if identity in self.seen:
                return None
            self._put(self.seen, identity)
            previous = self.recent.pop(key, None)
            if previous is not None:
                self.pending.pop(key, None)
                return self._pair(data, self_id, previous[2], [previous[1], mid], "question_then_mention")
            self._put(self.pending, key, mid)
        elif reason == "trigger" and observed and not ats and kinds == {"text"} \
                and text and len(text) <= self.MAX_TEXT:
            pending = self.pending.pop(key, None)
            if pending is not None:
                self.recent.pop(key, None)
                return self._pair(data, self_id, text, [pending[1], mid], "mention_then_question")
            self._put(self.recent, key, mid, text)
        elif reason == "trigger" and observed:
            self.forget(group, user)
        return None

    @staticmethod
    def _pair(data, self_id, text, ids, order):
        combined = {**data, "message": [{"type": "at", "data": {"qq": str(self_id)}},
                                        {"type": "text", "data": {"text": text}}]}
        return PairedInput(combined, ids, order)

    def admitted(self, group, user, message_ids):
        self._prune()
        key = (str(group), str(user))
        ids = {str(mid) for mid in message_ids}
        for entries in (self.recent, self.pending):
            if key in entries and entries[key][1] in ids:
                entries.pop(key, None)
