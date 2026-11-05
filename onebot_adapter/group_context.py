"""Opt-in, bounded, memory-only group background; never triggers a model call."""
from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from onebot_adapter.config import AdapterConfig

LIMITS = {"max_messages": 50, "max_age_seconds": 600,
          "max_message_chars": 600, "max_context_chars": 16000}


def validate_group_context(value, errors):
    if not isinstance(value, dict):
        errors.append("group_context must be a dict")
        return
    for group, rule in value.items():
        label = f"group_context.{group}"
        if not isinstance(rule, dict):
            errors.append(f"{label} must be a dict")
            continue
        if not isinstance(rule.get("enabled", False), bool):
            errors.append(f"{label}.enabled must be bool")
        for key, maximum in LIMITS.items():
            number = rule.get(key, maximum)
            if type(number) is not int or not 1 <= number <= maximum:
                errors.append(f"{label}.{key} must be an integer between 1 and {maximum}")


@dataclass(frozen=True)
class _Entry:
    received_at: float
    message: dict


class GroupContextBuffer:
    def __init__(self, config: AdapterConfig, *, clock=time.monotonic, wall_clock=time.time):
        self._clock = clock
        self._wall_clock = wall_clock
        self._buffers = {}
        self._scopes = {}
        self.configure(config)

    def configure(self, config):
        rules, scopes = {}, {}
        for group, raw in config.group_context.items():
            errors = []
            validate_group_context({group: raw}, errors)
            if errors or raw.get("enabled") is not True:
                continue
            group = str(group)
            gc = config.get_group_config(group)
            if not gc.enabled:
                continue
            rule = {key: raw.get(key, default) for key, default in LIMITS.items()}
            rules[group] = rule
            scopes[group] = (rule, gc.to_dict(), tuple(config.global_admins), config.bot_blacklist_enabled)
        # Revoked groups/users and changed retention policies must not leave old data.
        self._buffers = {g: rows for g, rows in self._buffers.items()
                         if g in scopes and self._scopes.get(g) == scopes[g]}
        self._rules, self._scopes, self._config = rules, scopes, config

    def _allowed(self, group, user, blacklist_match):
        cfg = self._config
        if group not in self._rules or not user or user == str(cfg.self_id):
            return False
        if not cfg.is_group_user_allowed(group, user):
            return False
        if cfg.bot_blacklist_enabled and not cfg.is_admin(user, group) and blacklist_match is not None:
            return blacklist_match(user, group) is None
        return True

    def _prune(self, group, blacklist_match):
        age = self._rules[group]["max_age_seconds"]
        now, wall = self._clock(), self._wall_clock()
        rows = self._buffers.get(group, ())
        self._buffers[group] = deque(
            row for row in rows if now - row.received_at < age
            and wall - row.message["timestamp"] < age
            and (row.message.get("bot") is True or self._allowed(group, row.message["user_id"], blacklist_match))
        )

    def before_event(self, data, *, blacklist_match=None):
        """Snapshot preceding messages, then remember this frame, without API I/O."""
        if not isinstance(data, dict) or data.get("post_type") != "message" \
                or data.get("message_type") != "group":
            return []
        group, user = str(data.get("group_id", "")), str(data.get("user_id", ""))
        if not self._allowed(group, user, blacklist_match):
            return []
        self._prune(group, blacklist_match)
        message_id = str(data.get("message_id", ""))
        if not message_id or len(message_id) > 80 or len(user) > 80:
            return []
        rule, rows = self._rules[group], self._buffers[group]
        # A duplicate transport delivery cannot include itself as prior context.
        snapshot = [dict(row.message) for row in rows if row.message["message_id"] != message_id]
        if any(row.message["message_id"] == message_id for row in rows):
            return snapshot
        segments = data.get("message", [])
        if not isinstance(segments, list):
            return snapshot
        parts, plain, reply = [], [], ""
        placeholders = {"image": "[图片，未识别]", "record": "[语音，未转写]",
                        "video": "[视频，未解析]", "file": "[文件，未读取]",
                        "forward": "[合并转发，未展开]", "face": "[表情]"}
        for segment in segments[:256]:
            if not isinstance(segment, dict) or not isinstance(segment.get("data", {}), dict):
                continue
            kind, detail = segment.get("type"), segment.get("data", {})
            if kind == "text":
                text = str(detail.get("text", ""))[:rule["max_message_chars"]]
                plain.append(text)
                parts.append(text)
            elif kind == "at":
                parts.append("@" + str(detail.get("qq", ""))[:80])
            elif kind == "reply":
                reply = str(detail.get("id", ""))[:80]
            elif kind in placeholders:
                parts.append(placeholders[kind])
        # Commands are never replayed as background or a new gateway action.
        text = "".join(parts).strip()
        if not text or "".join(plain).lstrip().startswith("/"):
            return snapshot
        try:
            raw_timestamp = data.get("time")
            timestamp = float(self._wall_clock() if raw_timestamp is None or raw_timestamp == 0 else raw_timestamp)
        except (ValueError, TypeError, OverflowError):
            return snapshot
        wall = self._wall_clock()
        if not math.isfinite(timestamp) or wall - timestamp >= rule["max_age_seconds"]:
            return snapshot
        timestamp = min(timestamp, wall)  # future sender clocks cannot extend retention
        sender = data.get("sender", {}) or {}
        if not isinstance(sender, dict):
            sender = {}
        message = {"message_id": message_id, "user_id": user,
                   "user_name": str(sender.get("card") or sender.get("nickname") or user)[:80],
                   "timestamp": timestamp, "text": text[:rule["max_message_chars"]]}
        if reply:
            message["reply_to_message_id"] = reply
        rows.append(_Entry(self._clock(), message))
        while len(rows) > rule["max_messages"] or self._size(rows) > rule["max_context_chars"]:
            rows.popleft()
        # Snapshot has its own budget, including before a long new message evicts rows.
        while snapshot and len(json.dumps(snapshot, ensure_ascii=False)) > rule["max_context_chars"]:
            snapshot.pop(0)
        return snapshot[-rule["max_messages"]:]

    @staticmethod
    def _size(rows):
        return len(json.dumps([row.message for row in rows], ensure_ascii=False))
