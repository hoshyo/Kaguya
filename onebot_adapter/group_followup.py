"""Bounded short-lived reply metadata; no chat text or cross-group history."""
from __future__ import annotations

import math
import time
from collections import OrderedDict, deque
from dataclasses import dataclass


@dataclass(frozen=True)
class FollowupRule:
    window: float
    interval: float
    budget_window: float
    maximum: int


class GroupFollowupTracker:
    def __init__(self, config, *, clock=time.monotonic):
        self._clock = clock
        self._recent = OrderedDict()
        self._outbox = OrderedDict()
        self._checks = OrderedDict()
        self.configure(config)

    def configure(self, config):
        self._rules = {}
        if not isinstance(config, dict):
            return
        for group, parent in config.items():
            if not isinstance(parent, dict) or parent.get("enabled") is not True:
                continue
            value = parent.get("followup", {})
            if not isinstance(value, dict) or value.get("enabled") is not True:
                continue
            try:
                rule = FollowupRule(float(value.get("window_seconds", 120)),
                                    float(value.get("min_interval_seconds", 3)),
                                    float(value.get("budget_window_seconds", 120)),
                                    value.get("max_checks", 6))
                if not all(math.isfinite(v) for v in (rule.window, rule.interval, rule.budget_window)):
                    continue
                if not 0 < rule.window <= 600 or rule.interval <= 0 or rule.budget_window <= 0:
                    continue
                if type(rule.maximum) is not int or rule.maximum < 1:
                    continue
                self._rules[str(group)] = rule
            except (ValueError, TypeError):
                continue

    @staticmethod
    def _put(mapping, key, value, maximum):
        mapping[key] = value
        mapping.move_to_end(key)
        while len(mapping) > maximum:
            mapping.popitem(last=False)

    def record_reply(self, group, user, message_id):
        if group not in self._rules or not message_id:
            return
        now = self._clock()
        if user:
            self._put(self._recent, (group, str(user)), now, 1024)
        self._put(self._outbox, (group, str(message_id)), now, 512)

    def forget_user(self, group, user):
        self._recent.pop((str(group), str(user)), None)

    def classify(self, group, user, segments, *, contextual=False, consume=True):
        """Return followup, blocked, or None (ordinary sampling may apply).

        Unknown quotes / any @other user close the sender's short window and
        are never randomly sampled. Explicit @bot was already admitted by the
        parser, so only unmentioned traffic reaches this seam.
        """
        rule = self._rules.get(group)
        if rule is None:
            return None
        now = self._clock()
        key = (group, str(user))
        quoted = [str(s.get("data", {}).get("id", "")) for s in segments if s.get("type") == "reply"]
        other = any(s.get("type") == "at" for s in segments)
        known_quote = quoted and all(
            now - self._outbox.get((group, mid), -math.inf) < rule.window for mid in quoted)
        if other or (quoted and not known_quote):
            self._recent.pop(key, None)
            return "blocked"
        # A transport-observed small ongoing exchange may continue without a
        # visible quote/@. Still require a real recent bot send in this group,
        # and retain the same per-sender quota and explicit-other exclusions.
        recent = now - self._recent.get(key, -math.inf) < rule.window
        recent = recent or contextual is True and any(
            g == group and now - at < rule.window for (g, _), at in self._outbox.items())
        if not known_quote and not recent:
            return None
        text = "".join(str(s.get("data", {}).get("text", ""))
                       for s in segments if s.get("type") == "text").strip()
        if not text or text.startswith("/"):
            return "blocked"
        # Social observation debounces a burst before judging it. Preflight
        # must not charge every keystroke/message or cancel an earlier candidate
        # merely because the same person immediately supplied a correction.
        if not consume:
            return "followup"
        checks = self._checks.get(key, deque())
        while checks and checks[0] <= now - rule.budget_window:
            checks.popleft()
        if len(checks) >= rule.maximum or (checks and now - checks[-1] < rule.interval):
            return "blocked"
        checks.append(now)
        self._put(self._checks, key, checks, 1024)
        return "followup"
