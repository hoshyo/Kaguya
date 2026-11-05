"""Local, pre-model sampling of opt-in ordinary group messages."""
from __future__ import annotations

import math
import random
import re
import time
from collections import deque


class GroupAutoReplyPolicy:
    def __init__(self, config, *, clock=time.monotonic, sample=random.random):
        self._clock = clock
        self._sample = sample
        self._attempts = {}
        self._last_activity = {}
        self.configure(config)

    def configure(self, config):
        self._rules = {}
        if not isinstance(config, dict):
            return
        for group, value in (config or {}).items():
            if not isinstance(value, dict) or value.get("enabled") is not True:
                continue
            try:
                cooldown = float(value.get("cooldown_seconds", 90))
                window = float(value.get("window_seconds", 600))
                probability = float(value.get("probability", .12))
                maximum = value.get("max_replies", 4)
                names = value.get("names", [])
                if not all(math.isfinite(x) for x in (cooldown, window, probability)):
                    continue
                if cooldown <= 0 or window <= 0 or not 0 <= probability <= 1:
                    continue
                if type(maximum) is not int or maximum < 1 or not isinstance(names, list):
                    continue
                patterns = tuple(re.compile(r"(?<![A-Za-z0-9_])" + re.escape(n) + r"(?![A-Za-z0-9_])", re.I)
                                 for n in names if isinstance(n, str) and n.strip())
                self._rules[str(group)] = (cooldown, window, probability, maximum, patterns)
            except (ValueError, TypeError):
                continue

    def note_activity(self, group):
        if group in self._rules:
            self._last_activity[group] = self._clock()

    def admit(self, group, segments):
        rule = self._rules.get(group)
        if rule is None or any(s.get("type") == "at" for s in segments):
            return False
        text = "".join(str(s.get("data", {}).get("text", ""))
                       for s in segments if s.get("type") == "text").strip()
        if not text or text.startswith("/") or not any(ch.isalpha() for ch in text):
            return False
        cooldown, window, probability, maximum, names = rule
        named = any(p.search(text) for p in names)
        if not named and (len(text) < 6 or self._sample() >= probability):
            return False
        now = self._clock()
        attempts = self._attempts.setdefault(group, deque())
        while attempts and attempts[0] <= now - window:
            attempts.popleft()
        last = self._last_activity.get(group)
        if len(attempts) >= maximum or (last is not None and now - last < cooldown):
            return False
        # Reserve before any await: failed/silent attempts still consume budget.
        attempts.append(now)
        self._last_activity[group] = now
        return True
