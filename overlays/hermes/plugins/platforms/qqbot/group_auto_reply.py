"""Local, pre-model admission for opt-in QQ full-group messages.

QQ's GROUP_MESSAGE_CREATE uses the already subscribed GROUP_AND_C2C_EVENT intent:
https://bot.q.qq.com/wiki/develop/api-v2/autogen/event/group_message_create.html
The group must separately enable QQ's receive-all-messages setting.
"""

from __future__ import annotations

import logging
import math
import random
import re
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

logger = logging.getLogger(__name__)
_AT_TOKEN = re.compile(r'<@!?([^>]+)>')


@dataclass(frozen=True)
class _Rule:
    cooldown_seconds: float
    window_seconds: float
    max_replies: int
    probability: float
    names: tuple[re.Pattern, ...]


def _rule(value: dict) -> _Rule:
    cooldown = float(value.get('cooldown_seconds', 90))
    window = float(value.get('window_seconds', 600))
    maximum = value.get('max_replies', 4)
    probability = float(value.get('probability', 0.12))
    if not all(math.isfinite(v) for v in (cooldown, window, probability)):
        raise ValueError('limits must be finite')
    if cooldown <= 0 or window <= 0 or not 0 <= probability <= 1:
        raise ValueError('cooldown/window must be positive and probability between 0 and 1')
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
        raise ValueError('max_replies must be a positive integer')
    names = value.get('names', [])
    if not isinstance(names, list) or any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError('names must be a list of nonempty strings')
    patterns = tuple(re.compile(r'(?<![A-Za-z0-9_])' + re.escape(name.strip()) + r'(?![A-Za-z0-9_])', re.IGNORECASE)
                     for name in names)
    return _Rule(cooldown, window, maximum, probability, patterns)


def mentions_bot(data: dict, content: str, bot_ids: set[str]) -> bool:
    if any(match.group(1) in bot_ids for match in _AT_TOKEN.finditer(content)):
        return True
    mentions = data.get('mentions')
    if not isinstance(mentions, list):
        return False
    return any(isinstance(user, dict) and any(str(user.get(key) or '') in bot_ids
               for key in ('id', 'member_openid', 'user_openid')) for user in mentions)


def strip_bot_mentions(content: str, bot_ids: set[str]) -> str:
    return _AT_TOKEN.sub(lambda match: '' if match.group(1) in bot_ids else match.group(0), content).strip()


class GroupAutoReplyPolicy:
    def __init__(self, config, *, clock: Callable[[], float] | None = None, sample: Callable[[], float] | None = None):
        self._clock = clock or time.monotonic
        self._sample = sample or random.random
        self._rules: dict[str, _Rule] = {}
        self._attempts: dict[str, deque[float]] = {}
        self._last_activity: dict[str, float] = {}
        if not isinstance(config, dict):
            return
        for group, value in config.items():
            if not isinstance(value, dict) or value.get('enabled', True) is not True:
                continue
            try:
                self._rules[str(group)] = _rule(value)
            except (TypeError, ValueError) as exc:
                logger.warning('Ignoring invalid QQ group_auto_reply limits for %s: %s', group, exc)

    def enabled(self, group: str) -> bool:
        return group in self._rules

    def note_activity(self, group: str) -> None:
        if self.enabled(group):
            self._last_activity[group] = self._clock()

    def admit(self, group: str, content: str) -> bool:
        rule = self._rules.get(group)
        text = content.strip()
        if rule is None or not text or _AT_TOKEN.search(text):
            return False
        if not any(ch.isalpha() for ch in text):
            return False
        named = any(pattern.search(text) for pattern in rule.names)
        if not named and (len(text) < 6 or self._sample() >= rule.probability):
            return False
        now = self._clock()
        attempts = self._attempts.setdefault(group, deque())
        while attempts and attempts[0] <= now - rule.window_seconds:
            attempts.popleft()
        last = self._last_activity.get(group)
        if len(attempts) >= rule.max_replies or (last is not None and now - last < rule.cooldown_seconds):
            return False
        # Reserve synchronously, before any media download, dispatch or model await.
        # Failed/silent attempts still cost budget: errors cannot create retry storms.
        attempts.append(now)
        self._last_activity[group] = now
        return True
