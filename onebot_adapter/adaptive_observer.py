"""Per-group engagement pacing; decisions are exclusively made by the observer agent."""
from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Shared transport policy, not persona-specific prompt knobs or history-tool quotas.
MICROBATCH_SECONDS = .25
MAX_PENDING = 256
MAX_BATCH_MESSAGES = 32
MAX_BATCH_CHARS = 16000
RECENT_EXCHANGE_SECONDS = 90
MAX_WAKE_CANDIDATES = 8
WAKE_MAX_AGE_SECONDS = 120


@dataclass
class Observation:
    data: dict
    emit: object
    received_at: float


@dataclass
class Engagement:
    score: float = 8.0
    updated: float = 0.0
    checked: float | None = None
    skips: int = 0
    pending: deque = field(default_factory=deque)
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    recent_members: dict = field(default_factory=dict)
    errors: int = 0
    retry_after: float = 0.0
    overflow: int = 0
    last_queue_seconds: float = 0.0
    last_judge_seconds: float = 0.0
    last_action: str | None = None

    def interval(self, now):
        self.score *= 0.5 ** (max(0, now - self.updated) / 180)
        self.updated = now
        for threshold, seconds in ((6, 0), (4, 10), (2, 30), (1, 60), (.5, 120)):
            if self.score >= threshold:
                return seconds
        return 180

    def result(self, wake, now):
        self.interval(now)
        if wake:
            # A proposed wake may end in NO_REPLY; it is not actual engagement.
            self.skips = 0
        else:
            self.skips += 1
            if self.skips % 3 == 0:
                self.score = max(0, self.score - 1)

    def replied(self, user, now, *, reward=True):
        self.interval(now)
        if reward:
            self.score = min(8, self.score + 2)
        self.skips = 0
        self.recent_members = {u: until for u, until in self.recent_members.items() if until > now}
        self.recent_members[str(user)] = now + RECENT_EXCHANGE_SECONDS
        while len(self.recent_members) > 32:
            self.recent_members.pop(next(iter(self.recent_members)))

    def priority(self, now):
        return any(self.recent_members.get(str(item.data.get("user_id")), 0) > now
                   for item in self.pending)


class AdaptiveObserver:
    def __init__(self, social):
        self.social = social
        self.states, self.tasks = {}, {}
        self.wakeups, self.wake_tasks = {}, {}

    @property
    def settings(self):
        return {"microbatch_seconds": MICROBATCH_SECONDS, "max_pending": MAX_PENDING,
                "max_batch_messages": MAX_BATCH_MESSAGES, "max_batch_chars": MAX_BATCH_CHARS,
                "max_message_chars": None, "batch_char_budget": "soft_preserve_single_message",
                "recent_exchange_seconds": RECENT_EXCHANGE_SECONDS,
                "max_wake_candidates": MAX_WAKE_CANDIDATES, "wake_max_age_seconds": WAKE_MAX_AGE_SECONDS,
                "engagement_reward": "successful_text_delivery"}

    def state(self, group):
        return self.states.setdefault(group, Engagement(updated=self.social.clock()))

    def boost(self, group):
        state = self.state(group)
        state.score, state.skips, state.updated = 8, 0, self.social.clock()
        state.changed.set()

    def note_reply(self, group, user, *, reward=True):
        self.state(group).replied(user, self.social.clock(), reward=reward)
        self.state(group).changed.set()

    def offer(self, data, emit):
        group = str(data["group_id"])
        state = self.state(group)
        if len(state.pending) >= MAX_PENDING:
            state.pending.popleft()
            state.overflow += 1
            self.social.stats["observer_queue_overflow"] += 1
        state.pending.append(Observation(data, emit, self.social.clock()))
        if state.interval(self.social.clock()) == 0 or state.priority(self.social.clock()):
            state.changed.set()
        if group not in self.tasks or self.tasks[group].done():
            self.tasks[group] = asyncio.create_task(self.run(group), name="observer-agent:" + group)

    async def run(self, group):
        state, social = self.state(group), self.social
        draining = False
        try:
            while state.pending and social.enabled(group):
                now = social.clock()
                interval = state.interval(now)
                priority = state.priority(now)
                delay = 0 if state.checked is None or draining or priority else max(
                    0, interval - (now - state.checked))
                delay = max(delay, state.retry_after - now)
                if delay:
                    state.changed.clear()
                    try:
                        await asyncio.wait_for(state.changed.wait(), delay)
                    except TimeoutError:
                        pass
                    continue
                # A fixed tiny collection window, never a reset-on-every-message debounce.
                # Explicit invitations have already dispatched independently.
                if not draining:
                    await asyncio.sleep(MICROBATCH_SECONDS)
                allowed, rows, chars = [], [], 0
                while state.pending and len(rows) < MAX_BATCH_MESSAGES:
                    item = state.pending[0]
                    data = item.data
                    permitted = social.guard_enabled_data(data) or (
                        data.get("_already_woken") and social.enabled(group) and social.observer._allowed(
                            group, str(data.get("user_id", "")), social.blacklist_match))
                    if not permitted:
                        state.pending.popleft()
                        continue
                    row = social.observation_row(data)
                    if state.recent_members.get(str(data.get("user_id")), 0) > item.received_at:
                        row["recent_exchange_with_role"] = True
                    size = len(json.dumps(row, ensure_ascii=False))
                    if rows and chars + size > MAX_BATCH_CHARS:
                        break
                    # A single long message travels intact in its own batch.
                    # This is a batching target, not permission to discard its tail.
                    state.pending.popleft()
                    allowed.append(item)
                    rows.append(row)
                    chars += size
                # Split a due backlog into bounded turns without waiting another
                # three minutes between its already-due chunks.
                draining = bool(state.pending)
                if not allowed:
                    continue
                unwoken = [item for item in allowed if not item.data.get("_already_woken")]
                anchor = (unwoken or allowed)[-1]
                data = anchor.data
                state.checked = social.clock()
                state.last_queue_seconds = max(0, state.checked - min(i.received_at for i in allowed))
                social.stats["observer_calls"] += 1
                social.stats["observer_messages"] += len(rows)
                if len(rows) > 1:
                    social.stats["observer_microbatches"] += 1
                omitted, state.overflow = state.overflow, 0
                try:
                    result = await social.judge({"observer_agent": True, "group_id": group,
                        "self_id": str(social.config.self_id), "current_message_id": str(data["message_id"]),
                        "messages": rows,
                        "observation_coverage": {"more_pending": draining,
                            "omitted_on_overflow": omitted, "archive_preserved": social.archive is not None,
                            "text_truncated": any(r.get("text_truncated") for r in rows)},
                        "recent_bot_messages": [r for r in social.rows(group) if r.get("bot")]})
                    if not isinstance(result, dict) or result.get("action") not in {"wake", "skip"}:
                        raise ValueError("invalid observer decision")
                    wake = result["action"] == "wake"
                except Exception:
                    social.stats["judge_error"] += 1
                    state.errors += 1
                    state.retry_after = social.clock() + min(30, 2 ** min(state.errors, 5))
                    state.last_action = "error"
                    logger.warning("observer unavailable: group=%s batch=%d error_streak=%d", group, len(rows),
                                   state.errors)
                    # Transport/model errors are not social rejections. No score
                    # penalty, no false skip statistic, no unsolicited error send.
                    continue
                finally:
                    state.last_judge_seconds = max(0, social.clock() - state.checked)
                state.errors, state.retry_after = 0, 0
                state.last_action = result["action"]
                logger.info("observer verdict: group=%s action=%s messages=%d queue_seconds=%.3f judge_seconds=%.3f",
                            group, result["action"], len(rows), state.last_queue_seconds, state.last_judge_seconds)
                if unwoken:
                    state.result(wake, social.clock())
                social.stats["wake" if wake else "skip"] += 1
                if wake and unwoken and social.rule["mode"] == "active":
                    candidates = self.wakeups.setdefault(group, deque())
                    known = {str(i.data["message_id"]) for i in candidates}
                    for item in unwoken:
                        if str(item.data["message_id"]) in known or not social.guard_enabled_data(item.data):
                            continue
                        if len(candidates) >= MAX_WAKE_CANDIDATES:
                            candidates.popleft()
                            social.stats["wake_candidate_overflow"] += 1
                        candidates.append(item)
                        known.add(str(item.data["message_id"]))
                    if group not in self.wake_tasks or self.wake_tasks[group].done():
                        self.wake_tasks[group] = asyncio.create_task(self.deliver(group))
        finally:
            if self.tasks.get(group) is asyncio.current_task():
                self.tasks.pop(group, None)

    async def deliver(self, group):
        while group in self.wakeups:
            current = self.wakeups[group]
            retained = deque()
            for item in current:
                data = item.data
                if self.social.clock() - item.received_at >= WAKE_MAX_AGE_SECONDS:
                    self.social.stats["wake_candidate_expired"] += 1
                elif self.social.wakeup_handled(group, str(data["message_id"])):
                    self.social.stats["wake_candidate_already_handled"] += 1
                elif self.social.guard_enabled_data(data):
                    retained.append(item)
            self.wakeups[group] = retained
            if not retained:
                self.wakeups.pop(group, None)
                break
            if self.social.busy(group):
                await asyncio.sleep(.5)
                continue
            candidates = self.wakeups.pop(group)
            data, emit = candidates[-1].data, candidates[-1].emit
            if self.social.guard_enabled_data(data):
                hint = self.social.issue(group, str(data["user_id"]), "answer", passive=True,
                                         message_id=str(data["message_id"]), unmentioned=True)
                turn = self.social.turns[hint["token"]]
                turn["wake_only"] = True
                turn["candidate_requests"] = [{"message_id": str(i.data["message_id"]),
                    "user_id": str(i.data["user_id"]), "timestamp": i.data.get("time", 0), "direct": False}
                    for i in candidates]
                hint = self.social._turn_hint(hint["token"], turn)
                self.social.stats["passive_wakeup_batches"] += 1
                self.social.stats["passive_wakeup_candidates"] += len(candidates)
                outcome = await emit(data, False, hint, [])
                if outcome == "dropped" and self.social.busy(group):
                    # A direct invitation may claim the group while parsing
                    # awaits QQ metadata. Retain candidates with their original
                    # age instead of silently losing this race or resetting TTL.
                    pending = self.wakeups.get(group, deque())
                    merged = {str(i.data["message_id"]): i for i in (*candidates, *pending)}
                    values = sorted(merged.values(), key=lambda i: i.received_at)
                    self.wakeups[group] = deque(values[-MAX_WAKE_CANDIDATES:])
                    self.social.stats["passive_dispatch_race_retries"] += 1

    async def close(self):
        tasks = [*self.tasks.values(), *self.wake_tasks.values()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.tasks.clear()
        self.wake_tasks.clear()
        self.wakeups.clear()
