"""Deterministic fake-model scheduling/transport checks; never call models or real QQ."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from onebot_adapter import adaptive_observer as scheduler
from onebot_adapter.message_archive import MessageArchive
from onebot_adapter.onebot.handler import OneBotHandler
from onebot_adapter.wakeup_batches import combine_wakeups
from tests.test_busy_wakeup_batches import finish, last_dispatch, runtime, wake  # noqa: F401
from tests.test_observer_agent_mode import cfg
from tests.test_social_chat import message, social


@pytest.fixture(autouse=True)
def short_collection_window(monkeypatch):
    monkeypatch.setattr(scheduler, "MICROBATCH_SECONDS", .001)


async def observe_all(s):
    await asyncio.wait_for(asyncio.gather(*list(s.adaptive.tasks.values())), 2)


async def deliver_all(s):
    await asyncio.wait_for(asyncio.gather(*list(s.adaptive.wake_tasks.values())), 2)


def offer(s, mid, *, user=100, group=42, text="虚构群聊", emit=None, already=False):
    data = message(mid, user=user, group=group, text=text)
    s.observe(data)
    if already:
        data = {**data, "_already_woken": True}
        s.adaptive.offer(data, emit or AsyncMock())
    else:
        s.offer(data, emit or AsyncMock())
    return data


@pytest.mark.parametrize("group", [42, 43])
async def test_hot_burst_is_one_call_preserving_all_authors_order_and_ids(group):
    s = social(cfg(), judge=AsyncMock(return_value={"action": "skip"}))
    try:
        for mid in range(1, 11):
            offer(s, mid, user=100 + mid, group=group)
        await observe_all(s)
        s.judge.assert_awaited_once()
        rows = s.judge.await_args.args[0]["messages"]
        assert [r["message_id"] for r in rows] == [str(i) for i in range(1, 11)]
        assert [r["user_id"] for r in rows] == [str(i + 100) for i in range(1, 11)]
        assert not s.adaptive.wakeups
    finally:
        await s.close()


async def test_due_backlog_is_split_by_message_and_character_budgets_without_cold_interval_wait():
    s = social(cfg(), judge=AsyncMock(return_value={"action": "skip"}))
    s.adaptive.state("42").score = .1
    try:
        for mid in range(1, 71):
            offer(s, mid, text="很长的虚构消息" * 2000)
        await observe_all(s)
        seen = []
        for call in s.judge.await_args_list:
            payload = call.args[0]
            rows = payload["messages"]
            assert len(rows) <= scheduler.MAX_BATCH_MESSAGES
            total_chars = sum(len(json.dumps(r, ensure_ascii=False)) for r in rows)
            assert total_chars <= scheduler.MAX_BATCH_CHARS or len(rows) == 1
            assert all(r["text"] == "很长的虚构消息" * 2000 for r in rows)
            assert not payload["observation_coverage"]["text_truncated"]
            seen.extend(r["message_id"] for r in rows)
        assert seen == [str(i) for i in range(1, 71)]
    finally:
        await s.close()


async def test_overflow_is_explicit_and_does_not_delete_received_archive():
    archive = MessageArchive()
    s = social(cfg(), archive=archive, judge=AsyncMock(return_value={"action": "skip"}))
    try:
        for mid in range(1, scheduler.MAX_PENDING + 12):
            offer(s, mid)
        assert len(s.adaptive.state("42").pending) == scheduler.MAX_PENDING
        assert archive.get("42", "1") is not None
        await observe_all(s)
        assert s.stats["observer_queue_overflow"] == 11
        assert s.judge.await_args_list[0].args[0]["observation_coverage"]["omitted_on_overflow"] == 11
        assert archive.coverage("42")["message_count"] == scheduler.MAX_PENDING + 11
    finally:
        await s.close()
        archive.close()


@pytest.mark.parametrize("action", ["wake", "skip"])
async def test_recent_member_is_observed_promptly_even_at_cold_tier_without_forcing_reply(action):
    now = [100.]
    s = social(cfg(), clock=lambda: now[0], judge=AsyncMock(return_value={"action": action}))
    state = s.adaptive.state("42")
    s.successful_reply("42", "100", "900", "今天吃什么？")
    state.score, state.checked = .1, now[0]
    emit = AsyncMock()
    try:
        offer(s, 1, text="汉堡", emit=emit)
        await observe_all(s)
        await deliver_all(s)
        s.judge.assert_awaited_once()
        assert s.judge.await_args.args[0]["recent_bot_messages"][-1]["text"] == "今天吃什么？"
        assert s.judge.await_args.args[0]["messages"][0]["recent_exchange_with_role"] is True
        assert emit.await_count == int(action == "wake")
        assert state.score <= .1  # A wake without an actual reply cannot manufacture engagement.
    finally:
        await s.close()


@pytest.mark.parametrize("group", [42, 43])
async def test_long_message_keeps_middle_and_tail_invitation_without_hiding_later_messages(group):
    s = social(cfg(), judge=AsyncMock(return_value={"action": "wake"}))
    emit = AsyncMock()
    text = "虚构活动安排，" * 500 + "中间也有重要信息。" + "交通方案，" * 4000 + "那你帮我查一下附近的活动"
    try:
        offer(s, 1, group=group, text=text, emit=emit)
        offer(s, 2, group=group, text="我说的是这周末", emit=emit)
        await observe_all(s)
        await deliver_all(s)
        rows = [r for call in s.judge.await_args_list for r in call.args[0]["messages"]]
        assert [r["message_id"] for r in rows] == ["1", "2"]
        assert rows[0]["text"] == text
        assert not any(r.get("text_truncated") for r in rows)
        assert s.adaptive.settings["max_message_chars"] is None
        assert emit.await_count >= 1
    finally:
        await s.close()


async def test_long_message_keeps_requests_after_256_segments_without_exposing_media_urls():
    s = social(cfg(), judge=AsyncMock(return_value={"action": "skip"}))
    try:
        data = message(1)
        data["message"] = [{"type": "text", "data": {"text": "备忘 "}} for _ in range(260)] + [
            {"type": "image", "data": {"url": "https://invalid.example/private-token"}},
            {"type": "text", "data": {"text": "那你帮我搜一下周末的活动"}}]
        s.observe(data)
        s.offer(data, AsyncMock())
        await observe_all(s)
        row = s.judge.await_args.args[0]["messages"][0]
        assert row["text"].endswith("[图片，未识别]那你帮我搜一下周末的活动")
        assert "private-token" not in row["text"] and not row.get("text_truncated")
    finally:
        await s.close()


async def test_recent_exchange_priority_expires_and_does_not_leak_to_other_users_or_groups():
    now = [100.]
    s = social(cfg(), clock=lambda: now[0], judge=AsyncMock(return_value={"action": "skip"}))
    try:
        s.successful_reply("42", "100", "900", "虚构回复")
        state = s.adaptive.state("42")
        state.score, state.checked = .1, now[0]
        offer(s, 1, user=200)
        await asyncio.sleep(.01)
        assert not s.judge.called and not state.priority(now[0])
        assert not s.adaptive.state("43").recent_members
        now[0] += scheduler.RECENT_EXCHANGE_SECONDS + 1
        offer(s, 2, user=100)
        await asyncio.sleep(.01)
        assert not s.judge.called and not state.priority(now[0])
    finally:
        await s.close()


async def test_personal_quiet_preference_beats_recent_exchange_priority():
    s = social(cfg(), judge=AsyncMock(return_value={"action": "wake"}))
    try:
        s.successful_reply("42", "100", "900", "今天吃什么？")
        s.preferences.set("42", "100", "mention_only")
        offer(s, 1, text="汉堡")
        assert not s.adaptive.tasks
        assert not s.judge.called
    finally:
        await s.close()


@pytest.mark.parametrize("result", [TimeoutError(), {"action": "wait"}, None])
async def test_observer_failure_is_not_a_skip_or_engagement_penalty(result):
    now = [100.]
    judge = AsyncMock(side_effect=result) if isinstance(result, Exception) else AsyncMock(return_value=result)
    s = social(cfg(), clock=lambda: now[0], judge=judge)
    state = s.adaptive.state("42")
    state.score, state.skips = 4., 2
    try:
        offer(s, 1)
        await observe_all(s)
        assert state.score == 4 and state.skips == 2
        assert state.errors == 1 and state.last_action == "error"
        assert state.retry_after == 102
        assert s.stats["judge_error"] == 1 and s.stats["skip"] == 0
        assert not s.adaptive.wakeups
    finally:
        await s.close()


async def test_mixed_batch_ending_in_already_woken_message_does_not_swallow_an_ordinary_wake():
    s = social(cfg(), judge=AsyncMock(return_value={"action": "wake"}))
    emit = AsyncMock()
    try:
        offer(s, 1, user=100, emit=emit)
        offer(s, 2, user=200, already=True)
        await observe_all(s)
        await deliver_all(s)
        assert s.judge.await_args.args[0]["current_message_id"] == "1"
        emit.assert_awaited_once()
        requests = emit.await_args.args[2]["candidate_requests"]
        assert [r["message_id"] for r in requests] == ["1"]
    finally:
        await s.close()


async def test_busy_main_retains_multiple_opportunities_and_dispatches_one_author_preserving_batch():
    busy = [True]
    s = social(cfg(), judge=AsyncMock(return_value={"action": "wake"}), busy=lambda _: busy[0])
    emit = AsyncMock()
    try:
        offer(s, 1, user=100, emit=emit)
        await observe_all(s)
        offer(s, 2, user=200, emit=emit)
        await observe_all(s)
        assert len(s.adaptive.wakeups["42"]) == 2 and not emit.called
        busy[0] = False
        await deliver_all(s)
        emit.assert_awaited_once()
        requests = emit.await_args.args[2]["candidate_requests"]
        assert [(r["message_id"], r["user_id"], r["direct"]) for r in requests] == [
            ("1", "100", False), ("2", "200", False)]
    finally:
        await s.close()


async def test_expired_handled_or_revoked_opportunities_are_not_sent():
    now, busy, handled = [100.], [True], set()
    s = social(cfg(), clock=lambda: now[0], busy=lambda _: busy[0],
               wakeup_handled=lambda g, m: (g, m) in handled,
               judge=AsyncMock(return_value={"action": "wake"}))
    emit = AsyncMock()
    try:
        offer(s, 1, user=100, emit=emit)
        offer(s, 2, user=200, emit=emit)
        offer(s, 3, user=300, emit=emit)
        await observe_all(s)
        handled.add(("42", "2"))
        s.preferences.set("42", "300", "mention_only")
        now[0] += scheduler.WAKE_MAX_AGE_SECONDS
        busy[0] = False
        await deliver_all(s)
        assert not emit.called and not s.adaptive.wakeups
        assert s.stats["wake_candidate_expired"] == 3
    finally:
        await s.close()


async def test_handled_and_quiet_candidates_are_filtered_independently_before_expiry():
    busy, handled = [True], set()
    s = social(cfg(), busy=lambda _: busy[0], wakeup_handled=lambda g, m: m in handled,
               judge=AsyncMock(return_value={"action": "wake"}))
    emit = AsyncMock()
    try:
        for mid in range(1, 4):
            offer(s, mid, user=100 + mid, emit=emit)
        await observe_all(s)
        handled.add("1")
        s.preferences.set("42", "102", "mention_only")
        busy[0] = False
        await deliver_all(s)
        emit.assert_awaited_once()
        assert [r["message_id"] for r in emit.await_args.args[2]["candidate_requests"]] == ["3"]
        assert s.stats["wake_candidate_already_handled"] == 1
    finally:
        await s.close()


async def test_busy_opportunities_are_bounded_without_automatic_extra_replies():
    s = social(cfg(), busy=lambda _: True, judge=AsyncMock(return_value={"action": "wake"}))
    emit = AsyncMock()
    try:
        for mid in range(1, 21):
            offer(s, mid, user=100 + mid, emit=emit)
        await observe_all(s)
        assert len(s.adaptive.wakeups["42"]) == scheduler.MAX_WAKE_CANDIDATES
        assert s.stats["wake_candidate_overflow"] == 20 - scheduler.MAX_WAKE_CANDIDATES
        assert not emit.called
    finally:
        await s.close()


async def test_direct_invitation_dispatches_without_waiting_for_the_observer_window():
    received = []
    s = social(cfg(), judge=AsyncMock(return_value={"action": "skip"}))
    handler = OneBotHandler(label="synthetic", config=cfg(), api=SimpleNamespace(),
        on_event=received.append, social_chat=s,
        name_resolver=SimpleNamespace(resolve_group_name=AsyncMock(return_value="虚构群")))
    try:
        data = message(1, segments=[{"type": "at", "data": {"qq": "999"}},
            {"type": "text", "data": {"text": "测试直接呼叫"}}])
        await handler.handle_event_text(json.dumps(data))
        assert len(received) == 1 and not received[0].passive_reply
        assert not s.judge.called  # Dispatch happens before the .25-second observation window.
        await observe_all(s)
        assert s.judge.await_args.args[0]["messages"][0]["already_woken"]
    finally:
        await s.close()


async def test_dispatch_busy_race_retains_candidate_instead_of_dropping_it():
    busy, calls = [False], []
    s = social(cfg(), busy=lambda _: busy[0], judge=AsyncMock(return_value={"action": "wake"}))

    async def emit(data, *args):
        calls.append(str(data["message_id"]))
        if len(calls) == 1:
            busy[0] = True
            asyncio.get_running_loop().call_later(.01, lambda: busy.__setitem__(0, False))
            return "dropped"
        return "broadcast"

    try:
        offer(s, 1, emit=emit)
        await observe_all(s)
        await deliver_all(s)
        assert calls == ["1", "1"] and not s.adaptive.wakeups
        assert s.stats["passive_dispatch_race_retries"] == 1
    finally:
        await s.close()


async def test_passive_batch_reaches_shared_relay_once_with_original_authors(runtime):  # noqa: F811
    relay = runtime
    s = relay._social_chat
    s.busy = lambda group: group in relay._busy_groups
    s.judge = AsyncMock(return_value={"action": "wake"})
    handler = OneBotHandler(label="synthetic", config=relay._config, api=SimpleNamespace(),
        on_event=relay.push_event, social_chat=s,
        name_resolver=SimpleNamespace(resolve_group_name=AsyncMock(return_value="虚构群")))
    await relay.push_event(wake(relay, 9, user=400))
    offer(s, 1, user=100, emit=handler._emit_social)
    offer(s, 2, user=200, emit=handler._emit_social)
    await observe_all(s)
    assert relay._broadcast_event.await_count == 1
    await finish(relay)
    await deliver_all(s)
    event = last_dispatch(relay)
    assert relay._broadcast_event.await_count == 2 and event.passive_reply
    assert [(r["message_id"], r["user_id"]) for r in event.wakeup_batch["requests"]] == [
        ("1", "100"), ("2", "200")]
    assert all(not r["direct"] for r in event.wakeup_batch["requests"])
    assert not s.turns[event.social_reply["token"]]["batch_invites"]


async def test_duplicate_success_receipts_do_not_repeatedly_raise_engagement():
    now = [100.]
    s = social(cfg(), clock=lambda: now[0])
    state = s.adaptive.state("42")
    state.score = 1.
    try:
        s.successful_reply("42", "100", "900", "虚构回复")
        assert state.score == 3.
        s.successful_reply("42", "100", "900", "虚构回复")
        assert state.score == 3.
    finally:
        await s.close()


@pytest.mark.parametrize("group", ["42", "43"])
async def test_successful_image_delivery_is_visible_to_observer_without_rewarding_extra_bubbles(group):
    now = [100.]
    archive = MessageArchive()
    s = social(cfg(), clock=lambda: now[0], archive=archive,
               judge=AsyncMock(return_value={"action": "wake"}))
    state = s.adaptive.state(group)
    state.score = 1.
    try:
        s.successful_reply(group, "100", "900", "画了一张银紫短发的娃")
        score = state.score
        now[0] += 40
        s.successful_reply(group, "100", "901", "", image=True)
        assert state.score < score  # Time decay only; an image is not a new text-participation reward.
        s.successful_reply(group, "100", "901", "", image=True)
        offer(s, 1, group=int(group), text="腿有点短了")
        await observe_all(s)
        await deliver_all(s)
        payload = s.judge.await_args.args[0]
        assert payload["messages"][0]["recent_exchange_with_role"] is True
        images = [r for r in payload["recent_bot_messages"] if r["message_id"] == "901"]
        assert len(images) == 1 and images[0]["text"] == "[已发送图片，观察者未识别画面]"
        assert archive.get(group, "901")["message"][0] == {"type": "image", "data": {}}
    finally:
        await s.close()
        archive.close()


async def test_passive_candidates_survive_handler_and_combining_without_borrowing_authority():
    received = []
    s = social(cfg())
    try:
        data = message(2, user=200)
        s.observe(message(1, user=100))
        s.observe(data)
        hint = s.issue("42", "200", "answer", passive=True, message_id="2")
        s.turns[hint["token"]]["candidate_requests"] = [
            {"message_id": "1", "user_id": "100", "timestamp": data["time"], "direct": False},
            {"message_id": "2", "user_id": "200", "timestamp": data["time"], "direct": False}]
        hint = s._turn_hint(hint["token"], s.turns[hint["token"]])
        handler = OneBotHandler(label="synthetic", config=cfg(), api=SimpleNamespace(),
            on_event=received.append, social_chat=s,
            name_resolver=SimpleNamespace(resolve_group_name=AsyncMock(return_value="虚构群")))
        await handler._emit_social(data, False, hint, [])
        received[0].is_admin = received[0].is_global_admin = True
        combined = combine_wakeups(received)
        assert [(r["message_id"], r["user_id"]) for r in combined.wakeup_batch["requests"]] == [
            ("1", "100"), ("2", "200")]
        assert all(not r["direct"] for r in combined.wakeup_batch["requests"])
        assert not combined.is_admin and not combined.is_global_admin
        s.prepare_wakeup_batch(combined)
        assert not s.turns[combined.social_reply["token"]]["batch_invites"]
        assert "不是所有人都分别在提问" in combined.text
        assert "明确承接你的新任务、追问或结果反馈应正常回应" in combined.text
    finally:
        await s.close()


def test_reported_policy_is_shared_not_copied_into_profiles():
    s = social(cfg())
    assert s.adaptive.settings["microbatch_seconds"] == scheduler.MICROBATCH_SECONDS
    assert s.adaptive.settings["max_pending"] == scheduler.MAX_PENDING
    assert s.adaptive.settings["engagement_reward"] == "successful_text_delivery"
    assert "profiles" not in s.adaptive.settings
