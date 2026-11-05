"""Busy-only, notification-only batching. No models, real QQ or real records."""
from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from onebot_adapter.hermes_plugin import adapter as plugin
from onebot_adapter.hermes_plugin.delivery import DeliveryBatches, expression_prompt, parse_replies
from onebot_adapter.relay.protocol import NormalizedEvent
from onebot_adapter.send_receipts import SendReceipts
from onebot_adapter.wakeup_batches import combine_wakeups
from tests.test_incremental_context import make_adapter
from tests.test_plugin_queue_idle import _make_adapter
from tests.test_relay_queue import _make_relay
from tests.test_reply_mentions import reply_event
from tests.test_social_chat import config, message, social


@pytest.fixture
async def runtime():
    cfg = config(observer_agent_enabled=True, delivery_enabled=True)
    relay, _, _ = _make_relay()
    relay._config = cfg
    relay._batch_clients.update(relay._clients)
    relay._social_chat = social(cfg)
    relay._send_receipts = SendReceipts(":memory:")
    relay._broadcast_event = AsyncMock(return_value=True)
    relay._api = SimpleNamespace(send_group_msg=AsyncMock(return_value={"message_id": "9001"}))
    yield relay
    if relay._watchdog_task:
        relay._watchdog_task.cancel()
        await asyncio.gather(relay._watchdog_task, return_exceptions=True)
    if relay._text_tasks:
        await asyncio.gather(*relay._text_tasks, return_exceptions=True)
    await relay._social_chat.close()
    relay._send_receipts.close()


def wake(relay, mid, user=100, group="42", **kwargs):
    raw = message(mid, user=user, group=int(group), text="PRIVATE_ORIGINAL_" + str(mid))
    relay._social_chat.observe(raw)
    event = NormalizedEvent(str(mid), "group:" + group, "group", str(user), "虚构成员",
                            raw["message"][0]["data"]["text"],
                            timestamp=1000 + mid, **kwargs)
    relay._social_chat.direct(group, event)
    return event


async def finish(relay, group="42", *, success=True, batch_id=None):
    batch_id = batch_id or relay._active_wakeup_batches[group]["id"]
    await relay._handle_idle({"group_id": group, "wakeup_batch_id": batch_id, "wakeup_processed": success})
    await asyncio.sleep(0)
    await asyncio.sleep(0)


def last_dispatch(relay):
    return relay._broadcast_event.await_args.args[0]


async def busy_batch(relay):
    await relay.push_event(wake(relay, 1))
    await relay.push_event(wake(relay, 2, 200))
    await relay.push_event(wake(relay, 3, 300))
    await finish(relay)
    return last_dispatch(relay)


def sender_for(relay, event, monkeypatch):
    monkeypatch.setattr(plugin, "SendResult", lambda **kw: SimpleNamespace(
        **{"message_id": None, "error": None, "retryable": True, **kw}))
    adapter = make_adapter()
    adapter.MAX_MESSAGE_LENGTH = 4500
    adapter.truncate_message = lambda text, max_length: [text]
    adapter._social_policy = {"enabled": True, "delivery_enabled": True,
                              "part_delay_min_seconds": 0, "part_delay_max_seconds": 0}

    async def request(action, **payload):
        ws = SimpleNamespace(send_json=AsyncMock())
        await relay._handle_send(ws, {"action": action, "req_id": "synthetic", **payload})
        return ws.send_json.await_args.args[0]

    adapter._request = AsyncMock(side_effect=request)
    native_event = reply_event(user=event.user_id, group=event.chat_id[6:])
    native_event.message_id = event.message_id
    native_event.raw_message.update(event.to_dict())
    return adapter, native_event


def envelope(replies):
    return "[QQ_REPLIES]" + json.dumps({"replies": replies}, ensure_ascii=False) + "[/QQ_REPLIES]"


def target(event, mid):
    return next(t["ref"] for t in event.social_reply["delivery_targets"] if t["message_id"] == str(mid))


async def test_idle_dispatch_has_no_collection_timer_or_original_body(runtime):
    original = wake(runtime, 1)
    assert await runtime.push_event(original) == "broadcast"
    event = last_dispatch(runtime)
    assert runtime._broadcast_event.await_count == 1
    assert len(event.wakeup_batch["requests"]) == 1
    assert "PRIVATE_ORIGINAL" not in event.text
    assert event.recent_group_context == [] and event.media_items == []
    assert original.wakeup_batch == {}  # replay-buffer entry is not mutated
    assert "先调用onebot_read_current_group_history" in plugin._channel_context(event.to_dict())


async def test_five_waiting_people_are_one_next_wakeup(runtime):
    await runtime.push_event(wake(runtime, 1))
    queued = []
    for mid in range(2, 7):
        queued.append(wake(runtime, mid, user=100 + mid))
        assert await runtime.push_event(queued[-1]) == "queued"
    assert runtime._broadcast_event.await_count == 1
    await finish(runtime)
    event = last_dispatch(runtime)
    assert runtime._broadcast_event.await_count == 2
    assert [r["message_id"] for r in event.wakeup_batch["requests"]] == list(map(str, range(2, 7)))
    assert len({r["user_id"] for r in event.wakeup_batch["requests"]}) == 5
    assert len(event.delivery_ids) == 5
    assert "PRIVATE_ORIGINAL" not in event.text
    assert event.message_id == "2" and event.user_id == "102"
    assert not runtime._queues
    await finish(runtime)
    assert all(runtime._send_receipts.wakeup_handled("42", str(i)) for i in range(1, 7))


async def test_same_sender_while_busy_joins_next_batch_not_native_redirect(runtime):
    await runtime.push_event(wake(runtime, 1))
    assert await runtime.push_event(wake(runtime, 2)) == "queued"
    assert await runtime.push_event(wake(runtime, 3, 200)) == "queued"
    assert runtime._broadcast_event.await_count == 1
    await finish(runtime)
    assert [r["message_id"] for r in last_dispatch(runtime).wakeup_batch["requests"]] == ["2", "3"]


async def test_new_messages_during_second_batch_wait_for_third(runtime):
    second = await busy_batch(runtime)
    assert await runtime.push_event(wake(runtime, 4, 400)) == "queued"
    assert [r["message_id"] for r in second.wakeup_batch["requests"]] == ["2", "3"]
    await finish(runtime)
    assert [r["message_id"] for r in last_dispatch(runtime).wakeup_batch["requests"]] == ["4"]
    assert runtime._broadcast_event.await_count == 3


async def test_groups_remain_independent(runtime):
    await runtime.push_event(wake(runtime, 1))
    await runtime.push_event(wake(runtime, 2, 200))
    assert await runtime.push_event(wake(runtime, 3, 300, group="43")) == "broadcast"
    assert set(runtime._active_wakeup_batches) == {"42", "43"}
    await finish(runtime, "43")
    assert "42" in runtime._busy_groups and "43" not in runtime._busy_groups
    await finish(runtime)
    assert last_dispatch(runtime).chat_id == "group:42"


async def test_stale_or_unscoped_idle_cannot_release_current_batch(runtime):
    await runtime.push_event(wake(runtime, 1))
    old_id = runtime._active_wakeup_batches["42"]["id"]
    await runtime.push_event(wake(runtime, 2, 200))
    await finish(runtime)
    current = runtime._active_wakeup_batches["42"]["id"]
    await finish(runtime, batch_id=old_id)
    await runtime._handle_idle({"group_id": "42"})
    assert runtime._active_wakeup_batches["42"]["id"] == current
    assert not runtime._send_receipts.wakeup_handled("42", "2")


async def test_failed_processing_does_not_mark_wakeup_handled(runtime):
    await runtime.push_event(wake(runtime, 1))
    await finish(runtime, success=False)
    assert not runtime._send_receipts.wakeup_handled("42", "1")
    assert "42" not in runtime._busy_groups


async def test_duplicate_received_id_and_completed_replay_do_not_wake_again(runtime):
    event = wake(runtime, 1)
    await runtime.push_event(event)
    assert await runtime.push_event(replace(event, delivery_id="new", delivery_ids=["new"])) == "handled"
    await finish(runtime)
    assert await runtime.push_event(replace(event, delivery_id="replay", delivery_ids=["replay"])) == "handled"
    assert runtime._broadcast_event.await_count == 1


async def test_new_mentions_do_not_cancel_current_direct_answer_parts(runtime):
    await runtime.push_event(wake(runtime, 1))
    token = last_dispatch(runtime).social_reply["token"]
    await runtime.push_event(wake(runtime, 2, 200))
    assert runtime._social_chat.guard("42", token, 0)
    assert runtime._social_chat.guard("42", token, 1)
    await finish(runtime)
    assert not runtime._batch_current("42", token)


async def test_no_batch_capability_keeps_legacy_same_user_behavior(runtime):
    runtime._batch_clients.clear()
    await runtime.push_event(wake(runtime, 1))
    assert await runtime.push_event(wake(runtime, 2)) == "broadcast"
    assert last_dispatch(runtime).wakeup_batch == {}


async def test_permission_revoked_while_waiting_does_not_enter_batch(runtime):
    await runtime.push_event(wake(runtime, 1))
    await runtime.push_event(wake(runtime, 2, 200))
    await runtime.push_event(wake(runtime, 3, 300))
    runtime._social_chat.blacklist_match = lambda user, group: "blocked" if user == "200" else None
    await finish(runtime)
    assert [r["user_id"] for r in last_dispatch(runtime).wakeup_batch["requests"]] == ["300"]


def test_batch_does_not_elevate_other_members_to_admin():
    one = NormalizedEvent("1", "group:42", "group", "100", "甲", "通知", wake_only=True,
                          is_admin=True, is_global_admin=True)
    two = replace(one, message_id="2", user_id="200", is_admin=False, is_global_admin=False)
    batch = combine_wakeups([one, two])
    assert not batch.is_admin and not batch.is_global_admin
    with pytest.raises(ValueError):
        combine_wakeups([one, replace(two, chat_id="group:43")])


async def test_multiple_answers_each_quote_and_mention_only_own_first_part(runtime, monkeypatch):
    batch = await busy_batch(runtime)
    adapter, event = sender_for(runtime, batch, monkeypatch)
    text = envelope([{"mode": "reply", "target": target(batch, 2), "text": "给甲的回答\n\n给甲的补充"},
                     {"mode": "reply", "target": target(batch, 3), "text": "给乙的回答"}])
    scope = plugin._reply_event_context.set(event)
    try:
        assert (await adapter.send("group:42", text, metadata={"notify": True})).success
        calls = runtime._api.send_group_msg.await_args_list
        assert len(calls) == 3
        assert calls[0].args[1][0] == {"type": "reply", "data": {"id": "2"}}
        assert calls[0].args[1][1] == {"type": "at", "data": {"qq": "200"}}
        assert [s["type"] for s in calls[1].args[1]] == ["text"]
        assert calls[2].args[1][0] == {"type": "reply", "data": {"id": "3"}}
        assert calls[2].args[1][1] == {"type": "at", "data": {"qq": "300"}}
        assert all("QQ_REPLIES" not in str(c.args) and "QQ_SEND" not in str(c.args) for c in calls)
        assert (await adapter.send("group:42", text, metadata={"notify": True})).success
        assert runtime._api.send_group_msg.await_count == 3
    finally:
        plugin._reply_event_context.reset(scope)


async def test_explicit_reply_to_waiting_message_removes_its_later_wakeup(runtime, monkeypatch):
    await runtime.push_event(wake(runtime, 1))
    first = last_dispatch(runtime)
    await runtime.push_event(wake(runtime, 2, 200))
    runtime._social_chat.history_targets({"group_id": "42", "user_id": "100", "message_id": "1"},
        {"messages": [{"message_id": "2", "user_id": "200", "user_name": "乙", "text": "虚构问题"}]})
    ref = next(t["ref"] for t in runtime._social_chat.turns[first.social_reply["token"]]["targets"]
               if t["message_id"] == "2")
    adapter, event = sender_for(runtime, first, monkeypatch)
    scope = plugin._reply_event_context.set(event)
    try:
        result = await adapter.send("group:42", '[QQ_SEND]' + json.dumps(
            {"mode": "reply", "target": ref}) + '[/QQ_SEND]\n给乙的回答', metadata={"notify": True})
        assert result.success and runtime._send_receipts.wakeup_handled("42", "2")
        await finish(runtime)
        assert runtime._broadcast_event.await_count == 1
        assert "42" not in runtime._queues
    finally:
        plugin._reply_event_context.reset(scope)


async def test_batch_group_speech_does_not_force_quote_or_mention(runtime, monkeypatch):
    batch = await busy_batch(runtime)
    adapter, event = sender_for(runtime, batch, monkeypatch)
    scope = plugin._reply_event_context.set(event)
    try:
        result = await adapter.send("group:42", envelope([
            {"mode": "group", "text": "群聊接话"},
            {"mode": "reply", "target": target(batch, 3), "notify": False, "text": "需要定位但不提醒"}
        ]), metadata={"notify": True})
        assert result.success
        one, two = [c.args[1] for c in runtime._api.send_group_msg.await_args_list]
        assert [s["type"] for s in one] == ["text"]
        assert [s["type"] for s in two] == ["reply", "text"]
    finally:
        plugin._reply_event_context.reset(scope)


async def test_unknown_batch_target_fails_without_guessing_a_recipient(runtime, monkeypatch):
    batch = await busy_batch(runtime)
    adapter, event = sender_for(runtime, batch, monkeypatch)
    scope = plugin._reply_event_context.set(event)
    try:
        result = await adapter.send("group:42", envelope([
            {"mode": "reply", "target": "h_0000000000000000", "text": "不应发送"}
        ]), metadata={"notify": True})
        assert not result.success and not result.retryable
        assert not runtime._api.send_group_msg.called
    finally:
        plugin._reply_event_context.reset(scope)


async def test_mention_only_members_can_answer_own_explicit_calls_inside_batch(runtime, monkeypatch):
    runtime._social_chat.preferences.set("42", "300", "mention_only")
    batch = await busy_batch(runtime)
    adapter, event = sender_for(runtime, batch, monkeypatch)
    scope = plugin._reply_event_context.set(event)
    try:
        result = await adapter.send("group:42", envelope([
            {"mode": "reply", "target": target(batch, 3), "text": "回应乙自己的明确呼叫"}
        ]), metadata={"notify": True})
        assert result.success and runtime._api.send_group_msg.await_count == 1
    finally:
        plugin._reply_event_context.reset(scope)


async def test_changed_member_preference_invalidates_earlier_batch_invitation(runtime):
    batch = await busy_batch(runtime)
    social = runtime._social_chat
    social.preferences.set("42", "300", "mention_only")
    social.preference_versions[("42", "300")] = 1
    with pytest.raises(PermissionError):
        social.delivery_fields("42", batch.social_reply["token"], {"mode": "reply", "target": target(batch, 3)})


async def test_multiple_replies_not_enabled_for_single_wakeup(runtime, monkeypatch):
    await runtime.push_event(wake(runtime, 1))
    adapter, event = sender_for(runtime, last_dispatch(runtime), monkeypatch)
    scope = plugin._reply_event_context.set(event)
    try:
        result = await adapter.send("group:42", envelope([{"mode": "group", "text": "不应发送"}]),
                                    metadata={"notify": True})
        assert not result.success and not runtime._api.send_group_msg.called
    finally:
        plugin._reply_event_context.reset(scope)


@pytest.mark.parametrize("text", [
    '[QQ_REPLIES]{', '前言[QQ_REPLIES]{"replies":[]}[/QQ_REPLIES]',
    '[QQ_REPLIES]{"replies":[]}[/QQ_REPLIES]',
    envelope([{"mode": "reply", "target": "all", "text": "错误目标"}]),
    envelope([{"mode": "group", "text": "NO_REPLY"}]),
    envelope([{"mode": "group", "text": "MEDIA:/tmp/private.png"}]),
    envelope([{"mode": "group", "text": "[QQ_SEND]嵌套控制符"}]),
    envelope([{"mode": "group", "text": "正常"}, {"mode": "group", "text": None}]),
])
def test_invalid_multi_reply_is_rejected_before_sending_anything(text):
    with pytest.raises(ValueError):
        parse_replies(text)


def test_routing_state_is_independent_per_answer_but_retains_global_piece_budget():
    state = DeliveryBatches()
    a, b = {"mode": "reply", "target": "current"}, {"mode": "group"}
    assert state.reserve("token", "甲", a, 2, reply_id=0) == (0, a)
    assert state.reserve("token", "乙", b, 1, reply_id=1) == (0, b)
    assert state.reserve("token", "甲", a, 2, reply_id=0) == (0, a)
    with pytest.raises(ValueError):
        state.reserve("token", "太多", b, 18, reply_id=2)


async def test_batch_prompt_and_native_restore_preserve_notification_only_semantics(runtime):
    batch = await busy_batch(runtime)
    hint = batch.social_reply
    prompt = expression_prompt("原角色", hint["delivery_targets"], hint["conversation"], batch.wakeup_batch)
    assert "QQ_REPLIES" in prompt and "必要前因不足时仍需自行查记录" in prompt
    one = expression_prompt("原角色", hint["delivery_targets"], hint["conversation"])
    assert prompt == one  # batching changes new-turn data, not the system prefix
    assert "request_count大于1" in one
    event = reply_event(user=batch.user_id)
    event.raw_message = None
    event._onebot_social_reply = hint
    restored = plugin._context_event_data(event)
    assert restored["wake_only"] and restored["wakeup_batch"] == batch.wakeup_batch


@pytest.mark.parametrize("status,processed", [("SUCCESS", True), ("FAILURE", False)])
async def test_processing_hook_reports_batch_identity_and_success(runtime, status, processed):
    await runtime.push_event(wake(runtime, 1))
    batch = last_dispatch(runtime)
    adapter = _make_adapter()
    adapter._session_has_followup = lambda _: False
    event = reply_event(user=batch.user_id)
    event.message_id = batch.message_id
    event.raw_message.update(batch.to_dict())
    await adapter.on_processing_complete(event, SimpleNamespace(name=status))
    adapter._send_idle.assert_awaited_once_with("group:42", wakeup_batch_id=batch.wakeup_batch["id"],
                                               wakeup_processed=processed)


def test_processing_receipts_are_durable_group_scoped_and_not_read_cursors(tmp_path):
    path = tmp_path / "receipts.db"
    receipts = SendReceipts(path)
    receipts.mark_wakeups_handled("42", ["1"])
    receipts.close()
    receipts = SendReceipts(path)
    assert receipts.wakeup_handled("42", "1")
    assert not receipts.wakeup_handled("43", "1")
    assert not receipts.wakeup_handled("42", "2")
    receipts.close()
