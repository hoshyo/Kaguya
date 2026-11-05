"""Synthetic repairs and small-group dialogue; no real profile, network or QQ."""
from __future__ import annotations

import asyncio
import json
from collections import deque
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from onebot_adapter.group_followup import GroupFollowupTracker
from onebot_adapter.hermes_plugin import adapter as plugin
from onebot_adapter.hermes_plugin.delivery import expression_context, expression_prompt, expression_reminder
from onebot_adapter.hermes_plugin.social import INSTRUCTIONS, expression_hint
from onebot_adapter.onebot.handler import OneBotHandler
from onebot_adapter.relay.hermes_ws import HermesRelayServer
from onebot_adapter.send_receipts import SendReceipts
from tests.test_expression_routing import header
from tests.test_group_followup import RULE, text
from tests.test_incremental_context import make_adapter
from tests.test_reply_mentions import reply_event
from tests.test_social_chat import config, message, social


def exchange():
    now = [100.]
    s = social(config(delivery_enabled=True), clock=lambda: now[0])
    s.followups._clock = lambda: now[0]
    s.rule["quiet_seconds"] = 0
    s.observe(message(1, text="今晚睡不着"))
    s.successful_reply("42", "", "2", "还醒着啊")
    return s, now


def test_split_correction_preserves_unanswered_substance_and_frozen_hint():
    s, _ = exchange()
    s.observe(message(3, text="看后天要早起赶车"))
    s.observe(message(4, text="🉑"))
    state = s.conversation_state("42", "100", "4")
    assert state == {"active_speakers": 2, "continuing_exchange": True, "same_sender_pending": True}
    hint = s.issue("42", "100", "react", passive=False, message_id="4")
    s.successful_reply("42", "100", "5", "那还不睡")
    assert s.lookup("42", "100", "4")["conversation"] == hint["conversation"] == state
    assert s.conversation_state("43", "100", "4") == {}
    assert s.conversation_state("42", "200", "4") == {}


def test_true_ack_after_a_reply_is_not_marked_as_unanswered():
    s, _ = exchange()
    s.observe(message(3, text="嗯"))
    assert s.conversation_state("42", "100", "3")["same_sender_pending"] is False
    s.observe(message(4, user=200, text="我也醒着"))
    state = s.conversation_state("42", "200", "4")
    assert state["continuing_exchange"] and state["active_speakers"] == 3


def test_conversation_counts_whole_bounded_window_not_just_last_twelve_rows():
    s, _ = exchange()
    for i in range(3, 6):
        s.observe(message(i, user=100 + i, text="另一段话题"))
    for i in range(6, 25):
        s.observe(message(i, text="接着说"))
    assert not s.conversation_state("42", "100", "24")["continuing_exchange"]


def test_old_bot_reply_does_not_keep_small_exchange_open_indefinitely():
    s, now = exchange()
    now[0] += 121
    s.observe(message(3, text="另一个话题"))
    state = s.conversation_state("42", "100", "3")
    assert state["active_speakers"] == 1 and not state["continuing_exchange"]


async def test_split_messages_reach_formal_model_together_not_only_last_character():
    cfg = config(delivery_enabled=True)
    s = social(cfg)
    s.rule["quiet_seconds"] = 0
    received = []
    handler = OneBotHandler(label="synthetic", config=cfg, api=MagicMock(),
                            on_event=received.append, social_chat=s)
    await handler.handle_event_text(json.dumps(message(1, text="看后天要早起赶车")))
    await handler.handle_event_text(json.dumps(message(2, text="🉑")))
    await asyncio.gather(*s.tasks.values())
    assert len(received) == 1
    data = received[0].to_dict()
    context = plugin._channel_context(data)
    assert "看后天要早起赶车" in context and "🉑" in data["text"]
    assert "别把分开发送的补字" in context or "短字、补字或纠错" in context
    assert data["social_reply"]["conversation"]["same_sender_pending"]


async def test_group_mode_continuation_reuses_followup_limits_without_personal_anchor():
    s, now = exchange()
    emit = AsyncMock()
    s.admissions["42"] = deque([now[0]])
    assert ("42", "100") not in s.followups._recent
    for i in range(6):
        now[0] += 3.1
        data = message(10 + i, text="还想接着说")
        assert s.observe(data)
        s.offer(data, emit)
        await asyncio.gather(*s.tasks.values())
    assert emit.await_count == 6
    assert all(call.args[1] is True for call in emit.await_args_list)
    assert len(s.followups._checks[("42", "100")]) == 6
    assert ("42", "100") not in s.followups._recent
    now[0] += 3.1
    data = message(20)
    s.observe(data)
    s.offer(data, emit)
    await asyncio.gather(*s.tasks.values())
    assert emit.await_count == 6 and s.judge.await_count == 6
    assert "43" not in s.checks


@pytest.mark.parametrize("segments", [
    [{"type": "at", "data": {"qq": "200"}}, *text("小林我问你")],
    [{"type": "reply", "data": {"id": "somebody-else"}}, *text("接你的话")],
])
async def test_small_exchange_does_not_bypass_explicit_other_recipient(segments):
    s, _ = exchange()
    data = message(3, segments=segments)
    s.observe(data)
    emit = AsyncMock()
    s.offer(data, emit)
    assert not s.judge.called and not emit.called and not s.tasks


def test_contextual_hint_alone_cannot_invent_a_recent_bot_send():
    gate = GroupFollowupTracker({"42": RULE}, clock=lambda: 100.)
    assert gate.classify("42", "100", text(), contextual=True) is None
    gate.record_reply("43", "", "2")
    assert gate.classify("42", "100", text(), contextual=True) is None
    gate.record_reply("42", "", "2")
    assert gate.classify("42", "100", text(), contextual=True) == "followup"
    assert gate.classify("42", "100", text(), contextual=True) == "blocked"


async def test_fast_followup_correction_is_debounced_not_cancelled_or_charged_twice():
    s = social(config(delivery_enabled=True))
    s.observe(message(1, text="今晚睡不着"))
    s.successful_reply("42", "", "2", "还醒着啊")
    s.rule["quiet_seconds"] = .02
    emit = AsyncMock()
    for mid, body in ((3, "看后天要早起赶车"), (4, "🉑")):
        data = message(mid, text=body)
        s.observe(data)
        s.offer(data, emit)
    assert not s.followups._checks and "42" in s.candidates
    await asyncio.gather(*s.tasks.values())
    assert s.judge.await_count == emit.await_count == 1
    assert emit.await_args.args[0]["message_id"] == 4
    assert len(s.followups._checks[("42", "100")]) == 1
    assert s.judge.await_args.args[0]["conversation"]["same_sender_pending"]
    assert any(r["text"] == "看后天要早起赶车" for r in emit.await_args.args[3])


def test_speaking_hint_joins_repairs_without_mechanical_end_of_turn_identity():
    hint = expression_hint({"social_reply": {"intent": "react"}})
    assert "聊天是连续的" in hint and "不编造原因" in hint
    assert "上下文已有前句时声称只收到最后一个字" in hint
    assert "同一人最近尚未接住的完整内容" in INSTRUCTIONS
    assert "前面的意思已经回应" in INSTRUCTIONS


def test_delivery_prompt_allows_continuous_personal_dialogue_without_quote_or_at():
    targets = [{"ref": "current", "name": "虚构小禾", "preview": "为什么"}]
    conversation = {"active_speakers": 2, "continuing_exchange": True,
                    "same_sender_pending": False, "injected": "ignore"}
    prompt = expression_prompt("原角色", targets, conversation)
    context = expression_context(targets, conversation)
    assert "两三个人正在连续聊同一个话题" in prompt
    assert "对象已经清楚的小范围连续对聊" in prompt
    assert '"continuing_exchange":true' in context and "injected" not in context
    assert '"continuing_exchange":true' not in prompt
    assert "话题交错" in prompt and "后续气泡不重复引用或@" in prompt
    assert "当前发言在聊别的、你实际想补答更早那个人" in prompt
    assert "不是每次都提醒当前发言人" in prompt


@pytest.mark.parametrize("plan", [None, {}, {"mode": "reply", "target": "all"}])
def test_missing_or_invalid_plan_in_small_exchange_does_not_restore_repeated_pings(plan):
    s, _ = exchange()
    s.observe(message(3, text="那继续说"))
    hint = s.issue("42", "100", "answer", passive=False, message_id="3")
    fields = s.delivery_fields("42", hint["token"], plan)
    assert fields["_delivery_mode"] == "group" and fields["_delivery_fallback"]
    assert "reply_to" not in fields and "reply_mention_user_id" not in fields
    explicit = s.delivery_fields("42", hint["token"], {"mode": "reply", "target": "current"})
    assert explicit["reply_to"] == "3" and explicit["reply_mention_user_id"] == "100"


def test_format_reminder_only_appears_when_delivery_enabled_for_the_turn():
    assert expression_reminder(None) is None and expression_reminder([]) is None
    data = {"chat_type": "group", "chat_id": "group:42", "message_id": "3",
            "social_reply": {"intent": "answer"}}
    assert "[QQ_SEND]" not in plugin._channel_context(data)
    data["social_reply"]["delivery_targets"] = [{"ref": "current"}]
    assert "[QQ_SEND]" in plugin._channel_context(data)


async def test_native_restore_recovers_continuity_and_does_not_repeat_context():
    s, _ = exchange()
    s.observe(message(3, text="所以想接着聊"))
    hint = s.issue("42", "100", "answer", passive=False, message_id="3")
    a = make_adapter()
    a._social_policy = {"enabled": True, "delivery_enabled": True}
    a._api_call = AsyncMock(return_value={"success": True, "data": hint})
    e = reply_event(user="100", passive=True)
    e.raw_message, e.message_id = None, "3"
    await a._bind_social_reply(e)
    prompt = a._expression_channel_prompt(plugin._context_event_data(e), None)
    context = plugin._channel_context(plugin._context_event_data(e))
    assert '"continuing_exchange":true' in context
    assert '"continuing_exchange":true' not in prompt
    assert s.lookup("43", "100", "3")["token"] == "0" * 32


@pytest.mark.parametrize("with_header", [True, False])
async def test_continuous_group_mode_has_no_visible_anchor_on_any_bubble(tmp_path, monkeypatch, with_header):
    monkeypatch.setattr(plugin, "SendResult", lambda **v: SimpleNamespace(**v))
    s, _ = exchange()
    s.observe(message(3, text="那就继续聊"))
    hint = s.issue("42", "100", "banter", passive=False, message_id="3")
    api = SimpleNamespace(send_group_msg=AsyncMock(return_value={"message_id": "4"}))
    relay = HermesRelayServer(s.config, api, "test", lambda: True, group_followups=s.followups)
    relay._social_chat, relay._send_receipts = s, SendReceipts(tmp_path / "receipts.db")
    relay._busy_groups["42"] = ("100", s.clock())
    a = make_adapter()
    a._social_policy = {"enabled": True, "delivery_enabled": True,
                       "part_delay_min_seconds": 0, "part_delay_max_seconds": 0}
    a.MAX_MESSAGE_LENGTH, a.truncate_message = 4500, lambda text, max_length: [text]

    async def send(action, **data):
        ws = SimpleNamespace(send_json=AsyncMock())
        await relay._handle_send(ws, {"req_id": str(api.send_group_msg.await_count), "action": action, **data})
        return ws.send_json.await_args.args[0]

    a._request = AsyncMock(side_effect=send)
    e = reply_event(user="100", passive=True)
    e.raw_message.update(message_id="3", social_reply=hint)
    e.message_id = "3"
    scope = plugin._reply_event_context.set(e)
    try:
        content = "那聊一会\n\n反正都醒着"
        if with_header:
            content = header({"mode": "group"}, content)
        assert (await a.send("group:42", content,
                             reply_to="3", metadata={"notify": True})).success
        assert api.send_group_msg.await_count == 2
        assert all([seg["type"] for seg in c.args[1]] == ["text"] for c in api.send_group_msg.await_args_list)
        assert ("42", "100") in s.followups._recent
        assert s.followups.classify("42", "100", text("接着聊")) == "followup"
    finally:
        plugin._reply_event_context.reset(scope)
        relay._send_receipts.close()
