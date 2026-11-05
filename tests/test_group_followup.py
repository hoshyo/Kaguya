"""Short followups remain optional, bounded, and scoped to the real conversation."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from onebot_adapter.config import AdapterConfig
from onebot_adapter.group_followup import GroupFollowupTracker
from onebot_adapter.hermes_plugin.adapter import _conversation_reply_fields, _msg_context, _unmentioned_context
from onebot_adapter.onebot.handler import OneBotHandler
from onebot_adapter.relay.hermes_ws import HermesRelayServer

RULE = {"enabled": True, "names": ["Kaguya"], "probability": 0,
        "cooldown_seconds": 90, "window_seconds": 600, "max_replies": 4,
        "followup": {"enabled": True, "window_seconds": 120, "max_checks": 6,
                     "min_interval_seconds": 3, "budget_window_seconds": 120}}


def text(value="为什么"):
    return [{"type": "text", "data": {"text": value}}]


def tracker():
    now = [0.0]
    return GroupFollowupTracker({"42": RULE}, clock=lambda: now[0]), now


def test_same_sender_short_followup_and_expiry():
    gate, now = tracker()
    assert gate.classify("42", "100", text()) is None
    gate.record_reply("42", "100", "bot-1")
    now[0] = 1
    assert gate.classify("42", "100", text()) == "followup"
    assert gate.classify("42", "200", text()) is None
    assert gate.classify("43", "100", text()) is None
    now[0] = 120
    assert gate.classify("42", "100", text()) is None


def test_only_actual_reply_extends_window():
    gate, now = tracker()
    gate.record_reply("42", "100", "bot-1")
    now[0] = 110
    assert gate.classify("42", "100", text()) == "followup"
    now[0] = 121
    assert gate.classify("42", "100", text()) is None
    gate.record_reply("42", "100", "bot-2")
    assert gate.classify("42", "100", text()) == "followup"


def test_other_user_mentions_close_window():
    gate, _ = tracker()
    gate.record_reply("42", "100", "bot-1")
    assert gate.classify("42", "100", [{"type": "at", "data": {"qq": "200"}}, *text()]) == "blocked"
    assert gate.classify("42", "100", text()) is None


def test_quotes_are_bot_owned_and_group_scoped():
    gate, now = tracker()
    gate.record_reply("42", "100", "bot-1")
    reply = [{"type": "reply", "data": {"id": "bot-1"}}, *text()]
    assert gate.classify("42", "200", reply) == "followup"
    assert gate.classify("43", "200", reply) is None
    other = [{"type": "reply", "data": {"id": "other-user-message"}}, *text()]
    assert gate.classify("42", "100", other) == "blocked"
    assert gate.classify("42", "100", text()) is None
    now[0] = 120
    assert gate.classify("42", "200", reply) == "blocked"


def test_quota_survives_replies_and_config_refresh():
    gate, now = tracker()
    for i in range(6):
        now[0] = i * 3
        gate.record_reply("42", "100", f"bot-{i}")
        assert gate.classify("42", "100", text()) == "followup"
        assert gate.classify("42", "100", text()) == "blocked"
        gate.configure({"42": RULE})
    now[0] = 18
    gate.record_reply("42", "100", "bot-last")
    assert gate.classify("42", "100", text()) == "blocked"
    now[0] = 120
    assert gate.classify("42", "100", text()) == "followup"


@pytest.mark.parametrize("items", [text("/model"), text(""), [{"type": "image", "data": {}}]])
def test_non_conversational_inputs_do_not_run(items):
    gate, _ = tracker()
    gate.record_reply("42", "100", "bot-1")
    assert gate.classify("42", "100", items) == "blocked"


def make_handler():
    gate, now = tracker()
    cfg = AdapterConfig(self_id="999", group_default_enabled=False, group_auto_reply={"42": RULE},
                        groups={"42": {"enabled": True}})
    api = MagicMock()
    api.get_msg = AsyncMock(return_value={"message_id": "other", "sender": {"user_id": 200},
                                         "message": text("other")})
    resolver = MagicMock()
    resolver.resolve = AsyncMock(return_value="Tester")
    resolver.resolve_group_name = AsyncMock(return_value="Test group")
    received = AsyncMock()
    handler = OneBotHandler(label="test", config=cfg, api=api, name_resolver=resolver,
                            on_event=received, group_followups=gate)
    handler._auto_reply.note_activity("42")
    return handler, received, gate, now


async def deliver(handler, items=None, user=100, group=42):
    await handler.handle_text(json.dumps({
        "post_type": "message", "message_type": "group", "group_id": group, "user_id": user,
        "sender": {"card": "Tester"}, "message_id": 7, "time": 1700000000,
        "message": text() if items is None else items,
    }))


async def test_handler_prioritizes_short_followup_without_bypassing_admission():
    handler, received, gate, _ = make_handler()
    gate.record_reply("42", "100", "bot-1")
    await deliver(handler)
    received.assert_awaited_once()
    event = received.call_args.args[0]
    assert event.passive_reply and event.followup_reply
    assert event.to_dict()["followup_reply"]
    await deliver(handler, user=200)
    await deliver(handler, group=43)
    assert received.await_count == 1
    handler.update_config(handler._config.with_overrides(groups={"42": {"enabled": False}}))
    await deliver(handler)
    assert received.await_count == 1


async def test_reply_to_other_people_never_falls_back_to_random_sampling():
    handler, received, gate, _ = make_handler()
    gate.record_reply("42", "100", "bot-1")
    handler._auto_reply._sample = lambda: 0
    await deliver(handler, [{"type": "reply", "data": {"id": "other"}}, *text("Kaguya 这件事情怎么看")])
    received.assert_not_awaited()


def test_final_reply_metadata_excludes_notices_and_interim_sends():
    token = _msg_context.set((False, "42", "100", False))
    try:
        assert not _conversation_reply_fields(None)
        assert not _conversation_reply_fields({"notify": True, "_interim_send": True})
        assert _conversation_reply_fields({"notify": True}) == {
            "conversation_reply": True, "conversation_user_id": "100"}
    finally:
        _msg_context.reset(token)
    assert not _conversation_reply_fields({"notify": True})
    assert "NO_REPLY" in _unmentioned_context({"followup_reply": True})


async def test_only_successful_owned_final_sends_open_followup_window():
    gate, _ = tracker()
    api = MagicMock()
    api.send_group_msg = AsyncMock(return_value={"message_id": "bot-1"})
    relay = HermesRelayServer(AdapterConfig(), api, adapter_version="test", onebot_connected_fn=lambda: True,
                              group_followups=gate)
    relay._busy_groups["42"] = ("100", 0)
    ws = MagicMock()
    ws.send_json = AsyncMock()
    data = {"req_id": "test", "action": "send_text", "chat_id": "group:42", "content": "hello"}
    await relay._handle_send(ws, data)
    assert gate.classify("42", "100", text()) is None
    await relay._handle_send(ws, {**data, "content": "reply", "conversation_reply": True,
                                  "conversation_user_id": "100"})
    assert gate.classify("42", "100", text()) == "followup"
    other, _ = tracker()
    relay._group_followups = other
    api.send_group_msg.side_effect = RuntimeError("synthetic send failure")
    await relay._handle_send(ws, {**data, "content": "failed", "conversation_reply": True,
                                  "conversation_user_id": "100"})
    assert other.classify("42", "100", text()) is None
    api.send_group_msg.side_effect = None
    await relay._handle_send(ws, {**data, "content": "mismatched user", "conversation_reply": True,
                                  "conversation_user_id": "200"})
    assert other.classify("42", "200", text()) is None
    await relay.stop()
