"""Notify the real addressed sender once; never ping random topic participants."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from onebot_adapter.config import AdapterConfig
from onebot_adapter.hermes_plugin import adapter as plugin
from onebot_adapter.relay.hermes_ws import HermesRelayServer, _build_send_segments, _send_fingerprint
from tests.test_incremental_context import event, make_adapter


def reply_event(*, user="200", group="42", passive=False, followup=False, **flags):
    value = event(prior=[], group=group, passive_reply=passive, followup_reply=followup)
    value.source.user_id = user
    value.raw_message["user_id"] = user
    value.reply_expected = False if passive else None
    value.internal = False
    for key, item in flags.items():
        setattr(value, key, item)
    return value


@pytest.mark.parametrize(("passive", "followup", "mention"), [
    (False, False, True), (True, True, True), (True, False, False),
])
def test_only_addressed_and_followup_final_replies_mention(passive, followup, mention):
    token = plugin._reply_event_context.set(reply_event(passive=passive, followup=followup))
    try:
        fields = plugin._conversation_reply_fields({"notify": True}, "group:42")
        assert fields["conversation_user_id"] == "200"
        assert (fields.get("reply_mention_user_id") == "200") is mention
        for metadata in (None, {}, {"notify": False}, {"notify": True, "_interim_send": True}):
            assert not plugin._conversation_reply_fields(metadata, "group:42")
        assert not plugin._conversation_reply_fields({"notify": True}, "group:43")
        assert not plugin._conversation_reply_fields({"notify": True}, "200")
    finally:
        plugin._reply_event_context.reset(token)


@pytest.mark.parametrize("user", ["all", "0", "-200", "２００", "200[CQ:at,qq=all]", "2" * 21, ""])
def test_invalid_or_mass_mention_target_is_never_used(user):
    token = plugin._reply_event_context.set(reply_event(user=user))
    try:
        assert not plugin._conversation_reply_fields({"notify": True}, "group:42")
    finally:
        plugin._reply_event_context.reset(token)


def test_system_notices_and_commands_do_not_mention():
    value = reply_event(internal=True)
    token = plugin._reply_event_context.set(value)
    try:
        assert "reply_mention_user_id" not in plugin._conversation_reply_fields({"notify": True}, "group:42")
        value.internal = False
        value.text = "/help"
        assert "reply_mention_user_id" not in plugin._conversation_reply_fields({"notify": True}, "group:42")
    finally:
        plugin._reply_event_context.reset(token)


def test_native_redirect_can_promote_passive_turn_to_addressed_reply():
    value = reply_event(passive=True)
    token = plugin._reply_event_context.set(value)
    try:
        assert "reply_mention_user_id" not in plugin._conversation_reply_fields({"notify": True}, "group:42")
        # Hermes absorb_reply_expected promotes the same live event on @redirect.
        value.reply_expected = None
        assert plugin._conversation_reply_fields({"notify": True}, "group:42")["reply_mention_user_id"] == "200"
    finally:
        plugin._reply_event_context.reset(token)


@pytest.mark.parametrize("passive,followup,mention", [(False, False, True), (True, False, False), (True, True, True)])
def test_native_restored_event_uses_trusted_source_and_persisted_channel_context(passive, followup, mention):
    value = reply_event(passive=passive, followup=followup)
    value.raw_message = None
    token = plugin._reply_event_context.set(value)
    try:
        fields = plugin._conversation_reply_fields({"notify": True}, "group:42")
        assert (fields.get("reply_mention_user_id") == "200") is mention
    finally:
        plugin._reply_event_context.reset(token)


async def test_dispatch_and_native_pending_turn_rebind_actual_recipient():
    adapter = make_adapter()
    first = reply_event(user="100")
    second = reply_event(user="200")
    captured = []

    async def handle(value):
        captured.append(plugin._conversation_reply_fields({"notify": True}, value.source.chat_id))

    adapter.handle_message = AsyncMock(side_effect=handle)
    await adapter._dispatch_event(first, (False, "42", "100", False), [])
    assert captured[0]["reply_mention_user_id"] == "100"
    assert plugin._reply_event_context.get() is None
    # Native restored turns can execute in an existing task's admin context.
    ctx_token = plugin._msg_context.set((False, "42", "100", False))
    event_token = plugin._reply_event_context.set(first)
    try:
        await adapter.on_processing_start(second)
        fields = plugin._conversation_reply_fields({"notify": True}, "group:42")
        assert fields["reply_mention_user_id"] == fields["conversation_user_id"] == "200"
    finally:
        plugin._reply_event_context.reset(event_token)
        plugin._msg_context.reset(ctx_token)


async def test_concurrent_groups_do_not_share_reply_recipients():
    async def fields(user, group):
        token = plugin._reply_event_context.set(reply_event(user=user, group=group))
        try:
            await asyncio.sleep(0)
            return plugin._conversation_reply_fields({"notify": True}, "group:" + group)
        finally:
            plugin._reply_event_context.reset(token)

    first, second = await asyncio.gather(fields("100", "42"), fields("200", "43"))
    assert first["reply_mention_user_id"] == "100"
    assert second["reply_mention_user_id"] == "200"


async def test_long_response_mentions_first_chunk_only(monkeypatch):
    monkeypatch.setattr(plugin, "SendResult", lambda **values: SimpleNamespace(**values))
    adapter = make_adapter()
    adapter.MAX_MESSAGE_LENGTH = 10
    adapter.truncate_message = lambda text, max_length: ["第一段", "第二段", "第三段"]
    adapter._request = AsyncMock(return_value={"success": True, "message_id": "bot-1"})
    token = plugin._reply_event_context.set(reply_event())
    try:
        result = await adapter.send("group:42", "长回复", reply_to="51", metadata={"notify": True})
        assert result.success
    finally:
        plugin._reply_event_context.reset(token)
    payloads = [call.kwargs for call in adapter._request.await_args_list]
    assert payloads[0]["reply_mention_user_id"] == "200"
    assert payloads[0]["reply_to"] == "51"
    assert all("reply_mention_user_id" not in value and "reply_to" not in value for value in payloads[1:])


def frame(**changes):
    return {"req_id": "reply-test", "action": "send_text", "chat_id": "group:42", "reply_to": "51",
            "content": "怎么个控制法，你说来听听", "conversation_reply": True,
            "conversation_user_id": "200", "reply_mention_user_id": "200", **changes}


def test_group_reply_has_real_at_segment_not_a_textual_nickname():
    segments = _build_send_segments("send_text", frame())
    assert segments == [
        {"type": "reply", "data": {"id": "51"}},
        {"type": "at", "data": {"qq": "200"}},
        {"type": "text", "data": {"text": " "}},
        {"type": "text", "data": {"text": "怎么个控制法，你说来听听"}},
    ]


@pytest.mark.parametrize("changes", [
    {"chat_id": "200"}, {"conversation_reply": False}, {"conversation_user_id": "100"},
    {"reply_mention_user_id": "all", "conversation_user_id": "all"},
    {"reply_mention_user_id": "２００", "conversation_user_id": "２００"},
    {"reply_mention_user_id": "0", "conversation_user_id": "0"},
    {"reply_mention_user_id": 200}, {"reply_mention_user_id": None},
])
def test_untrusted_or_non_group_send_cannot_add_at(changes):
    assert all(seg["type"] != "at" for seg in _build_send_segments("send_text", frame(**changes)))


async def test_retry_does_not_send_duplicate_mention():
    api = MagicMock()
    api.send_group_msg = AsyncMock(return_value={"message_id": "bot-1"})
    relay = HermesRelayServer(AdapterConfig(send_dedup_enabled=True), api,
                              adapter_version="test", onebot_connected_fn=lambda: True)
    ws = MagicMock()
    ws.send_json = AsyncMock()
    try:
        await relay._handle_send(ws, frame())
        await relay._handle_send(ws, frame(req_id="retried"))
        api.send_group_msg.assert_awaited_once()
        assert api.send_group_msg.await_args.args[1][1] == {"type": "at", "data": {"qq": "200"}}
        assert _send_fingerprint("send_text", frame()) != _send_fingerprint(
            "send_text", frame(reply_mention_user_id=None))
    finally:
        await relay.stop()
