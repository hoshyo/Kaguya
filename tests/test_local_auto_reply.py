"""No-network contracts for the migrated group auto-reply policy."""
import json
from unittest.mock import AsyncMock, MagicMock

from onebot_adapter.config import AdapterConfig
from onebot_adapter.group_auto_reply import GroupAutoReplyPolicy
from onebot_adapter.onebot.handler import OneBotHandler
from onebot_adapter.relay.hermes_ws import HermesRelayServer
from onebot_adapter.relay.protocol import NormalizedEvent

RULE = {"enabled": True, "names": ["Kaguya", "辉夜"], "probability": 1,
        "cooldown_seconds": 90, "window_seconds": 600, "max_replies": 4}


def segments(text="今天的这个话题挺有意思"):
    return [{"type": "text", "data": {"text": text}}]


def policy():
    now = [0.0]
    gate = GroupAutoReplyPolicy({"42": RULE}, clock=lambda: now[0], sample=lambda: 0)
    return gate, now


def test_cooldown_budget_and_group_isolation():
    gate, now = policy()
    assert not gate.admit("43", segments())
    for t in (0, 90, 180, 270):
        now[0] = t
        assert gate.admit("42", segments())
        assert not gate.admit("42", segments())
    now[0] = 360
    assert not gate.admit("42", segments())
    now[0] = 600
    assert gate.admit("42", segments())


def test_activity_and_hot_reload_preserve_limits():
    gate, now = policy()
    gate.note_activity("42")
    gate.configure({"42": RULE})
    now[0] = 89
    assert not gate.admit("42", segments("辉夜"))
    now[0] = 90
    assert gate.admit("42", segments("辉夜"))


def test_noise_slashes_and_mentions_never_passive():
    for items in (segments("嗯"), segments("12345"), segments("/model"),
                  [{"type": "image", "data": {"file": "a"}}],
                  [{"type": "at", "data": {"qq": "100"}}, *segments()]):
        gate, _ = policy()
        assert not gate.admit("42", items)


async def test_handler_samples_before_model_and_explicit_at_bypasses_cooldown():
    config = AdapterConfig(self_id="999", group_default_enabled=False,
                           groups={"42": {"enabled": True}}, group_auto_reply={"42": RULE})
    received = AsyncMock()
    resolver = MagicMock()
    resolver.resolve_group_name = AsyncMock(return_value="test group")
    handler = OneBotHandler(label="test", config=config, api=MagicMock(),
                            on_event=received, name_resolver=resolver)
    data = {"post_type": "message", "message_type": "group", "group_id": 42,
            "user_id": 100, "message_id": 1, "time": 1700000000,
            "sender": {"card": "Tester", "nickname": "Tester"}, "message": segments()}
    await handler.handle_text(json.dumps(data))
    assert received.await_count == 1
    assert received.call_args.args[0].to_dict()["passive_reply"] is True
    data["message_id"] = 2
    await handler.handle_text(json.dumps(data))
    assert received.await_count == 1
    data["message_id"] = 3
    data["message"] = [{"type": "at", "data": {"qq": "999"}}, *segments()]
    await handler.handle_text(json.dumps(data))
    assert received.await_count == 2
    assert not received.call_args.args[0].passive_reply
    data["group_id"] = 43
    await handler.handle_text(json.dumps(data))
    assert received.await_count == 2


async def test_passive_busy_event_is_not_queued_or_buffered():
    relay = HermesRelayServer(AdapterConfig(), MagicMock(), adapter_version="test",
                              onebot_connected_fn=lambda: True)
    relay._store_hermes_mode(False)
    client = MagicMock()
    client.close = AsyncMock()
    relay._clients.add(client)
    relay._broadcast_event = AsyncMock()
    relay._busy_groups["42"] = ("100", 0)
    event = NormalizedEvent("1", "group:42", "group", "100", "Tester", "hi", passive_reply=True)
    assert await relay.push_event(event) == "dropped"
    assert not relay._ring_buffer
    assert not relay._queues
    relay._broadcast_event.assert_not_awaited()
    await relay.stop()
