"""Synthetic explicit invitations; no production history, models or QQ sends."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from onebot_adapter.group_invitation import GroupInvitations
from onebot_adapter.hermes_plugin import adapter as plugin
from onebot_adapter.onebot.handler import OneBotHandler
from tests.test_social_chat import config, message, social


@pytest.fixture(params=["42", "43"])
async def pipeline(request):
    now, received = [1000.], []
    cfg = config(delivery_enabled=True)
    denied = set()
    def blacklist(user, group):
        return "synthetic denial" if user in denied else None
    s = social(cfg, clock=lambda: now[0], blacklist_match=blacklist)
    s.followups._clock = lambda: now[0]
    s.rule["quiet_seconds"] = 0
    s.judge.return_value = {"action": "skip", "intent": "answer"}
    resolver = SimpleNamespace(resolve_group_name=AsyncMock(return_value="虚构群"))
    handler = OneBotHandler(label="synthetic", config=cfg, api=MagicMock(), name_resolver=resolver,
                            on_event=received.append, social_chat=s, bot_blacklist_match_fn=blacklist)
    handler._invitations = GroupInvitations(clock=lambda: now[0])

    async def send(mid, text="这是什么意思", *, mention=False, user=100, group=None, segments=None):
        if segments is None:
            segments = ([{"type": "at", "data": {"qq": "999"}}] if mention else [])
            segments += [{"type": "text", "data": {"text": text}}]
        data = message(mid, user=user, group=int(group or request.param), segments=segments)
        await handler.handle_event_text(json.dumps(data))
        await asyncio.gather(*s.tasks.values())
        return data

    try:
        yield SimpleNamespace(group=request.param, clock=now, received=received,
                              social=s, handler=handler, send=send, denied=denied)
    finally:
        await s.close()


async def test_question_then_separate_mention_is_one_direct_request(pipeline):
    p = pipeline
    await p.send(1, "短线是什么")
    assert not p.received
    p.clock[0] += 6
    await p.send(2, " ", mention=True)
    assert len(p.received) == 1
    e = p.received[0]
    assert e.message_id == "2" and "短线是什么" in e.text
    assert e.context_message_ids == ["1", "2"]
    assert not e.passive_reply and not e.followup_reply
    assert p.social.judge.await_count == 1
    context = plugin._channel_context(e.to_dict())
    assert set(plugin._context_current_ids(e.to_dict())) == {"1", "2"}
    assert "onebot_current_ids_json" in context


async def test_separate_mention_then_question_waits_for_actual_content(pipeline):
    p = pipeline
    await p.send(1, " ", mention=True)
    assert not p.received and not p.social.judge.called
    p.clock[0] += 6
    await p.send(2, "短线是什么")
    assert len(p.received) == 1 and "短线是什么" in p.received[0].text
    assert p.received[0].context_message_ids == ["1", "2"]
    assert not p.received[0].passive_reply and not p.social.judge.called


@pytest.mark.parametrize("first_is_mention", [True, False])
async def test_pairing_expires_without_creating_a_reply(first_is_mention, pipeline):
    p = pipeline
    await p.send(1, " " if first_is_mention else "这是什么", mention=first_is_mention)
    p.clock[0] += 31
    await p.send(2, "这是什么" if first_is_mention else " ", mention=not first_is_mention)
    assert not p.received


async def test_pending_mention_never_binds_another_member_or_group(pipeline):
    p = pipeline
    await p.send(1, " ", mention=True)
    await p.send(2, "这是什么", user=200)
    other = "43" if p.group == "42" else "42"
    await p.send(3, "这是什么", group=other)
    assert not p.received
    await p.send(4, "自己的问题")
    assert len(p.received) == 1 and p.received[0].context_message_ids == ["1", "4"]


@pytest.mark.parametrize("segments", [
    [{"type": "at", "data": {"qq": "200"}}, {"type": "text", "data": {"text": "我问你"}}],
    [{"type": "reply", "data": {"id": "500"}}, {"type": "text", "data": {"text": "接他的话"}}],
    [{"type": "text", "data": {"text": "/model"}}],
    [{"type": "image", "data": {"file": "not-fetched"}}],
])
async def test_other_address_commands_and_unsupported_media_cancel_pending(pipeline, segments):
    p = pipeline
    await p.send(1, " ", mention=True)
    await p.send(2, segments=segments)
    await p.send(3, "再说一句")
    assert not p.received


@pytest.mark.parametrize("passive", [True, False])
async def test_already_dispatched_question_is_not_replayed(pipeline, passive):
    p = pipeline
    if passive:
        p.social.judge.return_value = {"action": "reply", "intent": "answer"}
    await p.send(1, "这是什么", mention=not passive)
    assert len(p.received) == 1
    await p.send(2, " ", mention=True)
    assert len(p.received) == 1


async def test_duplicate_bare_mention_does_not_repeat_or_rearm_invitation(pipeline):
    p = pipeline
    await p.send(1, "这是什么")
    await p.send(2, " ", mention=True)
    assert len(p.received) == 1
    await p.send(2, " ", mention=True)
    await p.send(3, "另一个普通消息")
    assert len(p.received) == 1


async def test_member_quiet_mode_still_accepts_their_own_explicit_invitation(pipeline):
    p = pipeline
    p.social.preferences.set(p.group, "100", "mention_only")
    await p.send(1, " ", mention=True)
    await p.send(2, "这是什么")
    assert len(p.received) == 1 and not p.received[0].passive_reply


async def test_quiet_request_after_bare_mention_cancels_pending(pipeline):
    p = pipeline
    await p.send(1, " ", mention=True)
    await p.send(2, "别主动给我发消息")
    await p.send(3, "这是普通聊天")
    assert not p.received


async def test_configuration_revocation_clears_pending_invitation(pipeline):
    p = pipeline
    await p.send(1, " ", mention=True)
    cfg = p.handler._config.with_overrides(groups={p.group: {"enabled": False}})
    p.handler.update_config(cfg)
    p.social.configure(cfg)
    await p.send(2, "这是什么")
    assert not p.received and not p.handler._invitations.pending


async def test_pending_invitation_does_not_override_user_allowlist(pipeline):
    p = pipeline
    await p.send(1, " ", mention=True)
    cfg = p.handler._config.with_overrides(groups={p.group: {
        "enabled": True, "group_user_filter_mode": "whitelist", "group_user_list": ["200"]}})
    p.handler.update_config(cfg)
    p.social.configure(cfg)
    await p.send(2, "这是什么")
    assert not p.received


async def test_pending_invitation_does_not_override_dynamic_blacklist(pipeline):
    p = pipeline
    cfg = p.handler._config.with_overrides(bot_blacklist_enabled=True)
    p.handler.update_config(cfg)
    p.social.configure(cfg)
    await p.send(1, " ", mention=True)
    p.denied.add("100")
    await p.send(2, "这是什么")
    assert not p.received


def test_invitation_cache_is_bounded_and_malformed_input_is_ignored():
    now = [100.]
    invitations = GroupInvitations(clock=lambda: now[0])
    for i in range(1100):
        data = message(i + 1, user=i + 1000, text="虚构问题")
        invitations.consider(data, self_id="999", reason="trigger", observed=True)
    assert len(invitations.recent) == invitations.MAXIMUM
    invitations.consider(message(2000, segments=[{"type": "at", "data": None}]),
                         self_id="999", reason="empty", observed=True)
    now[0] += 31
    invitations._prune()
    assert not invitations.recent
