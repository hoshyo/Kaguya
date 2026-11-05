"""Group/member consent and machine notices. Pure synthetic data; no home writes."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from onebot_adapter.hermes_plugin import adapter as plugin
from onebot_adapter.hermes_plugin.courtesy import internal_notice, preference_request
from onebot_adapter.member_preferences import MemberPreferences
from onebot_adapter.onebot.handler import OneBotHandler
from onebot_adapter.relay.hermes_ws import HermesRelayServer
from tests.test_incremental_context import make_adapter
from tests.test_social_chat import config, message, social

WARN = "⚠️ The model returned only a silence marker for a message that needed a reply. Try again or rephrase."
REVIEW = "💾 Self-improvement review: User profile updated · Memory updated"
HOME = "📬 No home channel is set for Onebot. A home channel is where Hermes delivers results.\nType /sethome"


def addressed(body, *, user=100, group=42, mid=1):
    return message(mid, user=user, group=group, segments=[{"type": "at", "data": {"qq": "999"}},
                                                        {"type": "text", "data": {"text": body}}])


@pytest.mark.parametrize("body", ["闭嘴", "没叫你别出来", "别随便插话", "大人说话，小孩别插嘴",
                                  "不要动不动主动发言", "禁止主动给我发消息",
                                  "别乱讲，只有让你说的时候才说，把这条写到记忆里"])
async def test_silence_request_is_processed_without_model_or_gateway_or_cascade(body):
    s = social()
    received, dropped = AsyncMock(), AsyncMock()
    handler = OneBotHandler(label="test", config=s.config, api=MagicMock(), social_chat=s,
                            on_event=received, on_dropped=dropped)
    await handler.handle_event_text(json.dumps(addressed(body)))
    assert s.preferences.mention_only("42", "100")
    assert not s.preferences.mention_only("43", "100")
    assert not s.preferences.mention_only("42", "200")
    assert not s.judge.called and not received.called and not dropped.called
    assert not s.tasks
    await s.close()


def test_preferences_persist_but_do_not_expand_to_other_members_or_groups(tmp_path):
    path = tmp_path / "prefs.sqlite3"
    first = MemberPreferences(path)
    first.set("42", "100", "mention_only")
    first.close()
    second = MemberPreferences(path)
    assert second.mention_only("42", "100")
    assert not second.mention_only("43", "100") and not second.mention_only("42", "200")
    second.set("42", "100", "normal")
    second.close()
    third = MemberPreferences(path)
    assert not third.mention_only("42", "100")
    third.close()


@pytest.mark.parametrize("body", ["他说闭嘴是什么意思", "告诉你别回复别人", "把其他群都关了",
                                  "他说不要主动给我发消息", "别再给大家发了，先回答这个问题"])
def test_quoted_or_third_person_text_does_not_edit_preferences(body):
    assert preference_request(body, addressed=True) is None


async def test_opted_out_user_is_not_judged_even_in_small_ongoing_exchange():
    s = social()
    s.preferences.set("42", "100", "mention_only")
    s.successful_reply("42", "100", "2", "前一条回复")
    data = message(3, text="今天打算摸一天")
    s.observe(data)
    s.offer(data, AsyncMock())
    assert not s.judge.called and not s.tasks
    assert s.stats["member_quiet_skip"] == 1
    await s.close()


@pytest.mark.parametrize("kind", ["at", "name", "quote"])
async def test_opted_out_member_can_still_explicitly_invite_a_reply(kind):
    s = social()
    s.preferences.set("42", "100", "mention_only")
    s.successful_reply("42", "", "20", "之前的回复")
    if kind == "at":
        data = addressed("帮我解释一下缓存", mid=21)
    elif kind == "name":
        data = message(21, text="Kaguya，帮我解释一下缓存")
    else:
        data = message(21, segments=[{"type": "reply", "data": {"id": "20"}},
                                     {"type": "text", "data": {"text": "这个是什么意思"}}])
    api = MagicMock()
    api.get_msg = AsyncMock(return_value={"message_id": 20, "group_id": 42, "user_id": 999,
                                          "sender": {"user_id": 999, "nickname": "Kaguya"},
                                          "message": [{"type": "text", "data": {"text": "之前的回复"}}]})
    received = []
    handler = OneBotHandler(label="test", config=s.config, api=api, on_event=received.append, social_chat=s)
    await handler.handle_event_text(json.dumps(data))
    assert len(received) == 1 and not received[0].passive_reply
    assert s.guard("42", received[0].social_reply["token"])
    assert s.preferences.mention_only("42", "100")  # one invitation does not revoke the preference
    await s.close()


async def test_quiet_request_revokes_old_direct_and_followup_drafts_only_for_its_author():
    s = social()
    direct = s.issue("42", "100", "answer", passive=False, message_id="1")["token"]
    followup = s.issue("42", "100", "answer", passive=False, message_id="1", unmentioned=True)["token"]
    other = s.issue("42", "200", "answer", passive=False, message_id="2")["token"]
    stop = addressed("闭嘴", mid=3)
    s.observe(stop)
    assert s.handle_preference(stop)
    assert not s.guard("42", direct) and not s.guard("42", followup)
    assert s.guard("42", other)
    new = s.issue("42", "100", "answer", passive=False, message_id="4")["token"]
    assert s.guard("42", new)
    await s.close()


async def test_other_people_cannot_indirectly_ping_opted_out_member():
    s = social(config(delivery_enabled=True))
    s.preferences.set("42", "100", "mention_only")
    s.observe(message(1, user=100))
    s.observe(message(2, user=200))
    hint = s.issue("42", "200", "answer", passive=False, message_id="2")
    target = next(t["ref"] for t in hint["delivery_targets"] if t["user_id"] == "100")
    with pytest.raises(PermissionError):
        s.delivery_fields("42", hint["token"], {"mode": "address", "target": target})
    await s.close()


async def test_member_can_reenable_only_their_own_proactive_replies():
    s = social()
    s.preferences.set("42", "100", "mention_only")
    s.preferences.set("42", "200", "mention_only")
    data = message(1, text="可以主动接我的话了")
    s.observe(data)
    assert s.handle_preference(data)
    assert not s.preferences.mention_only("42", "100")
    assert s.preferences.mention_only("42", "200")
    await s.close()


@pytest.mark.parametrize("notice", [WARN, REVIEW, HOME])
async def test_plugin_and_server_both_withhold_machine_notices_without_followup(notice, monkeypatch):
    monkeypatch.setattr(plugin, "SendResult", lambda **v: SimpleNamespace(**v))
    a = make_adapter()
    a._request = AsyncMock()
    assert (await a.send("group:42", notice, reply_to="1", metadata={"notify": True})).success
    assert not a._request.called
    s = social()
    api = SimpleNamespace(send_group_msg=AsyncMock())
    relay = HermesRelayServer(s.config, api, "test", lambda: True, group_followups=s.followups)
    relay._social_chat = s
    ws = SimpleNamespace(send_json=AsyncMock())
    await relay._handle_send(ws, {"action": "send_text", "chat_id": "group:42", "content": notice,
                                  "req_id": "test", "conversation_reply": True, "conversation_user_id": "100"})
    assert ws.send_json.await_args.args[0]["success"]
    assert not api.send_group_msg.called and not s.followups._outbox
    s.successful_reply("42", "100", "2", notice)
    assert not s.followups._outbox and not s.observer._buffers.get("42")
    await s.close()


@pytest.mark.parametrize("body", ["Hello, good morning", "⚠️ Request failed: upstream timeout",
                                  "Self-improvement review 是什么意思？", "这段英文提到了 silence marker"])
async def test_normal_english_explanations_and_real_failures_are_not_blindly_filtered(body, monkeypatch):
    monkeypatch.setattr(plugin, "SendResult", lambda **v: SimpleNamespace(**v))
    assert not internal_notice(body)
    a = make_adapter()
    a._request = AsyncMock(return_value={"success": True, "message_id": "1"})
    a.MAX_MESSAGE_LENGTH, a.truncate_message = 4500, lambda text, max_length: [text]
    assert (await a.send("group:42", body)).success
    assert a._request.await_count == 1


async def test_inflight_judgment_is_rechecked_after_member_opts_out():
    s = social()
    s.rule["quiet_seconds"] = 0

    async def change_preference(_):
        s.preferences.set("42", "100", "mention_only")
        return {"action": "reply", "intent": "react"}

    s.judge = change_preference
    data = message(1)
    s.observe(data)
    emit = AsyncMock()
    s.offer(data, emit)
    await asyncio.gather(*s.tasks.values())
    assert not emit.called
    await s.close()
