"""Voluntary textual +1 chains; fake verdicts/QQ only, no model requests."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from onebot_adapter.hermes_plugin import adapter as plugin
from onebot_adapter.hermes_plugin.delivery import expression_prompt
from onebot_adapter.hermes_plugin.observer_agent import INSTRUCTIONS
from onebot_adapter.hermes_plugin.prefill import GUIDANCE, render_context, verbatim_group_echo
from tests.test_expression_routing import header
from tests.test_incremental_context import make_adapter
from tests.test_observer_agent_mode import cfg
from tests.test_reply_mentions import reply_event
from tests.test_social_chat import message, social


def packet(text, *, bot=False, truncated=False):
    return {"chat_id": "group:42", "context_prefill": {"messages": [
        {"message_id": "1", "user_id": "100", "text": text, "bot": bot, "text_truncated": truncated}],
        "scope": "current_group_only"}, "recent_group_context": []}


def test_repeat_is_an_optional_exception_not_a_new_fixed_trigger_or_observer_action():
    for text in ("复读接龙是重复规则的例外", "三四条上下", "不固定第几条", "不逢三条/四条必触发",
                 "不要求每轮都参加", "单条本身就是好接的梗", "已加入这轮相同内容",
                 "不跟随一个人刷屏", "缺内容的媒体占位不算原话", "复读不扩大权限"):
        assert text in INSTRUCTIONS
    assert '只输出一个JSON对象' in INSTRUCTIONS
    assert '{"action":"wake"}' in INSTRUCTIONS and '{"action":"skip"}' in INSTRUCTIONS
    soul = (Path(__file__).resolve().parents[1] / "profiles/qq-observer/SOUL.md").read_text()
    assert "## 偶尔跟着复读" in soul and "不保证每轮参加" in soul
    assert "安静偏好" in soul and "不使用工具" in soul


def test_both_personas_get_the_same_verbatim_no_ping_instruction_without_new_routing_mode():
    root = Path(__file__).resolve().parents[1]
    capabilities = "\n".join((root / "profiles" / name / "SOUL.md").read_text() for name in ("aidog", "kig"))
    for text in ("不固定第几条", "一模一样的原句", "不引用不@", "不加引号", "隐私", "安静偏好"):
        assert capabilities.count(text) >= 2
    assert "已加入这轮就不再跟后续+1" in GUIDANCE
    prompt = expression_prompt("同一份发送规则", [{"ref": "current", "name": "甲", "preview": "彩蛋"}])
    assert "复读接龙时也选group" in prompt and "原句一次" in prompt
    assert "不改标点、大小写或原有表情" in prompt
    assert "mode=echo" not in prompt


@pytest.mark.parametrize("text", ["这也太巧了", "这也太巧了。", "**这也太巧了**。", "AＡ aA 🤣",
                                       "啊  啊", "第一行\n\n第二行"])
def test_literal_preservation_returns_original_text_not_formatted_or_rephrased(text):
    data = packet(text)
    assert verbatim_group_echo(text, data, self_id="999") == text
    assert verbatim_group_echo("我也来一个：" + text, data, self_id="999") is None


@pytest.mark.parametrize("text", ["NO_REPLY", "MEDIA:/not-a-file.png", "[图片，未识别]", "[语音，未转写]",
                                       "[QQ_SEND]", "/help", "x" * 601])
def test_literals_with_control_tags_media_placeholders_or_long_bodies_do_not_bypass_renderer(text):
    assert verbatim_group_echo(text, packet(text), self_id="999") is None


def test_copy_candidates_never_come_from_own_bot_text_truncated_or_arbitrary_unscoped_objects():
    assert verbatim_group_echo("**原句**", packet("**原句**", bot=True), self_id="999") is None
    assert verbatim_group_echo("**原句**", packet("**原句**", truncated=True), self_id="999") is None
    assert verbatim_group_echo("原句", packet("原句"), self_id="100") is None
    assert verbatim_group_echo("原句", {"messages": [{"text": "原句"}]}, self_id="999") is None
    assert verbatim_group_echo("原句", {"context_prefill": {"messages": None}}, self_id="999") is None


@pytest.mark.parametrize("group", [42, 43])
@pytest.mark.parametrize("text", ["**原样**。", "啊  啊 🤣", "第一行\n\n第二行"])
async def test_model_chosen_group_echo_is_one_exact_qq_text_bubble_without_quotes_or_mentions(monkeypatch, group, text):
    monkeypatch.setattr(plugin, "SendResult", lambda **v: SimpleNamespace(**v))
    adapter = make_adapter()
    adapter._self_id = "999"
    adapter.MAX_MESSAGE_LENGTH = 4500
    adapter.truncate_message = lambda value, max_length: [value]
    adapter._social_policy = {"enabled": True, "delivery_enabled": True}
    adapter._request = AsyncMock(return_value={"success": True, "message_id": "synthetic-sent"})
    event = reply_event(user="100", passive=True)
    event.raw_message.update(packet(text), chat_id=f"group:{group}")
    event.raw_message["social_reply"] = {"token": "a" * 32}
    scope = plugin._reply_event_context.set(event)
    try:
        result = await adapter.send(f"group:{group}", header({"mode": "group"}, text), metadata={"notify": True})
        assert result.success
    finally:
        plugin._reply_event_context.reset(scope)
    adapter._request.assert_awaited_once()
    args, sent = adapter._request.await_args
    assert args == ("send_text",)
    assert sent["content"] == text and sent["social_delivery"] == {"mode": "group", "target": None, "notify": False}
    assert "reply_to" not in sent and "reply_mention_user_id" not in sent


async def test_non_echo_reply_keeps_normal_markdown_formatting(monkeypatch):
    monkeypatch.setattr(plugin, "SendResult", lambda **v: SimpleNamespace(**v))
    adapter = make_adapter()
    adapter.MAX_MESSAGE_LENGTH = 4500
    adapter.truncate_message = lambda value, max_length: [value]
    adapter._request = AsyncMock(return_value={"success": True, "message_id": "synthetic"})
    await adapter.send("group:42", header({"mode": "group"}, "**并非原句**"))
    assert adapter._request.await_args.kwargs["content"] == "并非原句"


@pytest.mark.parametrize("group", [42, 43])
@pytest.mark.parametrize("count,action", [(1, "wake"), (3, "skip"), (4, "skip"), (5, "wake")])
async def test_repetition_count_never_overrides_luna_verdict(group, count, action):
    state = social(cfg(), judge=AsyncMock(return_value={"action": action}))
    state.adaptive.state(str(group)).score = .1  # inspect the cold batch without real timers/model calls
    emit = AsyncMock()
    try:
        for mid in range(1, count + 1):
            data = message(mid, user=100 + mid, group=group, text="这也太巧了")
            assert state.observe(data)
            state.offer(data, emit)
        await asyncio.gather(*list(state.adaptive.tasks.values()))
        await asyncio.gather(*list(state.adaptive.wake_tasks.values()))
        state.judge.assert_awaited_once()
        batch = state.judge.await_args.args[0]["messages"]
        assert len(batch) == count and len({r["message_id"] for r in batch}) == count
        assert {r["text"] for r in batch} == {"这也太巧了"}
        assert emit.await_count == int(action == "wake")
    finally:
        await state.close()


async def test_existing_bot_participation_is_visible_to_luna_and_other_groups_stay_separate():
    state = social(cfg(), judge=AsyncMock(return_value={"action": "skip"}))
    state.adaptive.state("42").score = .1
    state.successful_reply("42", "100", "900", "这也太巧了")
    emit = AsyncMock()
    try:
        data = message(10, user=101, text="这也太巧了")
        state.observe(data)
        state.offer(data, emit)
        await asyncio.gather(*list(state.adaptive.tasks.values()))
        payload = state.judge.await_args.args[0]
        assert payload["recent_bot_messages"][-1]["text"] == "这也太巧了"
        assert not state.rows("43")
        emit.assert_not_awaited()
    finally:
        await state.close()


async def test_quiet_preference_cannot_be_overridden_by_repeated_words():
    state = social(cfg(), judge=AsyncMock(return_value={"action": "wake"}))
    state.preferences.set("42", "100", "mention_only")
    emit = AsyncMock()
    try:
        for mid in range(1, 5):
            data = message(mid, text="这也太巧了")
            state.observe(data)
            state.offer(data, emit)
        state.judge.assert_not_awaited()
        emit.assert_not_awaited()
    finally:
        await state.close()


def test_repeated_originals_are_preserved_by_message_id_not_erased_as_duplicate_text():
    data = packet("原句")
    data["context_prefill"]["messages"] += [
        {"message_id": str(i), "user_id": str(100 + i), "text": "原句"} for i in (2, 3, 4)]
    rendered = render_context(data, {"1"})
    marker = rendered.split("<onebot_trigger_messages_json>")[1].split("</onebot_trigger_messages_json>")[0]
    originals = json.loads(marker)
    assert [row["message_id"] for row in originals] == ["2", "3", "4"]
    assert len({row["user_id"] for row in originals}) == 3
