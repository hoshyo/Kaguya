"""Synthetic only: durable observer dispatch, adaptive pacing and agent-owned history."""
from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from onebot_adapter.adaptive_observer import Engagement
from onebot_adapter.group_history import GroupHistoryReader
from onebot_adapter.hermes_plugin import adapter as plugin
from onebot_adapter.hermes_plugin.delivery import normalize_plan
from onebot_adapter.hermes_plugin.observer_agent import INSTRUCTIONS
from onebot_adapter.message_archive import MessageArchive
from onebot_adapter.onebot.handler import OneBotHandler
from tests.test_group_history import params, row
from tests.test_social_chat import config, message, social


def cfg():
    return config(observer_agent_enabled=True, delivery_enabled=True).with_overrides(
        group_history={"42": {"enabled": True}})


def test_engagement_tiers_decay_and_recovery():
    for score, expected in ((8, 0), (5, 10), (3, 30), (1.5, 60), (.75, 120), (.1, 180)):
        state = Engagement(score=score, updated=100)
        assert state.interval(100) == expected
    state = Engagement(updated=100)
    for _ in range(9):
        state.result(False, 100)
    assert state.interval(100) == 10
    state.result(True, 100)
    assert state.interval(100) == 10  # A wake alone is not an actual conversation.
    state.replied("100", 100)
    assert state.interval(100) == 0
    assert state.interval(100 + 3600) == 180


async def test_active_tier_microbatches_every_message_without_old_followup_heuristics():
    judge = AsyncMock(return_value={"action": "wake"})
    s = social(cfg(), judge=judge, busy=lambda _: True)
    emit = AsyncMock()
    for i in range(5):
        data = message(i + 1, user=100 + i, text="不同人也允许由观察Agent判断")
        s.observe(data)
        s.offer(data, emit)
    await asyncio.gather(*list(s.adaptive.tasks.values()))
    assert judge.await_count == 1
    assert [r["message_id"] for r in judge.await_args.args[0]["messages"]] == list(map(str, range(1, 6)))
    assert all(c.args[0]["observer_agent"] for c in judge.await_args_list)
    assert len(s.adaptive.wakeups["42"]) == 5  # Bounded opportunities preserve individual authors.
    assert not emit.called
    await s.close()


async def test_cold_tier_retains_batch_and_explicit_call_interrupts_sleep():
    now = [100.0]
    s = social(cfg(), clock=lambda: now[0], judge=AsyncMock(return_value={"action": "skip"}))
    state = s.adaptive.state("42")
    state.score, state.checked = .1, now[0]
    for i in range(3):
        data = message(i + 1)
        s.observe(data)
        s.offer(data, AsyncMock())
    await asyncio.sleep(.01)
    assert not s.judge.called and len(state.pending) == 3
    s.adaptive.boost("42")
    await asyncio.gather(*list(s.adaptive.tasks.values()))
    assert s.judge.await_count == 1
    assert len(s.judge.await_args.args[0]["messages"]) == 3
    await s.close()


async def test_skip_never_wakes_main_agent():
    s = social(cfg(), judge=AsyncMock(return_value={"action": "skip"}))
    data, emit = message(), AsyncMock()
    s.observe(data)
    s.offer(data, emit)
    await asyncio.gather(*list(s.adaptive.tasks.values()))
    assert not emit.called and not s.turns
    await s.close()


async def test_explicit_invitation_is_observed_without_duplicate_wakeup():
    s = social(cfg(), judge=AsyncMock(return_value={"action": "wake"}))
    emit = AsyncMock()
    data = {**message(text="Kaguya今天吃什么"), "_already_woken": True}
    s.adaptive.offer(data, emit)
    await asyncio.gather(*list(s.adaptive.tasks.values()))
    assert s.judge.await_args.args[0]["messages"][0]["already_woken"] is True
    assert not emit.called and not s.adaptive.wakeups
    assert s.adaptive.state("42").interval(s.clock()) == 0
    await s.close()


async def test_notification_does_not_forward_original_body_or_recent_background():
    from onebot_adapter.relay.protocol import NormalizedEvent
    s = social(cfg())
    s.observe(message(text="PRIVATE_ORIGINAL_BODY"))
    emitted = []
    handler = OneBotHandler(label="test", config=cfg(), api=MagicMock(),
                            on_event=lambda event: emitted.append(event), social_chat=s)
    event = NormalizedEvent("1", "group:42", "group", "100", "虚构群友", "PRIVATE_ORIGINAL_BODY")
    handler._parse = AsyncMock(return_value=event)
    hint = s.issue("42", "100", "answer", passive=True, message_id="1")
    await handler._emit_social(message(), False, hint, [])
    wire = emitted[0].to_dict()
    assert wire["wake_only"] and "PRIVATE_ORIGINAL_BODY" not in wire["text"]
    assert not wire.get("recent_group_context") and not wire["media_items"]
    assert "先调用onebot_read_current_group_history" in plugin._channel_context(wire)
    await s.close()


async def test_native_observer_branch_is_not_auxiliary_completion(monkeypatch):
    observer = SimpleNamespace(judge=AsyncMock(return_value={"action": "wake"}))
    a = plugin.OneBotAdapter.__new__(plugin.OneBotAdapter)
    a._social_home, a._observer_sessions = None, observer
    a._ws = SimpleNamespace(closed=False, send_json=AsyncMock())
    monkeypatch.setattr(plugin, "social_judge", AsyncMock(side_effect=AssertionError("old stateless path")))
    await a._judge_social({"req_id": "synthetic", "payload": {"observer_agent": True, "group_id": "42"}})
    assert observer.judge.await_count == 1
    assert a._ws.send_json.await_args.args[0]["result"] == {"action": "wake"}
    assert '"action":"wake"' in INSTRUCTIONS and "禁止reply、replay、wait" in INSTRUCTIONS


def test_archive_is_persistent_group_scoped_and_supports_multiple_members(tmp_path):
    path = tmp_path / "messages.sqlite3"
    archive = MessageArchive(path)
    for i in range(1, 7):
        archive.record(row(i, user=100 if i % 2 else 200))
    archive.record(row(7, group=43))
    archive.close()
    archive = MessageArchive(path)
    assert len(archive.query("42", start=1001, end=1006, users=["100", "200"])) == 6
    assert len(archive.query("42", users=["100"], count=2)) == 2
    assert path.stat().st_mode & 0o777 == 0o600
    archive.close()


async def test_range_without_count_returns_all_matches_and_progress():
    archive = MessageArchive()
    api = MagicMock()
    api.call = AsyncMock(return_value={"data": {"messages": [row(i, user=100 if i % 2 else 200)
                                                             for i in range(1, 20)]}})
    r = GroupHistoryReader(cfg(), api, archive=archive)
    result = await r.read(params(start_time=1001, end_time=1019, user_ids=["100", "200"]))
    assert result["returned_count"] == 19 and result["last_read_time_before_query"] is None
    second = await r.read(params(start_time=1010, end_time=1018, user_ids=["200"]))
    assert second["returned_count"] == 5 and second["last_read_time_before_query"] == 1019
    assert all(row["user_id"] == "200" for row in second["messages"])
    for _ in range(4):
        assert (await r.read(params(start_time=1001, end_time=1001)))["returned_count"] == 1  # no 3-call budget
    archive.close()


@pytest.mark.parametrize("changes", [{}, {"count": True}, {"count": 0}, {"start_time": True},
                                    {"start_time": 2000, "end_time": 1000},
                                    {"count": 2, "user_ids": ["not-an-id"]},
                                    {"group_id": "43", "count": 2}])
async def test_invalid_query_never_reads_native_history(changes):
    api = MagicMock()
    api.call = AsyncMock()
    archive = MessageArchive()
    r = GroupHistoryReader(cfg(), api, archive=archive)
    with pytest.raises((ValueError, PermissionError)):
        await r.read(params(**changes))
    api.call.assert_not_called()
    archive.close()


async def test_cross_group_native_reply_is_rejected():
    api = MagicMock()
    api.call = AsyncMock(return_value={"data": {"messages": [row(1, group=43)]}})
    archive = MessageArchive()
    r = GroupHistoryReader(cfg(), api, archive=archive)
    with pytest.raises(PermissionError):
        await r.read(params(start_time=1001, end_time=1001))
    archive.close()


def test_history_reply_reference_remains_server_validated():
    s = social(cfg())
    s.observe(message())
    hint = s.issue("42", "100", "answer", passive=True, message_id="1000")
    result = {"messages": [{"message_id": "7", "user_id": "101", "user_name": "小林",
                            "timestamp": time.time(), "text": "虚构问题"}]}
    s.history_targets(params(), result)
    ref = result["messages"][0]["reply_ref"]
    plan = {"mode": "reply", "target": ref, "notify": True}
    assert normalize_plan(plan) == plan
    fields = s.delivery_fields("42", hint["token"], plan)
    assert fields["reply_to"] == "7" and fields["reply_mention_user_id"] == "101"
    with pytest.raises(PermissionError):
        s.delivery_fields("42", hint["token"], {**plan, "target": "h_" + "0" * 16})


def test_new_tool_parameters_are_not_group_or_session_selectors():
    from onebot_adapter.hermes_plugin.onebot_tools import _CURRENT_GROUP_HISTORY_SCHEMA
    fields = _CURRENT_GROUP_HISTORY_SCHEMA["parameters"]["properties"]
    assert {"start_time", "end_time", "user_ids", "keyword", "keywords", "keyword_mode",
            "match_mode", "limit", "before", "include_media"} == set(fields)
    assert not _CURRENT_GROUP_HISTORY_SCHEMA["parameters"]["required"]
    assert "group_id" not in fields
    assert json.dumps(fields)


def test_quoted_image_prompt_requires_media_and_vision_for_both_profiles():
    prompts = [plugin._channel_context({"wake_only": True, "profile": name}) for name in ("aidog", "kig")]
    assert prompts[0] == prompts[1]
    for prompt in prompts:
        assert "明确引用图片并提问" in prompt
        assert "查询必须带include_media=true" in prompt
        assert "应继续带include_media=true获取原图" in prompt
        assert "调用识图工具查看再答" in prompt
        assert "未请求媒体不等于取图失败" in prompt
        assert "已经看过同一张图时可复用" in prompt
        assert "自主调整查询范围定位" in prompt


def test_history_tool_description_does_not_present_placeholders_as_unavailable_media():
    from onebot_adapter.hermes_plugin.onebot_tools import _CURRENT_GROUP_HISTORY_SCHEMA
    schema = _CURRENT_GROUP_HISTORY_SCHEMA
    assert "图片语音只有占位" not in schema["description"]
    assert "不表示附件无法获取" in schema["description"]
    assert "明确引用图片提问" in schema["description"]
    assert "调用识图工具再答" in schema["description"]
    assert "必须设true" in schema["parameters"]["properties"]["include_media"]["description"]


def test_shared_image_generation_prompt_requires_real_image_delivery_not_code_substitution():
    for profile in ("aidog", "kig"):
        prompt = plugin._channel_context({"wake_only": True, "profile": profile})
        assert "有image_generate工具时，应实际调用" in prompt
        assert "不擅自改成ASCII文字图或HTML/SVG代码" in prompt
        assert "MEDIA:<实际图片路径>" in prompt
        assert "不读取、发送其他本机文件" in prompt
        assert "失败时不声称已生成或已发送" in prompt


async def test_history_media_is_current_group_only_and_raw_urls_are_not_tool_output():
    from onebot_adapter.hermes_plugin import onebot_tools
    raw = row(1)
    raw["message"] = [{"type": "image", "data": {"url": "https://qq.invalid/private-signed-url"}}]
    api = MagicMock()
    api.call = AsyncMock(side_effect=[{"data": {"messages": [raw]}}, {"data": raw}])
    archive = MessageArchive()
    r = GroupHistoryReader(cfg(), api, archive=archive)
    data = await r.read(params(start_time=1001, end_time=1001, include_media=True))
    assert data["messages"][0]["media_items"]
    callback = AsyncMock(return_value={"success": True, "data": data})
    cache = AsyncMock(return_value=(["/synthetic/cached-image.jpg"], ["image/jpeg"]))
    scope = {**params(), "active": True, "indexed_history": True, "query": callback,
             "cache_media": cache, "deadline": time.monotonic() + 60}
    token = plugin._history_turn.set(scope)
    try:
        result = await onebot_tools._read_current_group_history({
            "start_time": 1001, "end_time": 1001, "include_media": True})
    finally:
        plugin._history_turn.reset(token)
    assert "private-signed-url" not in result
    assert "cached-image.jpg" in result
    archive.close()


async def test_quoted_image_placeholder_can_be_queried_again_with_media():
    from onebot_adapter.hermes_plugin import onebot_tools
    image, question = row(1), row(2)
    image["message"] = [{"type": "image", "data": {"url": "https://qq.invalid/private-signed-url"}}]
    question["message"] = [{"type": "reply", "data": {"id": "1"}},
                           {"type": "at", "data": {"qq": "999"}},
                           {"type": "text", "data": {"text": "写的什么"}}]
    api = MagicMock()
    api.call = AsyncMock(side_effect=[{"data": {"messages": [image, question]}},
                                     {"data": {"messages": [image, question]}}, {"data": image}])
    archive = MessageArchive()
    reader = GroupHistoryReader(cfg(), api, archive=archive)

    async def query(action, arguments):
        assert action == "adapter_read_current_group_history"
        return {"success": True, "data": await reader.read(arguments)}

    cache = AsyncMock(return_value=(["/synthetic/referenced-image.jpg"], ["image/jpeg"]))
    token = plugin._history_turn.set({**params(), "active": True, "indexed_history": True,
        "query": query, "cache_media": cache, "deadline": time.monotonic() + 60})
    try:
        plain = await onebot_tools._read_current_group_history({"start_time": 1001, "end_time": 1002})
        assert "图片，未识别" in plain and "attachments" not in plain
        assert "reply_to_message_id" in plain
        cache.assert_not_awaited()
        with_media = await onebot_tools._read_current_group_history({
            "start_time": 1001, "end_time": 1002, "include_media": True})
    finally:
        plugin._history_turn.reset(token)
        archive.close()
    cache.assert_awaited_once()
    assert "referenced-image.jpg" in with_media
    assert "private-signed-url" not in with_media


async def test_native_pagination_can_return_more_than_old_100_message_limit():
    api = MagicMock()
    pages = [[row(i) for i in range(101, 151)], [row(i) for i in range(51, 101)],
             [row(i) for i in range(1, 51)], []]
    api.call = AsyncMock(side_effect=[{"data": {"messages": p}} for p in pages])
    archive = MessageArchive()
    r = GroupHistoryReader(cfg(), api, archive=archive)
    result = await r.read(params(start_time=1000, end_time=1200))
    assert result["returned_count"] == 150 and api.call.await_count == 4
    archive.close()


def test_direct_slash_commands_are_not_replaced_by_observer_notifications():
    from onebot_adapter.relay.protocol import NormalizedEvent
    s = social(cfg())
    event = NormalizedEvent("1", "group:42", "group", "100", "小林", "/stop")
    s.direct("42", event)
    assert event.text == "/stop" and not event.wake_only
