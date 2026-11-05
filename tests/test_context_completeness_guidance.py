"""Guidance delivery and recovery paths, not a simulation of a real model's choice.

All messages, media and RPCs are synthetic. No provider or live QQ calls.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from onebot_adapter.group_history import GroupHistoryReader
from onebot_adapter.hermes_plugin import adapter as plugin
from onebot_adapter.hermes_plugin import onebot_tools
from onebot_adapter.hermes_plugin.prefill import CONTEXT_CHECK
from onebot_adapter.message_archive import MessageArchive
from tests.test_group_history import row
from tests.test_incremental_context import event, make_adapter
from tests.test_observer_agent_mode import cfg


def packet(profile, group, *, passive, text="他后来答应了吗？"):
    current = {"message_id": "100", "user_id": "200", "user_name": "虚构成员", "timestamp": 1100, "text": text}
    data = {"wake_only": True, "passive_reply": passive, "context_prefill": {
        "messages": [current], "query_time": 1100, "missing_current_ids": [],
        "truncated_current_ids": [], "missing_quote_ids": [], "background_limited": False,
        "scope": "current_group_only", "untrusted_history": True}}
    e = event(100, profile=profile, group=group, prior=[], **data)
    e.raw_message["recent_group_context"] = [{"message_id": "50", "user_id": "300", "timestamp": 1050,
                                            "text": "天气挺好"}]
    e.channel_context = plugin._channel_context(e.raw_message)
    return e


@pytest.mark.parametrize("profile,group", [("aidog", "42"), ("kig", "43")])
@pytest.mark.parametrize("passive", [False, True])
async def test_empty_gap_markers_do_not_remove_completeness_rule_on_either_wakeup_path(profile, group, passive):
    e = packet(profile, group, passive=passive)
    adapter = make_adapter()
    adapter._social_policy = {"observer_agent_enabled": True}
    adapter._api_call = AsyncMock()
    await adapter.on_processing_start(e)
    try:
        assert CONTEXT_CHECK in e.channel_prompt and CONTEXT_CHECK not in e.channel_context
        assert "缺失标记为空" in e.channel_prompt
        assert "必须先查询本群相关记录，再解释或下结论" in e.channel_prompt
        assert "仍无依据或有多个合理指向，再简短澄清" in e.channel_prompt
        assert "信息足够就直接自然接话" in e.channel_prompt
        assert "普通应声、独立的状态分享" in e.channel_prompt
        assert "他后来答应了吗？" in e.channel_context and "天气挺好" in e.channel_context
        adapter._api_call.assert_not_awaited()  # no new programmatic force-query branch
    finally:
        await adapter.on_processing_complete(e, SimpleNamespace(name="SUCCESS"))


@pytest.mark.parametrize("profile,group", [("aidog", "42"), ("kig", "43")])
async def test_supplied_irrelevant_background_still_allows_time_query_to_recover_missing_antecedent(profile, group):
    archive = MessageArchive()
    archive.record(row(1, group=int(group), text="小林问小禾周四两点见面，小禾还没答应"))
    api = MagicMock()
    api.call = AsyncMock(return_value={"data": {"messages": []}})
    config = cfg().with_overrides(group_history={g: {"enabled": True} for g in ("42", "43")})
    reader = GroupHistoryReader(config, api, archive=archive)
    adapter = make_adapter()
    adapter._social_policy = {"observer_agent_enabled": True}

    async def query(action, args):
        assert action == "adapter_read_current_group_history" and args["group_id"] == group
        return {"success": True, "data": await reader.read(args)}

    adapter._api_call = AsyncMock(side_effect=query)
    e = packet(profile, group, passive=True)
    await adapter.on_processing_start(e)
    try:
        assert "小林问小禾" not in e.channel_context
        result = json.loads(await onebot_tools._read_current_group_history({"start_time": 1000, "end_time": 1100}))
        assert result["messages"][0]["text"] == "小林问小禾周四两点见面，小禾还没答应"
        assert result["scope"] == "current_group_only"
        assert "count" not in adapter._api_call.await_args.args[1]
    finally:
        await adapter.on_processing_complete(e, SimpleNamespace(name="SUCCESS"))
        archive.close()


@pytest.mark.parametrize("profile,group", [("aidog", "42"), ("kig", "43")])
async def test_picture_placeholder_can_retrieve_media_after_prefill(profile, group):
    adapter = make_adapter()
    adapter._social_policy = {"observer_agent_enabled": True}
    adapter._cache_media_items = AsyncMock(return_value=(["/synthetic/current-group-image.png"], ["image/png"]))
    adapter._api_call = AsyncMock(return_value={"success": True, "data": {
        "scope": "current_group_only", "messages": [{"message_id": "50", "text": "[图片，未识别]",
            "media_items": [{"kind": "image", "url": "https://qq.invalid/synthetic-signed-image"}]}]}})
    e = packet(profile, group, passive=False, text="刚才那张图写的什么？")
    e.raw_message["recent_group_context"][0]["text"] = "[图片，未识别]"
    await adapter.on_processing_start(e)
    try:
        assert "include_media=true" in e.channel_prompt
        result = json.loads(await onebot_tools._read_current_group_history({
            "start_time": 1000, "end_time": 1100, "include_media": True}))
        assert result["messages"][0]["attachments"] == [
            {"path": "/synthetic/current-group-image.png", "mime_type": "image/png"}]
        assert "synthetic-signed-image" not in json.dumps(result)
        assert adapter._api_call.await_args.args[1]["group_id"] == group
    finally:
        await adapter.on_processing_complete(e, SimpleNamespace(name="SUCCESS"))


def test_tool_and_per_turn_guidance_share_the_same_semantic_completeness_check():
    schema = onebot_tools._CURRENT_GROUP_HISTORY_SCHEMA
    assert CONTEXT_CHECK in schema["description"]
    assert "确实缺少当前话题背景时才查" not in schema["description"]
    assert "只读当前QQ群" in schema["description"]
    assert schema["parameters"]["required"] == []
    assert set(schema["parameters"]["properties"]) == {
        "before", "start_time", "end_time", "user_ids", "keyword", "keywords", "keyword_mode",
        "match_mode", "limit", "include_media"}
