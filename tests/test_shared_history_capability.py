"""Shared current-group behavior with separate profile/group capabilities."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from onebot_adapter.group_history import GroupHistoryReader
from onebot_adapter.hermes_plugin import adapter as plugin
from onebot_adapter.hermes_plugin import onebot_tools
from tests.test_group_history import config, row
from tests.test_incremental_context import event, make_adapter


@pytest.mark.parametrize("profile,group", [("aidog", "42"), ("kig", "43")])
async def test_shared_guidance_is_in_current_turn_and_latest_read_is_scoped(profile, group):
    a = make_adapter()
    api = MagicMock()
    api.call = AsyncMock(return_value={"data": {"messages": [row(2, group=int(group), text="最新的群友讨论")]}})
    reader = GroupHistoryReader(config(), api)

    async def query(action, args):
        assert action == "adapter_read_current_group_history"
        return {"success": True, "data": await reader.read(args)}

    a._api_call = query
    e = event(profile=profile, group=group, prior=[])
    e.source.platform = "onebot"
    await a._dispatch_with_incremental_context(e)
    await a.on_processing_start(e)
    assert plugin._CURRENT_GROUP_HISTORY_GUIDANCE in e.channel_context
    assert "已有会话只包含你参与过的部分" in e.channel_context
    result = json.loads(await asyncio.to_thread(
        lambda: asyncio.run(onebot_tools._read_current_group_history({}))))
    assert result["messages"][0]["text"] == "最新的群友讨论"
    assert api.call.await_args.args[1]["group_id"] == group
    assert api.call.await_args.args[1]["count"] == 20
    assert api.call.await_args.args[1]["message_seq"] == "0"
    await a.on_processing_start(e)  # redirects/restores must not stack the same hint
    assert e.channel_context.count(plugin._CURRENT_GROUP_HISTORY_GUIDANCE) == 1
    await a.on_processing_complete(e, SimpleNamespace(name="SUCCESS"))
    assert "error" in json.loads(await onebot_tools._read_current_group_history({}))


@pytest.mark.parametrize("profile,group", [("aidog", "42"), ("kig", "43")])
async def test_ambiguous_what_does_this_mean_can_read_newer_others_messages(profile, group):
    """Verify the tool path, not pretend a mocked LLM proves semantic choice."""
    a = make_adapter()
    api = MagicMock()
    api.call = AsyncMock(return_value={"data": {"messages": [
        row(20, group=int(group), text="刚才他们说重试要用指数退避"),
        row(21, group=int(group), text="这是什么意思"),
    ]}})
    reader = GroupHistoryReader(config(), api)

    async def query(action, args):
        assert action == "adapter_read_current_group_history"
        return {"success": True, "data": await reader.read(args)}

    a._api_call = query
    e = event(profile=profile, group=group, prior=[])
    e.text = "这是什么意思"
    e.source.platform = "onebot"
    await a._dispatch_with_incremental_context(e)
    await a.on_processing_start(e)
    try:
        assert "对方问‘这是什么意思’" in e.channel_context
        assert "原话不在上下文或背景不足时" in e.channel_context
        assert "已有明确原文或引用时直接结合它回答" in e.channel_context
        assert "指数退避" not in e.channel_context  # no automatic rolling history injection
        result = json.loads(await asyncio.to_thread(
            lambda: asyncio.run(onebot_tools._read_current_group_history({}))))
        assert result["messages"][0]["text"] == "刚才他们说重试要用指数退避"
        params = api.call.await_args.args[1]
        assert params["group_id"] == group and params["count"] == 20 and params["message_seq"] == "0"
    finally:
        await a.on_processing_complete(e, SimpleNamespace(name="SUCCESS"))


def test_shared_hint_preserves_existing_deduplication_prefix_and_has_no_persona():
    existing = plugin._channel_context({"chat_type": "group", "chat_id": "group:42", "message_id": "3",
        "recent_group_context": [{"message_id": "1", "user_id": "100", "text": "未补入的新讨论"}]})
    hinted = plugin._history_context(existing)
    assert hinted.startswith(existing)
    assert plugin._context_ids_in_prompt(hinted) == {"1"}
    assert plugin._history_context(hinted) == hinted
    assert "animegao" not in plugin._CURRENT_GROUP_HISTORY_GUIDANCE
    assert "超时空" not in plugin._CURRENT_GROUP_HISTORY_GUIDANCE
