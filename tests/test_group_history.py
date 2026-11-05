"""Current-group read-only history: pagination, hard budgets and runtime scope."""
from __future__ import annotations

import asyncio
import importlib.util
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from onebot_adapter.app import AdapterService
from onebot_adapter.config import AdapterConfig, ConfigStore
from onebot_adapter.group_history import GroupHistoryReader
from onebot_adapter.hermes_plugin import adapter as plugin_adapter
from onebot_adapter.hermes_plugin import onebot_tools
from onebot_adapter.hermes_plugin.adapter import _history_turn
from onebot_adapter.onebot.handler import OneBotHandler
from tests.test_incremental_context import event, make_adapter


def config():
    return AdapterConfig(self_id="999", group_default_enabled=False,
                         groups={"42": {"enabled": True}, "43": {"enabled": True}},
                         group_history={"42": {"enabled": True}, "43": {"enabled": True}})


def row(mid, *, group=42, user=100, text=None):
    return {"message_id": mid, "group_id": group, "user_id": user, "time": 1000 + mid,
            "real_seq": mid, "message_type": "group", "sender": {"card": "群名片", "nickname": "QQ昵称"},
            "message": [{"type": "text", "data": {"text": text if text is not None else f"虚构聊天{mid}"}}]}


def reader(cfg=None):
    api = MagicMock()
    api.call = AsyncMock(return_value={"data": {"messages": [row(i) for i in range(1, 21)]}})
    return GroupHistoryReader(cfg or config(), api), api


def params(**changes):
    return {"group_id": "42", "user_id": "100", "turn_id": "trusted-turn", "message_id": "1000", **changes}


async def test_latest_page_is_chronological_and_only_issues_read_action():
    history, api = reader()
    result = await history.read(params())
    assert result["returned_count"] == 20
    assert result["messages"][0]["user_name"] == "群名片"
    assert result["calls_remaining"] == 2 and result["messages_remaining"] == 80
    assert result["next_before"] and result["untrusted_history"]
    api.call.assert_awaited_once_with("get_group_msg_history", {
        "group_id": "42", "message_seq": "0", "count": 20, "reverse_order": True,
        "disable_get_url": True, "parse_mult_msg": False})


async def test_cursor_reads_older_page_without_returning_boundary_again():
    history, api = reader()
    api.call.return_value = {"data": {"messages": [row(30), row(29), row(28)]}}
    first = await history.read(params(count=3))
    api.call.return_value = {"data": {"messages": [row(28), row(27), row(26)]}}
    second = await history.read(params(count=3, before=first["next_before"]))
    assert [value["message_id"] for value in second["messages"]] == ["26", "27"]
    assert api.call.call_args.args[1]["message_seq"] == "28"


async def test_20_plus_50_plus_30_then_no_more_queries():
    history, api = reader()
    await history.read(params(count=20))
    await history.read(params(count=50))
    third = await history.read(params(count=50))
    assert api.call.call_args.args[1]["count"] == 30
    assert third["calls_remaining"] == third["messages_remaining"] == 0
    with pytest.raises(PermissionError, match="额度"):
        await history.read(params(count=1))
    assert api.call.await_count == 3


async def test_failed_requests_consume_quota_and_reload_does_not_reset_it():
    cfg = config()
    history, api = reader(cfg)
    api.call.side_effect = OSError("synthetic failure with private internal details")
    for _ in range(3):
        with pytest.raises(RuntimeError, match="暂时无法读取"):
            await history.read(params())
        history.configure(cfg.with_overrides(log_level="DEBUG"))
    with pytest.raises(PermissionError):
        await history.read(params())
    assert api.call.await_count == 3


async def test_parallel_reads_cannot_exceed_three_calls():
    history, api = reader()
    result = await asyncio.gather(*(history.read(params(count=1)) for _ in range(10)), return_exceptions=True)
    assert sum(isinstance(value, dict) for value in result) == 3
    assert api.call.await_count == 3


@pytest.mark.parametrize("changes", [{"group_id": "44"}, {"user_id": ""}, {"turn_id": ""},
                                      {"count": 51}, {"count": True}, {"before": 12},
                                      {"action": "get_private_msg"}, {"session_id": "other"}])
async def test_invalid_or_out_of_scope_request_never_calls_api(changes):
    history, api = reader()
    with pytest.raises((ValueError, PermissionError)):
        await history.read(params(**changes))
    api.call.assert_not_awaited()


@pytest.mark.parametrize("changes", [{"group_id": "43"}, {"user_id": "200"},
                                      {"turn_id": "other-turn"}, {"message_id": "other-message"}])
async def test_cursors_are_bound_to_group_user_and_turn(changes):
    history, api = reader()
    first = await history.read(params())
    with pytest.raises(ValueError, match="翻页标识"):
        await history.read(params(before=first["next_before"], **changes))
    assert api.call.await_count == 1


async def test_private_or_other_group_response_is_rejected_before_disclosure():
    history, api = reader()
    api.call.return_value = {"data": {"messages": [row(1, group=43)]}}
    with pytest.raises(PermissionError, match="来源"):
        await history.read(params())
    api.call.return_value = {"data": {"messages": [{**row(1), "message_type": "private"}]}}
    with pytest.raises(PermissionError):
        await history.read(params())


async def test_group_user_filter_and_blacklist_are_enforced():
    cfg = config()
    cfg.groups["42"]["group_user_list"] = ["100"]
    history, api = reader(cfg)
    with pytest.raises(PermissionError):
        await history.read(params())
    api.call.assert_not_awaited()
    cfg.groups["42"]["group_user_list"] = []
    history.configure(cfg)
    history._blacklist_match = lambda user, group: {} if user == "100" else None
    with pytest.raises(PermissionError):
        await history.read(params())
    api.call.assert_not_awaited()


async def test_hot_disable_during_read_prevents_returning_records():
    cfg = config()
    history, api = reader(cfg)
    async def revoke(*args):
        history.configure(cfg.with_overrides(group_history={}))
        return {"data": {"messages": [row(1)]}}
    api.call.side_effect = revoke
    with pytest.raises(PermissionError, match="撤销"):
        await history.read(params())


async def test_media_is_placeholder_not_download_or_forward_expansion():
    history, api = reader()
    message = row(1)
    message["message"] = [
        {"type": "image", "data": {"url": "http://internal.invalid/secret"}},
        {"type": "record", "data": {"url": "http://internal.invalid/audio"}},
        {"type": "forward", "data": {"content": [{"text": "not expanded"}]}},
    ]
    api.call.return_value = {"data": {"messages": [message]}}
    result = await history.read(params())
    text = json.dumps(result, ensure_ascii=False)
    assert "未识别" in text and "未转写" in text and "未展开" in text
    assert "internal.invalid" not in text and "not expanded" not in text
    assert api.call.await_count == 1


async def test_trigger_message_and_same_turn_overlap_are_not_repeated():
    history, api = reader()
    api.call.return_value = {"data": {"messages": [row(1), row(2), row(3)]}}
    first = await history.read(params(count=3, message_id="3"))
    assert [value["message_id"] for value in first["messages"]] == ["1", "2"]
    second = await history.read(params(count=3, message_id="3"))
    assert second["messages"] == []


@pytest.mark.parametrize("invalid", [[], {"42": []}, {"42": {"enabled": "true"}},
                                        {"42": {"max_calls_per_turn": 4}},
                                        {"42": {"max_messages_per_turn": 101}}])
def test_invalid_history_policy_rejected(invalid):
    assert any("group_history" in value for value in config().with_overrides(group_history=invalid).validate())


async def test_app_local_action_reuses_reader_without_generic_qq_tools():
    app = AdapterService(ConfigStore(config()))
    app._group_history = SimpleNamespace(read=AsyncMock(return_value={"messages": []}))
    assert await app._handle_local_api_call("adapter_read_current_group_history", params()) == {"messages": []}
    app._group_history.read.assert_awaited_once_with(params())


async def test_tool_uses_runtime_group_and_rejects_arbitrary_scope_arguments():
    query = AsyncMock(return_value={"success": True, "data": {"messages": [], "calls_remaining": 2}})
    scope = {"active": True, "deadline": time.monotonic() + 600, "query": query,
             "group_id": "42", "user_id": "100", "turn_id": "trusted", "message_id": "5"}
    token = _history_turn.set(scope)
    try:
        for args in ({"group_id": "43"}, {"user_id": "100"}, {"action": "get_msg"}, {"count": True}):
            assert "error" in json.loads(await onebot_tools._read_current_group_history(args))
        query.assert_not_awaited()
        await onebot_tools._read_current_group_history({"count": 20})
        assert query.call_args.args == ("adapter_read_current_group_history", {
            "group_id": "42", "user_id": "100", "turn_id": "trusted", "message_id": "5", "count": 20, "before": None})
        scope["active"] = False
        assert "error" in json.loads(await onebot_tools._read_current_group_history({}))
    finally:
        _history_turn.reset(token)


async def test_no_runtime_group_means_no_history_capability():
    token = _history_turn.set(None)
    try:
        assert "error" in json.loads(await onebot_tools._read_current_group_history({}))
    finally:
        _history_turn.reset(token)


async def test_history_scope_survives_worker_thread_and_expires_after_turn():
    adapter = make_adapter()
    value = event(prior=[])
    adapter._api_call = AsyncMock(return_value={"success": True, "data": {"messages": []}})
    await adapter._dispatch_with_incremental_context(value)
    await adapter.on_processing_start(value)
    await asyncio.to_thread(lambda: asyncio.run(onebot_tools._read_current_group_history({"count": 1})))
    assert adapter._api_call.await_count == 1
    await adapter.on_processing_complete(value, SimpleNamespace(name="SUCCESS"))
    assert "error" in json.loads(await onebot_tools._read_current_group_history({}))
    assert adapter._api_call.await_count == 1


def test_dedicated_tool_explicit_opt_in_does_not_enable_general_catalog(monkeypatch):
    monkeypatch.setattr(onebot_tools, "_load_tool_policies", lambda: {
        **{name: {"registered": False} for name, _, _ in onebot_tools._TOOLS},
        "onebot_read_current_group_history": {"registered": True},
    })
    ctx = MagicMock()
    onebot_tools.register_tools(ctx)
    assert ctx.register_tool.call_count == 1
    assert ctx.register_tool.call_args.kwargs["name"] == "onebot_read_current_group_history"
    props = ctx.register_tool.call_args.kwargs["schema"]["parameters"]["properties"]
    assert set(props) == {"before", "start_time", "end_time", "user_ids", "keyword", "keywords",
                          "keyword_mode", "match_mode", "limit", "include_media"}
    assert ctx.register_tool.call_args.kwargs["schema"]["parameters"]["required"] == []
    assert len(onebot_tools._TOOLS) == 100


async def test_on_demand_mode_never_attaches_ordinary_background_to_trigger():
    from tests.test_group_context import frame

    cfg = config()
    api, resolver = MagicMock(), MagicMock()
    resolver.resolve_group_name = AsyncMock(return_value="test group")
    delivered = AsyncMock()
    handler = OneBotHandler(label="test", config=cfg, api=api, on_event=delivered, name_resolver=resolver)
    await handler.handle_text(json.dumps(frame()))
    delivered.assert_not_awaited()
    trigger = frame(2, user=200, text="上面在聊什么")
    trigger["message"].insert(0, {"type": "at", "data": {"qq": "999"}})
    await handler.handle_text(json.dumps(trigger))
    delivered.assert_awaited_once()
    normalized = delivered.call_args.args[0]
    assert normalized.recent_group_context == []
    assert "recent_group_context" not in normalized.to_dict()
    assert not api.mock_calls


def test_automatic_and_on_demand_modes_cannot_be_enabled_together():
    cfg = config().with_overrides(group_context={"42": {"enabled": True}})
    assert any("mutually exclusive" in error for error in cfg.validate())


def test_profile_plugin_copies_share_context_definition_not_turn_values():
    spec = importlib.util.spec_from_file_location("onebot_adapter.hermes_plugin.adapter_copy", plugin_adapter.__file__)
    copy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(copy)
    assert copy._history_turn is plugin_adapter._history_turn
    scope = {"group_id": "42", "active": True}
    token = copy._history_turn.set(scope)
    try:
        assert plugin_adapter._history_turn.get() is scope
    finally:
        copy._history_turn.reset(token)
