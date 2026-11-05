"""Automatic quote/forward recovery with synthetic records and mocked NapCat, never LLM calls."""
from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from onebot_adapter.group_history import GroupHistoryReader
from onebot_adapter.hermes_plugin import adapter as plugin
from onebot_adapter.hermes_plugin import onebot_tools
from onebot_adapter.history_references import HistoryReferences
from onebot_adapter.message_archive import MessageArchive
from tests.test_group_history import params, row
from tests.test_observer_agent_mode import cfg


def quoted(mid, parent, **kwargs):
    result = row(mid, **kwargs)
    result["message"].insert(0, {"type": "reply", "data": {"id": str(parent)}})
    return result


def forwarded(mid, *, resource="bundle", nodes=None, **kwargs):
    result = row(mid, **kwargs)
    data = {"id": resource} if resource is not None else {}
    if nodes is not None:
        data["content"] = nodes
    result["message"] = [{"type": "forward", "data": data}]
    return result


def setup(roots, *, extra=None, bundles=None, blacklist=None):
    archive = MessageArchive()
    extra, bundles = extra or {}, bundles or {}
    api = MagicMock()

    async def call(action, args):
        if action == "get_group_msg_history":
            return {"data": {"messages": roots}}
        if action == "get_msg":
            mid = str(args["message_id"])
            return {"data": extra.get(mid) or next((r for r in roots if str(r["message_id"]) == mid), {})}
        if action == "get_forward_msg":
            return {"data": {"messages": bundles[args["message_id"]]}}
        raise AssertionError("Unexpected action: " + action)

    api.call = AsyncMock(side_effect=call)
    config = cfg().with_overrides(group_history={g: {"enabled": True} for g in ("42", "43")},
                                  bot_blacklist_enabled=True)
    reader = GroupHistoryReader(config, api, archive=archive, blacklist_match=blacklist)
    return reader, api, archive


@pytest.mark.parametrize("group", [42, 43])
async def test_keyword_hit_follows_complete_quote_chain_outside_time_and_member_filters(group):
    root = quoted(100, 90, group=group, user=100, text="这个活动好多")
    extra = {"90": quoted(90, 80, group=group, user=200),
             "80": quoted(80, 70, group=group, user=300), "70": row(70, group=group, text="源头原文")}
    reader, api, archive = setup([root], extra=extra)
    try:
        args = params(group_id=str(group), start_time=1100, end_time=1100, user_ids=["100"], keyword="活动")
        result = await reader.read(args)
        assert [r["message_id"] for r in result["messages"]] == ["70", "80", "90", "100"]
        assert result["matched_count"] == 1 and result["referenced_count"] == 3
        assert result["returned_count"] == 4 and result["references_complete"]
        assert not result["reference_errors"]
        assert all(r["reference_only"] for r in result["messages"][:-1])
        assert [c.args[1]["message_id"] for c in api.call.await_args_list if c.args[0] == "get_msg"] == [
            "90", "80", "70"]
        assert archive.get(str(group), "70") is not None
        before = api.call.await_count
        again = await reader.read(args)
        assert again["returned_count"] == 4 and api.call.await_count == before + 1  # local source cache, no reread ban
    finally:
        archive.close()


async def test_quoted_owner_forward_is_readable_even_when_original_was_not_archived():
    source = forwarded(2, user=200, nodes=[row(3, text="下月同好聚会排期"), row(4, text="周末有两场")])
    reader, api, archive = setup([quoted(100, 2, text="活动这么多")], extra={"2": source})
    try:
        result = await reader.read(params(start_time=1100, end_time=1100, keyword="活动"))
        parent = result["messages"][0]
        assert parent["message_id"] == "2" and parent["reference_only"]
        assert [n["text"] for n in parent["forwarded_messages"]] == ["下月同好聚会排期", "周末有两场"]
        assert all(n["forwarded"] and "reply_ref" not in n for n in parent["forwarded_messages"])
        assert {str(r["message_id"]) for r in archive.query("42")} == {"2", "100"}
        assert not any(c.args[0] == "get_forward_msg" for c in api.call.await_args_list)  # inline content preferred
        assert result["references_complete"]
    finally:
        archive.close()


async def test_nested_forward_is_expanded_beyond_old_parser_depth_without_extra_resource_calls():
    nested = row(1, text="最里面的文字")
    for index in range(12):
        nested = forwarded(10 + index, resource="inline-reused-id", nodes=[nested])
    root = forwarded(100, resource="outer")
    reader, api, archive = setup([root], bundles={"outer": [nested]})
    try:
        result = await reader.read(params(start_time=1100, end_time=1100))
        current = result["messages"][0]
        for _ in range(13):
            current = current["forwarded_messages"][0]
        assert current["text"] == "最里面的文字"
        assert result["references_complete"]
        reads = [c.args[1]["message_id"] for c in api.call.await_args_list if c.args[0] == "get_forward_msg"]
        assert reads == ["outer"]
    finally:
        archive.close()


async def test_long_local_quote_chain_has_no_small_fixed_depth_limit_or_duplicate_messages():
    root = quoted(120, 119)
    reader, api, archive = setup([root])
    try:
        for mid in range(1, 120):
            archive.record(quoted(mid, mid - 1) if mid > 1 else row(mid, text="链头"))
        result = await reader.read(params(start_time=1120, end_time=1120))
        assert result["returned_count"] == 120 and result["referenced_count"] == 119
        assert result["references_complete"]
        assert result["messages"][0]["text"] == "链头"
        assert api.call.await_count == 1
    finally:
        archive.close()


async def test_shared_reference_is_deduplicated_and_pagination_uses_matched_root_not_old_reference():
    reader, api, archive = setup([quoted(100, 2), quoted(101, 2)], extra={"2": row(2)})
    try:
        result = await reader.read(params(start_time=1100, end_time=1101))
        assert [r["message_id"] for r in result["messages"]] == ["2", "100", "101"]
        assert result["next_before"] == "100"
        assert len([c for c in api.call.await_args_list if c.args[0] == "get_msg"]) == 1
    finally:
        archive.close()


async def test_indexed_source_does_not_silently_drop_text_after_256_segments():
    root = row(100)
    root["message"] = [{"type": "text", "data": {"text": "文字"}} for _ in range(270)]
    reader, _, archive = setup([root])
    try:
        result = await reader.read(params(start_time=1100, end_time=1100))
        assert result["messages"][0]["text"] == "文字" * 270
        assert not result["messages"][0].get("text_truncated")
    finally:
        archive.close()


@pytest.mark.parametrize("bad_parent", [row(2, group=43, text="跨群秘密"),
                                       {**row(2, text="错误消息秘密"), "message_id": 3}])
async def test_quoted_message_must_match_both_current_group_and_requested_id(bad_parent):
    reader, _, archive = setup([quoted(100, 2)], extra={"2": bad_parent})
    try:
        result = await reader.read(params(start_time=1100, end_time=1100))
        assert result["returned_count"] == 1 and not result["references_complete"]
        assert result["reference_errors"][0]["reason"] == "outside_current_group_or_wrong_message"
        assert "秘密" not in json.dumps(result, ensure_ascii=False)
        assert archive.get("42", "2") is None and archive.get("43", "2") is None
    finally:
        archive.close()


async def test_referenced_blacklisted_member_is_not_disclosed_even_from_local_archive():
    reader, api, archive = setup([quoted(100, 2)], blacklist=lambda user, group: "blocked" if user == "777" else None)
    archive.record(row(2, user=777, text="不可读的成员内容"))
    try:
        result = await reader.read(params(start_time=1100, end_time=1100))
        assert not result["references_complete"] and result["returned_count"] == 1
        assert "不可读的成员内容" not in json.dumps(result, ensure_ascii=False)
        assert result["reference_errors"][0]["reason"] == "member_not_authorized"
        assert api.call.await_count == 1
    finally:
        archive.close()


async def test_group_permission_revocation_during_reference_lookup_fails_closed():
    reader, api, archive = setup([quoted(100, 2)])
    original = api.call.side_effect

    async def revoke(action, args):
        if action == "get_msg":
            reader.configure(reader._config.with_overrides(group_history={}))
            return {"data": row(2)}
        return await original(action, args)

    api.call.side_effect = revoke
    try:
        with pytest.raises(PermissionError, match="撤销"):
            await reader.read(params(start_time=1100, end_time=1100))
        assert archive.get("42", "2") is None
    finally:
        archive.close()


async def test_missing_or_recalled_reference_has_an_explicit_breakpoint():
    reader, _, archive = setup([quoted(100, 2)])
    try:
        result = await reader.read(params(start_time=1100, end_time=1100))
        assert not result["references_complete"]
        assert result["reference_errors"][0]["message_id"] == "2"
        assert result["reference_errors"][0]["source_message_id"] == "100"
        assert result["returned_count"] == 1
    finally:
        archive.close()


async def test_quote_cycle_is_reported_and_each_physical_message_is_returned_once():
    reader, api, archive = setup([quoted(100, 90)], extra={"90": quoted(90, 100)})
    try:
        result = await reader.read(params(start_time=1100, end_time=1100))
        assert result["returned_count"] == 2 and not result["references_complete"]
        assert any(e["reason"] == "cycle" for e in result["reference_errors"])
        assert len([c for c in api.call.await_args_list if c.args[0] == "get_msg"]) == 1
    finally:
        archive.close()


@pytest.mark.parametrize("inline", [None, []])
async def test_forward_resource_cycle_with_absent_or_empty_inline_content_stops_safely(inline):
    node = forwarded(1, resource="same", nodes=inline)
    reader, api, archive = setup([forwarded(100, resource="same")], bundles={"same": [node]})
    try:
        result = await reader.read(params(start_time=1100, end_time=1100))
        assert not result["references_complete"] and result["reference_errors"][0]["reason"] == "cycle"
        assert len([c for c in api.call.await_args_list if c.args[0] == "get_forward_msg"]) == 1
    finally:
        archive.close()


@pytest.mark.parametrize("cyclic", [False, True])
async def test_forward_internal_quotes_use_visible_embedded_nodes_not_source_group_lookups(cyclic):
    second = quoted(2, 1) if cyclic else row(2, text="转发内部的原话")
    reader, api, archive = setup([forwarded(100, nodes=[quoted(1, 2), second])])
    try:
        result = await reader.read(params(start_time=1100, end_time=1100))
        assert result["references_complete"] is not cyclic
        assert api.call.await_count == 1
        assert len(result["messages"][0]["forwarded_messages"]) == 2
    finally:
        archive.close()


async def test_reference_timeout_retains_matched_original_and_reports_incomplete_chain():
    root = quoted(100, 2)
    reader, api, archive = setup([root])

    async def slow(action, args):
        await asyncio.sleep(.05)
        return {"data": row(2)}

    api.call.side_effect = slow
    try:
        result = await HistoryReferences(reader, "42", "100", timeout=.005).expand([root])
        assert [m["message_id"] for m in result["messages"]] == ["100"]
        assert not result["references_complete"]
        assert any(e["reason"] == "timeout" for e in result["reference_errors"])
    finally:
        archive.close()


@pytest.mark.parametrize("group", [42, 43])
async def test_media_in_nested_forward_and_quoted_source_is_cached_without_leaking_signed_urls(group):
    image = row(1, group=group)
    image["message"] = [{"type": "image", "data": {"url": "https://qq.invalid/nested-signed-secret"}}]
    parent = forwarded(2, group=group, nodes=[forwarded(3, nodes=[image])])
    reader, _, archive = setup([quoted(100, 2, group=group)], extra={"2": parent})
    data = await reader.read(params(group_id=str(group), start_time=1100, end_time=1100, include_media=True))
    callback = AsyncMock(return_value={"success": True, "data": data})
    cache = AsyncMock(return_value=(["/synthetic/forward-image.jpg"], ["image/jpeg"]))
    scope = {**params(group_id=str(group)), "active": True, "indexed_history": True, "query": callback,
             "cache_media": cache, "deadline": time.monotonic() + 60}
    token = plugin._history_turn.set(scope)
    try:
        result = json.loads(await onebot_tools._read_current_group_history({
            "start_time": 1100, "end_time": 1100, "include_media": True}))
        nested = result["messages"][0]["forwarded_messages"][0]["forwarded_messages"][0]
        assert nested["attachments"] == [{"path": "/synthetic/forward-image.jpg", "mime_type": "image/jpeg"}]
        assert "nested-signed-secret" not in json.dumps(result)
        cache.assert_awaited_once()
    finally:
        plugin._history_turn.reset(token)
        archive.close()


async def test_nested_raw_media_is_removed_even_when_no_cache_callback_exists():
    data = {"messages": [{"forwarded_messages": [{"media_items": [
        {"kind": "image", "url": "https://qq.invalid/private-signed-secret"}]}]}]}
    await onebot_tools._cache_history_media(data, None)
    assert "private-signed-secret" not in json.dumps(data)
    assert data["messages"][0]["forwarded_messages"][0]["media_unavailable"]
