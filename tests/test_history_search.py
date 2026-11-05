"""Enhanced shared history search: synthetic records and mocked RPCs only."""
from __future__ import annotations

import copy
import json
import sqlite3
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from onebot_adapter.group_history import GroupHistoryReader
from onebot_adapter.hermes_plugin import onebot_tools
from onebot_adapter.hermes_plugin.adapter import _history_turn
from onebot_adapter.history_search import HistoryMatcher
from onebot_adapter.message_archive import MessageArchive
from tests.test_group_history import params, row
from tests.test_history_references import forwarded, quoted, setup
from tests.test_observer_agent_mode import cfg


def indexed(rows=(), *, blacklist=None, native=()):
    archive = MessageArchive()
    for raw in rows:
        archive.record(raw)
    api = MagicMock()
    api.call = AsyncMock(return_value={"data": {"messages": list(native)}})
    config = cfg().with_overrides(group_history={g: {"enabled": True} for g in ("42", "43")},
                                  bot_blacklist_enabled=True)
    return GroupHistoryReader(config, api, archive=archive, blacklist_match=blacklist), api, archive


@pytest.mark.parametrize("text", ["BAO", "bao", "B A O", "B\nA\tO", "ＢＡＯ", "Ｂ　Ａ　Ｏ"])
@pytest.mark.parametrize("group", [42, 43])
async def test_default_normalization_handles_case_width_and_unicode_whitespace(text, group):
    reader, api, archive = indexed([row(1, group=group, text=text)])
    try:
        result = await reader.read(params(group_id=str(group), keyword="BAO"))
        assert result["messages"][0]["text"] == text  # matching never rewrites originals
        assert result["messages"][0]["search_hits"] == [{"kind": "message", "message_id": "1"}]
        assert result["search_filters"]["start_time"] is None
        assert result["search_filters"]["match_mode"] == "normalized"
        assert result["search_complete_in_index"] and not result["history_complete"]
        api.call.assert_awaited_once()
    finally:
        archive.close()


@pytest.mark.parametrize("args,expected", [
    ({"keyword": "蓝莓真香"}, [1, 2, 3]),
    ({"keyword": "蓝莓真香", "match_mode": "literal"}, [1]),
    ({"keyword": "蓝莓 真香", "match_mode": "literal"}, [2]),
    ({"keyword": r"蓝\s*莓\s*真香", "match_mode": "regex"}, [1, 2, 3]),
    ({"keyword": "蓝莓|草莓"}, []),
    ({"keyword": "蓝莓|草莓", "match_mode": "regex"}, [1, 2, 4]),
    ({"keywords": ["草莓", "无花果"]}, [4]),
    ({"keywords": ["蓝莓", "真香"], "keyword_mode": "all"}, [1, 2, 3]),
    ({"keyword": "草莓", "keywords": ["无花果"]}, [4]),
])
async def test_literal_regex_and_candidate_combination(args, expected):
    reader, _, archive = indexed([row(i, text=text) for i, text in enumerate(
        ["蓝莓真香", "蓝莓 真香", "蓝　莓\n真香", "草莓好甜"], 1)])
    try:
        result = await reader.read(params(**args))
        assert [int(r["message_id"]) for r in result["messages"]] == expected
    finally:
        archive.close()


async def test_keywords_never_join_different_people_or_different_forward_nodes():
    carrier = forwarded(3, nodes=[row(99, text="蓝莓"), row(100, text="真香")])
    reader, _, archive = indexed([row(1, text="蓝莓"), row(2, user=200, text="真香"), carrier])
    try:
        assert not (await reader.read(params(keyword="蓝莓真香")))["messages"]
        assert not (await reader.read(params(keywords=["蓝莓", "真香"], keyword_mode="all")))["messages"]
        assert (await reader.read(params(keywords=["蓝莓", "真香"])))["matched_count"] == 3
    finally:
        archive.close()


async def test_limit_is_after_keyword_and_author_permissions_not_last_n_input_messages():
    rows = [row(i, text="目标" if i in (1, 4, 10, 60) else "无关") for i in range(1, 101)]
    rows.append(row(101, user=777, text="目标：禁止展示"))
    reader, _, archive = indexed(rows, blacklist=lambda u, g: "denied" if u == "777" else None)
    try:
        first = await reader.read(params(keyword="目标", limit=2))
        assert [r["message_id"] for r in first["messages"]] == ["10", "60"]
        assert first["has_more"] and first["matching_count_in_index"] == 4
        second = await reader.read(params(keyword="目标", limit=2, before=first["next_before"]))
        assert [r["message_id"] for r in second["messages"]] == ["1", "4"]
        assert not second["has_more"]
        assert first["archive_coverage"]["message_count"] == 100
        assert "禁止展示" not in json.dumps(first, ensure_ascii=False)
    finally:
        archive.close()


async def test_latest_only_limit_and_equal_second_cursor_follow_native_sequence():
    rows = []
    for mid, seq in [(987, 1), (12, 2), (456, 3), (111, 4)]:
        raw = row(mid)
        raw.update(time=1001, real_seq=seq)
        rows.append(raw)
    reader, _, archive = indexed(rows)
    try:
        first = await reader.read(params(limit=2))
        assert [r["message_id"] for r in first["messages"]] == ["456", "111"]
        assert first["next_before"] == "456"
        second = await reader.read(params(limit=2, before=first["next_before"]))
        assert [r["message_id"] for r in second["messages"]] == ["987", "12"]
        assert not second["has_more"]
    finally:
        archive.close()


async def test_limit_keeps_every_reachable_quote_and_does_not_change_cursor_to_source():
    reader, _, archive = setup([quoted(100, 90, text="BAO")], extra={
        "90": quoted(90, 80), "80": quoted(80, 70), "70": row(70, text="真正源头")})
    try:
        result = await reader.read(params(keyword="BAO", limit=1))
        assert result["matched_count"] == 1 and result["referenced_count"] == 3
        assert result["returned_count"] == 4 and result["next_before"] == "100"
        assert result["references_complete"]
    finally:
        archive.close()


@pytest.mark.parametrize("args,expected", [
    ({"start_time": 1002}, [2, 3]),
    ({"end_time": 1002}, [1, 2]),
    ({"start_time": 1002, "end_time": 1002}, [2]),
    ({"user_ids": ["200"]}, [2]),
    ({"keyword": "BAO", "start_time": 2000}, []),
])
async def test_time_bounds_are_independently_optional_and_never_silently_ignored(args, expected):
    reader, _, archive = indexed([row(i, user=200 if i == 2 else 100, text="BAO") for i in range(1, 4)])
    try:
        result = await reader.read(params(**args))
        assert [int(r["message_id"]) for r in result["messages"]] == expected
        if args.get("start_time") == 2000:
            assert result["search_filters"]["start_time"] == 2000
            assert result["archive_coverage"]["earliest_time"] == 1001
    finally:
        archive.close()


async def test_unbounded_search_includes_old_archive_without_implicitly_crawling_entire_account():
    old = row(1, text="BAO")
    old["time"] = 1
    reader, api, archive = indexed([old], native=[row(i, text="无关") for i in range(100, 150)])
    try:
        result = await reader.read(params(keywords=["BAO"]))
        assert result["messages"][0]["timestamp"] == 1
        assert result["native_sync"] == "latest_page_only" and not result["requested_range_scanned"]
        assert not result["archive_coverage"]["continuous_history_guaranteed"]
        api.call.assert_awaited_once()
    finally:
        archive.close()


@pytest.mark.parametrize("args", [
    {}, {"include_media": True}, {"keywords": []}, {"keywords": "BAO"}, {"keywords": [None]},
    {"keyword": " "}, {"keyword": 1}, {"keywords": ["BAO", "\n\t"]},
    {"keyword": "BAO", "keyword_mode": "neither"}, {"keyword": "BAO", "match_mode": "fuzzy"},
    {"keyword": "(", "match_mode": "regex"}, {"keyword": "a" * 4097}, {"keywords": [str(i) for i in range(65)]},
    {"limit": True}, {"limit": 0}, {"limit": -1}, {"limit": 1.5}, {"limit": "50"},
    {"count": 50}, {"start_time": True}, {"start_time": float("nan")},
    {"start_time": "2026-10-28T10:00:00"}, {"start_time": 2, "end_time": 1},
    {"start_time": time.time() + 3600},
    {"user_ids": []}, {"user_ids": ["not-qq"]}, {"before": "foreign-id"},
    {"keyword": "BAO", "session_id": "foreign"},
])
async def test_invalid_parameters_fail_before_native_calls(args):
    reader, api, archive = indexed()
    try:
        with pytest.raises((ValueError, PermissionError)):
            await reader.read(params(**args))
        api.call.assert_not_awaited()
    finally:
        archive.close()


async def test_forward_body_is_searchable_and_preserved_after_text_only_native_refresh():
    child = forwarded(99, resource="inner", nodes=[row(999, text="Ｂ　Ａ　Ｏ 蓝莓真香")])
    carrier = forwarded(5, nodes=[child])
    stripped = forwarded(5)
    reader, _, archive = indexed([carrier], native=[stripped])
    try:
        result = await reader.read(params(keywords=["bao"], user_ids=["100"]))
        assert result["matched_count"] == 1
        root = result["messages"][0]
        nested = root["forwarded_messages"][0]["forwarded_messages"][0]
        assert root["search_hits"] == [{"kind": "forward", "message_id": nested["message_id"]}]
        assert nested["text"] == "Ｂ　Ａ　Ｏ 蓝莓真香" and "reply_ref" not in nested
        assert archive.get("42", "5")["message"][0]["data"]["content"]
        assert [str(r["message_id"]) for r in archive.query("42")] == ["5"]
    finally:
        archive.close()


async def test_forward_fetched_by_reference_is_indexed_for_later_keyword_search():
    carrier = forwarded(2)
    reader, api, archive = setup([quoted(100, 2, text="活动好多")], extra={"2": carrier},
                                 bundles={"bundle": [row(999, text="只有转发里才有的词")]})
    try:
        await reader.read(params(keyword="活动"))
        assert any(c.args[0] == "get_forward_msg" for c in api.call.await_args_list)
        second = await reader.read(params(keyword="只有转发里才有的词"))
        assert second["matched_count"] == 1
        assert second["messages"][0]["message_id"] == "2"
        assert second["messages"][0]["search_hits"][0]["kind"] == "forward"
        assert sum(c.args[0] == "get_forward_msg" for c in api.call.await_args_list) == 1
    finally:
        archive.close()


async def test_forward_sender_filter_is_the_carrier_author_not_an_embedded_identity():
    reader, _, archive = indexed([forwarded(1, user=100, nodes=[row(99, user=200, text="BAO")])])
    try:
        assert not (await reader.read(params(keyword="BAO", user_ids=["200"])))["messages"]
        root = (await reader.read(params(keyword="BAO", user_ids=["100"])))["messages"][0]
        assert root["user_id"] == "100" and root["forwarded_messages"][0]["user_id"] == "200"
    finally:
        archive.close()


def test_replaced_forward_updates_search_documents_without_stale_body_or_extra_live_rows():
    archive = MessageArchive()
    try:
        archive.record(forwarded(1, resource="old", nodes=[row(99, text="旧的正文")]))
        archive.record(forwarded(1, resource="new", nodes=[row(100, text="新的正文")]))
        assert not archive.search("42", matcher=HistoryMatcher({"keyword": "旧的正文"}))
        assert len(archive.search("42", matcher=HistoryMatcher({"keyword": "新的正文"}))) == 1
        assert archive.db.execute("SELECT count(*) FROM group_messages").fetchone()[0] == 1
    finally:
        archive.close()


def test_forward_cache_preservation_never_recursively_refills_cyclic_resources():
    archive = MessageArchive()
    try:
        raw = forwarded(1, resource="loop", nodes=[forwarded(99, resource="loop")])
        archive.record(raw)
        expected = archive.get("42", "1")
        for _ in range(3):
            archive.record(forwarded(1, resource="loop"))
            assert archive.get("42", "1") == expected
            archive.record(raw)  # inner id is reused but must not get the outer bundle as its content
            assert archive.get("42", "1") == expected
    finally:
        archive.close()


def test_additive_legacy_migration_preserves_original_rows_and_rebuilds_missing_documents(tmp_path):
    path = tmp_path / "legacy.sqlite3"
    raw = forwarded(1, nodes=[row(99, text="B A O")])
    payload = json.dumps(raw, ensure_ascii=False)
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE group_messages(group_id TEXT,message_id TEXT,user_id TEXT,timestamp REAL,"
                   "text TEXT,payload TEXT,PRIMARY KEY(group_id,message_id))")
        db.execute("INSERT INTO group_messages VALUES(?,?,?,?,?,?)", ("42", "1", "100", 1001, "", payload))
    for _ in range(2):
        archive = MessageArchive(path)
        try:
            matches = archive.search("42", matcher=HistoryMatcher({"keyword": "bao"}))
            assert len(matches) == 1 and matches[0][1][0]["kind"] == "forward"
            assert archive.db.execute("SELECT payload FROM group_messages").fetchone()[0] == payload
            assert archive.db.execute("SELECT count(*) FROM group_messages").fetchone()[0] == 1
            assert archive.db.execute("SELECT count(*) FROM group_search_documents").fetchone()[0] == 2
        finally:
            archive.close()


async def test_search_is_group_scoped_including_pagination_and_indexed_forward_sources():
    reader, api, archive = indexed([row(1, group=43, text="BAO私密对照"), row(2, text="无关")])
    try:
        assert not (await reader.read(params(keyword="BAO")))["messages"]
        api.call.reset_mock()
        with pytest.raises(ValueError, match="当前群"):
            await reader.read(params(keyword="BAO", before="1"))
        api.call.assert_not_awaited()
    finally:
        archive.close()


async def test_native_failure_still_searches_index_without_claiming_history_complete():
    reader, api, archive = indexed([row(1, text="B A O")])
    api.call.side_effect = RuntimeError("private-token-must-not-leak")
    try:
        result = await reader.read(params(keyword="BAO"))
        assert result["messages"] and result["native_history_unavailable"]
        assert not result["history_complete"]
        assert "private-token-must-not-leak" not in json.dumps(result)
    finally:
        archive.close()


async def test_regex_timeout_is_explicit_not_empty_success_and_sql_text_is_bound():
    reader, _, archive = indexed([row(1, text="a" * 100000 + "!")])
    try:
        started = time.monotonic()
        with pytest.raises(ValueError, match="超时"):
            await reader.read(params(keyword=r"(a+)+$", match_mode="regex"))
        assert time.monotonic() - started < 2
        assert not (await reader.read(params(keyword="x' OR 1=1 --")))["messages"]
    finally:
        archive.close()


@pytest.mark.parametrize("group", [42, 43])
async def test_plugin_accepts_new_filters_in_both_groups_and_caches_nested_forward_media(group):
    child = row(99, text="B A O")
    child["message"].append({"type": "image", "data": {"url": "https://qq.invalid/private-signed-image"}})
    carrier = forwarded(5, group=group, nodes=[child])
    reader, _, archive = setup([carrier])

    async def query(action, arguments):
        assert action == "adapter_read_current_group_history" and arguments["group_id"] == str(group)
        return {"success": True, "data": await reader.read(arguments)}

    cache = AsyncMock(return_value=(["/synthetic/forward.png"], ["image/png"]))
    scope = {**params(group_id=str(group)), "active": True, "indexed_history": True, "query": query,
             "cache_media": cache, "deadline": time.monotonic() + 60}
    token = _history_turn.set(scope)
    try:
        raw = await onebot_tools._read_current_group_history(
            {"keywords": ["BAO", "别的候选"], "limit": 1, "include_media": True})
        result = json.loads(raw)
        assert result["messages"][0]["forwarded_messages"][0]["attachments"][0]["path"] == "/synthetic/forward.png"
        assert "private-signed-image" not in raw
        cache.assert_awaited_once()
        before = copy.deepcopy(result["search_filters"])
        assert before["start_time"] is None and before["keywords"] == ["BAO", "别的候选"]
    finally:
        _history_turn.reset(token)
        archive.close()
