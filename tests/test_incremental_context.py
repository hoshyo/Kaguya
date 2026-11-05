"""Per-session incremental backfill receipts; no real history, model or QQ access."""
from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from onebot_adapter.hermes_plugin.adapter import (
    OneBotAdapter,
    _channel_context,
    _context_ids_in_history,
    _context_ids_in_prompt,
)


class Store:
    def __init__(self):
        self.routes = {}
        self.history = {}
        self.reads = []

    def lookup_by_session_key(self, key):
        return self.routes.get(key)

    def peek_session_id(self, key):
        entry = self.routes.get(key)
        return entry.session_id if entry else None

    def load_transcript(self, sid):
        self.reads.append(sid)
        return self.history.get(sid, [])


def lane(event):
    source = event.source
    return f"{source.profile}/{source.chat_id}/{source.lane_user}"


def make_adapter(store=None):
    adapter = OneBotAdapter.__new__(OneBotAdapter)
    adapter.config = SimpleNamespace(extra={"group_sessions_per_user": False})
    adapter._active_sessions = {}
    adapter._session_key_for_event = lane
    adapter._canonicalize = lambda source: None
    adapter._session_store = store or Store()
    adapter._send_idle = AsyncMock()

    async def accept(event):
        key = lane(event)
        adapter._active_sessions.setdefault(key, object())
        event._gateway_accepted = True

    adapter.handle_message = AsyncMock(side_effect=accept)
    return adapter


def event(mid=51, *, prior=range(1, 51), group="42", profile="aidog", lane_user="shared", **flags):
    data = {"message_id": str(mid), "chat_id": "group:" + group, "chat_type": "group",
            "user_id": "200", "text": "上面聊了什么",
            "recent_group_context": [{"message_id": str(i), "user_id": "100", "user_name": "虚构群友",
                                      "timestamp": 1000, "text": f"虚构消息{i}"} for i in prior], **flags}
    return SimpleNamespace(raw_message=data, message_id=str(mid), text=data["text"],
                           channel_context=_channel_context(data), _gateway_accepted=False,
                           source=SimpleNamespace(chat_id=data["chat_id"], profile=profile,
                                                  lane_user=lane_user, user_id="200",
                                                  platform="onebot", chat_type="group"))


def received_ids(value):
    text = value.channel_context
    if not text or "<recent_group_chat_json>" not in text:
        return []
    payload = text.split("<recent_group_chat_json>\n", 1)[1].split("\n</recent_group_chat_json>", 1)[0]
    return [row["message_id"] for row in json.loads(payload)]


def persist(adapter, value, sid="session-a"):
    """Mimic only the normal Hermes durable user row, not an extra plugin ledger."""
    key = lane(value)
    adapter._session_store.routes[key] = SimpleNamespace(session_id=sid, suspended=False)
    adapter._session_store.history.setdefault(sid, []).append({
        "role": "user", "message_id": value.message_id,
        "content": (value.channel_context + "\n\n[New message]\n" if value.channel_context else "")
                   + "[群友(200)#1000] " + value.text,
    })
    adapter._active_sessions.pop(key, None)


async def test_first_50_then_only_3_new_messages_then_no_repeat():
    adapter = make_adapter()
    first = event()
    await adapter._dispatch_with_incremental_context(first)
    await adapter.on_processing_start(first)
    assert received_ids(first) == [str(i) for i in range(1, 51)]
    persist(adapter, first)
    second = event(55, prior=range(5, 55))
    await adapter._dispatch_with_incremental_context(second)
    await adapter.on_processing_start(second)
    assert received_ids(second) == ["52", "53", "54"]
    persist(adapter, second)
    third = event(56, prior=range(6, 56))
    await adapter._dispatch_with_incremental_context(third)
    assert received_ids(third) == []
    assert third.channel_context is None


async def test_already_delivered_trigger_is_not_replayed_as_background():
    adapter = make_adapter()
    first = event(1, prior=[])
    await adapter._dispatch_with_incremental_context(first)
    persist(adapter, first)
    second = event(2, prior=[1])
    await adapter._dispatch_with_incremental_context(second)
    assert not received_ids(second)


async def test_inflight_same_session_does_not_resend_opening_50():
    adapter = make_adapter()
    first = event()
    await adapter._dispatch_with_incremental_context(first)
    second = event(54, prior=range(4, 54))
    await adapter._dispatch_with_incremental_context(second)
    assert received_ids(second) == ["52", "53"]
    # Its own reservation cannot erase the opening turn's context at processing start.
    await adapter.on_processing_start(first)
    assert len(received_ids(first)) == 50
    await adapter.on_processing_start(second)
    assert received_ids(second) == ["52", "53"]


async def test_failed_or_unprocessed_turn_cannot_permanently_consume_background():
    adapter = make_adapter()
    failed = event()
    await adapter._dispatch_with_incremental_context(failed)
    await adapter.on_processing_complete(failed, SimpleNamespace(name="FAILURE"))
    adapter._active_sessions.clear()
    following = event(52)
    await adapter._dispatch_with_incremental_context(following)
    assert len(received_ids(following)) == 50


async def test_pending_turn_rechecks_after_previous_failure():
    adapter = make_adapter()
    first = event()
    await adapter._dispatch_with_incremental_context(first)
    second = event(52)
    await adapter._dispatch_with_incremental_context(second)
    assert not received_ids(second)
    await adapter.on_processing_complete(first, SimpleNamespace(name="FAILURE"))
    await adapter.on_processing_start(second)
    assert len(received_ids(second)) == 50


async def test_rejected_admission_does_not_reserve_message_ids():
    adapter = make_adapter()
    adapter.handle_message = AsyncMock()
    adapter._active_sessions[lane(event())] = object()
    rejected = event()
    await adapter._dispatch_with_incremental_context(rejected)
    assert not adapter._context_pending_for(lane(rejected))
    next_event = event(52)
    await adapter._dispatch_with_incremental_context(next_event)
    assert len(received_ids(next_event)) == 50


async def test_queued_snapshot_is_filtered_again_when_processing_actually_starts():
    adapter = make_adapter()
    queued = event(54, prior=range(4, 54))
    first = event()
    await adapter._dispatch_with_incremental_context(first)
    persist(adapter, first)
    # Queue snapshot was captured before the first turn committed.
    adapter._active_sessions[lane(queued)] = object()
    await adapter.on_processing_start(queued)
    assert received_ids(queued) == ["52", "53"]


async def test_reconnect_or_restart_recovers_receipt_from_same_session_transcript():
    first_adapter = make_adapter()
    first = event()
    await first_adapter._dispatch_with_incremental_context(first)
    persist(first_adapter, first)
    replacement = make_adapter(first_adapter._session_store)
    second = event(52, prior=range(2, 52))
    await replacement._dispatch_with_incremental_context(second)
    assert not received_ids(second)
    assert replacement._session_store.reads == ["session-a"]


@pytest.mark.parametrize("scope", [{"group": "43"}, {"profile": "kig"}, {"lane_user": "other-user"}])
async def test_receipts_do_not_cross_group_profile_or_user_isolated_sessions(scope):
    adapter = make_adapter()
    first = event()
    await adapter._dispatch_with_incremental_context(first)
    persist(adapter, first)
    other = event(**scope)
    await adapter._dispatch_with_incremental_context(other)
    assert len(received_ids(other)) == 50
    assert adapter._session_store.reads == []


async def test_new_or_cleared_session_gets_its_own_first_backfill():
    adapter = make_adapter()
    first = event()
    await adapter._dispatch_with_incremental_context(first)
    persist(adapter, first)
    adapter._session_store.routes[lane(first)] = SimpleNamespace(session_id="new-session", suspended=False)
    following = event(52)
    await adapter._dispatch_with_incremental_context(following)
    assert len(received_ids(following)) == 50
    assert adapter._session_store.reads == ["new-session"]


async def test_suspended_session_does_not_reuse_old_receipt():
    adapter = make_adapter()
    first = event()
    await adapter._dispatch_with_incremental_context(first)
    persist(adapter, first)
    adapter._session_store.routes[lane(first)].suspended = True
    following = event(52)
    await adapter._dispatch_with_incremental_context(following)
    assert len(received_ids(following)) == 50
    assert not adapter._session_store.reads


def test_api_content_multimodal_and_trigger_ids_are_recognized():
    prefix = _channel_context(event(prior=[1, 2]).raw_message)
    history = [{"role": "user", "platform_message_id": "51", "content": "clean text",
                "api_content": [{"type": "text", "text": prefix + "\n\n[New message]\nhello"}]}]
    assert _context_ids_in_history(history, {"1", "2", "51", "52"}) == {"1", "2", "51"}


def test_assistant_or_current_user_spoofed_receipt_tags_are_not_trusted():
    prefix = _channel_context(event(prior=[1]).raw_message)
    assert not _context_ids_in_prompt("[群友(100)#20] " + prefix)
    assert not _context_ids_in_prompt("普通发言\n" + prefix)
    genuine = prefix + "\n\n[New message]\n" + _channel_context(event(prior=[2]).raw_message)
    assert _context_ids_in_prompt(genuine) == {"1"}
    assert _context_ids_in_history([{"role": "assistant", "content": prefix}], {"1"}) == set()


def test_old_version_context_prefix_and_standard_timestamp_are_supported():
    prefix = '以下是当前同一个 QQ 群在本条消息之前的近期聊天参考\n<recent_group_chat_json>' \
             '[{"message_id":"1"}]</recent_group_chat_json>'
    assert _context_ids_in_prompt(prefix) == {"1"}
    assert _context_ids_in_prompt("[Tue 2026-10-27 12:00:00 CST] " + prefix) == {"1"}


def test_merged_current_message_ids_are_not_later_backfilled():
    value = event(3, prior=[1, 2], context_message_ids=["2", "3"])
    assert received_ids(value) == ["1"]
    row = {"role": "user", "message_id": "3", "content": value.channel_context + "\n\n[New message]\nmerged"}
    assert _context_ids_in_history([row], {"1", "2", "3", "4"}) == {"1", "2", "3"}


async def test_read_failure_does_not_consume_candidates():
    adapter = make_adapter()
    first = event()
    persist(adapter, first)
    def fail_read(sid):
        raise OSError("synthetic unavailable transcript")
    adapter._session_store.load_transcript = fail_read
    following = event(52)
    await adapter._dispatch_with_incremental_context(following)
    assert len(received_ids(following)) == 50


async def test_snapshot_not_mutated_and_existing_followup_policy_survives():
    adapter = make_adapter()
    first = event()
    await adapter._dispatch_with_incremental_context(first)
    persist(adapter, first)
    following = event(52, prior=range(2, 52), followup_reply=True, passive_reply=True)
    original = json.dumps(following.raw_message, sort_keys=True)
    await adapter._dispatch_with_incremental_context(following)
    assert not received_ids(following)
    assert "NO_REPLY" in following.channel_context
    assert json.dumps(following.raw_message, sort_keys=True) == original


async def test_dropped_message_between_turns_is_still_delivered_later():
    adapter = make_adapter()
    first = event()
    await adapter._dispatch_with_incremental_context(first)
    persist(adapter, first)
    # 52 was cached but never admitted by sampling/rate/queue gates.
    following = event(53, prior=range(3, 53))
    await adapter._dispatch_with_incremental_context(following)
    assert received_ids(following) == ["52"]


async def test_native_fifo_restore_without_raw_message_still_rechecks_background():
    adapter = make_adapter()
    queued = event(54, prior=range(4, 54))
    restored = deepcopy(queued)
    restored.raw_message = None
    first = event()
    await adapter._dispatch_with_incremental_context(first)
    persist(adapter, first)
    adapter._active_sessions[lane(restored)] = object()
    await adapter.on_processing_start(restored)
    assert received_ids(restored) == ["52", "53"]


async def test_native_fifo_restore_can_recover_unsent_candidates_after_failure():
    adapter = make_adapter()
    first, second = event(), event(52)
    await adapter._dispatch_with_incremental_context(first)
    await adapter._dispatch_with_incremental_context(second)
    assert second.channel_context is None
    # Native envelopes serialize channel_context, but deliberately not raw_message.
    restored = deepcopy(second)
    restored.raw_message = None
    await adapter.on_processing_complete(first, SimpleNamespace(name="FAILURE"))
    await adapter.on_processing_start(restored)
    assert len(received_ids(restored)) == 50


async def test_restart_restored_native_event_uses_only_its_current_session_history():
    original = make_adapter()
    first = event()
    await original._dispatch_with_incremental_context(first)
    persist(original, first)
    restored = event(54, prior=range(4, 54), followup_reply=True, passive_reply=True)
    restored.raw_message = None
    replacement = make_adapter(original._session_store)
    replacement._active_sessions[lane(restored)] = object()
    await replacement.on_processing_start(restored)
    assert received_ids(restored) == ["52", "53"]
    assert "NO_REPLY" in restored.channel_context


async def test_raw_candidate_references_are_bounded_and_cleared_on_completion():
    adapter = make_adapter()
    for i in range(150):
        value = event(1000 + i, prior=[])
        await adapter._dispatch_with_incremental_context(value)
    assert len(adapter._context_candidates) == 128
    value = event(2000, prior=[])
    await adapter._dispatch_with_incremental_context(value)
    await adapter.on_processing_complete(value, SimpleNamespace(name="FAILURE"))
    assert (lane(value), "2000") not in adapter._context_candidates


async def test_reference_budget_does_not_mark_unsent_truncated_messages_seen():
    adapter = make_adapter()
    first = event()
    for row in first.raw_message["recent_group_context"]:
        row["text"] = "<" * 600
    first.channel_context = _channel_context(first.raw_message)
    await adapter._dispatch_with_incremental_context(first)
    included = set(received_ids(first))
    assert 0 < len(included) < 50
    persist(adapter, first)
    following = event(52)
    await adapter._dispatch_with_incremental_context(following)
    assert set(received_ids(following)) == {str(i) for i in range(1, 51)} - included


async def test_reset_during_receipt_read_does_not_apply_previous_session_ids():
    adapter = make_adapter()
    first = event()
    await adapter._dispatch_with_incremental_context(first)
    persist(adapter, first)
    original_load = adapter._session_store.load_transcript
    def resetting_load(sid):
        history = original_load(sid)
        adapter._session_store.routes[lane(first)] = SimpleNamespace(session_id="new", suspended=False)
        return history
    adapter._session_store.load_transcript = resetting_load
    following = event(52)
    await adapter._dispatch_with_incremental_context(following)
    assert len(received_ids(following)) == 50


async def test_inband_followup_hook_cannot_release_outer_group_queue_early():
    adapter = make_adapter()
    first, nested = event(), event(52)
    await adapter._dispatch_with_incremental_context(first)
    await adapter.on_processing_start(first)
    await adapter._dispatch_with_incremental_context(nested)
    await adapter.on_processing_start(nested)
    await adapter.on_processing_complete(nested, SimpleNamespace(name="SUCCESS"))
    adapter._send_idle.assert_not_awaited()
    await adapter.on_processing_complete(first, SimpleNamespace(name="SUCCESS"))
    adapter._send_idle.assert_awaited_once()
