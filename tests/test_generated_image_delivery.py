"""Synthetic generated-image delivery: native file entry and uncertain QQ receipts."""
from __future__ import annotations

import asyncio
import hashlib
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from onebot_adapter.config import AdapterConfig
from onebot_adapter.hermes_plugin import adapter as plugin
from onebot_adapter.hermes_plugin.adapter import OneBotAdapter, _result_timeout
from onebot_adapter.onebot import api as module
from onebot_adapter.onebot.api import OneBotApi, UploadOutcomeUnknownError, _matching_group_image_messages

NT_TIMEOUT = RuntimeError("OneBot API error send_group_msg: Timeout: NTEvent "
                          "serviceAndMethod:NodeIKernelMsgService/sendMsg")


def setup_api(tmp_path, monkeypatch):
    path = tmp_path / "generated.png"
    path.write_bytes(b"synthetic generated-image bytes, never sent")
    fingerprint = hashlib.md5(path.read_bytes()).hexdigest()
    message = [module.image_segment(str(path)), module.text_segment("测试图片")]
    row = {"group_id": 42, "user_id": 999, "message_id": 88, "real_seq": "123",
           "time": time.time(), "message": [module.image_segment(fingerprint.upper() + ".png"),
                                             module.text_segment("测试图片")]}
    api = OneBotApi(MagicMock())
    api._config = AdapterConfig(self_id="999")
    monkeypatch.setattr(module, "_IMAGE_SEND_CONFIRM_WINDOW", 0)
    return api, message, row, fingerprint


async def test_native_local_image_entry_delegates_to_real_qq_image_sender():
    adapter = OneBotAdapter.__new__(OneBotAdapter)
    result = object()
    adapter.send_image = AsyncMock(return_value=result)
    assert await adapter.send_image_file("group:42", "/synthetic/generated.png", caption="画好了",
        reply_to="12", metadata={"notify": True}) is result
    adapter.send_image.assert_awaited_once_with("group:42", "/synthetic/generated.png", caption="画好了",
                                               reply_to="12", metadata={"notify": True})


async def test_image_delivery_retains_turn_owner_without_adding_a_second_ping(monkeypatch):
    monkeypatch.setattr(plugin, "SendResult", lambda **v: SimpleNamespace(**v))
    from tests.test_reply_mentions import reply_event

    adapter = OneBotAdapter.__new__(OneBotAdapter)
    adapter._request = AsyncMock(return_value={"success": True, "message_id": "80"})
    token = plugin._reply_event_context.set(reply_event(user="200"))
    try:
        result = await adapter.send_image("group:42", "/synthetic.png", metadata={"notify": True})
        assert result.success
        payload = adapter._request.await_args.kwargs
        assert payload["conversation_reply"] is True and payload["conversation_user_id"] == "200"
        assert "reply_mention_user_id" not in payload
    finally:
        plugin._reply_event_context.reset(token)


def test_image_rpc_waits_for_bounded_confirmation_but_text_timeout_is_unchanged():
    assert _result_timeout("send", "send_image") >= module._IMAGE_SEND_REQUEST_TIMEOUT \
        + module._IMAGE_SEND_CONFIRM_WINDOW + 30
    assert _result_timeout("send", "send_text") == 30


@pytest.mark.parametrize("content", [
    "", " \n\t", "** **",
    '[QQ_SEND]{"mode":"group","target":null,"notify":false}[/QQ_SEND]',
    '[QQ_SEND]{"mode":"reply","target":"current","notify":true}[/QQ_SEND]\n',
])
async def test_empty_or_extracted_media_envelope_never_sends_a_text_bubble(monkeypatch, content):
    monkeypatch.setattr(plugin, "SendResult", lambda **v: SimpleNamespace(**v))
    adapter = OneBotAdapter.__new__(OneBotAdapter)
    adapter._request = AsyncMock()
    # Even when social delivery is active, a no-op must not reserve the
    # first-part quote/@ slot or issue an RPC.
    adapter._delivery_batches = MagicMock()
    adapter._social_policy = {"enabled": True, "delivery_enabled": True}
    result = await adapter.send("group:42", content, metadata={"notify": True})
    assert result.success
    adapter._request.assert_not_awaited()
    adapter._delivery_batches.reserve.assert_not_called()


async def test_image_rpc_timeout_is_not_retryable_and_releases_pending_future(monkeypatch):
    adapter = OneBotAdapter.__new__(OneBotAdapter)
    adapter._send_semaphore = asyncio.Semaphore(2)
    adapter._file_upload_timeout = 600
    adapter._ws = SimpleNamespace(closed=False, send_json=AsyncMock())
    adapter._futures = {}

    async def timeout(future, *, timeout):
        future.cancel()
        raise TimeoutError

    monkeypatch.setattr(plugin.asyncio, "wait_for", timeout)
    result = await adapter._rpc("send", "send_image", {"chat_id": "group:42", "image_url": "/synthetic.png"})
    assert result["success"] is False and result["retryable"] is False
    assert adapter._ws.send_json.await_count == 1 and not adapter._futures


async def test_image_connection_loss_after_submission_never_invites_a_resend(monkeypatch):
    monkeypatch.setattr(plugin, "SendResult", lambda **v: SimpleNamespace(**v))
    adapter = OneBotAdapter.__new__(OneBotAdapter)
    adapter._request = AsyncMock(side_effect=ConnectionError("connection lost after write"))
    result = await adapter.send_image("group:42", "/synthetic.png")
    assert not result.success and not result.retryable
    adapter._request.assert_awaited_once()


def simulated_confirmation_clock(monkeypatch):
    now = [time.monotonic()]
    start = now[0]
    sleeps = []

    async def sleep(delay):
        sleeps.append(delay)
        now[0] += delay

    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(module.asyncio, "sleep", sleep)
    monkeypatch.setattr(module, "_IMAGE_SEND_CONFIRM_WINDOW", 180)
    return lambda: now[0] - start, sleeps


async def test_late_two_minute_receipt_waits_and_confirms_without_resending(tmp_path, monkeypatch):
    api, message, row, _ = setup_api(tmp_path, monkeypatch)
    elapsed, sleeps = simulated_confirmation_clock(monkeypatch)
    pending = {**row, "message": [module.image_segment("generated.png"), module.text_segment("测试图片")]}

    async def call(action, params, timeout=None):
        if action == "send_group_msg":
            raise NT_TIMEOUT
        assert action == "get_group_msg_history"
        # A local pending image can misleadingly reuse the previous text's
        # sequence. Only the canonical MD5 image is a successful receipt.
        return {"data": {"messages": [pending if elapsed() < 140 else row]}}

    api.call = AsyncMock(side_effect=call)
    result = await api.send_group_msg(42, message)
    assert result == {"message_id": "88", "confirmed_after_timeout": True}
    assert elapsed() == 140 and sum(sleeps) == 140
    assert [c.args[0] for c in api.call.await_args_list].count("send_group_msg") == 1


async def test_receipt_confirmation_is_bounded_and_query_failures_never_resend(tmp_path, monkeypatch):
    api, message, _, _ = setup_api(tmp_path, monkeypatch)
    elapsed, sleeps = simulated_confirmation_clock(monkeypatch)
    api.call = AsyncMock(side_effect=TimeoutError("no response"))
    with pytest.raises(UploadOutcomeUnknownError, match="do not retry automatically"):
        await api.send_group_msg(42, message)
    assert elapsed() == 180 and sum(sleeps) == 180
    assert [c.args[0] for c in api.call.await_args_list].count("send_group_msg") == 1


async def test_slow_history_queries_count_towards_the_same_deadline(tmp_path, monkeypatch):
    api, message, _, _ = setup_api(tmp_path, monkeypatch)
    elapsed, sleeps = simulated_confirmation_clock(monkeypatch)
    monkeypatch.setattr(module, "_IMAGE_SEND_CONFIRM_WINDOW", 175)
    query_timeouts = []

    async def call(action, params, timeout=None):
        if action == "send_group_msg":
            raise NT_TIMEOUT
        query_timeouts.append(timeout)
        await module.asyncio.sleep(timeout)
        raise TimeoutError

    api.call = AsyncMock(side_effect=call)
    with pytest.raises(UploadOutcomeUnknownError):
        await api.send_group_msg(42, message)
    assert elapsed() == 175 and sum(sleeps) == 175
    assert all(0 < t <= module._GROUP_UPLOAD_CONFIRM_QUERY_TIMEOUT for t in query_timeouts)
    assert query_timeouts[-1] < module._GROUP_UPLOAD_CONFIRM_QUERY_TIMEOUT


async def test_image_timeout_confirms_one_exact_server_sequenced_image_without_resending(tmp_path, monkeypatch):
    api, message, row, _ = setup_api(tmp_path, monkeypatch)
    api.call = AsyncMock(side_effect=[NT_TIMEOUT, {"data": {"messages": [row]}}])
    result = await api.send_group_msg(42, message)
    assert result == {"message_id": "88", "confirmed_after_timeout": True}
    assert [c.args[0] for c in api.call.await_args_list] == ["send_group_msg", "get_group_msg_history"]


@pytest.mark.parametrize("change", [
    {"group_id": 43}, {"user_id": 100}, {"real_seq": "0"}, {"time": 1}, {"time": float("nan")},
    {"message": [module.image_segment("wrong-image.png"), module.text_segment("测试图片")]},
    {"message": [module.text_segment("测试图片")]},
])
async def test_unconfirmed_or_wrong_image_is_unknown_and_never_resent(tmp_path, monkeypatch, change):
    api, message, row, _ = setup_api(tmp_path, monkeypatch)
    api.call = AsyncMock(side_effect=[NT_TIMEOUT, {"data": {"messages": [{**row, **change}]}}])
    with pytest.raises(UploadOutcomeUnknownError, match="do not retry automatically"):
        await api.send_group_msg(42, message)
    assert api.call.await_count == 2


async def test_multiple_matching_images_are_not_a_unique_send_confirmation(tmp_path, monkeypatch):
    api, message, row, _ = setup_api(tmp_path, monkeypatch)
    api.call = AsyncMock(side_effect=[NT_TIMEOUT, {"data": {"messages": [row, {**row, "message_id": 89}]}}])
    with pytest.raises(UploadOutcomeUnknownError):
        await api.send_group_msg(42, message)


async def test_text_send_and_explicit_image_errors_do_not_probe_or_resend(tmp_path, monkeypatch):
    api, message, _, _ = setup_api(tmp_path, monkeypatch)
    api.call = AsyncMock(side_effect=RuntimeError("explicit permission denied"))
    with pytest.raises(RuntimeError, match="permission denied"):
        await api.send_group_msg(42, message)
    api.call.assert_awaited_once()
    api.call = AsyncMock(return_value={"data": {"message_id": 90}})
    assert await api.send_group_msg(42, [module.text_segment("普通文字")]) == {"message_id": 90}
    api.call.assert_awaited_once_with("send_group_msg", {"group_id": 42, "message": [module.text_segment("普通文字")]})


async def test_transport_image_timeout_uses_the_same_safe_confirmation(tmp_path, monkeypatch):
    api, message, row, _ = setup_api(tmp_path, monkeypatch)
    api.call = AsyncMock(side_effect=[TimeoutError(), {"data": {"messages": [row]}}])
    assert (await api.send_group_msg(42, message))["confirmed_after_timeout"]


def test_receipt_match_deduplicates_same_message_and_checks_caption(tmp_path, monkeypatch):
    _, _, row, fingerprint = setup_api(tmp_path, monkeypatch)
    fields = {"group_id": "42", "self_id": "999", "fingerprint": fingerprint,
              "text": "测试图片", "started": time.time()}
    assert _matching_group_image_messages([row, row], **fields) == ["88"]
    assert not _matching_group_image_messages([row], **{**fields, "text": "不同说明"})
