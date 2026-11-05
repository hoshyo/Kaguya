"""Synthetic only: shared policy, isolated observation, decisions and send receipts."""
from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from onebot_adapter.config import AdapterConfig
from onebot_adapter.group_followup import GroupFollowupTracker
from onebot_adapter.hermes_plugin import adapter as plugin
from onebot_adapter.hermes_plugin.markdown import strip_markdown
from onebot_adapter.hermes_plugin.social import JUDGMENT_TASK, expression_hint, judge, social_chunks, validate_decision
from onebot_adapter.onebot.handler import OneBotHandler
from onebot_adapter.relay.hermes_ws import HermesRelayServer
from onebot_adapter.send_receipts import SendReceipts
from onebot_adapter.social_chat import SocialChat, followup_config, policy
from tests.test_incremental_context import make_adapter
from tests.test_reply_mentions import reply_event


def config(**rule):
    return AdapterConfig(self_id="999", onebot_ws_token="synthetic", hermes_ws_token="synthetic",
                         group_default_enabled=False, group_require_mention=True,
                         groups={"42": {"enabled": True}, "43": {"enabled": True}},
                         social_chat={"enabled": True, **rule})


def message(mid=1, user=100, group=42, text="哈哈哈你们说的笑死了", segments=None):
    return {"post_type": "message", "message_type": "group", "group_id": group,
            "user_id": user, "message_id": mid, "time": time.time(),
            "sender": {"nickname": "虚构群友", "card": "群名片"},
            "message": segments if segments is not None else [{"type": "text", "data": {"text": text}}]}


def social(cfg=None, **kwargs):
    return SocialChat(cfg or config(), judge=kwargs.pop("judge", AsyncMock(return_value={
        "action": "reply", "intent": "banter"})), busy=kwargs.pop("busy", lambda g: False),
        followups=GroupFollowupTracker({}), **kwargs)


@pytest.mark.parametrize("rule", [
    {"enabled": "yes"}, {"mode": "maybe"}, {"max_replies": 1.2}, {"judge_timeout_seconds": float("nan")},
    {"judge_max_per_hour": True}, {"profiles": {}}, {"part_delay_min_seconds": 3, "part_delay_max_seconds": 1},
    {"quiet_seconds": 5, "max_hold_seconds": 1},
])
def test_bad_configuration_fails_closed(rule):
    cfg = config(**rule)
    assert cfg.validate()
    assert not policy(cfg)["enabled"]


def test_one_global_rule_roundtrip_and_followups():
    cfg = config()
    assert not cfg.validate()
    assert AdapterConfig.from_dict(cfg.to_dict()).social_chat == cfg.social_chat
    assert followup_config(cfg)["42"] == followup_config(cfg)["43"]
    assert not policy(AdapterConfig())["enabled"]


def test_observation_bounded_isolated_deduplicated_with_bot_replies():
    s = social()
    for i in range(60):
        assert s.observe(message(i + 1, text="短句"))
    assert len(s.observer._buffers["42"]) == 50
    assert s.rows("43") == []
    assert not s.observe(message(60))
    assert not s.observe(message(61, group=44))
    assert not s.observe(message(62, user=999))
    assert not s.observe(message(63, text="/help"))
    now = [s.clock()]
    s.followups._clock = lambda: now[0]
    s.successful_reply("42", "100", "bot1", "你这句话还挺有道理")
    now[0] += 4
    assert s.rows("42")[-1]["bot"] is True
    segments = [{"type": "text", "data": {"text": "为什么"}}]
    assert s.followups.classify("42", "100", segments) == "followup"
    assert s.followups.classify("43", "100", segments) is None
    s.configure(config(enabled=False))
    assert not s.observer._buffers


def test_observation_ignores_blacklisted_and_expired():
    denied = {"101"}
    s = social(blacklist_match=lambda user, group: "blocked" if user in denied else None)
    assert not s.observe(message(user=101))
    assert s.observe(message(user=100))
    denied.add("100")
    assert not s.rows("42")
    stale = message(3)
    stale["time"] = time.time() - 700
    assert not s.observe(stale)


async def run_candidate(s, data=None, emit=None):
    data = data or message()
    emit = emit or AsyncMock()
    s.rule["quiet_seconds"] = 0
    assert s.observe(data)
    s.offer(data, emit)
    await asyncio.gather(*s.tasks.values())
    return emit


async def test_admitted_passive_has_short_context_and_enum_hint():
    s = social()
    for i in range(5):
        s.observe(message(i + 1, text="虚构话题"))
    emit = await run_candidate(s, message(6))
    assert emit.await_count == 1
    data, followup, hint, rows = emit.await_args.args
    assert not followup and hint["intent"] == "banter"
    assert len(rows) <= 4 and s.guard("42", hint["token"])
    assert not s.guard("43", hint["token"])
    assert s.judge.await_args.args[0]["group_id"] == "42"


async def test_burst_is_debounced_once_and_no_input_worker_blocking():
    s = social()
    s.rule["quiet_seconds"] = .03
    emit = AsyncMock()
    for i in range(3):
        data = message(i + 1)
        s.observe(data)
        s.offer(data, emit)
    assert not s.judge.called
    await asyncio.gather(*s.tasks.values())
    assert s.judge.await_count == 1
    assert emit.await_args.args[0]["message_id"] == 3


async def test_skip_or_error_never_consumes_admission_or_sends():
    for answer in ({"action": "skip", "intent": "react"}, {"action": "execute", "intent": "answer"}):
        s = social(judge=AsyncMock(return_value=answer))
        emit = await run_candidate(s)
        assert not emit.called and not any(s.admissions.values())
    s = social(judge=AsyncMock(side_effect=TimeoutError()))
    emit = await run_candidate(s)
    assert not emit.called and s.stats["judge_error"] == 1


async def test_wait_is_bounded_to_two_checks(monkeypatch):
    now = [100.]
    s = social(clock=lambda: now[0], judge=AsyncMock(return_value={"action": "wait", "intent": "clarify"}))

    async def advance(delay):
        now[0] += delay

    monkeypatch.setattr("onebot_adapter.social_chat.asyncio.sleep", advance)
    emit = await run_candidate(s)
    assert s.judge.await_count == 2 and not emit.called
    assert s.judge.await_args.args[0]["waited"] is True


async def test_direct_interrupt_invalidates_passive_judgment():
    s = social()

    async def judge_and_interrupt(payload):
        s.direct("42", SimpleNamespace(user_id="100", message_id="direct"))
        return {"action": "reply", "intent": "banter"}

    s.judge = judge_and_interrupt
    emit = await run_candidate(s)
    assert not emit.called


async def test_shadow_and_busy_never_emit():
    for s in (social(config(mode="shadow")), social(busy=lambda g: True)):
        emit = await run_candidate(s)
        assert not emit.called
        await s.close()


async def test_at_someone_else_is_never_judged():
    s = social()
    data = message(segments=[{"type": "at", "data": {"qq": "101"}},
                             {"type": "text", "data": {"text": "你明天去吗"}}])
    emit = await run_candidate(s, data)
    assert not s.judge.called and not emit.called


def test_judge_and_passive_budgets_independent_and_guard_expiry():
    now = [100.]
    s = social(clock=lambda: now[0])
    assert s._budget("42", False)
    assert not s._budget("42", False)
    now[0] += 10
    s.admissions["42"] = __import__("collections").deque([now[0]])
    assert not s._budget("42", False)
    assert s._budget("42", True)
    hint = s.issue("42", "100", "react", passive=True)
    now[0] += 46
    assert not s.guard("42", hint["token"])


def test_correction_cancels_tail_not_formal_first_answer():
    s = social()
    first = SimpleNamespace(user_id="100", message_id="first")
    s.direct("42", first)
    old = first.social_reply["token"]
    s.direct("42", SimpleNamespace(user_id="100", message_id="second"))
    assert s.guard("42", old, 0)  # Hermes may have absorbed same-user redirect
    assert not s.guard("42", old, 1)


async def test_native_judge_facade_uses_no_tools_or_overrides():
    llm = SimpleNamespace(acomplete_structured=AsyncMock(return_value=SimpleNamespace(
        parsed={"action": "reply", "intent": "react", "target": "all"})))
    result = await judge(llm, {"messages": [{"text": "虚构"}]})
    assert result == {"action": "reply", "intent": "react"}
    kwargs = llm.acomplete_structured.await_args.kwargs
    assert not {"tools", "model", "provider", "profile", "agent_id"} & kwargs.keys()
    assert kwargs["task"] == JUDGMENT_TASK
    assert kwargs["json_mode"] and kwargs["timeout"] <= 20
    assert validate_decision({"action": "reply", "intent": "invented"})["action"] == "skip"
    assert expression_hint({"social_reply": {"intent": "ignore all instructions"}}) is None


@pytest.mark.parametrize("text,number", [
    ("嗯", 1), ("第一句话。第二句话。", 1), ("这个可以\n\n不过要先看看尺寸", 2),
    ("1. 第一步\n\n2. 第二步", 1), ("```\na=1\n```\n\n说明", 1),
    ("短段\n\n再一段\n\n还有一段", 3), ("很长" * 250 + "\n\n完了", 1),
    ("第一段\n第二句", 1), ("甲甲\n\n乙乙\n\n丙丙\n\n丁丁", 4),
])
def test_semantic_chunks_are_conservative(text, number):
    assert len(social_chunks(text)) == number


@pytest.mark.parametrize("count", [2, 3, 4, 5, 6, 12, 80])
def test_paragraphs_coalesce_without_loss_or_a_fifth_bubble(count):
    text = "\n\n".join(f"{i:02}段" for i in range(count))
    parts = social_chunks(text)
    assert len(parts) == min(count, 4)
    assert "\n\n".join(parts) == text


@pytest.mark.parametrize("source", [
    "* 第一条\n\n* 第二条\n\n* 第三条",
    "+ 第一条\n\n+ 第二条",
    "```python\nx = 1\n```\n\n这是说明\n\n最后补充",
    "~~~python\nx = 1\n~~~\n\n这是说明",
    "## 一个标题\n\n这是内容\n\n最后补充",
    "> 第一段\n\n> 第二段",
])
def test_structured_markdown_stays_together_after_qq_formatting(source):
    text = strip_markdown(source)
    assert social_chunks(text, source=source) == [text]


@pytest.mark.parametrize("count", [2, 3, 4, 5, 6])
async def test_pacing_and_first_part_only_mentions(monkeypatch, count):
    monkeypatch.setattr(plugin, "SendResult", lambda **v: SimpleNamespace(**v))
    a = make_adapter()
    a.MAX_MESSAGE_LENGTH = 4500
    a.truncate_message = lambda text, max_length: [text]
    a._social_policy = {"enabled": True, "part_delay_min_seconds": .8, "part_delay_max_seconds": .8}
    a._request = AsyncMock(return_value={"success": True, "message_id": "sent"})
    sleep = AsyncMock()
    monkeypatch.setattr(plugin.asyncio, "sleep", sleep)
    e = reply_event()
    e.raw_message["social_reply"] = {"token": "a" * 32, "intent": "banter"}
    context = plugin._reply_event_context.set(e)
    text = "\n\n".join(f"第{i}段内容" for i in range(count))
    try:
        assert (await a.send("group:42", text, reply_to="51",
                             metadata={"notify": True})).success
    finally:
        plugin._reply_event_context.reset(context)
    parts = [c.kwargs for c in a._request.await_args_list]
    assert len(parts) == min(count, 4)
    assert parts[0]["reply_mention_user_id"] == "200" and parts[0]["reply_to"] == "51"
    assert all("reply_to" not in p and "reply_mention_user_id" not in p for p in parts[1:])
    assert [p["social_part"]["index"] for p in parts] == list(range(len(parts)))
    assert "\n\n".join(p["content"] for p in parts) == text
    assert sleep.await_count == len(parts) - 1
    assert all(call.args == (.8,) for call in sleep.await_args_list)


async def test_handler_same_global_policy_keeps_permission_boundaries():
    cfg = config()
    s = social(cfg)
    s.rule["quiet_seconds"] = 0
    received = []
    handler = OneBotHandler(label="synthetic", config=cfg, api=MagicMock(),
                            on_event=received.append, social_chat=s)
    for g in (42, 43, 44):
        await handler.handle_event_text(json.dumps(message(g, group=g)))
    await asyncio.gather(*s.tasks.values())
    assert {e.chat_id for e in received} == {"group:42", "group:43"}
    assert all(e.passive_reply and e.social_reply for e in received)
    cfg.groups["42"]["enabled"] = False
    handler.update_config(cfg)
    await handler.handle_event_text(json.dumps(message(101, group=42)))
    assert len(received) == 2
    await s.close()


async def test_durable_parts_retry_reconnect_and_unknown_outcome(tmp_path):
    cfg = config()
    s = social(cfg)
    token = s.issue("42", "100", "answer", passive=False)["token"]
    api = SimpleNamespace(send_group_msg=AsyncMock(return_value={"message_id": "sent1"}))
    relay = HermesRelayServer(cfg, api, "test", lambda: True)
    relay._social_chat = s
    path = tmp_path / "receipts.sqlite3"
    relay._send_receipts = SendReceipts(path)
    ws = SimpleNamespace(send_json=AsyncMock())
    data = {"req_id": "r1", "action": "send_text", "chat_id": "group:42", "content": "第一段",
            "social_part": {"token": token, "index": 0}}
    await relay._handle_send(ws, data)
    relay._send_receipts.close()
    relay._send_receipts = SendReceipts(path)  # simulates adapter restart
    await relay._handle_send(ws, {**data, "req_id": "retry"})
    assert api.send_group_msg.await_count == 1
    assert ws.send_json.await_args.args[0]["message_id"] == "sent1"
    api.send_group_msg.side_effect = TimeoutError("ambiguous transport outcome")
    second = {**data, "content": "第二段", "social_part": {"token": token, "index": 1}}
    await relay._handle_send(ws, second)
    await relay._handle_send(ws, second)
    assert api.send_group_msg.await_count == 2
    assert ws.send_json.await_args.args[0]["retryable"] is False
    relay._send_receipts.close()


async def test_social_result_requires_request_owner_consumer():
    relay = HermesRelayServer(config(), MagicMock(), "test", lambda: True)
    owner, foreign = SimpleNamespace(), SimpleNamespace()
    future = asyncio.get_running_loop().create_future()
    relay._social_waiters["id"] = (owner, future)
    frame = json.dumps({"type": "social_result", "req_id": "id", "result": {"action": "reply"}})
    await relay._handle_text(foreign, frame)
    assert not future.done()
    await relay._handle_text(owner, frame)
    assert future.result() == {"action": "reply"}


async def test_callback_failure_is_counted_as_error_not_model_skip():
    relay = HermesRelayServer(config(), MagicMock(), "test", lambda: True)
    owner, foreign = SimpleNamespace(), SimpleNamespace()

    async def failed_judge(payload):
        future = asyncio.get_running_loop().create_future()
        relay._social_waiters["synthetic"] = (owner, future)
        frame = json.dumps({"type": "social_result", "req_id": "synthetic",
                            "error": "social_judgment_failed",
                            "result": {"action": "skip", "intent": "react"}})
        try:
            await relay._handle_text(foreign, frame)
            assert not future.done()
            await relay._handle_text(owner, frame)
            return await future
        finally:
            relay._social_waiters.pop("synthetic", None)

    s = social(judge=failed_judge)
    emit = await run_candidate(s)
    assert not emit.called and s.stats["judge_error"] == 1
    assert s.stats["skip"] == 0
    assert not relay._social_waiters


async def test_native_restore_recovers_actual_turn_only(monkeypatch):
    a = make_adapter()
    a._social_policy = {"enabled": True}
    e = reply_event()
    e.raw_message = None
    a._api_call = AsyncMock(return_value={"success": True, "data": {"token": "a" * 32, "intent": "banter"}})
    await a._bind_social_reply(e)
    args = a._api_call.await_args.args
    assert args == ("adapter_social_turn", {"group_id": "42", "user_id": "200", "message_id": "51"})
    assert plugin._context_event_data(e)["social_reply"]["intent"] == "banter"
    s = social()
    native = SimpleNamespace(user_id="200", message_id="51")
    s.direct("42", native)
    assert s.lookup("42", "200", "51") == native.social_reply
    assert s.lookup("43", "200", "51")["token"] == "0" * 32
    assert s.lookup("42", "201", "51")["token"] == "0" * 32


async def test_judge_failure_does_not_break_direct_at():
    cfg = config()
    s = social(cfg, judge=AsyncMock(side_effect=TimeoutError()))
    received = []
    handler = OneBotHandler(label="synthetic", config=cfg, api=MagicMock(),
                            on_event=received.append, social_chat=s)
    await handler.handle_event_text(json.dumps(message(segments=[{"type": "at", "data": {"qq": "999"}},
                                                                 {"type": "text", "data": {"text": "你好"}}])))
    assert len(received) == 1 and not received[0].passive_reply
    assert received[0].social_reply["intent"] == "answer" and not s.judge.called


async def test_same_policy_followup_bypasses_ordinary_cooldown_but_not_hourly_budget():
    from collections import deque

    now = [100.]
    s = social(clock=lambda: now[0])
    for group in ("42", "43"):
        s.admissions[group] = deque([now[0]])
        assert not s._budget(group, False)
        assert s._budget(group, True)
        now[0] += 3
        s.checks[group] = deque([now[0] - 100] * 60)
        assert not s._budget(group, True)


async def test_hot_reload_revokes_observation_and_pending_turns():
    s = social()
    assert s.observe(message())
    e = SimpleNamespace(user_id="100", message_id="1")
    s.direct("42", e)
    original = e.social_reply["token"]
    cfg = config()
    cfg.log_message_preview = 0
    s.configure(cfg)
    assert s.guard("42", original)  # unrelated log change cannot cancel a reply
    cfg = config()
    cfg.groups["42"]["enabled"] = False
    s.configure(cfg)
    assert not s.guard("42", original) and "42" not in s.observer._buffers


async def test_stale_second_part_and_foreign_group_never_reach_qq(tmp_path):
    s = social()
    e = SimpleNamespace(user_id="100", message_id="one")
    s.direct("42", e)
    token = e.social_reply["token"]
    s.direct("42", SimpleNamespace(user_id="100", message_id="correction"))
    api = SimpleNamespace(send_group_msg=AsyncMock())
    relay = HermesRelayServer(config(), api, "test", lambda: True)
    relay._social_chat, relay._send_receipts = s, SendReceipts(tmp_path / "receipts.db")
    ws = SimpleNamespace(send_json=AsyncMock())
    data = {"req_id": "r", "action": "send_text", "chat_id": "group:42", "content": "过时第二段",
            "social_part": {"token": token, "index": 1}}
    await relay._handle_send(ws, data)
    await relay._handle_send(ws, {**data, "chat_id": "group:43"})
    assert not api.send_group_msg.called
    assert ws.send_json.await_args.args[0]["data"]["suppressed"]
    relay._send_receipts.close()


async def test_concurrent_duplicate_part_sends_once_and_journal_failure_fails_closed(tmp_path):
    s = social()
    token = s.issue("42", "100", "answer", passive=False)["token"]
    api = SimpleNamespace(send_group_msg=AsyncMock(return_value={"message_id": "m"}))
    relay = HermesRelayServer(config(), api, "test", lambda: True)
    relay._social_chat, relay._send_receipts = s, SendReceipts(tmp_path / "receipts.db")
    ws = SimpleNamespace(send_json=AsyncMock())
    data = {"req_id": "r", "action": "send_text", "chat_id": "group:42", "content": "第一段",
            "social_part": {"token": token, "index": 0}}
    await asyncio.gather(relay._handle_send(ws, data), relay._handle_send(ws, data))
    assert api.send_group_msg.await_count == 1
    relay._send_receipts.close()
    await relay._handle_send(ws, {**data, "social_part": {"token": token, "index": 1}})
    assert api.send_group_msg.await_count == 1
    assert ws.send_json.await_args.args[0]["retryable"] is False


async def test_judgment_protocol_and_capability_handshake_are_nonblocking():
    relay = HermesRelayServer(config(), MagicMock(), "test", lambda: True)
    ws = MagicMock()
    ws.send_json = AsyncMock()
    relay._clients.add(ws)
    await relay._handle_text(ws, json.dumps({"type": "plugin_info", "plugin_version": "test",
                                           "capabilities": ["social_chat_v1"]}))
    task = asyncio.create_task(relay.request_social_judgment({"messages": []}))
    await asyncio.sleep(0)
    request = ws.send_json.await_args.args[0]
    assert request["type"] == "social_judge"
    await relay._handle_text(ws, json.dumps({"type": "social_result", "req_id": request["req_id"],
                                           "result": {"action": "reply", "intent": "react"}}))
    assert (await task)["action"] == "reply"
    assert not relay._social_waiters


def test_social_part_fingerprint_is_turn_scoped():
    from onebot_adapter.relay.hermes_ws import _send_fingerprint

    first = {"content": "嗯", "social_part": {"token": "a" * 32, "index": 0}}
    second = {**first, "social_part": {"token": "b" * 32, "index": 0}}
    assert _send_fingerprint("send_text", first) != _send_fingerprint("send_text", second)
