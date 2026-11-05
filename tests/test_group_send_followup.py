"""Unmentioned group replies retain dialogue candidates, not public pings."""
from __future__ import annotations

import asyncio
from collections import deque
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from onebot_adapter.relay.hermes_ws import HermesRelayServer
from onebot_adapter.send_receipts import SendReceipts
from tests.test_social_chat import config, message, social


@pytest.fixture(params=["42", "43"])
async def exchange(request, tmp_path):
    group = request.param
    now = [100.0]
    cfg = config(delivery_enabled=True)
    s = social(cfg, clock=lambda: now[0])
    s.followups._clock = lambda: now[0]
    s.rule["quiet_seconds"] = 0
    for mid, user in enumerate((100, 101, 102, 103), 10):
        assert s.observe(message(mid, user=user, group=int(group), text="虚构群聊"))
    event = SimpleNamespace(user_id="100", message_id="10")
    s.direct(group, event)
    api = SimpleNamespace(send_group_msg=AsyncMock(return_value={"message_id": "50"}))
    relay = HermesRelayServer(cfg, api, "test", lambda: True, group_followups=s.followups)
    relay._social_chat = s
    relay._send_receipts = SendReceipts(tmp_path / "receipts.db")
    relay._busy_groups[group] = ("100", now[0])
    try:
        yield SimpleNamespace(group=group, now=now, social=s, relay=relay, api=api, emit=AsyncMock(),
                              token=event.social_reply["token"])
    finally:
        await s.close()
        await relay.stop()
        relay._send_receipts.close()


async def send_group(e):
    ws = SimpleNamespace(send_json=AsyncMock())
    await e.relay._handle_send(ws, {
        "action": "send_text", "req_id": "synthetic", "chat_id": "group:" + e.group,
        "content": "有什么好玩的？", "conversation_reply": True, "conversation_user_id": "100",
        "social_part": {"token": e.token, "index": 0},
        "social_delivery": {"mode": "group", "target": None, "notify": False},
    })
    e.relay._busy_groups.clear()
    return ws.send_json.await_args.args[0]


async def offer_answer(e, *, seconds=41, segments=None):
    e.now[0] += seconds
    data = message(99, user=100, group=int(e.group), text="给你玩好玩的带你吃好吃的", segments=segments)
    assert e.social.observe(data)
    e.social.offer(data, e.emit)
    await asyncio.gather(*e.social.tasks.values())
    return data


async def test_group_send_retains_dialogue_member_without_quotes_or_pings(exchange):
    e = exchange
    assert (await send_group(e))["success"]
    assert [s["type"] for s in e.api.send_group_msg.await_args.args[1]] == ["text"]
    assert (e.group, "100") in e.social.followups._recent
    assert (e.group, "101") not in e.social.followups._recent
    e.social.admissions[e.group] = deque([e.now[0]])  # ordinary 90-second cooldown is active
    data = await offer_answer(e)
    state = e.social.conversation_state(e.group, "100", "99")
    assert state["active_speakers"] == 5 and not state["continuing_exchange"]
    e.social.judge.assert_awaited_once()
    assert e.social.judge.await_args.args[0]["followup"] is True
    e.emit.assert_awaited_once()
    assert e.emit.await_args.args[1] is True
    other = "43" if e.group == "42" else "42"
    assert e.social.followups.classify(other, "100", data["message"]) is None


async def test_group_send_followup_can_still_be_skipped_by_judge(exchange):
    e = exchange
    assert (await send_group(e))["success"]
    e.social.judge.return_value = {"action": "skip", "intent": "acknowledge"}
    await offer_answer(e)
    e.social.judge.assert_awaited_once()
    assert e.social.judge.await_args.args[0]["followup"] is True
    assert e.social.stats["skip"] == 1 and not e.social.admissions.get(e.group)
    assert not e.emit.called


async def test_group_send_followup_respects_member_quiet_preference(exchange):
    e = exchange
    assert (await send_group(e))["success"]
    e.social.preferences.set(e.group, "100", "mention_only")
    await offer_answer(e)
    assert not e.social.judge.called and e.social.stats["member_quiet_skip"] == 1


async def test_group_send_followup_expires_and_returns_to_ordinary_cooldown(exchange):
    e = exchange
    assert (await send_group(e))["success"]
    e.social.admissions[e.group] = deque([e.now[0] + 121])
    await offer_answer(e, seconds=121)
    assert not e.social.judge.called and e.social.stats["budget_or_stale_skip"] == 1


@pytest.mark.parametrize("quota", ["member", "hourly"])
async def test_group_send_followup_keeps_existing_judge_quotas(exchange, quota):
    e = exchange
    assert (await send_group(e))["success"]
    if quota == "member":
        e.social.followups._checks[(e.group, "100")] = deque([e.now[0]] * 6)
    else:
        e.social.checks[e.group] = deque([e.now[0]] * 60)
    await offer_answer(e)
    assert not e.social.judge.called
    key = "followup_budget_skip" if quota == "member" else "budget_or_stale_skip"
    assert e.social.stats[key] == 1


async def test_group_send_does_not_follow_when_member_addresses_someone_else(exchange):
    e = exchange
    assert (await send_group(e))["success"]
    await offer_answer(e, segments=[{"type": "at", "data": {"qq": "101"}},
                                    {"type": "text", "data": {"text": "我在和你说"}}])
    assert not e.social.judge.called and e.social.stats["addressed_elsewhere_skip"] == 1
    assert (e.group, "100") not in e.social.followups._recent


async def test_failed_group_send_does_not_create_dialogue_anchor(exchange):
    e = exchange
    e.api.send_group_msg.side_effect = RuntimeError("synthetic send failed")
    assert not (await send_group(e))["success"]
    assert not e.social.followups._recent and not e.social.followups._outbox


@pytest.mark.parametrize("success", [True, False])
async def test_image_receipt_updates_observer_only_after_successful_send(exchange, success):
    e = exchange
    e.social.rule["observer_agent_enabled"] = True
    if not success:
        e.api.send_group_msg.side_effect = RuntimeError("synthetic image send failed")
    ws = SimpleNamespace(send_json=AsyncMock())
    await e.relay._handle_send(ws, {
        "action": "send_image", "req_id": "synthetic-image", "chat_id": "group:" + e.group,
        "image_url": "/synthetic/generated.png", "conversation_reply": True, "conversation_user_id": "100",
    })
    assert ws.send_json.await_args.args[0]["success"] is success
    rows = [r for r in e.social.rows(e.group) if r.get("bot")]
    assert bool(rows) is success
    if success:
        assert rows[-1]["text"] == "[已发送图片，观察者未识别画面]"
        assert "/synthetic" not in str(rows)
