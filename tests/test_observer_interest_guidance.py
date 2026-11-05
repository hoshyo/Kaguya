"""Interest-based admission contract, with mocked verdicts, never real model decisions."""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from onebot_adapter.hermes_plugin.observer_agent import INSTRUCTIONS
from tests.test_observer_agent_mode import cfg
from tests.test_social_chat import message, social


def test_observer_distinguishes_social_interest_from_assistance_and_keyword_matching():
    for guidance in ("不是有没有人需要帮助", "平等参与者", "不需要问句、@、点名、求助或知识价值",
                     "一个人连发分享也可能是在向全群说话", "有自然接话空间", "倾向wake",
                     "不按美食、游戏等关键词强制唤醒", "同一分享的几条补充合起来判断",
                     "历史skip只是旧决定", "个人安静偏好和权限限制仍须尊重",
                     "图片占位不是实际图片", "不必等观察者先看清图片", "只向另一个人追问的封闭交流"):
        assert guidance in INSTRUCTIONS
    assert '{"action":"wake"}' in INSTRUCTIONS and '{"action":"skip"}' in INSTRUCTIONS
    assert "不能调用任何工具" in INSTRUCTIONS and "不能生成群聊正文" in INSTRUCTIONS
    assert "若本次只有已经唤醒的消息就skip" in INSTRUCTIONS


def test_observer_soul_reinforces_interest_without_changing_its_role_or_protocol():
    soul = (Path(__file__).resolve().parents[1] / "profiles/qq-observer/SOUL.md").read_text()
    for guidance in ("没有问句、没有@、没人点名", "角色可以出于兴趣参与", "不必提供帮助或知识",
                     "不把这些话题做成必触发关键词", "是否真说话仍由聊天角色决定", "尊重安静偏好"):
        assert guidance in soul
    assert '{"action":"wake"}' in soul and '{"action":"skip"}' in soul
    assert "不使用工具，不更新角色记忆或技能" in soul
    assert "不代写消息，不决定它说什么或回复谁" in soul


def test_explicit_follow_on_tasks_win_over_unsolicited_restraint_and_old_skip_decisions():
    for guidance in ("体验优先", "必须wake", "那你自己去搜一下最近有什么这个活动",
                     "那帮我找找杭州的", "同一作者或同一话题之后的新消息已被处理",
                     "一份回答分成两条发送也不等于连续抢话", "不要因自己没有搜索等工具就skip任务",
                     "recent_exchange_with_role=true", "它不是无条件wake开关"):
        assert guidance in INSTRUCTIONS
    soul = (Path(__file__).resolve().parents[1] / "profiles/qq-observer/SOUL.md").read_text()
    assert "体验优先于减少调用" in soul and "必须 wake" in soul
    assert "不只识别这一句" in soul and "个人安静偏好" in soul
    assert "腿有点短了" in soul and "腿有点短了" in INSTRUCTIONS
    assert "观察者不必先看懂图片" in INSTRUCTIONS


def test_slightly_more_social_bias_keeps_a_concrete_opening_and_recent_bot_crowding_check():
    for guidance in ("只小幅放宽边界", "接一句也自然、不接也可以", "优先wake一次",
                     "不要求话题特别精彩", "同一话题不等于重复", "相关短句不因字少就跳过",
                     "有实际接点", "没有连续抢话", "群友没有承接时优先skip",
                     "不为刷存在感重复点评", "群友在接角色的话或回答它的问题则正常考虑唤醒"):
        assert guidance in INSTRUCTIONS
    soul = (Path(__file__).resolve().parents[1] / "profiles/qq-observer/SOUL.md").read_text()
    for guidance in ("只小幅放宽边界", "优先 wake 一次", "仍须有上下文依据",
                     "近期角色已连续主动插话而群友没承接时优先 skip", "不机械压掉续聊"):
        assert guidance in soul


def test_open_questions_are_not_default_helpdesk_tasks_or_a_punctuation_filter():
    for phrase in ("开放提问不等于向角色求助", "降低主动抢答倾向，通常skip",
                   "不能只因角色能答、问题清楚或有问号就wake", "不按问句关键词硬拦截",
                   "自然续聊的新问题", "轻松选择、聊天邀请", "自然分享和接梗稍积极一点",
                   "不放宽连续抢话的限制"):
        assert phrase in INSTRUCTIONS
    soul = (Path(__file__).resolve().parents[1] / "profiles/qq-observer/SOUL.md").read_text()
    for phrase in ("向全群开放的知识、技术和咨询提问", "没有点名或续聊关系时，通常旁观",
                   "不能按问句或关键词一刀切", "不机械补足次数"):
        assert phrase in soul


@pytest.mark.parametrize("group", [42, 43])
@pytest.mark.parametrize("text", ["草莓酸奶里果肉好多，真好吃", "刚画完这张画，颜色终于调对了", "这首歌的副歌真上头"])
async def test_unaddressed_sharing_can_wake_either_group_without_a_request_for_help(group, text):
    state = social(cfg(), judge=AsyncMock(return_value={"action": "wake"}))
    emit = AsyncMock()
    data = message(group=group, text=text)
    try:
        assert state.observe(data)
        state.offer(data, emit)
        await asyncio.gather(*list(state.adaptive.tasks.values()))
        await asyncio.gather(*list(state.adaptive.wake_tasks.values()))
        state.judge.assert_awaited_once()
        assert state.judge.await_args.args[0]["messages"][0]["text"] == text
        assert state.judge.await_args.args[0]["group_id"] == str(group)
        emit.assert_awaited_once()
        original, followup, hint, _ = emit.await_args.args
        assert original == data and not followup
        assert state.turns[hint["token"]]["wake_only"]
        assert state.turns[hint["token"]]["unmentioned"]
    finally:
        await state.close()


@pytest.mark.parametrize("text", ["美食 游戏 新作品", "草莓酸奶真香", "我也想尝尝"])
async def test_interest_words_do_not_override_a_skip_verdict(text):
    state = social(cfg(), judge=AsyncMock(return_value={"action": "skip"}))
    emit = AsyncMock()
    data = message(text=text)
    try:
        state.observe(data)
        state.offer(data, emit)
        await asyncio.gather(*list(state.adaptive.tasks.values()))
        state.judge.assert_awaited_once()
        emit.assert_not_awaited()
        assert not state.turns
    finally:
        await state.close()


async def test_personal_quiet_preference_still_blocks_interest_admission_before_luna():
    state = social(cfg(), judge=AsyncMock(return_value={"action": "wake"}))
    state.preferences.set("42", "100", "mention_only")
    emit = AsyncMock()
    data = message(text="这个甜点真好吃")
    try:
        state.observe(data)
        state.offer(data, emit)
        assert not state.adaptive.tasks
        state.judge.assert_not_awaited()
        emit.assert_not_awaited()
        assert state.stats["member_quiet_skip"] == 1
    finally:
        await state.close()


async def test_cold_observation_batch_keeps_sharing_fragments_and_recent_bot_context():
    state = social(cfg(), judge=AsyncMock(return_value={"action": "wake"}))
    state.adaptive.state("42").score = .1
    state.successful_reply("42", "100", "900", "你刚才说的是哪种酸奶？")
    emit = AsyncMock()
    try:
        for mid, text in enumerate(("新买的草莓酸奶", "果肉好多", "好吃"), 1):
            data = message(mid, text=text)
            state.observe(data)
            state.offer(data, emit)
        await asyncio.gather(*list(state.adaptive.tasks.values()))
        await asyncio.gather(*list(state.adaptive.wake_tasks.values()))
        state.judge.assert_awaited_once()
        payload = state.judge.await_args.args[0]
        assert [row["text"] for row in payload["messages"]] == ["新买的草莓酸奶", "果肉好多", "好吃"]
        assert payload["recent_bot_messages"][-1]["text"] == "你刚才说的是哪种酸奶？"
        assert payload["current_message_id"] == "3"
        emit.assert_awaited_once()
        assert state.adaptive.state("42").interval(state.clock()) == 30  # +2 actual reply, not a wake reward
    finally:
        await state.close()
