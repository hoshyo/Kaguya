"""Candidate cues are shared, per-turn and advisory; never truncate model output."""
from copy import deepcopy

import pytest

from onebot_adapter.hermes_plugin.adapter import _channel_context
from onebot_adapter.hermes_plugin.social import PASSIVE_PARTICIPATION_HINT, participation_hint


def event(*direct):
    return {"chat_type": "group", "chat_id": "group:42", "wake_only": True,
            "wakeup_batch": {"id": "synthetic", "requests": [
                {"message_id": str(i + 1), "user_id": str(100 + i), "direct": value}
                for i, value in enumerate(direct)]}}


@pytest.mark.parametrize("direct", [(False,), (True, False), (False, True)])
def test_passive_or_mixed_batch_gets_the_same_cue_for_both_profiles(direct):
    data = event(*direct)
    before = deepcopy(data)
    assert participation_hint(data) == PASSIVE_PARTICIPATION_HINT
    outputs = [_channel_context({**data, "profile": profile}) for profile in ("aidog", "kig")]
    assert outputs[0] == outputs[1]
    assert outputs[0].count(PASSIVE_PARTICIPATION_HINT) == 1
    assert data == before


def test_explicit_only_does_not_get_the_unsolicited_answer_cue():
    assert participation_hint(event(True, True)) is None
    assert PASSIVE_PARTICIPATION_HINT not in _channel_context(event(True))


def test_batch_survives_restore_without_the_legacy_passive_flag():
    data = event(False)
    data["social_reply"] = {"wakeup_batch": data.pop("wakeup_batch")}
    assert participation_hint(data) == PASSIVE_PARTICIPATION_HINT


@pytest.mark.parametrize("data", [
    {"chat_type": "private", "chat_id": "private:100", "passive_reply": True},
    {"chat_type": "group", "chat_id": "group:42"},
    {"chat_type": "group", "chat_id": "group:42", "wakeup_batch": {"requests": "false"}},
])
def test_no_candidate_or_non_group_does_not_get_the_cue(data):
    assert participation_hint(data) is None


def test_cue_preserves_invited_followups_fact_checks_and_no_reply_without_length_caps():
    for phrase in ("明确点名/@你、接着问你或继续交代任务", "不因没有重新@而敷衍或漏答",
                   "不默认展开步骤、条件清单、整套建议", "对方继续找你展开时再跟着讲",
                   "不设硬性字数句数", "不省掉影响理解的关键信息", "没有自然接点可NO_REPLY",
                   "查证、取图、权限和个人安静偏好照常遵守"):
        assert phrase in PASSIVE_PARTICIPATION_HINT


def test_prefill_fast_path_still_gets_the_shared_passive_scope_cue():
    data = {**event(False), "context_prefill": {"messages": []}}
    assert PASSIVE_PARTICIPATION_HINT in _channel_context(data)
