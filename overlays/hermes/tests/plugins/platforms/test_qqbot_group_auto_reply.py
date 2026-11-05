"""Opt-in QQ full-group intake must remain quiet, bounded and group-scoped."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import PlatformConfig


def make_adapter(**overrides):
    from plugins.platforms.qqbot.adapter import QQAdapter

    extra = {
        'app_id': 'bot-app', 'client_secret': 'test-secret',
        'group_policy': 'allowlist', 'group_allow_from': ['group-a', 'group-b'],
        'group_auto_reply': {
            'group-a': {
                'probability': 1, 'cooldown_seconds': 90,
                'window_seconds': 600, 'max_replies': 4,
                'names': ['Kaguya', '辉夜'],
            },
        },
    }
    extra.update(overrides)
    return QQAdapter(PlatformConfig(enabled=True, extra=extra))


def group_message(message_id='message-1', group='group-a', content='今天这个话题还挺有意思的'):
    return {
        'id': message_id, 'group_openid': group, 'content': content,
        'author': {'member_openid': 'member-1'},
        'timestamp': '2026-10-26T12:00:00+00:00',
    }


@pytest.fixture
def policy():
    from plugins.platforms.qqbot.group_auto_reply import GroupAutoReplyPolicy

    clock = [1000.0]
    rules = {'group-a': {'probability': 1, 'names': ['Kaguya', '辉夜']}}
    value = GroupAutoReplyPolicy(rules, clock=lambda: clock[0], sample=lambda: 0.0)
    return value, clock


def test_no_implicit_group_opt_in(policy):
    gate, _ = policy
    assert gate.admit('group-b', 'Kaguya 你怎么看这个问题') is False


def test_cooldown_is_reserved_before_next_message(policy):
    gate, clock = policy
    assert gate.admit('group-a', 'Kaguya 你怎么看') is True
    assert gate.admit('group-a', '辉夜 再说两句') is False
    clock[0] += 89
    assert gate.admit('group-a', 'Kaguya 你怎么看') is False
    clock[0] += 1
    assert gate.admit('group-a', 'Kaguya 你怎么看') is True


def test_rolling_budget_includes_named_messages(policy):
    gate, clock = policy
    for _ in range(4):
        assert gate.admit('group-a', 'Kaguya 你怎么看') is True
        clock[0] += 90
    assert gate.admit('group-a', 'Kaguya 继续聊') is False
    clock[0] = 1600
    assert gate.admit('group-a', 'Kaguya 继续聊') is True


def test_explicit_reply_activity_delays_unsolicited_reply(policy):
    gate, clock = policy
    gate.note_activity('group-a')
    assert gate.admit('group-a', '这个话题现在可以聊一聊') is False
    clock[0] += 90
    assert gate.admit('group-a', '这个话题现在可以聊一聊') is True


def test_sampling_does_not_spend_budget_but_names_still_can_trigger():
    from plugins.platforms.qqbot.group_auto_reply import GroupAutoReplyPolicy

    gate = GroupAutoReplyPolicy({'group-a': {'probability': 0, 'names': ['Kaguya']}}, sample=lambda: 0)
    assert gate.admit('group-a', '今天有个很有意思的新发现') is False
    assert gate.admit('group-a', 'kaguya 你怎么看') is True


@pytest.mark.parametrize('text', ['', '嗯', '哈哈', '......', '123456789', '<@member-2> 你觉得呢'])
def test_noise_and_messages_addressed_to_others_are_not_sampled(policy, text):
    gate, _ = policy
    assert gate.admit('group-a', text) is False


@pytest.mark.parametrize('invalid', [{'probability': 2}, {'cooldown_seconds': 0}, {'max_replies': 0}, {'window_seconds': -1}])
def test_invalid_limits_fail_closed(invalid):
    from plugins.platforms.qqbot.group_auto_reply import GroupAutoReplyPolicy

    gate = GroupAutoReplyPolicy({'group-a': invalid})
    assert gate.admit('group-a', 'Kaguya 你好') is False


def test_group_budgets_do_not_cross_groups():
    from plugins.platforms.qqbot.group_auto_reply import GroupAutoReplyPolicy

    gate = GroupAutoReplyPolicy({'group-a': {'probability': 1}, 'group-b': {'probability': 1}}, sample=lambda: 0)
    assert gate.admit('group-a', '这是第一个群自己的聊天内容') is True
    assert gate.admit('group-b', '这是第二个群自己的聊天内容') is True


@pytest.mark.asyncio
async def test_full_group_event_reaches_existing_ingest_with_silence_allowed():
    adapter = make_adapter()
    adapter._ingest = AsyncMock()
    message = group_message()
    await adapter._on_message('GROUP_MESSAGE_CREATE', message)
    adapter._ingest.assert_awaited_once()
    assert adapter._ingest.call_args.kwargs['auto_reply'] is True
    assert adapter._ingest.call_args.kwargs['chat_id'] == 'group-a'
    assert adapter._ingest.call_args.kwargs['user_id'] == 'member-1'


@pytest.mark.asyncio
async def test_unconfigured_group_full_event_is_ignored_but_at_event_still_works():
    adapter = make_adapter()
    adapter._ingest = AsyncMock()
    await adapter._on_message('GROUP_MESSAGE_CREATE', group_message(group='group-b'))
    adapter._ingest.assert_not_awaited()
    await adapter._on_message('GROUP_AT_MESSAGE_CREATE', group_message('at-1', 'group-b'))
    adapter._ingest.assert_awaited_once()


@pytest.mark.asyncio
async def test_acl_and_sender_principal_checked_before_media_or_model():
    adapter = make_adapter(group_allow_from=['group-b'])
    adapter._ingest = AsyncMock()
    await adapter._on_message('GROUP_MESSAGE_CREATE', group_message())
    adapter._ingest.assert_not_awaited()
    message = group_message('empty-author', 'group-b')
    message['author'] = {}
    await adapter._on_message('GROUP_MESSAGE_CREATE', message)
    adapter._ingest.assert_not_awaited()


@pytest.mark.asyncio
async def test_duplicate_full_and_at_events_do_not_reply_twice():
    adapter = make_adapter()
    adapter._ingest = AsyncMock()
    message = group_message()
    await adapter._on_message('GROUP_MESSAGE_CREATE', message)
    await adapter._on_message('GROUP_AT_MESSAGE_CREATE', message)
    adapter._ingest.assert_awaited_once()


@pytest.mark.asyncio
async def test_plain_commands_and_explicit_mentions_bypass_passive_cooldown():
    adapter = make_adapter()
    adapter._ingest = AsyncMock()
    await adapter._on_message('GROUP_MESSAGE_CREATE', group_message())
    await adapter._on_message('GROUP_MESSAGE_CREATE', group_message('stop-1', content='/stop'))
    await adapter._on_message('GROUP_MESSAGE_CREATE', group_message('at-1', content='<@bot-app> 帮我看看'))
    assert adapter._ingest.await_count == 3
    assert not adapter._ingest.call_args.kwargs.get('auto_reply')


@pytest.mark.asyncio
async def test_inflight_session_is_not_interrupted_by_auto_reply():
    from gateway.platforms.event import MessageEvent

    adapter = make_adapter()
    event = MessageEvent(text='普通群消息', source=adapter.build_source(chat_id='group-a', chat_type='group', user_id='member-1'))
    event.metadata['qqbot_auto_reply'] = True
    adapter._active_sessions['active-key'] = object()
    with patch.object(adapter, '_drop_unresolved', return_value=False), patch.object(adapter, '_event_session_key', return_value='active-key'), patch('gateway.platforms.base.BasePlatformAdapter.handle_message', new_callable=AsyncMock) as deliver:
        await adapter.handle_message(event)
    deliver.assert_not_awaited()


@pytest.mark.asyncio
async def test_full_mode_marks_unaddressed_event_without_changing_sender():
    adapter = make_adapter()
    adapter.handle_message = AsyncMock()
    await adapter._on_message('GROUP_MESSAGE_CREATE', group_message())
    event = adapter.handle_message.call_args.args[0]
    assert event.reply_expected is False
    assert event.allow_gateway_control is False
    assert event.source.user_id == 'member-1'
    assert event.source.chat_id == 'group-a'
    assert '未@机器人' in event.channel_context


@pytest.mark.asyncio
async def test_real_websocket_dispatch_accepts_full_group_event():
    adapter = make_adapter()
    received = asyncio.Event()
    adapter._ingest = AsyncMock(side_effect=lambda *args, **kwargs: received.set())
    adapter._dispatch_payload({'op': 0, 't': 'GROUP_MESSAGE_CREATE', 's': 12, 'd': group_message()})
    async with asyncio.timeout(2):
        await received.wait()
    assert adapter._last_seq == 12
    assert adapter._ingest.call_args.kwargs['auto_reply'] is True


def test_user_yaml_setting_reaches_adapter_through_gateway_loader(tmp_path):
    from gateway.config import GatewayConfig, Platform
    from hermes_cli.config import atomic_config_write
    from hermes_cli.config_effective import load_user_config_effective
    from plugins.platforms.qqbot.adapter import QQAdapter

    path = tmp_path / 'config.yaml'
    atomic_config_write(path, {'platforms': {'qqbot': {'enabled': True, 'extra': {
        'app_id': 'test-bot', 'client_secret': 'test-secret',
        'group_policy': 'allowlist', 'group_allow_from': ['group-a', 'group-b'],
        'group_auto_reply': {'group-a': {'probability': 0, 'names': ['Kaguya']}},
    }}}})
    config = GatewayConfig.from_dict(load_user_config_effective(path, fail_closed=True))
    adapter = QQAdapter(config.platforms[Platform.QQBOT])
    assert adapter._group_auto_reply.admit('group-a', 'Kaguya 你怎么看') is True
    assert adapter._group_auto_reply.admit('group-b', 'Kaguya 你怎么看') is False
    assert adapter._group_allow_from == ['group-a', 'group-b']
