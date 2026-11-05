"""Deployment contract: unknown groups and DMs must not reach Hermes."""
from onebot_adapter.config import AdapterConfig
from onebot_adapter.onebot.parser import parse_event
from onebot_adapter.relay.protocol import DroppedEvent, NormalizedEvent


def event(group_id=None):
    data = {
        "post_type": "message", "message_type": "group" if group_id else "private",
        "user_id": 100, "message_id": 1, "time": 1700000000,
        "sender": {"nickname": "Tester", "user_id": 100},
        "message": [{"type": "text", "data": {"text": "hello"}}],
    }
    if group_id:
        data["group_id"] = group_id
    return data


async def test_deployment_group_allowlist():
    config = AdapterConfig(
        group_default_enabled=False, group_require_mention=False,
        groups={"42": {"enabled": True}},
    )
    unknown = await parse_event(event(43), self_id="999", config=config, group_require_mention=False)
    allowed = await parse_event(event(42), self_id="999", config=config, group_require_mention=False)
    assert isinstance(unknown, DroppedEvent)
    assert isinstance(allowed, NormalizedEvent)
    assert allowed.chat_id == "group:42"


async def test_deployment_dm_allowlist():
    config = AdapterConfig(dm_policy="deny", dm_whitelist=["101"])
    rejected = await parse_event(event(), self_id="999", config=config, group_require_mention=True)
    assert isinstance(rejected, DroppedEvent)
    config.dm_whitelist = ["100"]
    allowed = await parse_event(event(), self_id="999", config=config, group_require_mention=True)
    assert isinstance(allowed, NormalizedEvent)


def test_invalid_default_group_policy_rejected():
    assert "group_default_enabled must be bool" in AdapterConfig(group_default_enabled="yes").validate()


def test_unlisted_slash_commands_are_not_allowed():
    config = AdapterConfig(
        command_filter_enabled=True, command_filter_unknown=True,
        command_default_permission="disabled", command_permissions={"help": "everyone"},
    )
    assert config.check_command_permission("42", "100", "help", True)[0]
    assert not config.check_command_permission("42", "100", "model", True)[0]
    assert not config.check_command_permission(None, "100", "new-command", False)[0]
