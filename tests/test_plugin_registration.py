from __future__ import annotations

import logging
import sys
from dataclasses import make_dataclass
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from onebot_adapter.hermes_plugin import adapter, onebot_tools
from onebot_adapter.hermes_plugin.social import JUDGMENT_TASK

_AUTH_OPTIONS = {
    "group_allowed_chats_env": "ONEBOT_GROUP_ALLOWED_CHATS",
    "allowlist_suppresses_pairing": True,
}


@pytest.mark.parametrize("supported", [
    (),
    ("group_allowed_chats_env",),
    ("allowlist_suppresses_pairing",),
    tuple(_AUTH_OPTIONS),
])
@pytest.mark.parametrize("configured", [False, True])
def test_registration_adapts_to_host_fields(monkeypatch, caplog, supported, configured):
    # Model the real registration boundary: unknown PlatformEntry keywords
    # raise TypeError, unlike a permissive MagicMock register_platform().
    entry_type = make_dataclass("PlatformEntry", [(name, object) for name in supported])
    registry = ModuleType("gateway.platform_registry")
    registry.PlatformEntry = entry_type
    monkeypatch.setitem(sys.modules, "gateway.platform_registry", registry)
    monkeypatch.delenv("ONEBOT_GROUP_ALLOWED_CHATS", raising=False)
    if configured:
        monkeypatch.setenv("ONEBOT_GROUP_ALLOWED_CHATS", "group:12345")
    captured = []

    def register_platform(**kwargs):
        options = {name: value for name, value in kwargs.items() if name in _AUTH_OPTIONS}
        captured.append((kwargs, entry_type(**options)))

    ctx = SimpleNamespace(register_platform=register_platform, register_auxiliary_task=Mock())
    register_tools = Mock()
    monkeypatch.setattr(onebot_tools, "register_tools", register_tools)
    with caplog.at_level(logging.WARNING, logger=adapter.__name__):
        adapter.register(ctx)
    assert len(captured) == 1
    kwargs, entry = captured[0]
    assert kwargs["name"] == "onebot"
    assert kwargs["required_env"] == ["ONEBOT_ADAPTER_URL", "ONEBOT_ADAPTER_TOKEN"]
    for name, value in _AUTH_OPTIONS.items():
        if name in supported:
            assert getattr(entry, name) == value
        else:
            assert name not in kwargs
    register_tools.assert_called_once_with(ctx)
    needs_upgrade = configured and "group_allowed_chats_env" not in supported
    assert ("upgrade Hermes" in caplog.text) == needs_upgrade
    assert "group:12345" not in caplog.text


def test_registration_options_without_hermes(monkeypatch):
    monkeypatch.setitem(sys.modules, "gateway.platform_registry", None)
    monkeypatch.delenv("ONEBOT_GROUP_ALLOWED_CHATS", raising=False)
    assert adapter._group_auth_registration_kwargs() == {}


def test_social_judgment_registers_independent_host_task(monkeypatch):
    ctx = SimpleNamespace(register_platform=Mock(), register_auxiliary_task=Mock())
    monkeypatch.setattr(onebot_tools, "register_tools", Mock())
    adapter.register(ctx)
    ctx.register_auxiliary_task.assert_called_once()
    args, kwargs = ctx.register_auxiliary_task.call_args
    assert args == (JUDGMENT_TASK,)
    assert kwargs["defaults"] == {"timeout": 18}
    assert not {"provider", "model", "api_key"} & kwargs["defaults"].keys()
