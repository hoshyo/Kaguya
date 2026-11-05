"""Real WS callback scheduling with synthetic, context-local Hermes host services."""
from __future__ import annotations

import asyncio
import contextvars
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from onebot_adapter.hermes_plugin import adapter as plugin
from onebot_adapter.hermes_plugin.social import INSTRUCTIONS


def test_answer_to_bots_own_question_is_not_automatically_an_ending():
    assert '同一人紧接着提供答案、选择或补充' in INSTRUCTIONS
    assert '不能只因答案短、没问号就skip+acknowledge' in INSTRUCTIONS
    assert '明确结束、要求安静或转向别人时仍应skip' in INSTRUCTIONS


@pytest.fixture
def host_scope(monkeypatch):
    home = contextvars.ContextVar("synthetic_home", default="caller")
    secrets = contextvars.ContextVar("synthetic_secrets", default=None)
    secrets.set({"key": "caller-only"})
    reads = []
    constants = ModuleType("hermes_constants")
    constants.set_hermes_home_override = home.set
    constants.reset_hermes_home_override = home.reset
    secret_scope = ModuleType("agent.secret_scope")

    def bind(mapping, *, profile_home):
        assert profile_home == home.get()
        return secrets.set(mapping)

    secret_scope.set_secret_scope = bind
    secret_scope.reset_secret_scope = secrets.reset
    policy = ModuleType("tui_gateway.launch_profile_policy")

    def owner_scope(owner):
        assert home.get() == str(owner)
        reads.append(str(owner))
        return {"key": str(owner) + "-only"}

    policy.owner_secret_scope = owner_scope
    for name, module in ((constants.__name__, constants), (secret_scope.__name__, secret_scope),
                         (policy.__name__, policy)):
        monkeypatch.setitem(sys.modules, name, module)
    return SimpleNamespace(home=home, secrets=secrets, reads=reads, policy=policy)


def adapter(home="/synthetic/transport"):
    a = plugin.OneBotAdapter.__new__(plugin.OneBotAdapter)
    a._social_home = Path(home)
    a._push_tasks = set()
    a._social_llm = SimpleNamespace(acomplete_structured=AsyncMock())
    a._ws = SimpleNamespace(closed=False, send_json=AsyncMock())
    return a


def frame():
    return {"type": "social_judge", "req_id": "synthetic-id", "profile": "untrusted-other",
            "payload": {"messages": [{"user_id": "100", "text": "汉堡"}]}}


async def test_receiver_background_task_binds_owner_not_ambient_or_frame(host_scope):
    a = adapter()

    async def complete(**kwargs):
        assert host_scope.home.get() == str(a._social_home)
        assert host_scope.secrets.get() == {"key": str(a._social_home) + "-only"}
        await asyncio.sleep(0)
        assert host_scope.home.get() == str(a._social_home)
        assert not {"provider", "model", "profile", "agent_id"} & kwargs.keys()
        return SimpleNamespace(parsed={"action": "reply", "intent": "react"})

    a._social_llm.acomplete_structured.side_effect = complete
    await a._handle_text(json.dumps(frame()))
    await asyncio.gather(*a._push_tasks)
    assert host_scope.reads == [str(a._social_home)]
    assert host_scope.home.get() == "caller"
    assert host_scope.secrets.get() == {"key": "caller-only"}
    assert a._ws.send_json.await_args.args[0] == {
        "type": "social_result", "v": 1, "req_id": "synthetic-id",
        "result": {"action": "reply", "intent": "react"}}


async def test_cancellation_restores_native_scopes_without_sending(host_scope):
    a = adapter()
    a._social_llm.acomplete_structured.side_effect = asyncio.CancelledError
    with pytest.raises(asyncio.CancelledError):
        await a._judge_social(frame())
    assert host_scope.home.get() == "caller"
    assert host_scope.secrets.get() == {"key": "caller-only"}
    assert not a._ws.send_json.called


async def test_failure_is_distinct_and_logs_only_code_locations(host_scope, caplog):
    a = adapter()
    sensitive = "secret-token https://private.invalid raw-chat-body"
    a._social_llm.acomplete_structured.side_effect = RuntimeError(sensitive)
    await a._judge_social(frame())
    response = a._ws.send_json.await_args.args[0]
    assert response["error"] == "social_judgment_failed"
    assert response["result"]["action"] == "skip"  # older receivers still fail closed
    assert "RuntimeError" in caplog.text and "_judge_social" in caplog.text
    for secret in sensitive.split():
        assert secret not in caplog.text
        assert secret not in json.dumps(response)
    assert host_scope.home.get() == "caller"
    assert host_scope.secrets.get() == {"key": "caller-only"}


async def test_scope_load_failure_never_borrows_ambient_credentials(host_scope, caplog):
    a = adapter()

    def unavailable(owner):
        raise OSError("sensitive-file-content")

    host_scope.policy.owner_secret_scope = unavailable
    await a._judge_social(frame())
    assert not a._social_llm.acomplete_structured.called
    assert a._ws.send_json.await_args.args[0]["error"] == "social_judgment_failed"
    assert "sensitive-file-content" not in caplog.text
    assert host_scope.home.get() == "caller"
    assert host_scope.secrets.get() == {"key": "caller-only"}


async def test_concurrent_transport_owners_do_not_share_scopes(host_scope):
    adapters = [adapter("/synthetic/default"), adapter("/synthetic/secondary")]

    async def run(a):
        async def complete(**kwargs):
            await asyncio.sleep(0)
            assert host_scope.home.get() == str(a._social_home)
            assert host_scope.secrets.get()["key"] == str(a._social_home) + "-only"
            return SimpleNamespace(parsed={"action": "reply", "intent": "react"})

        a._social_llm.acomplete_structured.side_effect = complete
        await a._handle_text(json.dumps(frame()))
        await asyncio.gather(*a._push_tasks)

    await asyncio.gather(*(run(a) for a in adapters))
    assert all("error" not in a._ws.send_json.await_args.args[0] for a in adapters)
    assert host_scope.home.get() == "caller"
    assert host_scope.secrets.get() == {"key": "caller-only"}
