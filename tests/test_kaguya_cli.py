from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from kaguya_toolkit import cli


def test_public_presets_do_not_embed_live_bindings():
    for name in ("aidog", "kig"):
        preset = json.loads((cli.REPO_ROOT / "config" / "presets" / f"{name}.json").read_text())
        assert "group_id" not in preset["group"]
        assert "name" not in preset["group"]
        assert not cli.PROFILE_ID_PATTERN.search(json.dumps(preset, ensure_ascii=False))


def test_public_memory_seeds_contain_no_member_profiles():
    for name in ("aidog", "kig"):
        for filename in ("MEMORY.md", "USER.md"):
            text = (cli.REPO_ROOT / "profiles" / name / "memories" / filename).read_text()
            assert not cli.PROFILE_ID_PATTERN.search(text)
            assert "space.bilibili.com/" not in text
            assert "谁喜欢谁" not in text


def test_privacy_check_rejects_media(tmp_path: Path, monkeypatch, capsys):
    (tmp_path / "context.svg").write_text("<svg/>", encoding="utf-8")
    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)
    assert cli._privacy_check(argparse.Namespace()) == 1
    assert "forbidden file: context.svg" in capsys.readouterr().out


def test_privacy_check_rejects_hermes_session_context(tmp_path: Path, monkeypatch, capsys):
    session = tmp_path / "profiles" / "aidog" / "sessions" / "request_dump_synthetic.json"
    session.parent.mkdir(parents=True)
    session.write_text('{"messages":["synthetic prior turn"]}', encoding="utf-8")
    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)
    assert cli._privacy_check(argparse.Namespace()) == 1
    output = capsys.readouterr().out
    assert "runtime conversation/session record" in output
    assert "request_dump_synthetic.json" in output


def test_systemd_argument_quoting():
    assert cli._unit_quote('/tmp/Kaguya Bot/"qq"') == '"/tmp/Kaguya Bot/\\"qq\\""'


def test_bind_writes_only_local_adapter_and_hermes_configs(tmp_path: Path, monkeypatch):
    adapter = tmp_path / "config" / "onebot.json"
    hermes = tmp_path / "hermes"
    cli._atomic_json(adapter, {})
    cli._save_yaml(hermes / "config.yaml", {})
    monkeypatch.setattr(cli, "ADAPTER_CONFIG", adapter)
    monkeypatch.setattr(cli, "HERMES_HOME", hermes)
    monkeypatch.setattr(cli, "_save_adapter_dict", lambda data: cli._atomic_json(adapter, data))

    args = argparse.Namespace(group="123456", profile="aidog", name=None)
    assert cli._bind(args) == 0

    local = json.loads(adapter.read_text())
    assert local["groups"]["123456"]["name"] == "aidog"
    root = cli._load_yaml(hermes / "config.yaml")
    assert root["gateway"]["profile_routes"][0]["chat_id"] == "group:123456"
    assert root["gateway"]["profile_routes"][0]["profile"] == "aidog"


def test_napcat_patch_uses_generated_local_token_and_keeps_backup(tmp_path: Path, monkeypatch):
    adapter = tmp_path / "onebot.json"
    target = tmp_path / "onebot11.json"
    config_home = tmp_path / "kaguya-config"
    cli._atomic_json(adapter, {
        "onebot_ws_token": "synthetic-local-token",
        "onebot_reverse_ws_port": 18800,
        "onebot_reverse_ws_path": "/onebot",
    })
    cli._atomic_json(target, {"network": {"websocketClients": [{"name": "unrelated"}]}})
    monkeypatch.setattr(cli, "ADAPTER_CONFIG", adapter)
    monkeypatch.setattr(cli, "CONFIG_HOME", config_home)

    assert cli._configure_napcat(argparse.Namespace(config=str(target))) == 0
    patched = json.loads(target.read_text())
    clients = patched["network"]["websocketClients"]
    assert clients[0]["name"] == "unrelated"
    assert clients[1]["name"] == "kaguya-local"
    assert clients[1]["token"] == "synthetic-local-token"
    assert list((config_home / "backups" / "napcat").rglob("onebot11.json"))


def test_same_named_profile_backups_do_not_collide(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(cli, "CONFIG_HOME", tmp_path / "config")
    monkeypatch.setattr(cli, "_now", lambda: "20260101T000000Z")
    first = tmp_path / "aidog" / "SOUL.md"
    second = tmp_path / "kig" / "SOUL.md"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text("aidog")
    second.write_text("kig")

    one = cli._backup_existing(first)
    two = cli._backup_existing(second)
    assert one != two
    assert one.read_text() == "aidog"
    assert two.read_text() == "kig"


def test_fresh_init_activates_all_four_public_materials(tmp_path: Path, monkeypatch):
    hermes = tmp_path / "hermes"
    config_home = tmp_path / "config"
    monkeypatch.setattr(cli, "HERMES_HOME", hermes)
    monkeypatch.setattr(cli, "CONFIG_HOME", config_home)
    monkeypatch.setattr(cli, "ADAPTER_CONFIG", config_home / "onebot.json")
    monkeypatch.setattr(cli, "SETTINGS_PATH", config_home / "settings.json")
    monkeypatch.setattr(cli, "USER_UNITS", tmp_path / "units")
    monkeypatch.setattr(cli, "_hermes_binary", lambda required=True: Path("/bin/true"))
    monkeypatch.setattr(cli, "_ensure_adapter_config", lambda: SimpleNamespace(
        hermes_ws_port=18810,
        hermes_ws_path="/hermes",
        hermes_ws_token="synthetic-token",
    ))

    def fake_run(argv, **_kwargs):
        if "setup" in [str(x) for x in argv]:
            cli._save_yaml(hermes / "config.yaml", {})
            (hermes / "SOUL.md").write_text("generated default")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def fake_profile(name):
        (hermes / "profiles" / name).mkdir(parents=True)
        return True

    monkeypatch.setattr(cli, "_run", fake_run)
    monkeypatch.setattr(cli, "_ensure_profile", fake_profile)
    monkeypatch.setattr(cli, "_configure_root", lambda: None)
    monkeypatch.setattr(cli, "_merge_profile_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "_write_units", lambda *_args, **_kwargs: None)

    args = argparse.Namespace(force_existing=False, napcat_executable=None)
    assert cli._init(args) == 0
    assert (hermes / "SOUL.md").resolve() == (cli.REPO_ROOT / "profiles/default/SOUL.md").resolve()
    for name in ("aidog", "kig", "qq-observer"):
        assert (hermes / "profiles" / name / "SOUL.md").resolve() == (
            cli.REPO_ROOT / "profiles" / name / "SOUL.md"
        ).resolve()
