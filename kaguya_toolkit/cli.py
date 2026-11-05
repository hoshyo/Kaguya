from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from ruamel.yaml import YAML

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_HOME = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")).expanduser() / "kaguya"
ADAPTER_CONFIG = CONFIG_HOME / "onebot.json"
SETTINGS_PATH = CONFIG_HOME / "settings.json"
HERMES_HOME = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes")).expanduser()
USER_UNITS = Path.home() / ".config" / "systemd" / "user"
YAML = YAML(typ="rt")
YAML.preserve_quotes = True

SOCIAL_CHAT_DEFAULT = {
    "enabled": True,
    "delivery_enabled": True,
    "mode": "active",
    "quiet_seconds": 1.5,
    "max_hold_seconds": 5.0,
    "judge_interval_seconds": 10.0,
    "judge_max_per_hour": 60,
    "judge_timeout_seconds": 60.0,
    "cooldown_seconds": 90.0,
    "max_replies": 4,
    "window_seconds": 600.0,
    "followup_seconds": 120.0,
    "followup_interval_seconds": 3.0,
    "followup_max_checks": 6,
    "passive_expiry_seconds": 45.0,
    "part_delay_min_seconds": 0.8,
    "part_delay_max_seconds": 2.0,
    "observer_agent_enabled": True,
    "context_prefill_enabled": True,
}

PROFILE_DESCRIPTIONS = {
    "aidog": "Kaguya group-chat persona with isolated local memory",
    "kig": "Kaguya persona for animegao kigurumi communities with isolated local memory",
    "qq-observer": "Private wake/skip observer used by the OneBot participation flow",
}


def _now() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _atomic_text(path: Path, text: str, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(raw)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        if mode is not None:
            os.chmod(tmp, mode)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _atomic_json(path: Path, data: object, mode: int = 0o600) -> None:
    _atomic_text(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n", mode)


def _load_json(path: Path, default: object | None = None):
    if not path.exists():
        return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    value = YAML.load(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _save_yaml(path: Path, data: dict, mode: int = 0o600) -> None:
    from io import StringIO

    buf = StringIO()
    YAML.dump(data, buf)
    _atomic_text(path, buf.getvalue(), mode)


def _hermes_binary(required: bool = True) -> Path | None:
    candidates = [
        shutil.which("hermes"),
        HERMES_HOME / "hermes-agent" / ".hermes" / "bin" / "hermes",
        HERMES_HOME / "bin" / "hermes",
        Path.home() / ".local" / "bin" / "hermes",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return Path(candidate)
    if required:
        raise SystemExit("Hermes is not installed. Run `./kaguya install` first.")
    return None


def _run(argv: list[str | Path], *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(x) for x in argv], check=check, text=True,
        capture_output=capture,
    )


def _download_verified(url: str, sha256: str, label: str) -> Path:
    cache = CONFIG_HOME / "downloads"
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"{label}-{sha256[:12]}.sh"
    if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != sha256:
        fd, raw = tempfile.mkstemp(prefix=f".{label}.", dir=cache)
        os.close(fd)
        tmp = Path(raw)
        try:
            with urllib.request.urlopen(url, timeout=60) as response, tmp.open("wb") as out:
                shutil.copyfileobj(response, out)
            got = hashlib.sha256(tmp.read_bytes()).hexdigest()
            if got != sha256:
                raise SystemExit(f"{label} installer checksum mismatch: expected {sha256}, got {got}")
            os.chmod(tmp, 0o700)
            os.replace(tmp, target)
        finally:
            tmp.unlink(missing_ok=True)
    return target


def _lock() -> dict:
    return _load_json(REPO_ROOT / "dependencies.lock.json")


def _install_external(args: argparse.Namespace) -> int:
    lock = _lock()
    hermes = _hermes_binary(required=False)
    if hermes is None:
        spec = lock["hermes"]
        installer = _download_verified(spec["installer_url"], spec["installer_sha256"], "hermes")
        command = [
            "bash", installer, "--commit", spec["commit"], "--skip-setup",
            "--skip-browser", "--skip-computer-use",
        ]
        print("Installing pinned Hermes Agent...")
        _run(command)
    else:
        print(f"Hermes already installed: {hermes}")

    if args.with_napcat:
        spec = lock["napcat_installer"]
        installer = _download_verified(spec["installer_url"], spec["installer_sha256"], "napcat")
        print("Starting the pinned NapCat installer. QQ login remains local to this machine.")
        _run(["bash", installer, "--docker", "n", "--cli", "y", "--github-proxy", "0"])

    return _init(args)


def _env_merge(path: Path, updates: dict[str, str]) -> None:
    existing: list[str] = []
    if path.exists():
        existing = path.read_text(encoding="utf-8", errors="replace").splitlines()
    keys = set(updates)
    kept = [line for line in existing if line.partition("=")[0].strip() not in keys]
    for key, value in updates.items():
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        kept.append(f'{key}="{escaped}"')
    _atomic_text(path, "\n".join(kept).rstrip() + "\n", 0o600)


def _ensure_adapter_config() -> object:
    os.environ["ONEBOT_ADAPTER_CONFIG"] = str(ADAPTER_CONFIG)
    from onebot_adapter.config import AdapterConfig, ensure_tokens, load_config, save_config

    if ADAPTER_CONFIG.exists():
        cfg = load_config(ADAPTER_CONFIG)
    else:
        cfg = AdapterConfig()
    cfg = cfg.with_overrides(
        group_default_enabled=False,
        social_chat=cfg.social_chat or dict(SOCIAL_CHAT_DEFAULT),
        dm_policy="deny",
        automation_api_enabled=False,
        automation_api_key_hash="",
        log_message_preview=0,
        log_file_message_mode="none",
        usage_stats_enabled=False,
        reaction_emoji_enabled=False,
        rate_limit_enabled=True,
        command_filter_enabled=True,
        command_filter_unknown=True,
        command_default_permission="disabled",
        command_permissions={"help": "everyone", "whoami": "everyone", "stop": "everyone"},
        hermes_install_dir=str(HERMES_HOME),
    )
    cfg = ensure_tokens(cfg, source="kaguya", reason="public_bundle_init", actor="local_operator")
    errors = cfg.validate()
    if errors:
        raise SystemExit("Generated adapter config is invalid:\n- " + "\n- ".join(errors))
    save_config(
        cfg, source="kaguya", reason="public_bundle_init", actor="local_operator",
        submitted_fields=sorted(cfg.to_dict()),
    )
    os.chmod(ADAPTER_CONFIG, 0o600)
    return cfg


def _ensure_profile(name: str) -> bool:
    home = HERMES_HOME / "profiles" / name
    if home.exists():
        return False
    hermes = _hermes_binary()
    _run([
        hermes, "profile", "create", name, "--clone-from", "default", "--clone",
        "--description", PROFILE_DESCRIPTIONS[name],
    ])
    return True


def _backup_existing(path: Path) -> Path:
    fingerprint = hashlib.sha256(str(path.absolute()).encode()).hexdigest()[:10]
    backup = CONFIG_HOME / "backups" / _now() / f"{fingerprint}-{path.name}"
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(path, backup)
    return backup


def _link(src: Path, dest: Path, *, force: bool) -> None:
    src = src.resolve()
    if dest.is_symlink() and dest.resolve(strict=False) == src:
        return
    if os.path.lexists(dest):
        if not force:
            raise SystemExit(f"Refusing to replace existing {dest}; rerun init with --force-existing")
        backed_up = _backup_existing(dest)
        print(f"Backed up {dest} to {backed_up}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.symlink_to(src, target_is_directory=src.is_dir())


def _merge_profile_config(home: Path, *, observer: bool = False) -> None:
    path = home / "config.yaml"
    cfg = _load_yaml(path)
    cfg.setdefault("auth", {})["adopt_external_logins"] = False
    if observer:
        overlay = _load_yaml(REPO_ROOT / "profiles" / "qq-observer" / "config.template.yaml")
        root_cfg = _load_yaml(HERMES_HOME / "config.yaml")
        if isinstance(root_cfg.get("model"), dict):
            cfg["model"] = dict(root_cfg["model"])
        for key, value in overlay.items():
            cfg[key] = value
    else:
        cfg.setdefault("memory", {}).update(memory_enabled=True, user_profile_enabled=True)
        cfg.setdefault("known_plugin_toolsets", {})["onebot"] = ["onebot"]
        cfg.setdefault("platform_toolsets", {})["onebot"] = [
            "web", "vision", "todo", "memory", "onebot", "no_mcp",
        ]
        onebot = cfg.setdefault("platforms", {}).setdefault("onebot", {})
        onebot["enabled"] = False
        onebot["gateway_restart_notification"] = False
    _save_yaml(path, cfg)


def _configure_root() -> None:
    path = HERMES_HOME / "config.yaml"
    cfg = _load_yaml(path)
    cfg.setdefault("auth", {})["adopt_external_logins"] = False
    cfg["group_sessions_per_user"] = False
    cfg.setdefault("known_plugin_toolsets", {})["onebot"] = ["onebot"]
    cfg.setdefault("platform_toolsets", {})["onebot"] = ["web", "vision", "todo", "onebot", "no_mcp"]
    cfg.setdefault("plugins", {}).setdefault("entries", {}).setdefault("onebot", {})
    onebot = cfg.setdefault("platforms", {}).setdefault("onebot", {})
    onebot["enabled"] = True
    onebot["gateway_restart_notification"] = False
    onebot.setdefault("extra", {}).update({
        "allow_admin_from": ["__local_management_only__"],
        "group_allow_admin_from": ["__local_management_only__"],
        "user_allowed_commands": ["help", "whoami", "stop"],
        "group_user_allowed_commands": ["help", "whoami", "stop"],
    })
    onebot.setdefault("channel_prompts", {})
    cfg.setdefault("gateway", {}).setdefault("profile_routes", [])
    _save_yaml(path, cfg)


def _copy_memory_seed(name: str) -> None:
    src = REPO_ROOT / "profiles" / name / "memories"
    dest = HERMES_HOME / "profiles" / name / "memories"
    dest.mkdir(parents=True, exist_ok=True)
    for filename in ("MEMORY.md", "USER.md"):
        target = dest / filename
        if not target.exists() or not target.read_text(encoding="utf-8", errors="replace").strip():
            shutil.copyfile(src / filename, target)
            os.chmod(target, 0o600)


def _unit_quote(value: str | Path) -> str:
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _write_units(napcat_executable: str | None = None) -> None:
    USER_UNITS.mkdir(parents=True, exist_ok=True)
    python = REPO_ROOT / ".venv" / "bin" / "python"
    adapter_unit = f"""[Unit]
Description=Kaguya OneBot adapter
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory={_unit_quote(REPO_ROOT)}
ExecStart={_unit_quote(python)} -m onebot_adapter --host 127.0.0.1
Environment={_unit_quote(f"ONEBOT_ADAPTER_CONFIG={ADAPTER_CONFIG}")}
Restart=on-failure
RestartSec=5
UMask=0077
NoNewPrivileges=yes
PrivateTmp=yes

[Install]
WantedBy=default.target
"""
    _atomic_text(USER_UNITS / "kaguya-adapter.service", adapter_unit, 0o644)
    if napcat_executable:
        exe = Path(napcat_executable).expanduser().resolve()
        napcat_unit = f"""[Unit]
Description=Kaguya NapCat QQ bridge
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory={_unit_quote(exe.parent)}
ExecStart=/usr/bin/xvfb-run -a -s \"-screen 0 1280x720x24 -nolisten tcp\" {_unit_quote(exe)} --no-sandbox --disable-gpu
Restart=on-failure
RestartSec=10
TimeoutStopSec=30
KillMode=control-group
UMask=0077

[Install]
WantedBy=default.target
"""
        _atomic_text(USER_UNITS / "kaguya-napcat.service", napcat_unit, 0o644)


def _detect_napcat() -> str | None:
    settings = _load_json(SETTINGS_PATH, {})
    configured = settings.get("napcat_executable") if isinstance(settings, dict) else None
    candidates = [configured, Path.home() / "Napcat" / "opt" / "QQ" / "qq"]
    for candidate in candidates:
        if candidate and Path(candidate).expanduser().is_file():
            return str(Path(candidate).expanduser().resolve())
    return None


def _init(args: argparse.Namespace) -> int:
    hermes = _hermes_binary()
    fresh_setup = not (HERMES_HOME / "config.yaml").exists()
    if fresh_setup:
        print("Hermes has no user config yet; starting its setup wizard.")
        _run([hermes, "setup"])
    cfg = _ensure_adapter_config()
    created = {name: _ensure_profile(name) for name in PROFILE_DESCRIPTIONS}

    profile_links = {
        "aidog": (REPO_ROOT / "profiles" / "aidog" / "SOUL.md", HERMES_HOME / "profiles" / "aidog" / "SOUL.md"),
        "kig": (REPO_ROOT / "profiles" / "kig" / "SOUL.md", HERMES_HOME / "profiles" / "kig" / "SOUL.md"),
        "qq-observer": (
            REPO_ROOT / "profiles" / "qq-observer" / "SOUL.md",
            HERMES_HOME / "profiles" / "qq-observer" / "SOUL.md",
        ),
    }
    for name, (src, dest) in profile_links.items():
        _link(src, dest, force=args.force_existing or created[name])
    _link(
        REPO_ROOT / "profiles" / "default" / "SOUL.md",
        HERMES_HOME / "SOUL.md",
        force=args.force_existing or fresh_setup,
    )
    for name in ("aidog", "kig"):
        _link(
            REPO_ROOT / "profiles" / name / "profile.yaml",
            HERMES_HOME / "profiles" / name / "profile.yaml",
            force=args.force_existing or created[name],
        )
        _copy_memory_seed(name)

    plugin_src = REPO_ROOT / "onebot_adapter" / "hermes_plugin"
    for home in (HERMES_HOME, HERMES_HOME / "profiles" / "aidog", HERMES_HOME / "profiles" / "kig"):
        _link(plugin_src, home / "plugins" / "onebot", force=args.force_existing or created.get(home.name, False))
        _env_merge(home / ".env", {
            "ONEBOT_ADAPTER_URL": f"ws://127.0.0.1:{cfg.hermes_ws_port}{cfg.hermes_ws_path}",
            "ONEBOT_ADAPTER_TOKEN": cfg.hermes_ws_token,
        })

    _configure_root()
    _merge_profile_config(HERMES_HOME / "profiles" / "aidog")
    _merge_profile_config(HERMES_HOME / "profiles" / "kig")
    _merge_profile_config(HERMES_HOME / "profiles" / "qq-observer", observer=True)

    napcat = args.napcat_executable or _detect_napcat()
    _atomic_json(SETTINGS_PATH, {"repo_root": str(REPO_ROOT), "napcat_executable": napcat})
    _write_units(napcat)
    print(f"Initialized Kaguya. Local adapter config: {ADAPTER_CONFIG}")
    print("Next: ./kaguya auth, ./kaguya bind --group <QQ_GROUP_ID> --profile aidog|kig")
    return 0


def _preset(profile: str) -> dict:
    return _load_json(REPO_ROOT / "config" / "presets" / f"{profile}.json")


def _save_adapter_dict(data: dict) -> None:
    os.environ["ONEBOT_ADAPTER_CONFIG"] = str(ADAPTER_CONFIG)
    from onebot_adapter.config import AdapterConfig, save_config

    cfg = AdapterConfig.from_dict(data)
    errors = cfg.validate()
    if errors:
        raise SystemExit("Binding would make config invalid:\n- " + "\n- ".join(errors))
    save_config(
        cfg, source="kaguya", reason="local_group_binding", actor="local_operator",
        submitted_fields=["groups", "group_history", "social_chat"],
    )
    os.chmod(ADAPTER_CONFIG, 0o600)


def _bind(args: argparse.Namespace) -> int:
    if not re.fullmatch(r"[1-9][0-9]{4,12}", args.group):
        raise SystemExit("--group must be a numeric QQ group ID")
    if not ADAPTER_CONFIG.exists():
        raise SystemExit("Run `./kaguya init` first")
    data = _load_json(ADAPTER_CONFIG)
    preset = _preset(args.profile)
    group = dict(preset["group"])
    group.update(group_id=args.group, name=args.name or args.profile)
    data.setdefault("groups", {})[args.group] = group
    data.setdefault("group_history", {})[args.group] = dict(preset.get("history", {"enabled": True}))
    data["social_chat"] = data.get("social_chat") or dict(SOCIAL_CHAT_DEFAULT)
    _save_adapter_dict(data)

    config_path = HERMES_HOME / "config.yaml"
    cfg = _load_yaml(config_path)
    routes = cfg.setdefault("gateway", {}).setdefault("profile_routes", [])
    routes[:] = [r for r in routes if not (isinstance(r, dict) and r.get("platform") == "onebot"
                                           and r.get("chat_id") == f"group:{args.group}")]
    routes.append({
        "name": f"onebot-{args.profile}-{hashlib.sha256(args.group.encode()).hexdigest()[:8]}",
        "platform": "onebot", "chat_id": f"group:{args.group}", "profile": args.profile,
    })
    prompts = cfg.setdefault("platforms", {}).setdefault("onebot", {}).setdefault("channel_prompts", {})
    prompts[args.group] = group["custom_prompt"]
    _save_yaml(config_path, cfg)
    print(f"Bound one local QQ group to profile {args.profile}. The group ID is stored only outside Git.")
    return 0


def _unbind(args: argparse.Namespace) -> int:
    data = _load_json(ADAPTER_CONFIG)
    for key in ("groups", "group_history", "group_context", "group_auto_reply"):
        if isinstance(data.get(key), dict):
            data[key].pop(args.group, None)
    _save_adapter_dict(data)
    path = HERMES_HOME / "config.yaml"
    cfg = _load_yaml(path)
    routes = cfg.setdefault("gateway", {}).setdefault("profile_routes", [])
    routes[:] = [r for r in routes if not (isinstance(r, dict) and r.get("platform") == "onebot"
                                           and r.get("chat_id") == f"group:{args.group}")]
    cfg.setdefault("platforms", {}).setdefault("onebot", {}).setdefault("channel_prompts", {}).pop(
        args.group, None,
    )
    _save_yaml(path, cfg)
    print("Removed the local binding")
    return 0


def _napcat_targets(raw: str) -> list[Path]:
    path = Path(raw).expanduser().resolve()
    if path.is_file():
        return [path]
    if path.is_dir():
        found = sorted(path.glob("onebot11*.json"))
        if found:
            return found
    raise SystemExit(f"No onebot11*.json configuration found at {path}")


def _configure_napcat(args: argparse.Namespace) -> int:
    data = _load_json(ADAPTER_CONFIG)
    token = data.get("onebot_ws_token")
    if not token:
        raise SystemExit("Adapter token missing; run `./kaguya init`")
    url = f"ws://127.0.0.1:{data.get('onebot_reverse_ws_port', 18800)}{data.get('onebot_reverse_ws_path', '/onebot')}"
    for target in _napcat_targets(args.config):
        body = _load_json(target)
        network = body.setdefault("network", {})
        clients = network.setdefault("websocketClients", [])
        clients[:] = [x for x in clients if not (isinstance(x, dict) and x.get("name") == "kaguya-local")]
        clients.append({
            "name": "kaguya-local", "enable": True, "url": url,
            "messagePostFormat": "array", "reportSelfMessage": False,
            "reconnectInterval": 5000, "heartInterval": 30000, "token": token,
        })
        backup = CONFIG_HOME / "backups" / "napcat" / _now() / target.name
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, backup)
        mode = stat.S_IMODE(target.stat().st_mode)
        _atomic_text(target, json.dumps(body, ensure_ascii=False, indent=2) + "\n", mode)
        print(f"Configured {target}; backup: {backup}")
    return 0


def _auth(_args: argparse.Namespace) -> int:
    hermes = _hermes_binary()
    status = _run([hermes, "auth", "list"], check=False, capture=True)
    if "credentials):" not in status.stdout:
        print("No Hermes model credential is configured; starting the Hermes setup wizard.")
        return _run([hermes, "setup"], check=False).returncode
    print(status.stdout.rstrip())
    return 0


def _systemctl(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return _run(["systemctl", "--user", *args], check=check, capture=not check)


def _on(_args: argparse.Namespace) -> int:
    _write_units(_detect_napcat())
    _systemctl("daemon-reload")
    hermes = _hermes_binary()
    _run([hermes, "gateway", "install"], check=False)
    _run([hermes, "gateway", "start"], check=False)
    _systemctl("enable", "--now", "kaguya-adapter.service")
    if (USER_UNITS / "kaguya-napcat.service").exists():
        _systemctl("enable", "--now", "kaguya-napcat.service")
    print("Kaguya services started")
    return 0


def _off(_args: argparse.Namespace) -> int:
    for name in ("kaguya-napcat.service", "kaguya-adapter.service"):
        _systemctl("stop", name, check=False)
    hermes = _hermes_binary(required=False)
    if hermes:
        _run([hermes, "gateway", "stop"], check=False)
    print("Kaguya services stopped")
    return 0


def _status(_args: argparse.Namespace) -> int:
    hermes = _hermes_binary(required=False)
    if hermes:
        _run([hermes, "gateway", "status"], check=False)
    for name in ("kaguya-adapter.service", "kaguya-napcat.service"):
        result = _systemctl("is-active", name, check=False)
        print(f"{name}: {result.stdout.strip() or 'not installed'}")
    return 0


def _doctor(_args: argparse.Namespace) -> int:
    checks: list[tuple[str, bool, str]] = []
    hermes = _hermes_binary(required=False)
    checks.append(("Hermes", hermes is not None, str(hermes or "missing")))
    checks.append(("adapter config", ADAPTER_CONFIG.is_file(), str(ADAPTER_CONFIG)))
    if ADAPTER_CONFIG.exists():
        data = _load_json(ADAPTER_CONFIG)
        checks.append(("bound groups", bool(data.get("groups")), f"count={len(data.get('groups', {}))}"))
    for name in PROFILE_DESCRIPTIONS:
        checks.append((f"profile {name}", (HERMES_HOME / "profiles" / name / "SOUL.md").exists(), ""))
    checks.append(("NapCat executable", _detect_napcat() is not None, _detect_napcat() or "not configured"))
    if hermes:
        auth = _run([hermes, "auth", "list"], check=False, capture=True)
        checks.append(("model credential", "credentials):" in auth.stdout, "run ./kaguya auth if missing"))
    failed = False
    for label, ok, detail in checks:
        failed |= not ok
        print(f"{'OK' if ok else 'MISSING':7} {label:22} {detail}")
    return 1 if failed else 0


FORBIDDEN_SUFFIXES = {
    ".db", ".sqlite", ".sqlite3", ".log", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".avif",
    ".bmp", ".svg", ".svgz", ".zip", ".7z", ".tar", ".gz",
}
SECRET_PATTERNS = {
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "OpenAI-like key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"),
    "GitHub token": re.compile(r"\bgh[ps]_[A-Za-z0-9]{20,}"),
    "bearer token": re.compile(r"(?i)authorization\s*[:=]\s*bearer\s+[A-Za-z0-9._~-]{16,}"),
}
PROFILE_ID_PATTERN = re.compile(r"(?:QQ|群号|group:)\s*[=:：]?\s*[1-9][0-9]{5,}", re.IGNORECASE)


def _tracked_candidates() -> list[Path]:
    result = subprocess.run(["git", "-C", str(REPO_ROOT), "ls-files", "-co", "--exclude-standard"],
                            capture_output=True, text=True)
    if result.returncode == 0:
        return [REPO_ROOT / line for line in result.stdout.splitlines() if line]
    return [p for p in REPO_ROOT.rglob("*") if p.is_file() and ".git" not in p.parts]


def _privacy_check(_args: argparse.Namespace) -> int:
    problems: list[str] = []
    for path in _tracked_candidates():
        rel = path.relative_to(REPO_ROOT)
        lower = path.name.lower()
        if lower in {".env", "auth.json", "group_messages.sqlite3"} or path.suffix.lower() in FORBIDDEN_SUFFIXES:
            problems.append(f"forbidden file: {rel}")
            continue
        try:
            raw = path.read_bytes()
        except OSError as exc:
            problems.append(f"unreadable file: {rel}: {exc}")
            continue
        if b"\x00" in raw:
            continue
        text = raw.decode("utf-8", errors="replace")
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                problems.append(f"{label}: {rel}")
        if rel.parts and rel.parts[0] == "profiles" and PROFILE_ID_PATTERN.search(text):
            problems.append(f"real-looking QQ/group identifier in public profile: {rel}")
        if rel.parts[:3] in {("profiles", "aidog", "memories"), ("profiles", "kig", "memories")}:
            if re.search(r"本群成员|space\.bilibili\.com/\d+|谁喜欢谁|老公|主人", text):
                problems.append(f"member-specific memory in public seed: {rel}")
    if problems:
        print("Privacy check failed:")
        for problem in sorted(set(problems)):
            print(f"- {problem}")
        return 1
    print("Privacy check passed: no credentials, live IDs, databases, logs, archives, or media detected")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kaguya", description="Kaguya QQ bot installer and operator CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("install", help="Install pinned external components, then initialize")
    p.add_argument("--with-napcat", action="store_true", help="also run the pinned NapCat installer")
    p.add_argument("--force-existing", action="store_true")
    p.add_argument("--napcat-executable")
    p.set_defaults(func=_install_external)

    p = sub.add_parser("init", help="Create profiles, local config, symlinks and user services")
    p.add_argument("--force-existing", action="store_true", help="back up and replace existing managed links/files")
    p.add_argument("--napcat-executable")
    p.set_defaults(func=_init)

    p = sub.add_parser("bind", help="Bind one local QQ group to a profile")
    p.add_argument("--group", required=True)
    p.add_argument("--profile", required=True, choices=("aidog", "kig"))
    p.add_argument("--name", help="local display name; never written to Git")
    p.set_defaults(func=_bind)

    p = sub.add_parser("unbind", help="Remove one local QQ group binding")
    p.add_argument("--group", required=True)
    p.set_defaults(func=_unbind)

    p = sub.add_parser("napcat", help="Configure NapCat reverse WebSocket with the generated local token")
    p.add_argument("--config", required=True, help="onebot11 JSON file or its config directory")
    p.set_defaults(func=_configure_napcat)

    for name, func, help_text in (
        ("auth", _auth, "Configure/check the Hermes model credential"),
        ("on", _on, "Start Hermes, adapter and configured NapCat services"),
        ("off", _off, "Stop Kaguya services"),
        ("status", _status, "Show service status"),
        ("doctor", _doctor, "Check local readiness without changing state"),
        ("privacy-check", _privacy_check, "Reject secrets, live IDs, databases, archives and media"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.set_defaults(func=func)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
