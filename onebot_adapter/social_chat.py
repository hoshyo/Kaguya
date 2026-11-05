"""One global social policy; bounded, isolated per-group runtime state.

Observation is not history backfill. Only a few relevant rows on an admitted
passive turn are handed to Hermes' existing incremental context mechanism.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import time
import uuid
from collections import Counter, deque

from onebot_adapter.group_context import GroupContextBuffer
from onebot_adapter.hermes_plugin.courtesy import called_by_name, internal_notice, preference_request
from onebot_adapter.hermes_plugin.delivery import normalize_plan, valid_identity
from onebot_adapter.hermes_plugin.social import INTENT_HINTS
from onebot_adapter.member_preferences import MemberPreferences

logger = logging.getLogger(__name__)

DEFAULTS = {
    "enabled": False, "delivery_enabled": False, "observer_agent_enabled": False,
    "context_prefill_enabled": False, "context_background_messages": 20, "context_background_chars": 6000,
    "mode": "active", "quiet_seconds": 1.5, "max_hold_seconds": 5.0,
    "judge_interval_seconds": 10.0, "judge_max_per_hour": 60, "judge_timeout_seconds": 20.0,
    "cooldown_seconds": 90.0, "max_replies": 4, "window_seconds": 600.0,
    "followup_seconds": 120.0, "followup_interval_seconds": 3.0, "followup_max_checks": 6,
    "passive_expiry_seconds": 45.0, "part_delay_min_seconds": 0.8, "part_delay_max_seconds": 2.0,
}
BOUNDS = {
    "context_background_messages": (1, 50), "context_background_chars": (1000, 16000),
    "quiet_seconds": (0.1, 5), "max_hold_seconds": (1, 10),
    "judge_interval_seconds": (3, 120), "judge_max_per_hour": (1, 120),
    "judge_timeout_seconds": (3, 60), "cooldown_seconds": (10, 600),
    "max_replies": (1, 10), "window_seconds": (60, 3600), "followup_seconds": (10, 300),
    "followup_interval_seconds": (1, 30), "followup_max_checks": (1, 12),
    "passive_expiry_seconds": (10, 120), "part_delay_min_seconds": (0, 3),
    "part_delay_max_seconds": (0, 5),
}
INTENTS = set(INTENT_HINTS)


def validate_social_chat(value, errors):
    if not isinstance(value, dict):
        errors.append("social_chat must be a dict")
        return
    if set(value) - set(DEFAULTS):
        errors.append("social_chat has unknown keys")
    if type(value.get("enabled", False)) is not bool:
        errors.append("social_chat.enabled must be bool")
    if type(value.get("delivery_enabled", False)) is not bool:
        errors.append("social_chat.delivery_enabled must be bool")
    if type(value.get("observer_agent_enabled", False)) is not bool:
        errors.append("social_chat.observer_agent_enabled must be bool")
    if type(value.get("context_prefill_enabled", False)) is not bool:
        errors.append("social_chat.context_prefill_enabled must be bool")
    if value.get("mode", "active") not in {"active", "shadow"}:
        errors.append("social_chat.mode must be active or shadow")
    for key, (low, high) in BOUNDS.items():
        v = value.get(key, DEFAULTS[key])
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not low <= v <= high:
            errors.append(f"social_chat.{key} must be between {low} and {high}")
        elif isinstance(DEFAULTS[key], int) and type(v) is not int:
            errors.append(f"social_chat.{key} must be int")
    if not errors:
        rule = {**DEFAULTS, **value}
        if rule["part_delay_min_seconds"] > rule["part_delay_max_seconds"]:
            errors.append("social_chat part delay bounds reversed")
        if rule["quiet_seconds"] > rule["max_hold_seconds"]:
            errors.append("social_chat quiet exceeds max hold")


def policy(config):
    raw = getattr(config, "social_chat", {})
    errors = []
    validate_social_chat(raw, errors)
    return {**DEFAULTS, **raw} if not errors else dict(DEFAULTS)


def followup_config(config):
    rule = policy(config)
    if not rule["enabled"]:
        return config.group_auto_reply
    # Runtime derivation only: no per-profile/per-group copies in configuration.
    return {str(g): {"enabled": True, "followup": {
        "enabled": True, "window_seconds": rule["followup_seconds"],
        "min_interval_seconds": rule["followup_interval_seconds"],
        "budget_window_seconds": rule["followup_seconds"], "max_checks": rule["followup_max_checks"],
    }} for g in config.groups if config.get_group_config(str(g)).enabled}


def decision(value):
    if not isinstance(value, dict) or value.get("action") not in {"reply", "wait", "skip"} \
            or value.get("intent") not in INTENTS:
        return {"action": "skip", "intent": "react"}
    return {"action": value["action"], "intent": value["intent"]}


class SocialChat:
    def __init__(self, config, *, judge, busy, followups, blacklist_match=None, clock=time.monotonic,
                 preferences=None, archive=None, wakeup_handled=None):
        self.judge, self.busy, self.followups = judge, busy, followups
        self.blacklist_match, self.clock = blacklist_match, clock
        self.tasks, self.candidates, self.last_seen = {}, {}, {}
        self.checks, self.admissions, self.epochs, self.turns = {}, {}, {}, {}
        self.seen = {}
        self.preferences = preferences if preferences is not None else MemberPreferences()
        self.preference_versions = {}
        self.stats = Counter()
        self.observer = None
        self.archive = archive
        self.wakeup_handled = wakeup_handled or (lambda group, mid: False)
        from onebot_adapter.adaptive_observer import AdaptiveObserver
        self.adaptive = AdaptiveObserver(self)
        self.configure(config)

    def configure(self, config):
        scope = json.dumps({"policy": policy(config), "groups": config.groups,
                            "self_id": config.self_id, "admins": config.global_admins,
                            "blacklist": config.bot_blacklist_enabled}, sort_keys=True)
        changed = hasattr(self, "_scope") and self._scope != scope
        self._scope = scope
        self.config, self.rule = config, policy(config)
        if changed:
            for task in (*self.adaptive.tasks.values(), *self.adaptive.wake_tasks.values()):
                task.cancel()
            self.adaptive.states.clear()
            self.adaptive.wakeups.clear()
            self.adaptive.tasks.clear()
            self.adaptive.wake_tasks.clear()
            for task in self.tasks.values():
                task.cancel()
            self.tasks.clear()
            self.candidates.clear()
            self.turns.clear()  # outstanding guards fail closed on revocation
            self.seen.clear()
        groups = {str(g): {"enabled": True} for g in config.groups
                  if self.rule["enabled"] and config.get_group_config(str(g)).enabled}
        observation_config = config.with_overrides(group_context=groups)
        if self.observer is None:
            self.observer = GroupContextBuffer(observation_config, clock=self.clock)
        else:
            self.observer.configure(observation_config)
        self.followups.configure(followup_config(config))

    def enabled(self, group):
        return self.rule["enabled"] and str(group) in self.config.groups \
            and self.config.get_group_config(str(group)).enabled

    def observe(self, data):
        if not isinstance(data, dict):
            return False
        group, mid = str(data.get("group_id", "")), str(data.get("message_id", ""))
        if not self.enabled(group):
            return False
        if self.rule["observer_agent_enabled"] and self.archive is not None:
            self.archive.record(data)
        self.observer.before_event(data, blacklist_match=self.blacklist_match)
        rows = self.observer._buffers.get(group, ())
        valid = any(r.message["message_id"] == mid and r.message["user_id"] == str(data.get("user_id"))
                    for r in rows)
        if valid:
            self.last_seen[group] = self.clock()
        seen = self.seen.setdefault(group, deque(maxlen=100))
        duplicate = mid in seen
        if valid and not duplicate:
            seen.append(mid)
        return valid and not duplicate

    def rows(self, group):
        # Prune expired or newly blacklisted rows at use time, not just receipt.
        self.observer._prune(group, self.blacklist_match)
        rows = [dict(e.message) for e in self.observer._buffers.get(group, ())][-12:]
        while rows and len(json.dumps(rows, ensure_ascii=False)) > 2500:
            rows.pop(0)
        return rows

    def conversation_state(self, group, user, message_id):
        """Bounded transport facts, not a model verdict that a topic is shared.

        No new bodies/cursors are stored and no old history is fetched. The
        snapshot is pinned to the actual current message, including on restore.
        """
        if not self.enabled(group):
            return {}
        self.observer._prune(group, self.blacklist_match)
        now = self.clock()
        recent = [e.message for e in self.observer._buffers.get(group, ())
                  if now - e.received_at < self.rule["followup_seconds"]]
        current = next((i for i in range(len(recent) - 1, -1, -1)
                        if recent[i]["message_id"] == str(message_id) and recent[i]["user_id"] == str(user)), None)
        if current is None:
            return {}
        recent = recent[:current + 1]
        last_bot = next((i for i in range(len(recent) - 1, -1, -1) if recent[i].get("bot")), -1)
        speakers = {r["user_id"] for r in recent}
        pending = any(r["user_id"] == str(user) and not r.get("bot")
                      for r in recent[last_bot + 1:-1])
        return {"active_speakers": len(speakers),
                "continuing_exchange": last_bot >= 0 and len(speakers) <= 3,
                "same_sender_pending": pending}

    def direct(self, group, event):
        self.epochs[group] = self.epochs.get(group, 0) + 1
        self.candidates.pop(group, None)
        # An ongoing judgment is invalidated by epoch, not allowed to emit later.
        event.social_reply = self.issue(group, event.user_id, "answer", passive=False, message_id=event.message_id)
        if self.rule["observer_agent_enabled"] and not event.text.lstrip().startswith("/"):
            self.adaptive.boost(group)
            self.make_wakeup(event, direct=True)

    def make_wakeup(self, event, *, direct=False):
        event.wake_only = True
        # These are coalesced internal control notifications. The legacy
        # 12-messages/10-min limiter is for admitted user-message turns, not
        # the replacement adaptive observer/wakeup policy.
        event.rate_limit_eligible = False
        event.text = ("群里有人明确找你。" if direct else "本群有值得查看的新动态。") + (
            "请先自行读取当前群聊天记录，结合上次读取时间判断需要查看的范围，再决定是否回应以及回应谁。"
            "这是一条唤醒通知，不是群友原话；不必回复这条通知本身，不合适参与时保持沉默。")
        if self.archive is not None:
            event.text += (f"\n当前Unix时间：{time.time()}；本群上次读取到的消息时间："
                           f"{self.archive.progress(event.chat_id.removeprefix('group:'))}。")
        event.reply_to_text = None
        event.reply_to_message_id = None
        event.media_items = []
        event.recent_group_context = []
        token = event.social_reply.get("token")
        if token in self.turns:
            self.turns[token]["wake_only"] = True

    def prepare_wakeup_batch(self, event):
        """Refresh the capability at dispatch, not when an entry joined the queue."""
        group = event.chat_id.removeprefix("group:")
        event.social_reply = self.issue(group, event.user_id, "answer", passive=event.passive_reply,
                                        message_id=event.message_id)
        self.make_wakeup(event, direct=not event.passive_reply)
        turn = self.turns[event.social_reply["token"]]
        turn["wakeup_batch"] = event.wakeup_batch
        turn["serial_wakeup"] = not event.passive_reply
        turn["batch_invites"] = {r["user_id"]: self.preference_versions.get((group, r["user_id"]), 0)
                                 for r in event.wakeup_batch["requests"] if r["direct"]}
        for request in event.wakeup_batch["requests"]:
            # History-style refs remain usable after querying the actual text.
            ref = "h_" + hashlib.sha256(request["message_id"].encode()).hexdigest()[:16]
            if not any(t["ref"] == ref for t in turn["targets"]):
                turn["targets"].append({"ref": ref, "message_id": request["message_id"],
                                        "user_id": request["user_id"], "name": "本批次发言人", "preview": ""})
        event.social_reply = self._turn_hint(event.social_reply["token"], turn)
        event.text += "\n本轮待处理消息索引（只含定位信息，原话仍需自行查记录）：" + json.dumps(
            event.wakeup_batch, ensure_ascii=False, separators=(",", ":"))
        event.text += ("\n先核对本批次各人的新发言，相关内容可以一起回应，不必机械逐条答；"
                       "不同问题需要分别回应时，可在同一次输出中使用频道规定的多回复格式。"
                       "已明确回应过的内容不要再答，决定不参与的内容可保持沉默。")
        if event.passive_reply:
            event.text += ("\n本批可能包含明确对你的续聊请求，也可能只是可参与话题。明确承接你的新任务、"
                           "追问或结果反馈应正常回应；其余只是主动参与候选，选合适的话头自然参与，"
                           "不必逐条补答，不是所有人都分别在提问。")

    def prefill_wakeup(self, event):
        from onebot_adapter.context_prefill import prefill_context

        group = event.chat_id.removeprefix("group:")
        def allowed(user):
            return user == str(self.config.self_id) or self.observer._allowed(group, user, self.blacklist_match)
        prefill_context(event, self.archive, allowed, self_id=str(self.config.self_id),
                        background_messages=self.rule["context_background_messages"],
                        background_chars=self.rule["context_background_chars"])
        self.history_targets({"group_id": group, "user_id": event.user_id, "message_id": event.message_id},
                             {"messages": event.context_prefill["messages"] + event.recent_group_context})
        token = event.social_reply["token"]
        self.turns[token]["context_prefill"] = True
        event.social_reply = self._turn_hint(token, self.turns[token])
        event.text = ("本群有新的明确呼叫。" if not event.passive_reply else "本群有值得参与的新发言。") + (
            "本轮原话和增量背景由频道上下文提供；信息足够就自然接话，不必先调用查记录工具。"
            "缺少更早背景、引用或媒体时，再按时间查询本群记录。"
            "这是本轮消息批次的控制通知，不是群友原话，不要向群里解释这套机制。"
            "\n本轮待处理消息索引：" + json.dumps(event.wakeup_batch, ensure_ascii=False))
        if event.passive_reply:
            event.text += ("\n本批可能包含明确对你的续聊请求，也可能只是可参与话题。明确承接你的新任务、"
                           "追问或结果反馈应正常回应；其余只是主动参与候选，选合适的话头自然参与，"
                           "不必逐条补答，不是所有人都分别在提问。")

    def observation_row(self, data, *, max_chars=None):
        from onebot_adapter.group_history import _render
        # Observer admission needs the whole current utterance, including a
        # request at its end. Numeric limits remain available to legacy callers.
        row = _render(data, max_chars, max_segments=None if max_chars is None else 256)
        if data.get("_already_woken"):
            row["already_woken"] = True
        return row

    @staticmethod
    def plain_text(data):
        return "".join(str(s.get("data", {}).get("text", "")) for s in data.get("message", [])
                       if isinstance(s, dict) and s.get("type") == "text").strip()

    def invited(self, data):
        ats = [str(s.get("data", {}).get("qq", "")) for s in data.get("message", []) if s.get("type") == "at"]
        if any(u != str(self.config.self_id) for u in ats):
            return False
        quotes = [str(s.get("data", {}).get("id", "")) for s in data.get("message", []) if s.get("type") == "reply"]
        known = quotes and all((str(data.get("group_id", "")), mid) in self.followups._outbox for mid in quotes)
        if quotes and not known and not ats:
            return False
        return bool(ats or known or called_by_name(self.plain_text(data)))

    def handle_preference(self, data):
        group, user = str(data.get("group_id", "")), str(data.get("user_id", ""))
        if not self.enabled(group) or not self.observer._allowed(group, user, self.blacklist_match):
            return False
        ats = [str(s.get("data", {}).get("qq", "")) for s in data.get("message", []) if s.get("type") == "at"]
        if any(u != str(self.config.self_id) for u in ats):
            return False
        quotes = [str(s.get("data", {}).get("id", "")) for s in data.get("message", []) if s.get("type") == "reply"]
        known = quotes and all((group, mid) in self.followups._outbox for mid in quotes)
        if quotes and not known:
            return False
        mode = preference_request(self.plain_text(data), addressed=self.invited(data))
        if mode is None:
            return False
        try:
            self.preferences.set(group, user, mode)
        except Exception:
            logger.exception("member preference persistence failed group=%s", group)
        self.followups.forget_user(group, user)
        self.preference_versions[(group, user)] = self.preference_versions.get((group, user), 0) + 1
        candidate = self.candidates.get(group)
        if candidate and str(candidate[0].get("user_id", "")) == user:
            self.candidates.pop(group, None)
        self.stats["member_preference_saved"] += 1
        return True  # handled silently before any model/gateway run

    def issue(self, group, user, intent, *, passive, message_id="", unmentioned=None):
        now = self.clock()
        self.turns = {k: v for k, v in self.turns.items() if now - v["at"] < 3600}
        token = uuid.uuid4().hex
        self.turns[token] = {"group": group, "user": user, "epoch": self.epochs.get(group, 0),
                             "at": now, "passive": passive, "message_id": str(message_id), "intent": intent,
                             "unmentioned": passive if unmentioned is None else unmentioned}
        self.turns[token]["preference_version"] = self.preference_versions.get((group, str(user)), 0)
        self.turns[token]["conversation"] = self.conversation_state(group, user, message_id)
        if self.rule["delivery_enabled"]:
            rows = self.rows(group)
            current = next((r for r in rows if r["message_id"] == str(message_id)), {})
            targets = [{"ref": "current", "message_id": str(message_id), "user_id": str(user),
                        "name": current.get("user_name", "当前发言人"), "preview": current.get("text", "")[:100]}]
            for row in rows:
                if row.get("bot") or row["message_id"] == str(message_id) or len(targets) >= 12:
                    continue
                targets.append({"ref": "r" + token[:6] + "_" + str(len(targets)), "message_id": row["message_id"],
                                "user_id": row["user_id"], "name": row["user_name"], "preview": row["text"][:100]})
            self.turns[token]["targets"] = [t for t in targets if valid_identity(t["user_id"], t["message_id"])
                                            and t["user_id"] != str(self.config.self_id)]
        return self._turn_hint(token, self.turns[token])

    @staticmethod
    def _turn_hint(token, turn):
        hint = {"token": token, "intent": turn["intent"]}
        if turn.get("conversation"):
            hint["conversation"] = dict(turn["conversation"])
        if "targets" in turn:
            hint["delivery_targets"] = [dict(t) for t in turn["targets"]]
        if turn.get("wakeup_batch"):
            hint["wakeup_batch"] = turn["wakeup_batch"]
        if turn.get("context_prefill"):
            hint["context_prefill"] = True
        if turn.get("candidate_requests"):
            hint["candidate_requests"] = [dict(r) for r in turn["candidate_requests"]]
        return hint

    def lookup(self, group, user, mid):
        if self.enabled(group) and self.observer._allowed(group, user, self.blacklist_match):
            for token, turn in reversed(self.turns.items()):
                if turn["group"] == group and turn["user"] == user and turn["message_id"] == mid:
                    return self._turn_hint(token, turn)
        # A native replay after adapter restart has no live admission capability.
        # Suppress rather than re-issuing a send capability for an old turn.
        return {"token": "0" * 32, "intent": "answer"}

    def history_targets(self, params, result):
        if not self.rule["observer_agent_enabled"]:
            return
        import hashlib
        for turn in reversed(self.turns.values()):
            if (turn["group"], turn["user"], turn["message_id"]) != (
                str(params.get("group_id")), str(params.get("user_id")), str(params.get("message_id"))):
                continue
            targets = turn.setdefault("targets", [])
            for row in result.get("messages", []):
                row["participation_preference"] = ("mention_only" if self.preferences.mention_only(
                    turn["group"], row["user_id"]) else "normal")
                if row["user_id"] == str(self.config.self_id):
                    continue
                ref = "h_" + hashlib.sha256(row["message_id"].encode()).hexdigest()[:16]
                row["reply_ref"] = ref
                if not any(t["ref"] == ref for t in targets):
                    targets.append({"ref": ref, "message_id": row["message_id"], "user_id": row["user_id"],
                                    "name": row["user_name"], "preview": row["text"][:100]})
            break

    def delivery_fields(self, group, token, plan, *, index=0, anchor_override=None):
        turn = self.turns.get(token)
        if turn is None or turn["group"] != group or not self.rule["delivery_enabled"]:
            raise PermissionError("delivery turn unavailable")
        chosen = normalize_plan(plan)
        fallback = chosen is None
        if chosen is None:
            if turn.get("wake_only") or turn.get("conversation", {}).get("continuing_exchange") is True:
                # The model may omit routing on a tiny natural utterance. Do
                # not revert an ongoing small exchange to per-message pings.
                # No new/guessed recipient is introduced by this fallback.
                chosen = {"mode": "group", "target": None, "notify": False}
            else:
                chosen = {"mode": "reply", "target": "current", "notify": True}
        fields = {"_delivery_mode": chosen["mode"], "_delivery_recipient": "",
                  "_delivery_fallback": fallback}
        if chosen["mode"] == "group":
            return fields
        target = next((t for t in turn.get("targets", []) if t["ref"] == chosen["target"]), None)
        if target is None or not valid_identity(target["user_id"], target["message_id"]) \
                or not self.observer._allowed(group, target["user_id"], self.blacklist_match):
            # Never fall back an unknown/stale target to an unrelated person.
            raise PermissionError("delivery target is not an allowed member of this turn")
        invited = not turn["unmentioned"] and target["user_id"] == turn["user"]
        batch_invite = turn.get("batch_invites", {}).get(target["user_id"])
        invited = invited or batch_invite is not None and batch_invite == self.preference_versions.get(
            (group, target["user_id"]), 0)
        if self.preferences.mention_only(group, target["user_id"]) and not invited:
            raise PermissionError("member did not invite this reply")
        if chosen["mode"] == "reply" and chosen["target"] == "current" and anchor_override:
            # Native busy redirects may retarget the same user's running turn.
            # Only actually admitted messages from this same group/user qualify.
            if not any(t["group"] == group and t["user"] == turn["user"]
                       and t["message_id"] == anchor_override for t in self.turns.values()):
                raise PermissionError("native reply anchor unavailable")
            target = {**target, "message_id": anchor_override}
        fields["_delivery_recipient"] = target["user_id"]
        fields["_delivery_message_id"] = target["message_id"]
        if index == 0:
            if chosen["mode"] == "reply":
                fields["reply_to"] = target["message_id"]
            if chosen["notify"]:
                fields["reply_mention_user_id"] = target["user_id"]
                fields["_validated_delivery_user"] = target["user_id"]
        return fields

    def guard(self, group, token, part_index=0):
        turn = self.turns.get(token)
        if not self.enabled(group) or turn is None or turn["group"] != group \
                or not self.observer._allowed(group, turn["user"], self.blacklist_match):
            return False
        if turn.get("unmentioned") and self.preferences.mention_only(group, turn["user"]):
            return False
        if turn.get("preference_version", 0) != self.preference_versions.get((group, turn["user"]), 0):
            return False
        if self.epochs.get(group, 0) != turn["epoch"] and (turn["passive"] or part_index > 0) \
                and not turn.get("serial_wakeup"):
            return False
        return turn.get("wake_only", False) or not turn["passive"] or (
            self.clock() - turn["at"] < self.rule["passive_expiry_seconds"])

    def successful_reply(self, group, user, mid, text, *, image=False):
        if not self.enabled(group) or not mid or internal_notice(text):
            return
        # Only acknowledged deliveries enter observation, never pending drafts.
        # Images expose the fact of delivery, not a local path, URL or guessed scene.
        from onebot_adapter.group_context import _Entry
        body = (str(text).strip() + " [已发送图片，观察者未识别画面]").strip() if image else str(text)
        rows = self.observer._buffers.setdefault(group, deque())
        if not any(e.message["message_id"] == mid for e in rows):
            rows.append(_Entry(self.clock(), {"message_id": mid, "user_id": str(self.config.self_id),
                        "user_name": "Kaguya", "timestamp": time.time(), "text": body[:600], "bot": True}))
            while len(rows) > 50:
                rows.popleft()
            if self.rule["observer_agent_enabled"] and body.strip():
                self.adaptive.note_reply(group, user, reward=not image)
        self.followups.record_reply(group, user, mid)
        if self.rule["observer_agent_enabled"] and self.archive is not None:
            self.archive.record({"message_type": "group", "group_id": group, "message_id": mid,
                                 "user_id": str(self.config.self_id), "time": time.time(),
                                 "sender": {"nickname": "Kaguya"},
                                 "message": ([{"type": "image", "data": {}}] if image else []) +
                                            [{"type": "text", "data": {"text": text}}]})

    def offer(self, data, emit):
        group, user = str(data.get("group_id", "")), str(data.get("user_id", ""))
        if self.preferences.mention_only(group, user):
            self.stats["member_quiet_skip"] += 1
            return
        if self.rule["observer_agent_enabled"]:
            if self.enabled(group):
                self.adaptive.offer(data, emit)
            return
        if not self.enabled(group) or self.busy(group):
            self.stats["busy_skip"] += 1
            return
        state = self.conversation_state(group, user, data.get("message_id", ""))
        kind = self.followups.classify(group, user, data.get("message", []) or [],
                                      contextual=state.get("continuing_exchange", False), consume=False)
        if kind == "blocked":
            self.candidates.pop(group, None)
            self.stats["addressed_elsewhere_skip"] += 1
            return
        now = self.clock()
        self.candidates[group] = (data, emit, kind == "followup", self.epochs.get(group, 0))
        if group not in self.tasks or self.tasks[group].done():
            task = asyncio.create_task(self._run(group, now), name=f"social-chat:{group}")
            self.tasks[group] = task
            task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)

    def _budget(self, group, followup):
        now, rule = self.clock(), self.rule
        checks = self.checks.setdefault(group, deque())
        while checks and now - checks[0] >= 3600:
            checks.popleft()
        interval = rule["followup_interval_seconds"] if followup else rule["judge_interval_seconds"]
        if len(checks) >= rule["judge_max_per_hour"] or checks and now - checks[-1] < interval:
            return False
        admissions = self.admissions.setdefault(group, deque())
        while admissions and now - admissions[0] >= rule["window_seconds"]:
            admissions.popleft()
        if not followup and (len(admissions) >= rule["max_replies"]
                             or admissions and now - admissions[-1] < rule["cooldown_seconds"]):
            return False
        checks.append(now)
        return True

    async def _run(self, group, started):
        try:
            # A busy stream has a maximum hold, never an unbounded debounce.
            while True:
                delay = min(self.rule["quiet_seconds"] - (self.clock() - self.last_seen.get(group, started)),
                            self.rule["max_hold_seconds"] - (self.clock() - started))
                if delay <= 0:
                    break
                await asyncio.sleep(delay)
            waited = False
            while group in self.candidates:
                data, emit, followup, epoch = self.candidates.pop(group)
                if self.busy(group) or epoch != self.epochs.get(group, 0):
                    self.stats["budget_or_stale_skip"] += 1
                    return
                state = self.conversation_state(group, str(data["user_id"]), data["message_id"])
                if followup and self.followups.classify(
                    group, str(data["user_id"]), data.get("message", []) or [],
                    contextual=state.get("continuing_exchange", False),
                ) != "followup":
                    self.stats["followup_budget_skip"] += 1
                    return
                if not self._budget(group, followup):
                    self.stats["budget_or_stale_skip"] += 1
                    return
                snapshot = self.rows(group)
                payload = {"group_id": group, "self_id": str(self.config.self_id),
                           "followup": followup, "current_message_id": str(data["message_id"]),
                           "messages": snapshot, "waited": waited,
                           "conversation": state}
                try:
                    result = decision(await asyncio.wait_for(self.judge(payload), self.rule["judge_timeout_seconds"]))
                except Exception:
                    self.stats["judge_error"] += 1
                    logger.warning("social judgment unavailable; passive skipped group=%s", group)
                    return
                self.stats[result["action"]] += 1
                logger.info("social judgment: group=%s action=%s intent=%s", group, result["action"], result["intent"])
                # No stale judgment of a superseded message may enter the formal turn.
                if epoch != self.epochs.get(group, 0) or self.busy(group):
                    return
                if group in self.candidates:
                    if waited:
                        return
                    waited = True
                    await asyncio.sleep(self.rule["judge_interval_seconds"])
                    continue
                if result["action"] == "wait" and not waited:
                    waited = True
                    self.candidates[group] = (data, emit, followup, epoch)
                    await asyncio.sleep(self.rule["judge_interval_seconds"])
                    continue
                if result["action"] != "reply" or self.rule["mode"] != "active":
                    return
                if not self.guard_enabled_data(data):
                    return
                self.admissions.setdefault(group, deque()).append(self.clock()) if not followup else None
                hint = self.issue(group, str(data["user_id"]), result["intent"], passive=not followup,
                                  message_id=str(data["message_id"]), unmentioned=True)
                # Tiny selected window, incremental IDs deduplicated by existing Hermes hooks.
                await emit(data, followup, hint, snapshot[-4:])
                return
        finally:
            self.candidates.pop(group, None)

    def guard_enabled_data(self, data):
        group, user = str(data.get("group_id", "")), str(data.get("user_id", ""))
        return self.enabled(group) and self.observer._allowed(group, user, self.blacklist_match) \
            and not self.preferences.mention_only(group, user)

    async def close(self):
        await self.adaptive.close()
        tasks = list(self.tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.tasks.clear()
        self.candidates.clear()
        self.preferences.close()
