"""Notification-only batches. Never concatenate different members' original text."""
from __future__ import annotations

import uuid
from dataclasses import replace

from onebot_adapter.hermes_plugin.delivery import valid_identity


def combine_wakeups(events):
    """Seal only the already-waiting snapshot; there is no collection timer."""
    first = events[0]
    requests, seen, delivery_ids = [], set(), []
    passive_candidates = False
    for event in events:
        if event.chat_id != first.chat_id or not event.wake_only or event.is_system_notice:
            raise ValueError("cannot batch unrelated events")
        if not valid_identity(str(event.user_id), str(event.message_id)):
            raise ValueError("invalid wakeup identity")
        candidates = event.wakeup_batch.get("requests", []) if event.passive_reply else []
        passive_candidates = passive_candidates or bool(candidates)
        for request in candidates:
            message_id, user = str(request.get("message_id", "")), str(request.get("user_id", ""))
            if not valid_identity(user, message_id) or request.get("direct") is not False:
                raise ValueError("invalid passive wakeup candidate")
            if message_id not in seen:
                requests.append({"message_id": message_id, "user_id": user,
                                 "timestamp": request.get("timestamp", event.timestamp), "direct": False})
                seen.add(message_id)
        for message_id in dict.fromkeys([*event.context_message_ids, event.message_id]):
            message_id = str(message_id)
            if message_id not in seen and valid_identity(str(event.user_id), message_id):
                requests.append({"message_id": message_id, "user_id": str(event.user_id),
                                 "timestamp": event.timestamp, "direct": not event.passive_reply})
                seen.add(message_id)
        delivery_ids.extend(event.delivery_ids or ([event.delivery_id] if event.delivery_id else []))
    return replace(first, media_items=[], recent_group_context=[], reply_to_text=None,
                   reply_to_message_id=None, delivery_ids=list(dict.fromkeys(delivery_ids)),
                   context_message_ids=[r["message_id"] for r in requests],
                   # Candidate authors did not individually pass a command
                   # admission. Never lend them the anchor author's privilege.
                   is_admin=not passive_candidates and all(e.is_admin for e in events),
                   is_global_admin=not passive_candidates and all(e.is_global_admin for e in events),
                   social_reply=dict(first.social_reply),
                   wakeup_batch={"id": uuid.uuid4().hex, "requests": requests})
