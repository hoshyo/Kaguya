"""Local-only current originals plus bounded candidates; transcript dedup happens in the plugin."""
from __future__ import annotations

import time

from onebot_adapter.group_history import _order, _render


def packet_row(raw, limit):
    row = _render(raw, limit)
    row["sequence"] = _order(raw)[1]
    return row


def prefill_context(event, archive, allowed, *, self_id, background_messages=20, background_chars=6000):
    group = event.chat_id.removeprefix("group:")
    requests = event.wakeup_batch["requests"]
    current, missing, truncated = [], [], []
    current_ids = {r["message_id"] for r in requests}
    per_message_chars = min(16000, 64000 // max(1, len(requests)))
    for request in requests:
        raw = archive.get(group, request["message_id"]) if archive is not None else None
        if not raw or str(raw.get("user_id")) != request["user_id"] or not allowed(request["user_id"]):
            missing.append(request["message_id"])
            continue
        row = packet_row(raw, per_message_chars)
        current.append(row)
        if row.get("text_truncated"):
            truncated.append(row["message_id"])

    # Native replies already in the current transcript are filtered by ID later.
    candidates = archive.query(group, count=background_messages + len(current_ids) + 1) if archive else []
    candidates = [r for r in candidates if str(r.get("message_id")) not in current_ids
                  and allowed(str(r.get("user_id", "")))
                  and not _render(r, 32)["text"].lstrip().startswith("/")]
    candidates.sort(key=_order)
    limited = len(candidates) > background_messages
    chosen, selected, remaining = [], set(current_ids), background_chars
    quote_gaps = []
    for row in current:
        parent, visited = row.get("reply_to_message_id"), set()
        while parent and parent not in visited:
            visited.add(parent)
            raw = archive.get(group, parent) if archive else None
            if not raw or not allowed(str(raw.get("user_id", ""))):
                quote_gaps.append(parent)
                break
            quoted = packet_row(raw, max(1, remaining))
            if parent not in selected:
                if remaining <= 0 or len(chosen) >= background_messages or quoted.get("text_truncated"):
                    quote_gaps.append(parent)
                    limited = True
                    break
                quoted["quoted_reference"] = True
                chosen.append(quoted)
                selected.add(parent)
                remaining -= len(quoted["text"])
            parent = quoted.get("reply_to_message_id")

    for raw in reversed(candidates):
        mid = str(raw.get("message_id"))
        if mid in selected:
            continue
        if len(chosen) >= background_messages or remaining <= 0:
            limited = True
            break
        row = packet_row(raw, max(1, remaining))
        if row.get("text_truncated"):
            limited = True
        row["bot"] = row["user_id"] == self_id
        chosen.append(row)
        selected.add(mid)
        remaining -= len(row["text"])
    current.sort(key=lambda r: (r["timestamp"], r["sequence"]))
    chosen.sort(key=lambda r: (r["timestamp"], r["sequence"]))
    event.recent_group_context = chosen
    event.context_prefill = {"messages": current, "query_time": time.time(),
        "missing_current_ids": missing, "truncated_current_ids": truncated,
        "missing_quote_ids": list(dict.fromkeys(quote_gaps)), "background_limited": limited,
        "background_start_time": min((r["timestamp"] for r in chosen), default=None),
        "background_end_time": max((r["timestamp"] for r in chosen), default=None),
        "background_gap_seconds": max(0, time.time() - max(r["timestamp"] for r in chosen)) if chosen else None,
        "background_max_messages": background_messages, "background_max_chars": background_chars,
        "scope": "current_group_only", "untrusted_history": True}
