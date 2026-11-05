"""Resolve only current-group quotes and the contents of forwards posted there.

Query filters select roots, not their necessary sources. Embedded forward authors
are untrusted quoted identities, never new current-group members or reply targets.
Traversal is iterative, cycle checked and time bounded, with explicit breakpoints.
"""
from __future__ import annotations

import asyncio
import copy
import time
from collections import deque

from onebot_adapter.group_history import _order, _render
from onebot_adapter.history_search import forward_id


def render_source(raw):
    return _render(raw, 2**31, max_segments=None)


class HistoryReferences:
    def __init__(self, reader, group, user, *, include_media=False, timeout=30.0, clock=time.monotonic):
        self.reader, self.group, self.user = reader, group, user
        self.include_media, self.clock = include_media, clock
        self.deadline = clock() + timeout
        self.scope = reader._scopes[group]
        self.raw, self.rows, self.errors, self.links = {}, {}, [], []
        self.fetched, self.failed, self.forward_cache = set(), set(), {}
        self.embedded_ids, self.forward_links = {}, []

    def check_scope(self):
        if self.reader._scopes.get(self.group) != self.scope or not self.reader._allowed_user(self.group, self.user):
            raise PermissionError("当前群查询权限已撤销")

    def allowed(self, raw):
        user = str(raw.get("user_id") or (raw.get("sender") or {}).get("user_id", ""))
        return user == str(self.reader._config.self_id) or self.reader._allowed_user(self.group, user)

    def error(self, kind, message_id, reason, *, source=None):
        item = {"kind": kind, "message_id": str(message_id), "reason": reason}
        if source is not None:
            item["source_message_id"] = str(source)
        if item not in self.errors:
            self.errors.append(item)

    async def call(self, action, params):
        self.check_scope()
        remaining = self.deadline - self.clock()
        if remaining <= 0:
            raise TimeoutError
        result = await asyncio.wait_for(self.reader._api.call(action, params), remaining)
        self.check_scope()
        data = result.get("data") if isinstance(result, dict) else None
        if not isinstance(data, dict):
            raise ValueError("invalid reference response")
        return data

    async def fetch(self, mid, *, source=None, refresh=False):
        mid = str(mid)
        self.check_scope()
        existing = self.raw.get(mid) or self.reader.archive.get(self.group, mid)
        if existing and (not refresh or mid in self.fetched):
            if not self.allowed(existing):
                self.error("reply", mid, "member_not_authorized", source=source)
                return None
            self.raw[mid] = existing
            return existing
        if mid in self.failed:
            return None
        try:
            raw = await self.call("get_msg", {"message_id": mid})
        except PermissionError:
            raise
        except Exception as exc:
            self.failed.add(mid)
            self.error("message", mid, "timeout" if isinstance(exc, TimeoutError) else "unavailable", source=source)
            return None
        if raw.get("message_type") != "group" or str(raw.get("group_id", "")) != self.group \
                or str(raw.get("message_id", "")) != mid:
            self.failed.add(mid)
            self.error("reply", mid, "outside_current_group_or_wrong_message", source=source)
            return None
        if not self.allowed(raw):
            self.failed.add(mid)
            self.error("reply", mid, "member_not_authorized", source=source)
            return None
        self.fetched.add(mid)
        self.reader.archive.record(raw)
        raw = self.reader.archive.get(self.group, mid) or raw
        self.raw[mid] = raw
        return raw

    def media(self, raw, rendered):
        if not self.include_media:
            return
        items = []
        for segment in raw.get("message", []):
            if not isinstance(segment, dict) or segment.get("type") not in {"image", "record", "video"}:
                continue
            data = segment.get("data")
            if isinstance(data, dict) and isinstance(data.get("url"), str) and data["url"]:
                items.append({"kind": segment["type"], "url": data["url"]})
        if items:
            rendered["media_items"] = items
        elif any(mark in rendered["text"] for mark in ("[图片", "[视频", "[语音")):
            rendered["media_unavailable"] = True

    @staticmethod
    def node(raw):
        if not isinstance(raw, dict):
            return None
        if raw.get("type") == "node" and isinstance(raw.get("data"), dict):
            raw = raw["data"]
        segments = raw.get("message", raw.get("content"))
        if not isinstance(segments, list):
            return None
        sender = raw.get("sender") if isinstance(raw.get("sender"), dict) else {}
        user = raw.get("user_id", sender.get("user_id", raw.get("uin", "")))
        sender = {**sender, "user_id": user,
                  "nickname": sender.get("nickname") or raw.get("nickname") or raw.get("name") or str(user)}
        return {**raw, "message": segments, "user_id": user, "sender": sender}

    async def forwards(self, raw, rendered):
        work = deque([(raw, rendered, (), frozenset(), frozenset())])
        while work:
            self.check_scope()
            current, output, path, resource_ancestors, object_ancestors = work.popleft()
            if self.clock() >= self.deadline:
                self.error("forward", output["message_id"], "timeout")
                break
            self.media(current, output)
            for segment_index, segment in enumerate(current.get("message", [])):
                if not isinstance(segment, dict) or segment.get("type") != "forward":
                    continue
                data = segment.get("data")
                if not isinstance(data, dict):
                    self.error("forward", output["message_id"], "unsupported_format")
                    continue
                resource = str(data.get("id") or data.get("resid") or "")
                inline = data.get("content")
                if (resource and resource in resource_ancestors and (not isinstance(inline, list) or not inline)) or (
                    isinstance(inline, list) and id(inline) in object_ancestors
                ):
                    self.error("forward", output["message_id"], "cycle")
                    continue
                if not isinstance(inline, list) or not inline:
                    if not resource:
                        self.error("forward", output["message_id"], "unavailable")
                        continue
                    if resource not in self.forward_cache:
                        try:
                            response = await self.call("get_forward_msg", {"message_id": resource})
                            nodes = response.get("messages", response.get("message"))
                            if not isinstance(nodes, list):
                                raise ValueError("invalid forward response")
                            self.forward_cache[resource] = nodes
                        except PermissionError:
                            raise
                        except Exception as exc:
                            self.forward_cache[resource] = None
                            self.error("forward", output["message_id"],
                                       "timeout" if isinstance(exc, TimeoutError) else "unavailable")
                    inline = self.forward_cache[resource]
                    if inline is None:
                        continue
                    # Keep fetched text with its real current-group carrier, not as new live messages.
                    data["content"] = inline
                if not inline:
                    output["forwarded_messages"] = []
                    continue
                children = output.setdefault("forwarded_messages", [])
                for index, value in enumerate(inline):
                    node = self.node(value)
                    if node is None:
                        self.error("forward", output["message_id"], "unsupported_node")
                        continue
                    child_path = (*path, segment_index, index)
                    original_id = str(node.get("message_id") or "")
                    node = {**node, "message_id": forward_id(rendered["message_id"], child_path)}
                    child = render_source(node)
                    child.update(forwarded=True, container_message_id=rendered["message_id"],
                                 untrusted_history=True)
                    if original_id:
                        child["original_message_id"] = original_id
                        self.embedded_ids.setdefault(original_id, child["message_id"])
                    if child.get("reply_to_message_id"):
                        self.forward_links.append((child["message_id"], child["reply_to_message_id"]))
                    children.append(child)
                    work.append((node, child, child_path,
                                 resource_ancestors | ({resource} if resource else set()),
                                 object_ancestors | {id(inline)}))

    def cycles(self):
        edges = dict(self.links)
        done = set()
        for start in edges:
            current, trail = start, set()
            while current in edges and current not in done:
                if current in trail:
                    self.error("reply", current, "cycle", source=edges[current])
                    break
                trail.add(current)
                current = edges[current]
            done.update(trail)

    async def expand(self, roots):
        matched = {str(raw["message_id"]) for raw in roots}
        self.raw = {str(raw["message_id"]): copy.deepcopy(raw) for raw in roots}
        # Every matched original remains visible even if later reference I/O times out.
        self.rows = {mid: render_source(raw) for mid, raw in self.raw.items()}
        pending, processed = deque(self.raw), set()
        while pending:
            self.check_scope()
            if self.clock() >= self.deadline:
                self.error("resolution", pending[0], "timeout")
                break
            mid = pending.popleft()
            if mid in processed:
                continue
            processed.add(mid)
            raw = self.raw[mid]
            if self.include_media and any(isinstance(s, dict) and s.get("type") in {
                "image", "record", "video", "forward"} for s in raw.get("message", [])):
                fresh = await self.fetch(mid, refresh=True)
                if fresh:
                    raw = fresh
            output = self.rows[mid] = render_source(raw)
            if mid not in matched:
                output["reference_only"] = True
            await self.forwards(raw, output)
            self.reader.archive.record(raw)
            parent = output.get("reply_to_message_id")
            links = [(mid, parent)] if parent else []
            # Forward-local quoted nodes are already included; never query their original source group.
            for source, parent in self.forward_links:
                if parent in self.embedded_ids:
                    self.links.append((source, self.embedded_ids[parent]))
                else:
                    links.append((source, parent))
            self.forward_links.clear()
            for source, parent in links:
                self.links.append((source, parent))
                if parent in self.raw or parent in processed:
                    continue
                quoted = await self.fetch(parent, source=source)
                if quoted is not None:
                    self.rows[parent] = render_source(quoted)
                    self.rows[parent]["reference_only"] = True
                    pending.append(parent)
        self.check_scope()
        self.cycles()
        messages = [self.rows[mid] for mid in sorted(self.rows, key=lambda mid: _order(self.raw[mid]))]
        return {"messages": messages, "matched_count": len(matched),
                "referenced_count": sum(mid not in matched for mid in self.rows),
                "references_complete": not self.errors, "reference_errors": self.errors}
