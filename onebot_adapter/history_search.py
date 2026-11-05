"""Group-history matching, never shell commands or cross-message concatenation."""
from __future__ import annotations

import hashlib
import time
import unicodedata


def normalized(text):
    return "".join(unicodedata.normalize("NFKC", text).casefold().split())


def order_key(raw):
    from onebot_adapter.group_history import _order

    return (*_order(raw), str(raw.get("message_id", "")))


def forward_id(container, path):
    opaque = hashlib.sha256((str(container) + repr(path)).encode()).hexdigest()[:20]
    return "forward_" + opaque


def documents(raw):
    """One searchable document per original/forwarded message, not per bundle."""
    pending, seen = [(raw, ())], set()
    while pending:
        current, path = pending.pop()
        if not isinstance(current, dict) or id(current) in seen:
            continue
        seen.add(id(current))
        if current.get("type") == "node" and isinstance(current.get("data"), dict):
            current = current["data"]
        segments = current.get("message", current.get("content", []))
        if not isinstance(segments, list):
            yield path, ""
            continue
        text = []
        for index, segment in enumerate(segments):
            if not isinstance(segment, dict) or not isinstance(segment.get("data"), dict):
                continue
            data = segment["data"]
            if segment.get("type") == "text":
                text.append(str(data.get("text", "")))
            elif segment.get("type") == "forward" and isinstance(data.get("content"), list):
                pending.extend((node, (*path, index, child)) for child, node in enumerate(data["content"]))
        yield path, "".join(text)


class HistoryMatcher:
    """Literal/normalized SQL matches, and explicitly opt-in, time-bounded regex."""
    def __init__(self, params):
        keyword, candidates = params.get("keyword"), params.get("keywords", [])
        if keyword is not None and (not isinstance(keyword, str) or not keyword.strip()):
            raise ValueError("keyword须为非空文字")
        if not isinstance(candidates, list) or ("keywords" in params and not candidates) \
                or any(not isinstance(value, str) or not value.strip() for value in candidates):
            raise ValueError("keywords须为非空候选词数组")
        self.terms = list(dict.fromkeys(([keyword] if keyword is not None else []) + candidates))
        if len(self.terms) > 64 or any(len(term) > 4096 for term in self.terms):
            raise ValueError("候选词最多64个，每个最多4096个字符")
        self.mode, self.combination = params.get("match_mode", "normalized"), params.get("keyword_mode", "any")
        if self.mode not in ("normalized", "literal", "regex"):
            raise ValueError("match_mode须为normalized、literal或regex")
        if self.combination not in ("any", "all"):
            raise ValueError("keyword_mode须为any或all")
        self.needles = [normalized(term) for term in self.terms] if self.mode == "normalized" else self.terms
        self.patterns, self.failure = [], None
        self.deadline = 0.0
        if self.mode == "regex" and self.terms:
            import regex

            try:
                self.patterns = [regex.compile(term, regex.VERSION1) for term in self.terms]
            except regex.error:
                raise ValueError("正则表达式不合法，请修正或改用literal模式") from None

    def begin(self):
        self.deadline = time.monotonic() + 5.0
        self.failure = None

    def regex_match(self, index, text):
        try:
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError
            return int(self.patterns[index].search(text, timeout=min(0.05, remaining)) is not None)
        except TimeoutError:
            self.failure = "正则检索超时，请缩小范围、简化表达式或改用normalized模式；不能据此断言没有记录"
            raise

    def sql(self):
        if not self.terms:
            return "1", []
        if self.mode == "regex":
            clauses, values = [f"history_regex({i},d.body)>0" for i in range(len(self.terms))], []
        else:
            column = "d.normalized_body" if self.mode == "normalized" else "d.body"
            clauses, values = [f"instr({column},?)>0" for _ in self.needles], list(self.needles)
        join = " OR " if self.combination == "any" else " AND "
        return "(" + join.join(clauses) + ")", values
