"""Transport courtesy rules. No model instructions or private data stored."""
from __future__ import annotations

import re

_NOTICE = re.compile(
    r"\s*(?:⚠️?\s*)?The model returned only a silence marker for a message that needed a reply\."
    r"(?:\s*Try again or rephrase\.)?\s*\Z|"
    r"\s*💾\s*Self-improvement review:\s+[\s\S]+\Z|"
    r"\s*📬\s*No home channel is set for Onebot\.[\s\S]+\Z", re.IGNORECASE,
)
_NAME = re.compile(r"^\s*(?:Kaguya|辉夜|かぐや)\s*(?:[,，:：!！?？、\s]|$)", re.IGNORECASE)
_PERSONAL_QUIET = re.compile(
    r"^(?:请|请你|请记住|记住)?(?:你)?(?:不要|别|禁止|不许)(?:再|随便|动不动)?主动"
    r"(?:给我发(?:送)?消息|给我发言|找我(?:聊天)?|@我|艾特我|接我的话|回复我)[。.!！\s]*$",
)
_QUIET = re.compile(
    r"^(?:请|请你)?(?:你)?(?:先)?(?:闭嘴|安静(?:点|一下)?|别说(?:话|了)|不要说话|别插(?:话|嘴|画)|"
    r"别随便插(?:话|画)|不要随便主动发送消息|不要动不动主动发言|没叫你别出来|"
    r"大人说话[，,、\s]*小孩别插嘴|(?:别乱讲[，,\s]*)?只有(?:我)?(?:让你说|叫你|点名)"
    r"(?:的时候|时)?才(?:说|说话|出来|回复)(?:[，,\s]*把这条写到记忆里)?)[。.!！\s]*$",
)
_RESUME = re.compile(
    r"^(?:你)?(?:可以|允许|恢复)(?:再)?主动(?:给我发(?:送)?消息|找我(?:聊天)?|接我的话|回复我)"
    r"(?:了)?[。.!！\s]*$",
)


def internal_notice(text):
    """Known machine notices only, not English answers or genuine task failures."""
    return isinstance(text, str) and _NOTICE.fullmatch(text) is not None


def called_by_name(text):
    return isinstance(text, str) and _NAME.match(text) is not None


def preference_request(text, *, addressed):
    """Only a member's own clear preference; not quotes or third-person text."""
    value = text.strip()
    if called_by_name(value):
        value = _NAME.sub("", value, count=1).strip()
    if _RESUME.fullmatch(value):
        return "normal"
    if _PERSONAL_QUIET.fullmatch(value) or addressed and _QUIET.fullmatch(value):
        return "mention_only"
    return None
