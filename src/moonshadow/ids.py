"""内容寻址 ID。

原文档用 ``mem_20260930_001`` 递增序列，多进程写入会撞号、重跑会产生重复卡。
改成内容寻址后：同一段原文 + 同一 tier + 同一摘要 => 同一 ID，天然幂等。
"""

from __future__ import annotations

import base64
import hashlib
import re
from datetime import date
from typing import Iterable

CARD_ID_RE = re.compile(r"^mem_[0-9]{8}_[a-z2-7]{10}$")


def _b32(digest: bytes, size: int = 10) -> str:
    """base32 小写、去填充，字符集恰为 a-z2-7，与 schema.json 的 pattern 一致。"""
    return base64.b32encode(digest).decode("ascii").lower().rstrip("=")[:size]


def _day_token(day: str | date) -> str:
    text = day.isoformat() if isinstance(day, date) else str(day)
    return text.replace("-", "")


def _digest(*parts: str) -> bytes:
    return hashlib.sha256("|".join(parts).encode("utf-8")).digest()


def content_id(prefix: str, day: str | date, *parts: str) -> str:
    """通用内容寻址 ID：``<prefix>_<YYYYMMDD>_<hash10>``。"""
    return f"{prefix}_{_day_token(day)}_{_b32(_digest(_day_token(day), *parts))}"


def message_id(day: str | date, session_id: str, line_no: int) -> str:
    """原始消息 ID。line_no 参与寻址，因此 ID 与字节区间一一对应。"""
    return content_id("msg", day, session_id, str(line_no))


def card_id(
    day: str | date,
    session_id: str,
    source_ids: Iterable[str],
    tier: str,
    summary: str,
) -> str:
    """记忆卡 ID。

    注意：只由 (日期, 会话, 来源) 决定是**错的**——同一段原文会合法地抽出多张卡
    （决策 + 待办 + 约束），它们会撞成同一个 ID 并被静默丢弃。因此把 tier 与
    summary 一并纳入身份：

    - 同一段原文 + 同一输出 => 同一 ID，重跑即重放（幂等）
    - 同一段原文的不同产出 => 不同 ID，可以并存
    - 同 span 同 tier 但摘要变了的产出 => 由 store.put_card 标记取代关系，不重复激活
    """
    joined = ",".join(sorted(set(source_ids)))
    return content_id("mem", day, session_id, joined, tier, summary)


def claim_id(day: str | date, card: str, kind: str, predicate: str, obj: str | None) -> str:
    return content_id("clm", day, card, kind, predicate, obj or "")


def dedup_key(tier: str, source_ids: Iterable[str], summary: str) -> str:
    """卡级去重键：tier + 来源 + 摘要。SQLite 唯一索引，保证插入幂等。"""
    joined = ",".join(sorted(set(source_ids)))
    return hashlib.sha256("|".join([tier, joined, summary]).encode("utf-8")).hexdigest()
