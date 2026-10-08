"""记忆卡构造：把「字段很多容易漏」的活交给代码，而不是交给提示词。

make_card 负责补齐所有默认值与派生字段（id / dedup_key / half_life_days /
observed_at），保证每张卡都过 schema 校验。
"""

from __future__ import annotations

from datetime import date
from typing import Any, Iterable

from .clock import Clock, parse_ts, to_iso, to_utc
from .ids import card_id, dedup_key
from .schema import validate_card
from .tiers import half_life_for

LIST_FIELDS = ("entities", "keywords", "facts", "decisions", "todos", "constraints")


def make_card(
    *,
    clock: Clock,
    session_id: str,
    tier: str,
    importance: int,
    network: str,
    source_ids: Iterable[str],
    summary: str,
    entities: Iterable[str] = (),
    keywords: Iterable[str] = (),
    facts: Iterable[str] = (),
    decisions: Iterable[str] = (),
    todos: Iterable[str] = (),
    constraints: Iterable[str] = (),
    claims: Iterable[dict] = (),
    raw_quote: str | None = None,
    confidence: float | None = None,
    valid_from: str | None = None,
    valid_to: str | None = None,
    observed_at: str | None = None,
    validate: bool = True,
) -> dict[str, Any]:
    """构造并（默认）校验一张记忆卡。

    ``tier`` 决定 ``half_life_days``；``id`` 与 ``dedup_key`` 由内容派生，
    因此重复压缩同一段原文必然得到同一张卡（幂等）。
    """
    ids = list(dict.fromkeys(source_ids))  # 去重且保持顺序
    if not ids:
        raise ValueError("source_ids 不能为空：每张卡必须可回溯到原文")

    moment = to_utc(clock.now()) if observed_at is None else parse_ts(observed_at)
    stamp = to_iso(moment)
    day: date = moment.date()

    card: dict[str, Any] = {
        "id": card_id(day, session_id, ids, tier, summary),
        "schema_version": 1,
        "session_id": session_id,
        "tier": tier,
        "importance": importance,
        "network": network,
        "observed_at": stamp,
        "valid_from": valid_from if valid_from is not None else day.isoformat(),
        "valid_to": valid_to,
        "half_life_days": half_life_for(tier),
        "last_access": None,
        "access_count": 0,
        "status": "active",
        "source_ids": ids,
        "supersedes": [],
        "superseded_by": None,
        "summary": summary,
        "raw_quote": raw_quote,
        "confidence": confidence,
        "dedup_key": dedup_key(tier, ids, summary),
    }
    for name in LIST_FIELDS:
        card[name] = list(locals()[name])
    card["claims"] = [
        {
            "kind": claim["kind"],
            "subject_entity": claim.get("subject_entity"),
            "predicate": claim["predicate"],
            "object": claim.get("object"),
            "polarity": int(claim.get("polarity", 1)),
            "due_at": claim.get("due_at"),
            "valid_from": claim.get("valid_from", card["valid_from"]),
            "valid_to": claim.get("valid_to"),
        }
        for claim in claims
    ]

    if validate:
        validate_card(card)
    return card
