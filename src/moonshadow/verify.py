"""确定性精度校验（Phase 2.5 第一级）。

原则：**先用正则判死，再让模型判活。** v1.0 §8 只写了「用另一个模型检查」，
那是软校验：模型说没问题就没问题。这里先做一道不依赖模型、可回归的硬关卡。

作用域很关键：一段原文会合法地支撑多张卡（T0 记约束、T2 记决策、T3 记待办），
所以校验的对象是「原文 → 引用它的整批卡」，而不是「原文 → 某一张卡」。
否则会大量误报（例如 12 万只写进 T0 卡，却被判定为 T2 卡丢字段）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

#: 必须原样保留的字段类型 —— 这些一旦丢失，摘要就不可信。
HARD_PATTERNS: dict[str, str] = {
    "date": r"\d{4}-\d{2}-\d{2}",
    "amount": r"\d+(?:\.\d+)?\s*(?:万|亿|千|元|块|美元|美金|USD|CNY|RMB)",
    "url": r"https?://[^\s，。；、）)]+",
    "code": r"`[^`]+`",
}

#: 软信号：原文有否定/条件，整批卡却一条都没有 => 提示人复核，不作为失败。
SOFT_PATTERNS: dict[str, str] = {
    "negation": r"(?:不要|不得|不能|不会|不需要|无需|禁止|避免|除非|没有|不是)",
    "condition": r"(?:如果|若|一旦|当.+时|前提是|除非)",
}

CARD_TEXT_FIELDS = (
    "summary",
    "raw_quote",
    "facts",
    "decisions",
    "todos",
    "constraints",
    "entities",
    "keywords",
)


@dataclass
class Report:
    """校验结果。``missing`` 非空即判失败。"""

    missing: dict[str, list[str]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing

    def summary(self) -> str:
        if self.ok and not self.warnings:
            return "PASS：关键字段无丢失"
        parts = []
        if self.missing:
            total = sum(len(v) for v in self.missing.values())
            details = "；".join(f"{k} 缺 {v}" for k, v in self.missing.items())
            parts.append(f"FAIL：{total} 个关键字段丢失（{details}）")
        else:
            parts.append("PASS：关键字段无丢失")
        parts.extend(f"WARN：{w}" for w in self.warnings)
        return " | ".join(parts)


def normalize(text: str) -> str:
    """比对前归一化：去掉空格与千分位逗号，避免「12 万」/「12万」误判。"""
    return re.sub(r"[\s,，]", "", text)


def extract_key_fields(text: str) -> dict[str, list[str]]:
    """按类型抽取关键字段，保持出现顺序并去重。"""
    found: dict[str, list[str]] = {}
    for name, pattern in HARD_PATTERNS.items():
        seen: list[str] = []
        for match in re.findall(pattern, text):
            if match not in seen:
                seen.append(match)
        if seen:
            found[name] = seen
    return found


def card_text(card: Mapping[str, Any]) -> str:
    """把一张卡里所有「可承载字面值」的字段拼成一段文本用于比对。"""
    chunks: list[str] = []
    for name in CARD_TEXT_FIELDS:
        value = card.get(name)
        if value is None:
            continue
        if isinstance(value, str):
            chunks.append(value)
        elif isinstance(value, Iterable):
            chunks.extend(str(item) for item in value)
    return "\n".join(chunks)


def verify_no_silent_loss(
    raw_text: str,
    cards: Sequence[Mapping[str, Any]],
    *,
    soft: bool = True,
) -> Report:
    """校验一段原文里的关键字段，是否至少被一张卡保留下来。"""
    report = Report()
    haystack = normalize(card_text_many(cards))
    for name, values in extract_key_fields(raw_text).items():
        lost = [value for value in values if normalize(value) not in haystack]
        if lost:
            report.missing[name] = lost

    if soft and cards:
        raw_soft = {k: re.findall(p, raw_text) for k, p in SOFT_PATTERNS.items()}
        for name, hits in raw_soft.items():
            if hits and not re.search(SOFT_PATTERNS[name], card_text_many(cards)):
                report.warnings.append(
                    f"原文含 {name}（{hits[:3]}），但没有任何一张卡记录该{name}语义，建议复核"
                )
    return report


def card_text_many(cards: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join(card_text(card) for card in cards)
