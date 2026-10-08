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

#: **定性内容判据**用的三类关键词。这是 v1.3 Phase 0 补的第二道关卡：
#: 硬字段（日期/金额/URL/代码）保住 ≠ 内容保住——「决定先做读缓存」「绝不带病上线」
#: 这类句子一个硬字段都没有，丢光了也不会让硬关卡变红。
#:
#: 与 `compress.TIER_KEYWORDS` 的 `T0`/`T2` 是**同一批词**：`compress` 依赖 `verify`
#: （`extract_key_fields`），因此 `verify` 不能反向 import，只能复制一份；
#: `tests/test_precision.py` 有同步断言，改了任一侧就会红。
QUALITATIVE_KEYWORDS: dict[str, tuple[str, ...]] = {
    # T2 的判定词表：决策/结论
    "decision": ("决定", "采用", "确定", "拍板", "选定", "结论", "同意", "敲定"),
    # T0 的判定词表：硬约束
    "constraint": ("必须", "不得", "不要", "禁止", "避免", "硬性", "合规", "法律", "安全", "不可"),
    # 与 SOFT_PATTERNS["negation"] 同一批词
    "negation": ("不要", "不得", "不能", "不会", "不需要", "无需", "禁止", "避免", "除非", "没有", "不是"),
}

QUALITATIVE_PATTERNS: dict[str, re.Pattern[str]] = {
    kind: re.compile("|".join(re.escape(word) for word in words))
    for kind, words in QUALITATIVE_KEYWORDS.items()
}

#: 句子切分。与 `compress.SENTENCE_SPLIT_RE` 同一套（同样因循环依赖只能复制，
#: 由 `tests/test_precision.py` 的同步断言钉住）。
SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?\n])")

#: 哪些层级的卡**不算**定性内容的载体。寒暄卡与噪音卡可以引用原文，但把它们
#: 当作「决策有载体」是空头支票：T8/T9 的内容按定义就是要丢掉的。
CARRY_FORBIDDEN_TIERS: tuple[str, ...] = ("T8", "T9")

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


def qualitative_sentences(text: str) -> list[tuple[str, str]]:
    """挑出原文里的定性句子，返回 `(kind, 句子)` 列表（按出现顺序，同句只取一次）。"""
    found: list[tuple[str, str]] = []
    for sentence in SENTENCE_SPLIT_RE.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        for kind, pattern in QUALITATIVE_PATTERNS.items():
            if pattern.search(sentence):
                found.append((kind, sentence))
                break
    return found


def verify_qualitative_retention(
    messages: Sequence[Mapping[str, Any]],
    cards: Sequence[Mapping[str, Any]],
) -> Report:
    """**定性内容必须有载体**：决策 / 硬约束 / 否定句所在的消息，必须被某张卡引用。

    为什么必须有这一条：硬关卡只认日期/金额/URL/代码。v1.3 Phase 0 实测过——
    一个「只保留含硬字段的句子」的抽取器在硬关卡上 `recall = 1.0000`、只产生软告警，
    却把全部决策与条件丢光。这样一来，**正常的抽象**与**错误的信息丢失**在关卡上不可区分。

    判据的粒度落在**消息**上，与句子粒度等价：载体要求本身就是按 `source_id` 追溯的
    （措辞可以不同，抽象是允许的），所以同一个消息里的三条定性句子要么都有载体、
    要么都没有。逐句报只会把同一条失败重复三遍。

    两道边界：

    - **载体不能是 `T8/T9` 卡**（`CARRY_FORBIDDEN_TIERS`）。寒暄卡与噪音卡按定义就是要丢的，
      让它们充当载体等于把这条判据变成空头支票。
    - **只看「有没有卡引用」，不看卡里写了什么**。措辞可以不同，这是有意的：
      这一条是必要条件，不是充分条件——它拦不住「引用了却没写进去」，
      但那属于模型模糊校验（第二级）的职责，硬关卡不该假装能判语义。
    """
    cited: dict[str, list[str]] = {}
    for card in cards:
        tier = str(card.get("tier", ""))
        for source_id in card.get("source_ids", ()):
            cited.setdefault(str(source_id), []).append(tier)

    report = Report()
    uncarried: list[str] = []
    for message in messages:
        source_id = str(message.get("id", ""))
        text = str(message.get("text", ""))
        flagged = qualitative_sentences(text)
        if not flagged:
            continue
        carriers = [
            tier for tier in cited.get(source_id, ()) if tier not in CARRY_FORBIDDEN_TIERS
        ]
        if carriers:
            continue
        kinds = "、".join(dict.fromkeys(kind for kind, _ in flagged))
        detail = "；".join(sentence[:40] for _, sentence in flagged[:2])
        if cited.get(source_id):
            uncarried.append(
                f"{source_id}（{kinds}）只有 {'/'.join(sorted(set(cited[source_id])))} 卡引用，"
                f"不算载体 → {detail}"
            )
        else:
            uncarried.append(f"{source_id}（{kinds}）没有任何卡引用 → {detail}")

    if uncarried:
        report.missing["qualitative"] = uncarried
    return report
