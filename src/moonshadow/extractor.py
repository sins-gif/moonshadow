"""v1.3 Phase 1 的抽取器契约 + **规则版 stub**。

**STUB：非生产路径。** 本模块不被 `scoring.select()` 使用，也不被 `BaselineExtractor`
调用；唯一的调用方是探针 `tools/run_extractor_stub.py`。它存在的目的不是提供能力，
而是把 v1.3 C5 的接口形状**先跑起来**，让接口漏洞以实测数据的形式暴露——
接口签名应当由这些实测固化，而不是先写文档再猜。

契约来源：`docs/v1.3-spec.md` §2 C5 与 §3 Q1–Q3（输入单元 = 一批消息；
批内归并归抽取器、跨批归并归管线；`T0/T1` 硬字段进 `summary`/`facts`，**不写 `raw_quote`**）。

**规则版 stub 做了什么**（全部复用既有定义，不引入新逻辑）：

- tier：逐条消息用 `compress.classify_tier` 判定，再按 tier 分成组（一张卡一个 tier）；
  `tier_hint` 只在与判定结果冲突时被记录，不覆盖判定——见 `extract_batch` 的说明。
- 信息项归并：批内按 `verify.HARD_PATTERNS` 抽出信息项（日期/金额/URL/反引号代码），
  `normalize` 去重，每个信息项只保留**一处载体句子**（包含它的最长句子）。
- `facts` 写承载句（每句一次）；`summary` 只写一句概述——**不与 `facts` 复述同一批句子**，
  那正是基线膨胀的来源。硬字段因此全部落在 `facts`，满足 C5。
- `decisions`/`todos`/`constraints` 用 `compress.TIER_KEYWORDS` 的同一批词命中。
- `source_ids` 收录该组**全部**消息 id（可追溯性判据要求每条消息都有载体）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .compress import (
    SENTENCE_SPLIT_RE,
    TIER_IMPORTANCE,
    TIER_KEYWORDS,
    baseline_network,
    classify_tier,
)
from .pack import estimate_tokens
from .tiers import TIER_ORDER
from .verify import extract_key_fields, normalize

#: tier 关键词表在 `compress.TIER_KEYWORDS` 里是「顺序即优先级」的元组，
#: 这里按用途取用，不重新定义词表。
_CONSTRAINT_WORDS = dict(TIER_KEYWORDS)["T0"]
_DECISION_WORDS = dict(TIER_KEYWORDS)["T2"]
_TODO_WORDS = dict(TIER_KEYWORDS)["T3"]


@dataclass(frozen=True)
class Message:
    """抽取器的输入单元：一条原始消息。

    `ts` 留给模型判断时序（规则版不用）；`id` 是 `source_ids` 的唯一来源。
    """

    id: str
    text: str
    ts: str | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "Message":
        return cls(
            id=str(payload["id"]),
            text=str(payload.get("text", "")),
            ts=payload.get("created_at") or payload.get("ts"),
        )


@dataclass
class MemoryCard:
    """抽取器的输出单元：一张**尚未补齐派生字段**的卡（`cards.make_card` 负责补齐）。

    **刻意不含 `raw_quote`**：v1.3 Q3 已把它移出卡文本，逐字原文由 store 与
    `source_ids` 回链保证。契约要求 `T0/T1` 的硬字段落在 `summary` 或 `facts`。
    """

    tier: str
    summary: str
    source_ids: list[str]
    importance: int | None = None
    network: str | None = None
    facts: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    todos: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    confidence: float | None = None

    def to_draft(self) -> dict[str, Any]:
        """转成 `cards.make_card` 能吃的草稿。**这里就是「卡里没有 `raw_quote`」的落点。**"""
        return {
            "tier": self.tier,
            "importance": self.importance if self.importance is not None
            else TIER_IMPORTANCE[self.tier],
            "network": self.network or baseline_network(self.summary, self.tier),
            "summary": self.summary,
            "facts": list(self.facts),
            "decisions": list(self.decisions),
            "todos": list(self.todos),
            "constraints": list(self.constraints),
            "entities": list(self.entities),
            "keywords": list(self.keywords),
            "confidence": self.confidence,
            "source_ids": list(self.source_ids),
        }


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in SENTENCE_SPLIT_RE.split(text) if part.strip()]


def _keywords_hit(sentence: str, words: Iterable[str]) -> bool:
    return any(word in sentence for word in words)


@dataclass
class BatchDiagnostics:
    """一次 `extract_batch` 调用暴露出来的接口事实（供探针统计，不参与产出）。

    这些字段就是「stub 不得不猜接口」的地方：契约没规定时必须先记下来，
    而不是让实现悄悄替契约做决定。
    """

    messages: int = 0
    tiers: tuple[str, ...] = ()
    tier_hint_used: bool = False
    hint_mismatch: bool = False
    messages_without_items: int = 0
    items: int = 0
    items_multi_message: int = 0
    cards: int = 0


_LAST_BATCH_DIAGNOSTICS: list[BatchDiagnostics] = []


def last_batch_diagnostics() -> list[BatchDiagnostics]:
    """最近若干次 `extract_batch` 的诊断记录（探针用；无状态语义，仅追加）。"""
    return list(_LAST_BATCH_DIAGNOSTICS)


def reset_diagnostics() -> None:
    _LAST_BATCH_DIAGNOSTICS.clear()


def extract_batch(
    messages: Sequence[Message],
    tier_hint: str = "",
    *,
    sentence_fields: bool = True,
) -> list[MemoryCard]:
    """一批消息（`≤10` 条）→ 卡列表。签名即 Phase 2 的 LLM 抽取器要满足的形状。

    **`tier_hint` 的语义是契约里尚未定清的一处**：签名假设「一批对应一个 tier」，
    但一批消息天然可能跨多个 tier。规则版的处理是——**逐条判定 tier，按 tier 分组，
    一个 tier 一张卡**；`tier_hint` 只在与判定冲突时被记录（`BatchDiagnostics`），
    绝不覆盖判定。为什么不让 hint 覆盖：hint 由管线给出，而管线的 `compile_session`
    今天**根本不产出该参数**（探针只能传空串），把它当作权威等于让一个没人填的参数
    决定卡上的 tier。这条冲突的实测数据见 `tools/run_extractor_stub.py` 的接口诊段。

    信息项归并（批内）：按 `HARD_PATTERNS` 抽项、`normalize` 去重，每个信息项只留
    **一处载体句子**（包含它的最长句子），因此同一信息项在同一批里只写一次。

    ``sentence_fields=False`` 时不写 `decisions`/`todos`/`constraints`。
    这个开关是给探针做**归因**用的：把「批内归并」与「逐句复述」两种效应分开量。
    它同时暴露一处契约空白——**契约没有规定同一句话能不能既进 `facts` 又进
    `decisions`**，而这个未定项在实测里值 `1.25x` 的压缩率。
    """
    if not messages:
        return []

    tiers_of = [(message, classify_tier(message.text)) for message in messages]
    diagnostics = BatchDiagnostics(
        messages=len(messages),
        tiers=tuple(dict.fromkeys(tier for _, tier in tiers_of)),
        tier_hint_used=bool(tier_hint),
        hint_mismatch=bool(tier_hint) and tier_hint not in {t for _, t in tiers_of},
    )

    cards: list[MemoryCard] = []
    for tier in TIER_ORDER:
        group = [message for message, own in tiers_of if own == tier]
        if not group:
            continue
        cards.append(_merge_group(group, tier, diagnostics, sentence_fields=sentence_fields))

    diagnostics.cards = len(cards)
    _LAST_BATCH_DIAGNOSTICS.append(diagnostics)
    return cards


def _merge_group(
    group: Sequence[Message],
    tier: str,
    diagnostics: BatchDiagnostics,
    *,
    sentence_fields: bool = True,
) -> MemoryCard:
    """把一个 (批, tier) 组里的信息项归并成一张卡。"""
    sentences: list[tuple[str, int]] = []  # (句子, 该句 token)
    for message in group:
        for sentence in _sentences(message.text):
            sentences.append((sentence, estimate_tokens(sentence)))

    item_messages: dict[str, set[str]] = {}
    for message in group:
        for values in extract_key_fields(message.text).values():
            for value in values:
                item_messages.setdefault(normalize(value), set()).add(message.id)
    diagnostics.items += len(item_messages)
    diagnostics.items_multi_message += sum(
        1 for where in item_messages.values() if len(where) >= 2
    )
    diagnostics.messages_without_items += sum(
        1 for message in group if not extract_key_fields(message.text)
    )

    # 每个信息项只留一处载体：包含它的最长句子；同句只写一次。
    carriers: dict[str, int] = {}
    for item in item_messages:
        best: int | None = None
        for index, (sentence, tokens) in enumerate(sentences):
            if item and item in normalize(sentence):
                if best is None or tokens > sentences[best][1]:
                    best = index
        if best is not None:
            carriers[item] = best

    facts: list[str] = []
    seen_sentences: set[str] = set()
    for index in sorted(set(carriers.values())):
        sentence = sentences[index][0]
        key = normalize(sentence)
        if key not in seen_sentences:
            seen_sentences.add(key)
            facts.append(sentence)

    joined = "\n".join(message.text for message in group)
    return MemoryCard(
        tier=tier,
        # summary 只写概述：与 facts 复述同一批句子正是基线膨胀的来源。
        summary=f"{tier} 组，{len(group)} 条消息。",
        source_ids=[message.id for message in group],
        importance=TIER_IMPORTANCE[tier],
        network=baseline_network(joined, tier),
        facts=facts,
        decisions=[
            sentence
            for sentence in _sentences(joined)
            if _keywords_hit(sentence, _DECISION_WORDS)
        ]
        if sentence_fields
        else [],
        todos=[
            sentence for sentence in _sentences(joined) if _keywords_hit(sentence, _TODO_WORDS)
        ]
        if sentence_fields
        else [],
        constraints=[
            sentence
            for sentence in _sentences(joined)
            if _keywords_hit(sentence, _CONSTRAINT_WORDS)
        ]
        if sentence_fields
        else [],
    )


def canonical_item(value: str) -> str:
    """信息项的字面值 → 「同一信息项」的粗略规范化，供探针统计跨批归并的难点。

    规则版只能按**字面值**判同一性（`normalize` 去空格与千分位）。真实语料里
    「2026-09-30」与「9 月 30 日」是同一个信息项，字面却不同——跨批注册表若按字面值
    建索引，这类就会各留一处载体。本函数把数字部分与单位部分分开折算，
    用它统计这批用例里**同一信息项存在几种写法**。

    **小数点是数值的一部分，不能被抹掉**：`1.2 万`（12,000）与 `12 万`（120,000）
    是两个不同的信息项。这里用正则取数并去掉尾随零，避免把两者混为一谈。
    """
    match = re.search(r"\d+(?:\.\d+)?", value)
    if match is None:
        return f"|{normalize(value)}"
    number = match.group(0).rstrip("0").rstrip(".") if "." in match.group(0) else match.group(0)
    unit = normalize(value.replace(match.group(0), ""))
    return f"{number}|{unit}"
