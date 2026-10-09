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

import json
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
from .verify import (
    SUMMARY_MENTION_CHARS,
    SUMMARY_SHARE_MAX,
    SUMMARY_SHARE_MIN,
    extract_key_fields,
    normalize,
)

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
    #: LLM 输出里 `claims` 是**字符串**（而非 `make_card` 要的对象数组）的条数。
    #: 解析器把它们**显式降级**进 `facts` 并计数：不猜 `kind`、不猜 `predicate`。
    #: 猜错就是一条静默的错标，而「猜不准就不猜」是本仓库一贯的取舍。
    #: 这个计数**只进诊断、不进 `CaseResult`**——它是抽取器质量问题，不是契约违规，
    #: 混进 `CaseResult` 会污染 C8 的豁免/违规机制。
    unstructured_claims: int = 0


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
    summary_style: str = "digest",
    allow_duplication: bool = False,
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
        cards.append(
            _merge_group(
                group, tier, diagnostics,
                sentence_fields=sentence_fields,
                summary_style=summary_style,
                allow_duplication=allow_duplication,
            )
        )

    diagnostics.cards = len(cards)
    _LAST_BATCH_DIAGNOSTICS.append(diagnostics)
    return cards


def _merge_group(
    group: Sequence[Message],
    tier: str,
    diagnostics: BatchDiagnostics,
    *,
    sentence_fields: bool = True,
    summary_style: str = "digest",
    allow_duplication: bool = False,
) -> MemoryCard:
    """把一个 (批, tier) 组里的信息项归并成一张卡（**遵守 C6：卡内字段互斥**）。"""
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

    # 每个信息项只留一处载体：包含它的最长句子。
    carriers: set[str] = set()
    for item in item_messages:
        best: int | None = None
        for index, (sentence, tokens) in enumerate(sentences):
            if item and item in normalize(sentence):
                if best is None or tokens > sentences[best][1]:
                    best = index
        if best is not None:
            carriers.add(normalize(sentences[best][0]))

    # C6：每条句子按**语义类别**只进一个字段。一句话只能有一个载体字段，
    # 所以这里先按整组去重句子，再逐句定归属，绝不把同一句写进第二个字段。
    # 优先级（约束/否定 > 决策 > 待办 > 事实）是契约没规定、实现先猜的一处，见探针的接口诊段。
    facts: list[str] = []
    decisions: list[str] = []
    todos: list[str] = []
    constraints: list[str] = []
    seen: set[str] = set()
    for sentence, _ in sentences:
        key = normalize(sentence)
        if not key or key in seen:
            continue
        seen.add(key)
        if sentence_fields and _keywords_hit(sentence, _CONSTRAINT_WORDS):
            constraints.append(sentence)
        elif sentence_fields and _keywords_hit(sentence, _DECISION_WORDS):
            decisions.append(sentence)
        elif sentence_fields and _keywords_hit(sentence, _TODO_WORDS):
            todos.append(sentence)
        elif key in carriers:
            facts.append(sentence)
        # 其余句子（不含硬字段、也不是关键字句）不写进任何字段：那是抽象，规则版不做。

    if allow_duplication and sentence_fields:
        # **故意违反 C6** 的对照：把关键字句再往 facts 里写一遍（C6 之前的形状）。
        # 只给探针做归因用——让「C6 到底值多少压缩率」在同一个 summary 口径下可比。
        for sentence in (*constraints, *decisions, *todos):
            key = normalize(sentence)
            if key and all(normalize(item) != key for item in facts):
                facts.append(sentence)

    joined = "\n".join(message.text for message in group)
    summary = _build_summary(
        tier, facts, decisions, todos, constraints, style=summary_style
    )
    return MemoryCard(
        tier=tier,
        summary=summary,
        source_ids=[message.id for message in group],
        importance=TIER_IMPORTANCE[tier],
        network=baseline_network(joined, tier),
        facts=facts,
        decisions=decisions,
        todos=todos,
        constraints=constraints,
    )


def _build_summary(
    tier: str,
    facts: Sequence[str],
    decisions: Sequence[str],
    todos: Sequence[str],
    constraints: Sequence[str],
    *,
    style: str = "digest",
) -> str:
    """按 C8 生成 `summary`：**必须提及每条要点，且长度占字段 token 的 10%–40%**。

    规则版能做的只有「片段摘要」：从每条 `decisions`/`todos`/`constraints` 里取一段
    ≥ `SUMMARY_MENTION_CHARS` 字的开头片段拼起来。它满足 C8 的两条机械判据，
    但**不是语义摘要**——这正是漏洞 #10 的那条缝，也是 Phase 2 的 LLM 必须跨过的门槛。
    探针里保留 `style="minimal"` 等变体，用来把「summary 的成本」单独量出来。

    两条判据在小卡上会**互相冲突**（要提及 N 条要点所需的最小长度 > 字段合计的 40%）：
    那时片段按顺序尽量放，放不下的条目会被记进 C8 违规——
    这不是 stub 偷懒，是契约本身没有规定「提及」与「长度上限」谁优先。
    """
    if style == "minimal":
        return tier
    if style == "counts":
        return (
            f"{tier} 组：{len(facts)} 条事实、{len(decisions)} 条决策、"
            f"{len(todos)} 条待办、{len(constraints)} 条约束。"
        )

    entries = [normalize(value) for value in (*decisions, *todos, *constraints, *facts)]
    entries = [value for value in entries if value]
    if not entries:
        return tier
    field_tokens = sum(
        estimate_tokens(value)
        for values in (facts, decisions, todos, constraints)
        for value in values
    )
    if not field_tokens:
        return "；".join(entries)[:12]

    low = SUMMARY_SHARE_MIN * field_tokens
    high = SUMMARY_SHARE_MAX * field_tokens
    # ①「提及」优先：每条要点各取最短合法片段（`SUMMARY_MENTION_CHARS` 字）。
    length = SUMMARY_MENTION_CHARS
    fragments = [value[:length] for value in entries]
    # ② 不足 10% 就把片段一起加长，但**只在不超过 40% 上限时**加长。
    while estimate_tokens("；".join(fragments)) < low and length < 48:
        longer = [value[: length + 6] for value in entries]
        if estimate_tokens("；".join(longer)) > high:
            break
        length += 6
        fragments = longer
    # ③ **不为了压到 40% 以下而丢条目**：那会牺牲「提及」这条实质要求。
    #    小卡上两条判据互斥（提及全部要点所需长度 > 字段合计的 40%），此时保提及、让长度违约——
    #    契约缺一条优先级规定，这一条记在 `docs/v1.3-spec.md` §6.2。
    # ③ **硬要点必须出现**（C8.2 方案 B）：片段前缀未必含硬值（`预算先按18` 里就没有 `18 万元`），
    #    所以显式补上三字段里全部 HARD_PATTERNS 匹配；长度可能超 40%，由 C8.3 允许。
    text = "；".join(fragments)
    for value in (*decisions, *todos, *constraints):
        for item in (hit for hits in extract_key_fields(str(value)).values() for hit in hits):
            if normalize(item) not in normalize(text):
                text = f"{text}；{item}" if text else item
    return text


def parse_llm_cards(
    raw_text: str, diagnostics: BatchDiagnostics | None = None
) -> list[MemoryCard]:
    """LLM 分支：raw 文本 → `list[MemoryCard]`。**接口签名从真实输出里长出来的**。

    三批真实 dump 定下的四条事实（不是推测）：

    1. **整体是合法 JSON 数组**：无 markdown 代码块、无解释性前言（三批 0 例外），
       所以解析层不需要剥离外壳。哪天出现外壳，这里要加剥离——那属于接口的一部分。
    2. **`claims` 是字符串数组**，而 `cards.make_card` 要的是对象数组（按 `claim["kind"]` 取键）。
       **决策：显式降级**——str claim 原样进 `facts`，`unstructured_claims` 计数；
       不猜 `kind`/`predicate`、不构造对象。`claims` 字段本身不进 `MemoryCard`（置空）。
       理由：`kind` 是语义判断（decision/opinion/todo），规则版猜不准，猜错是**静默错标**；
       宁可降级，不可猜错。降级不触发 C6（`facts` 本就是陈述句容器），C1 追溯路径不变。
    3. **LLM 会用满 `network` 四类**（`observation` 出现 3 次），而规则版 `baseline_network`
       永不输出 `observation`。所以 `network` 分布是**抽取器相关的**，不是不变量。
    4. **`summary` 长度跨度极大**（16–233 字符），C8 的长度条款会在这里第一次被压到。

    参数 `diagnostics` 用于把降级计数写到既有的批次诊断里；不传则新建一个。
    """
    container = diagnostics if diagnostics is not None else BatchDiagnostics()
    payload = json.loads(raw_text)
    if not isinstance(payload, list):
        raise ValueError(f"LLM 输出顶层不是数组，而是 {type(payload).__name__}")

    cards: list[MemoryCard] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError(f"卡元素不是对象，而是 {type(item).__name__}")
        facts = [str(value) for value in item.get("facts") or ()]
        for claim in item.get("claims") or ():
            if isinstance(claim, str):
                # 显式降级：原样进 facts，计数，**不猜结构**
                if claim not in facts:
                    facts.append(claim)
                container.unstructured_claims += 1
        cards.append(
            MemoryCard(
                tier=str(item["tier"]),
                summary=str(item.get("summary") or ""),
                source_ids=[str(value) for value in item.get("source_ids") or ()],
                importance=int(item["importance"]) if item.get("importance") is not None else None,
                network=str(item["network"]) if item.get("network") else None,
                facts=facts,
                decisions=[str(value) for value in item.get("decisions") or ()],
                todos=[str(value) for value in item.get("todos") or ()],
                constraints=[str(value) for value in item.get("constraints") or ()],
                entities=[str(value) for value in item.get("entities") or ()],
                keywords=[str(value) for value in item.get("keywords") or ()],
                confidence=(
                    float(item["confidence"]) if item.get("confidence") is not None else None
                ),
            )
        )
    container.cards = len(cards)
    if diagnostics is None:
        _LAST_BATCH_DIAGNOSTICS.append(container)
    return cards


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
