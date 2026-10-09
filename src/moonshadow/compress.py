"""Phase 2：Retain 抽取管道。

设计约束（都来自 SPEC-v1.1）：
- **抽取器可替换**：``Extractor`` 只负责把消息变成「卡草稿」，派生字段（id / dedup_key /
  half_life_days / 校验）一律由 ``cards.make_card`` 补齐，避免两套契约。
- **快路径优先**：能由规则确定是 T8 寒暄 / T9 噪音的，绝不进模型（§8.1）。
- **幂等**：卡按内容寻址，重跑是重放；进度靠 ``cursor`` 续跑（§8.1）。
- **禁止编造**：基线条不做实体猜测——猜错的实体比不猜更糟（§3.2）。

本模块不含任何网络调用：模型抽取器由调用方注入，测试用假抽取器即可全量覆盖。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol, Sequence, runtime_checkable

from .cards import make_card
from .clock import Clock, SystemClock, to_iso, to_utc
from .pack import estimate_tokens
from .store import Store
from .verify import extract_key_fields

#: 提示词模板。对应 v1.0 §7A，并已修正其中两处会误导模型的地方：
#: 1) 明确「时间规则只影响保留多少细节，不影响 tier 判定」——否则模型会把「时间久」当成降级依据；
#: 2) 时间戳与 tier 规则分列，避免两套规则互相侵蚀。
PROMPT_TEMPLATE = """你是「时间加权记忆压缩器」。输入是带时间戳的对话/文档，输出 JSON 数组，每个元素是一张记忆卡草稿。

必须原样保留：日期、数字、金额、人名、地名、产品名、URL、代码、否定词、条件、因果、承诺、偏好、决策、待办、截止时间。

时间规则（只影响保留多少细节，**不影响 tier 判定**）：
1. 最近 24 小时：保留细节和关键原话。
2. 最近 7 天：保留事件、结论、待办、变更。
3. 更早：只保留长期结论、偏好、关系、项目状态、未完成事项。

tier 规则（由重要性决定，不因时间久而降级）：
T0 身份/硬约束/安全/法律；T1 长期偏好/核心目标/关系；T2 项目决策/重要承诺；
T3 近期待办/截止；T4 一般方案/讨论结论；T5 日常事实；T6 短期细节；
T7 临时草稿；T8 寒暄确认；T9 噪音。

实体规则：必须消解所有代词为具体实体（「他」→「张三」），不得残留代词。
**entities / keywords 是可选字段**：只有当实体或关键词**对召回有区分度**时才写
（例如专有名词、单号、模块名）；日常词汇、泛化描述、卡内已出现的重复词一律**不写**，
没有就写空数组 `[]`。这两栏会被计入卡 token，滥写直接抬高压缩成本。
冲突规则：冲突不合并；输出新卡，并把被取代的旧卡 id 放进 supersedes。
不确定的字段写 null 或 "不确定"，禁止编造。
每张卡必须带 source_ids，且只能来自输入消息的 id。

**summary 长度规则：每张卡的 summary 不超过 60 个字。** 它是概述，不是第二份字段：
不得逐字复述 facts/decisions/todos/constraints 里的句子，必须逐个覆盖这些字段里的
日期、金额、URL、反引号代码（硬要点），其余内容可以概括。

**claims 必须是对象数组**，每个元素形如
`{{"kind": "decision|opinion|todo", "predicate": "简短谓词", "object": "要点"}}`；
**不得输出字符串数组**（字符串 claim 会被降级进 facts 并计入质量诊断）。示例：
`"claims": [{{"kind": "decision", "predicate": "采用", "object": "限流加扩容双管齐下"}}]`

输出字段：tier, importance(1-10), network(world|experience|opinion|observation),
summary(≤60 字), entities, keywords, facts, decisions, todos, constraints, claims(对象数组),
confidence, source_ids。**不要输出 raw_quote。** 只输出 JSON 数组，不要任何解释文字。

输入消息：
{messages}
"""


def build_prompt(messages: Sequence[Mapping[str, Any]]) -> str:
    """把消息渲染进提示词模板。只喂 id / 时间 / 正文，减少模型被无关字段干扰。"""
    lines = []
    for message in messages:
        text = str(message.get("text", "")).replace("\n", "\\n")
        lines.append(f'- id={message.get("id")} ts={message.get("created_at") or message.get("ts")}: {text}')
    return PROMPT_TEMPLATE.format(messages="\n".join(lines))


def prompt_hash(prompt: str) -> str:
    """提示词指纹，写入 extraction_run，便于日后判断「结果变了是模型变了还是提示词变了」。"""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- 快路径

NOISE_RE = re.compile(r"^[\s。，,.!?！？~～…、*\-_=+()（）\[\]]*$")
GREETINGS = {
    "好的", "好", "收到", "谢谢", "多谢", "感谢", "嗯", "嗯嗯", "可以", "行", "了解", "明白",
    "辛苦了", "在吗", "你好", "您好", "hi", "hello", "ok", "okay", "thanks", "拜拜", "再见", "稍等",
}
_PUNCT_RE = re.compile(r"[\s。，,.!?！？~～…、*\-_=+()（）\[\]]")
DIGIT_RE = re.compile(r"\d")


def _is_pure_greeting(stripped: str) -> bool:
    """判断整串是否由问候词拼成（「嗯，好的，谢谢！」这类复合寒暄很常见）。"""
    rest = stripped
    for word in sorted(GREETINGS, key=len, reverse=True):
        if rest.startswith(word):
            rest = rest[len(word):]
        rest = rest.replace(word, "")
        if not rest:
            return True
    return not rest


def classify_fast_path(text: str) -> str | None:
    """能由规则确定的层级直接返回，否则 None（交给抽取器）。

    只覆盖 T8/T9：这两类的判定不需要语义理解，进模型纯属浪费。
    """
    if NOISE_RE.match(text):
        return "T9"
    stripped = _PUNCT_RE.sub("", text).lower()
    if stripped and not DIGIT_RE.search(text) and _is_pure_greeting(stripped):
        return "T8"
    return None


# --------------------------------------------------------------------------- 契约


@runtime_checkable
class Extractor(Protocol):
    """抽取器契约：消息 → 卡草稿（缺 id/dedup_key 等派生字段，由 make_card 补齐）。"""

    def __call__(self, messages: Sequence[Mapping[str, Any]]) -> Iterable[Mapping[str, Any]]:
        ...


@dataclass
class CompileReport:
    """一次压缩运行的结果。"""

    session_id: str
    processed: int = 0
    inserted: int = 0
    replayed: int = 0
    superseded: list[str] = field(default_factory=list)
    fast_path: int = 0
    extracted_batches: int = 0
    cursor: tuple[str, int] | None = None
    run_id: str | None = None

    def summary(self) -> str:
        return (
            f"处理 {self.processed} 条消息 → 新增 {self.inserted} 张卡"
            f"（重放 {self.replayed}，取代 {len(self.superseded)}，"
            f"规则快路径 {self.fast_path} 条，模型批次 {self.extracted_batches}）"
        )


# --------------------------------------------------------------------------- 基线抽取器


TIER_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    # 顺序即优先级：硬约束优先于决策，决策优先于待办
    ("T0", ("必须", "不得", "不要", "禁止", "避免", "硬性", "合规", "法律", "安全", "不可")),
    ("T1", ("长期", "以后都", "以后一直", "我一直", "习惯", "偏好", "原则", "总是", "永久")),
    ("T2", ("决定", "采用", "确定", "拍板", "选定", "结论", "同意", "敲定")),
    ("T3", ("待办", "负责", "需要", "截止", "交付", "之前", "安排", "跟进", "报价")),
)

TIER_IMPORTANCE = {
    "T0": 10, "T1": 9, "T2": 8, "T3": 6, "T4": 5,
    "T5": 4, "T6": 3, "T7": 2, "T8": 1, "T9": 1,
}

OPINION_RE = re.compile(r"(觉得|认为|看法|倾向于|不喜欢|更喜欢)")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?\n])")

#: 四张网络（v1.0 借自 y8y 的 `network` 字段）。
#: v1.2 提案想用它门控时间下限 `f`，因此它的赋值质量必须先被测出来——见 `network_eval.py`。
NETWORKS: tuple[str, ...] = ("world", "experience", "opinion", "observation")


def baseline_network(text: str, tier: str) -> str:
    """规则版网络赋值（由 `BaselineExtractor` 提取而来，行为不变）。

    现状记录：这条规则**永远不会输出 `observation`**——四张网络里有一张不可达。
    v1.2 提案给 `observation` 配的正是最激进的遗忘参数（`f` 0.20–0.45），
    因此「该类别在基线上不可达」是那条路线的前置问题。
    """
    if OPINION_RE.search(text):
        return "opinion"
    if tier == "T0":
        return "world"
    return "experience"

#: 时间词也算事实性内容：「会议安排在明天下午」与「正在梳理推导」不是一类。
TEMPORAL_WORDS = (
    "今天", "明天", "后天", "昨天", "本周", "下周", "上周", "本月", "下月",
    "月底", "月初", "本季度", "今年", "明年", "上午", "下午", "晚上",
)


def classify_tier(text: str) -> str:
    """规则版层级判定：先过快路径，再按关键词优先级。"""
    fast = classify_fast_path(text)
    if fast:
        return fast
    for tier, keywords in TIER_KEYWORDS:
        if any(keyword in text for keyword in keywords):
            return tier
    if OPINION_RE.search(text):
        return "T5"
    # 有数字或时间词 => 日常事实（T5）；纯叙述/无信息量 => 短期上下文（T6）
    factual = bool(DIGIT_RE.search(text)) or any(word in text for word in TEMPORAL_WORDS)
    return "T5" if factual else "T6"


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in SENTENCE_SPLIT_RE.split(text) if part.strip()]


def build_summary(text: str, limit: int = 120) -> str:
    """压缩摘要，但**不允许把关键字段截掉**：截断后若有丢失，追加回关键字段。"""
    head = text[:limit]
    lost: list[str] = []
    for values in extract_key_fields(text).values():
        lost.extend(value for value in values if value not in head)
    if lost:
        head = f"{head} …关键字段：{'、'.join(dict.fromkeys(lost))}"
    return head


class BaselineExtractor:
    """纯规则基线抽取器：不调模型，但保证关键字段不丢。

    它的价值是**评测基线**：关键字段保留率应当接近 1.0，代价是压缩率很差。
    后续模型抽取器必须在这个基线上把压缩率做上去，同时保持保留率。
    不做实体消解（宁可不猜，也不编造），实体消解属于 Phase 4。
    """

    name = "baseline-rules"

    def __call__(self, messages: Sequence[Mapping[str, Any]]) -> Iterable[Mapping[str, Any]]:
        for message in messages:
            yield self._card_draft(str(message.get("text", "")), [str(message["id"])])

    def _card_draft(self, text: str, source_ids: list[str]) -> dict[str, Any]:
        tier = classify_tier(text)
        sentences = split_sentences(text)
        key_values = [value for values in extract_key_fields(text).values() for value in values]

        facts = list(dict.fromkeys(key_values + [s for s in sentences if DIGIT_RE.search(s)]))
        decisions = [s for s in sentences if any(k in s for k in TIER_KEYWORDS[2][1])]
        todos = [s for s in sentences if any(k in s for k in TIER_KEYWORDS[3][1])]
        constraints = [s for s in sentences if any(k in s for k in TIER_KEYWORDS[0][1])]

        claims: list[dict[str, Any]] = []
        for sentence in decisions:
            claims.append({"kind": "decision", "predicate": "decision", "object": sentence})
        for sentence in todos:
            due = next(iter(extract_key_fields(sentence).get("date", [])), None)
            claims.append({"kind": "todo", "predicate": "todo", "object": sentence, "due_at": due})

        return {
            "tier": tier,
            "importance": TIER_IMPORTANCE[tier],
            "network": baseline_network(text, tier),
            "summary": build_summary(text),
            "raw_quote": text if tier in ("T0", "T1") else None,
            "facts": facts,
            "decisions": decisions,
            "todos": todos,
            "constraints": constraints,
            "claims": claims,
            "confidence": 1.0 if tier not in ("T5", "T6", "T7") else 0.6,
            "source_ids": source_ids,
            "entities": [],
            "keywords": [],
        }


# --------------------------------------------------------------------------- 管道


def _batches(
    messages: Sequence[Mapping[str, Any]],
    *,
    max_messages: int,
    max_tokens: int,
) -> list[list[Mapping[str, Any]]]:
    """按消息数与累计 token 数切批。token 用 pack.estimate_tokens，口径与装箱一致。"""
    batches: list[list[Mapping[str, Any]]] = []
    current: list[Mapping[str, Any]] = []
    used = 0
    for message in messages:
        cost = estimate_tokens(str(message.get("text", "")))
        if current and (len(current) >= max_messages or used + cost > max_tokens):
            batches.append(current)
            current, used = [], 0
        current.append(message)
        used += cost
    if current:
        batches.append(current)
    return batches


def compile_session(
    store: Store,
    session_id: str,
    extractor: Extractor,
    *,
    clock: Clock | None = None,
    max_messages: int = 10,
    max_tokens: int = 4096,
    model: str | None = None,
) -> CompileReport:
    """把会话里尚未压缩的消息压成记忆卡，并推进游标。

    幂等保证：卡 ID 与 dedup_key 都由内容派生，因此**重复调用同一段输入**
    只会得到 ``replayed`` 计数增加，不产生新卡、不改变状态。
    """
    moment = to_utc((clock or store.clock or SystemClock()).now())
    report = CompileReport(session_id=session_id)
    pending = store.pending_messages(session_id)
    if not pending:
        return report

    for batch in _batches(pending, max_messages=max_messages, max_tokens=max_tokens):
        leftovers: list[Mapping[str, Any]] = []
        prompt = ""
        for message in batch:
            text = store.get_raw(str(message["id"]))
            tier = classify_fast_path(text)
            if tier is None:
                leftovers.append({**message, "text": text})
                continue
            draft = {
                "tier": tier,
                "importance": TIER_IMPORTANCE[tier],
                "network": "observation",
                "summary": build_summary(text),
                "raw_quote": text,
                "confidence": 1.0,
                "source_ids": [str(message["id"])],
            }
            _write(store, session_id, draft, moment, report)
            report.fast_path += 1

        if leftovers:
            prompt = build_prompt(leftovers)
            for draft in extractor(leftovers):
                _write(store, session_id, dict(draft), moment, report)
            report.extracted_batches += 1

        last = batch[-1]
        store.set_cursor(
            session_id,
            last_source_id=str(last["id"]),
            last_day=str(last["day"]),
            last_line=int(last["line_no"]),
        )
        report.processed += len(batch)
        report.cursor = (str(last["id"]), int(last["line_no"]))
        report.run_id = store.record_run(
            session_id=session_id,
            from_source=str(batch[0]["id"]),
            to_source=str(last["id"]),
            model=model or getattr(extractor, "name", type(extractor).__name__),
            prompt_hash=prompt_hash(prompt) if prompt else None,
            status="ok",
        )
    return report


def _write(
    store: Store,
    session_id: str,
    draft: Mapping[str, Any],
    moment: Any,
    report: CompileReport,
) -> None:
    """把卡草稿补齐成合法卡并写入。草稿缺字段是正常的，派生字段一律由 make_card 负责。"""
    payload = dict(draft)
    payload.setdefault("session_id", session_id)
    card = make_card(clock=store.clock, observed_at=to_iso(moment), **payload)  # type: ignore[arg-type]
    result = store.put_card(card)
    if result.inserted:
        report.inserted += 1
        report.superseded.extend(result.superseded)
    else:
        report.replayed += 1
