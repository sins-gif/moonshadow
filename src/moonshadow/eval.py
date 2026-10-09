"""Phase 2.5：评测关卡 + 压缩率的分层读数。

没有这一层，抽取器改好改坏只能靠感觉——而本系统的全部价值押在「精度不丢」上。
因此把它做成硬关卡：``python tools/run_eval.py`` 不达标就退出码非 0。

指标是**一对**而非一个：

- **关键字段保留率**（必须 ≥ 阈值）：日期、金额、URL、代码不能丢；
- **压缩率**（越大越好）：原文 token / 卡 token。

只盯其中一个都会走偏：基线抽取器保留率 1.0 却没有压缩价值；激进摘要压缩率很高却会丢字段。
两个数一起看，才知道一个抽取器相对基线是「更好」还是「只是更短」。

**压缩率必须同时报两个口径**，否则会被读错：

- ``EvalReport.compression_ratio``：**逐用例算术平均**（历史口径，3 条短用例时够用）；
- ``EvalReport.weighted_compression``：**token 加权**（全部原文 token / 全部卡 token）。

用例长短差异一大，前者就被短用例主导：一条 2000 token 的会话与一条 40 token 的会话
在均值里权重相同。只说「平均压缩率」而不说口径，等于没说。

**分层读数**（``EvalReport.tier_table``）：按卡上的 ``tier`` 归集卡数、原文 token、卡 token，
于是「膨胀出在哪一层」是可指的，而不是一个总数。原文 token 走 ``source_ids`` 归属——
那是「这张卡从哪来」的权威记录，拿卡里的文字去和原文做匹配是猜。
未产出卡的层级显示为「—」而不是 ``0.00x``：前者是「不可观测」，后者会被读成「压缩得很差」。

**适用域（v1.3 Phase 0）**：压缩率是输入规模的函数。原文本身只有「一句话 + 一个日期」时，
任何正确系统都做不出 > 1x——没有可抽象的冗余，而精度关卡又要求逐字保留日期与金额。
旧金标准集 3 条、每条 28–45 token 正落在这个退化区间，因此本版把金标准集扩成 **25 条**
（其中 22 条长会话，原文 `527–1,927` token，合计 `25,630` token，含跨消息冗余与多类关键字段）。

同一份金标准集上的两条参照线（**一次性探针，仓库里没有对应命令**；逐用例平均 / token 加权）：

| 参照线 | 逐用例平均 | token 加权 | 卡数 |
|---|---|---|---|
| 基线 | `0.498x` | `0.487x` | `474` |
| 只留 `summary` | `1.022x` | `1.026x` | `474` |

两条参照线卡数相同（一条消息一张卡）——**卡数不变、被去掉的只是「逐句复述字段」**，
差额 `0.54` 就是那一部分。所以**只把每张卡写短也不够**：要真正超过 `1`
必须减少卡数（把跨消息冗余归并成更少的卡）。

（v1.2 时这里还有第三条「去掉 `raw_quote`」的参照线。v1.3 决定 Q3 之后它**不再是变量**：
`raw_quote` 已移出 `CARD_TEXT_FIELDS`，把它置空与基线**实测完全同值**
（`0.4976x` / `0.4868x`）——这正说明口径变更生效了。）

**一处口径相关的旧数**：文档里曾写「只留 summary 恰好回到 `1.0000x`」，也曾写
「本流水线上同一变体读 `0.9123x`」。两个数各自在一套口径下成立：`1.0000x` 是**不计
`raw_quote`** 时的读数（Q3 之后本流水线在旧 3 条上实测正是 `1.0000x`），`0.9123x` 是
**计 `raw_quote`** 时的读数（快路径给 `T8/T9` 卡补了它）。引用时必须带上口径。

**基线为什么膨胀**：`facts` 把关键字段复述一遍，`decisions`/`todos`/`constraints`
再把整句抄第二遍，而基线是「一条消息一张卡」，同一批字段按消息数重复 N 次。
（`T0/T1` 卡上的 `raw_quote` 已不计入卡文本——见上。这是**符合规范的基线代价**，不是 bug：
它标出了「不丢字段」的召回上限，模型抽取器的任务是在保持该保留率的前提下把压缩率做上去。）
"""

from __future__ import annotations

import json
import pathlib
import shutil
import sys
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .clock import FixedClock, parse_ts
from .compress import (
    SENTENCE_SPLIT_RE,
    BaselineExtractor,
    Extractor,
    classify_tier,
    compile_session,
)
from .pack import estimate_tokens
from .store import Store
from .tiers import TIER_ORDER
from .verify import (
    card_text_many,
    extract_key_fields,
    normalize,
    verify_field_mutual_exclusion,
    verify_no_silent_loss,
    verify_qualitative_retention,
    verify_summary_substance,
)

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_GOLD_DIR = ROOT / "eval" / "gold"
WORK_DIR = ROOT / ".tmp" / "eval"

#: 阈值来自 SPEC-v1.1 §9.2。未达标即视为关卡未通过。
DEFAULT_MIN_RECALL = 0.98

#: **C6.1 基线豁免**（`docs/v1.3-spec.md` §2 C6.1）：被指定为参照点的抽取器，
#: 其 C6 违规记入 `CaseResult.expected_violations`，**不作为失败条件**。
#: 理由：基线的存在意义就是展示「不遵守契约的形状是什么样的」，它必须保持原地不动；
#: 修它会让 v1.2 与 v1.3 失去可比性。**豁免只对这些名字生效**——任何其他抽取器
#: （stub、Phase 2 的 LLM）的 C6 违规仍然让 `run_eval` 退出 `1`。
#: 按名字而不是按调用点识别：这样「谁被豁免」写在数据里，不靠调用方自觉。
BASELINE_IDENTIFIERS: tuple[str, ...] = ("baseline-rules",)


def is_baseline_extractor(extractor: object) -> bool:
    """该抽取器是否属于被指定为参照点、因而享受 C6.1 豁免的那一类。"""
    name = getattr(extractor, "name", type(extractor).__name__)
    return str(name) in BASELINE_IDENTIFIERS


@dataclass
class GoldCase:
    """一条金标准：一段输入 + 必须活下来的东西。"""

    id: str
    messages: list[str]
    day: str = "2026-10-01"
    expected_key_fields: list[str] = field(default_factory=list)
    expected_tiers: list[str] = field(default_factory=list)
    note: str = ""

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "GoldCase":
        expected = payload.get("expected", {})
        return cls(
            id=str(payload["id"]),
            messages=[str(item) for item in payload["messages"]],
            day=str(payload.get("day", "2026-10-01")),
            expected_key_fields=[str(item) for item in expected.get("key_fields", [])],
            expected_tiers=[str(item) for item in expected.get("tiers_present", [])],
            note=str(payload.get("note", "")),
        )


def load_gold(directory: str | pathlib.Path | None = None) -> list[GoldCase]:
    """加载金标准集，按 id 排序保证报告可复现。"""
    target = pathlib.Path(directory) if directory is not None else DEFAULT_GOLD_DIR
    cases = []
    for path in sorted(target.glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            cases.append(GoldCase.from_json(json.load(handle)))
    return sorted(cases, key=lambda case: case.id)


@dataclass
class TierStat:
    """某一层级上的 token 读数（分层统计的原子单位）。

    口径（与 `run_case` 里算压缩率的口径完全一致，不另立一套）：

    - ``card_tokens`` = 该层级每张卡 ``card_text()`` 的 ``estimate_tokens`` 之和；
    - ``raw_tokens`` = 该层级卡片所引用的 ``source_ids`` 的原文 token 之和，
      同一段原文在同一层级只计一次；
    - ``shared_sources`` = 被**两个及以上层级**同时引用的原文条数。这部分原文
      在每个层级都会被各计一次（因为「这段原文支撑了 T0 卡与 T2 卡」两件事都真），
      所以**各层级 raw_tokens 之和 ≥ 全部原文 token**，差额就是它。报告里必须写明，
      否则把分层读数加起来会对不上总数。
    """

    tier: str
    cards: int = 0
    raw_tokens: int = 0
    card_tokens: int = 0
    sources: int = 0
    shared_sources: int = 0

    @property
    def compression(self) -> float:
        """该层级的压缩率；没有卡时为 0（调用方应显示为「—」而不是 0.00x）。"""
        return self.raw_tokens / self.card_tokens if self.card_tokens else 0.0


@dataclass
class CaseResult:
    case_id: str
    ok: bool
    recall: float
    total_fields: int
    lost: dict[str, list[str]] = field(default_factory=dict)
    missing_expected: list[str] = field(default_factory=list)
    missing_tiers: list[str] = field(default_factory=list)
    compression: float = 0.0
    cards: int = 0
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    raw_tokens: int = 0
    card_tokens: int = 0
    tier_stats: dict[str, TierStat] = field(default_factory=dict)
    shared_sources: int = 0
    #: 定性内容（决策/硬约束/否定）没有载体的条目。见 `verify.verify_qualitative_retention`。
    uncarried: list[str] = field(default_factory=list)
    #: 违反 C6（卡内字段互斥）的条目。见 `verify.verify_field_mutual_exclusion`。
    field_violations: list[str] = field(default_factory=list)
    #: 违反 C8（`summary` 必须承载内容）的条目。见 `verify.verify_summary_substance`。
    summary_violations: list[str] = field(default_factory=list)
    #: C6.1 豁免：被指定为参照点的抽取器（基线）的卡契约违规（C6 + C8）记在这里，不计入失败。
    expected_violations: list[str] = field(default_factory=list)

    def summary(self) -> str:
        verdict = "PASS" if self.ok else "FAIL"

        def brief(items: list[str], limit: int = 2) -> str:
            """违规清单可能上百条；打印前几条 + 总数，否则一次冒烟会刷掉几百行。"""
            if len(items) <= limit:
                return str(items)
            return f"{items[:limit]}…（共 {len(items)} 条）"

        head = (
            f"{verdict}  {self.case_id:<28} 保留率={self.recall:.3f} "
            f"压缩率={self.compression:.3f}x 卡数={self.cards} "
            f"原文={self.raw_tokens:,}tok 卡={self.card_tokens:,}tok"
        )
        details = []
        if self.lost:
            details.append(f"丢字段={brief(list(self.lost.values()))}")
        if self.field_violations:
            details.append(f"C6 字段互斥违规={brief(self.field_violations)}")
        if self.summary_violations:
            details.append(f"C8 summary 承载违规={brief(self.summary_violations)}")
        if self.expected_violations:
            details.append(
                f"卡契约违规（预期，C6.1 基线豁免 {len(self.expected_violations)} 条）="
                f"{brief(self.expected_violations, 1)}"
            )
        if self.uncarried:
            details.append(f"定性内容无载体={brief(self.uncarried)}")
        if self.missing_expected:
            details.append(f"缺预期字段={self.missing_expected}")
        if self.missing_tiers:
            details.append(f"缺预期层级={self.missing_tiers}")
        if self.error:
            details.append(f"异常={self.error}")
        return head + (("\n      " + "；".join(details)) if details else "")


@dataclass
class EvalReport:
    results: list[CaseResult]
    min_recall: float = DEFAULT_MIN_RECALL

    @property
    def key_field_recall(self) -> float:
        total = sum(r.total_fields for r in self.results)
        if total == 0:
            return 1.0
        lost = sum(sum(len(v) for v in r.lost.values()) for r in self.results)
        return (total - lost) / total

    @property
    def compression_ratio(self) -> float:
        """逐用例压缩率的**算术平均**。

        这是历史口径（3 条短用例时它够用）。用例长短差异大以后它会失真：
        一条 2000 token 的会话与一条 40 token 的会话在均值里权重相同。
        因此凡是要说「整体压缩得怎么样」，必须同时读 ``weighted_compression``。
        """
        ratios = [r.compression for r in self.results if r.compression > 0]
        return sum(ratios) / len(ratios) if ratios else 0.0

    @property
    def weighted_compression(self) -> float:
        """token 加权总体压缩率 = 全部原文 token / 全部卡 token。

        与逐用例均值并列报告：两者含义不同，缺一个就会被读错。
        """
        raw = sum(r.raw_tokens for r in self.results)
        cards = sum(r.card_tokens for r in self.results)
        return raw / cards if cards else 0.0

    @property
    def expected_violations(self) -> int:
        """C6.1 豁免掉的违规条数（参照点抽取器）。**打印出来，不静默。**"""
        return sum(len(r.expected_violations) for r in self.results)

    @property
    def field_violations(self) -> int:
        """真正导致失败的 C6 违规条数。"""
        return sum(len(r.field_violations) for r in self.results)

    @property
    def summary_violations(self) -> int:
        """真正导致失败的 C8 违规条数。"""
        return sum(len(r.summary_violations) for r in self.results)

    def tier_stats(self) -> list[TierStat]:
        """把逐用例的层级读数合并成 T0–T9 全表（未出现的层级保留 0 值行）。"""
        merged: dict[str, TierStat] = {tier: TierStat(tier=tier) for tier in TIER_ORDER}
        for result in self.results:
            for tier, stat in result.tier_stats.items():
                target = merged.setdefault(tier, TierStat(tier=tier))
                target.cards += stat.cards
                target.raw_tokens += stat.raw_tokens
                target.card_tokens += stat.card_tokens
                target.sources += stat.sources
                target.shared_sources += stat.shared_sources
        return [merged[tier] for tier in TIER_ORDER]

    def tier_table(self) -> list[str]:
        """分层读数表。**未产出卡的层级必须显示为「—」**，不能显示 0.00x——
        0.00x 读起来像「压缩得很差」，而事实是「这一层根本没有卡，度量不可观测」。"""
        stats = self.tier_stats()
        total = sum(stat.card_tokens for stat in stats)
        # 共享原文按**用例**去重后求和：同一段原文若被 3 个层级引用，
        # 逐层级的 shared_sources 会各记 1，加起来是 3 而事实是 1 段。
        shared = sum(result.shared_sources for result in self.results)
        lines = [
            "分层读数（按卡上的 tier 归集；原文 token 按 source_ids 归属）",
            f"  {'tier':<5}{'卡数':>6}{'原文token':>11}{'卡token':>11}{'压缩率':>9}{'卡token占比':>11}",
        ]
        for stat in stats:
            ratio = f"{stat.compression:.3f}x" if stat.card_tokens else "—"
            share = stat.card_tokens / total if total else 0.0
            lines.append(
                f"  {stat.tier:<5}{stat.cards:>6}{stat.raw_tokens:>11,}"
                f"{stat.card_tokens:>11,}{ratio:>9}{share:>10.1%}"
            )
        empty = [stat.tier for stat in stats if stat.cards == 0]
        if empty:
            lines.append(
                f"  · 未产出任何卡的层级：{'、'.join(empty)}"
                "（该层级在本轮不可观测，不等于「没有内容」）"
            )
        if shared:
            lines.append(
                f"  · 被多个层级共同引用的原文 {shared} 段：它在各层级各计一次，"
                "故分层原文 token 之和大于全部原文 token"
            )
        return lines

    @property
    def ok(self) -> bool:
        return all(r.ok for r in self.results) and self.key_field_recall >= self.min_recall

    def summary(self) -> str:
        lines = [result.summary() for result in self.results]
        verdict = "PASS" if self.ok else "FAIL"
        lines.append(
            f"{verdict}  合计：关键字段保留率={self.key_field_recall:.4f}"
            f"（阈值 {self.min_recall}），"
            f"平均压缩率={self.compression_ratio:.3f}x（{len(self.results)} 条用例的算术平均），"
            f"总体压缩率={self.weighted_compression:.3f}x（原文 token / 卡 token，按长度加权），"
            f"用例 {sum(1 for r in self.results if r.ok)}/{len(self.results)} 通过"
        )
        lines.extend(self.tier_table())
        if self.expected_violations:
            lines.append(
                f"C6.1 基线豁免：卡契约违规 {self.expected_violations} 条（**预期**）——"
                "参照点抽取器按定义不遵守 v1.3 的卡契约（C6 字段互斥 + C8 summary 承载），"
                "它保持原地不动；豁免只对它生效，见 `eval.BASELINE_IDENTIFIERS` 与 spec C6.1。"
            )
        if self.field_violations:
            lines.append(
                f"FAIL  C6（卡内字段互斥）违规 {self.field_violations} 条："
                "同一句话只能有一个载体字段，`summary` 不得逐字复述（spec C6）。"
            )
        if self.summary_violations:
            lines.append(
                f"FAIL  C8（summary 必须承载内容）违规 {self.summary_violations} 条："
                "`summary` 必须提及每条决策/待办/约束，且长度占字段 token 的 10%–40%（spec C8）。"
            )
        if self.compression_ratio < 1.0 or self.weighted_compression < 1.0:
            lines.append(
                "WARN  压缩率 < 1.0：抽取器在**膨胀** token。PASS 只说明没丢字段，"
                "不说明压缩有效——模型抽取器必须在保持保留率的同时把这一项做到 > 1。"
            )
        return "\n".join(lines)


def _tier_stats(
    cards: Sequence[Mapping[str, Any]], store: Store
) -> tuple[dict[str, TierStat], int]:
    """按卡上的 tier 归集 token 读数。

    原文 token 走 ``source_ids`` 而不是把卡里的文字拿去和原文做匹配：
    只有 source_ids 是「这张卡从哪来」的权威记录，匹配是猜。

    返回 ``(分层读数, 被多个层级共同引用的原文段数)``。
    """
    stats: dict[str, TierStat] = {}
    texts: dict[str, str] = {}
    tiers_of_source: dict[str, set[str]] = {}

    for card in cards:
        tier = str(card["tier"])
        stat = stats.setdefault(tier, TierStat(tier=tier))
        stat.cards += 1
        stat.card_tokens += estimate_tokens(card_text_many([card]))
        for source_id in card.get("source_ids", ()):
            sid = str(source_id)
            if sid not in texts:
                texts[sid] = store.get_raw(sid)
            tiers_of_source.setdefault(sid, set()).add(tier)

    for source_id, source_tiers in tiers_of_source.items():
        cost = estimate_tokens(texts[source_id])
        for tier in source_tiers:
            stat = stats.setdefault(tier, TierStat(tier=tier))
            stat.raw_tokens += cost
            stat.sources += 1
            if len(source_tiers) > 1:
                stat.shared_sources += 1

    shared = sum(1 for source_tiers in tiers_of_source.values() if len(source_tiers) > 1)
    return stats, shared


def run_case(
    case: GoldCase,
    extractor: Extractor,
    *,
    session_id: str | None = None,
    workdir: str | pathlib.Path | None = None,
    is_baseline: bool | None = None,
) -> CaseResult:
    """跑一条金标准：写入原文 → 压缩 → 校验关键字段与预期层级 → 算压缩率。

    ``is_baseline=None`` 时按抽取器的 `name` 自动判定（见 `BASELINE_IDENTIFIERS`）：
    基线享受 C6.1 豁免，其 C6 违规进 `expected_violations` 而不是 `field_violations`。
    显式传 `is_baseline` 可以覆盖——测试用它钉住「豁免只对基线生效」。
    """
    session = session_id or f"gold-{case.id}"
    work = pathlib.Path(workdir) if workdir is not None else WORK_DIR / case.id
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)

    clock = FixedClock(parse_ts(f"{case.day}T09:00:00+00:00"))
    store = Store(work, clock=clock)
    try:
        for text in case.messages:
            store.append_message(session, text)
        compile_session(store, session, extractor, clock=clock)

        cards = store.cards(session_id=session)
        raw_text = "\n".join(case.messages)
        # 回链侧：`source_id → 原文`。v1.3 决定 Q3 之后，`raw_quote` 不再进卡文本，
        # 硬字段在卡内查不到时按这些原文块兜底（见 `verify.verify_no_silent_loss`）。
        raw_messages = [
            {"id": row["id"], "text": store.get_raw(str(row["id"]))}
            for row in store.messages_for_session(session)
        ]
        sources = {str(row["id"]): str(row["text"]) for row in raw_messages}
        report = verify_no_silent_loss(raw_text, cards, sources=sources)
        # 定性内容判据：硬字段保住 ≠ 内容保住（决策/约束/否定一个硬字段都没有）。
        qualitative = verify_qualitative_retention(raw_messages, cards)
        # C6：卡内字段互斥（同一句话只能有一个载体字段；summary 不得逐字复述）。
        mutual = verify_field_mutual_exclusion(cards)
        # C8：summary 必须承载内容（提及每条要点 + 长度占字段 token 的 10%–40%）。
        substance = verify_summary_substance(cards)

        extracted = extract_key_fields(raw_text)
        total_fields = sum(len(values) for values in extracted.values())
        lost_count = sum(len(values) for values in report.missing.values())
        recall = (total_fields - lost_count) / total_fields if total_fields else 1.0

        # 金标准集的期望字段走**卡内文本**：这条不走回链，所以「引用了却没写进去」
        # 在这里仍会被判失败——回链放宽的是全局扫描，不是金标准的期望。
        produced = normalize(card_text_many(cards))
        missing_expected = [
            value for value in case.expected_key_fields if normalize(value) not in produced
        ]
        tiers = {str(card["tier"]) for card in cards}
        missing_tiers = [tier for tier in case.expected_tiers if tier not in tiers]

        raw_tokens = sum(estimate_tokens(text) for text in case.messages)
        card_tokens = sum(estimate_tokens(card_text_many([card])) for card in cards) or 1
        tier_stats, shared = _tier_stats(cards, store)
        # C6.1：参照点抽取器的卡契约违规（C6 + C8）是**预期**，记在另一个字段里、不判失败。
        exempt = is_baseline_extractor(extractor) if is_baseline is None else is_baseline
        c6 = list(mutual.missing.get("field_mutual_exclusion", []))
        c8 = list(substance.missing.get("summary_substance", []))
        return CaseResult(
            case_id=case.id,
            ok=(
                report.ok
                and qualitative.ok
                and (exempt or (not c6 and not c8))
                and not missing_expected
                and not missing_tiers
            ),
            recall=recall,
            total_fields=total_fields,
            lost=dict(report.missing),
            missing_expected=missing_expected,
            missing_tiers=missing_tiers,
            compression=raw_tokens / card_tokens,
            cards=len(cards),
            warnings=list(report.warnings),
            raw_tokens=raw_tokens,
            card_tokens=card_tokens,
            tier_stats=tier_stats,
            shared_sources=shared,
            uncarried=list(qualitative.missing.get("qualitative", [])),
            field_violations=[] if exempt else c6,
            summary_violations=[] if exempt else c8,
            expected_violations=(c6 + c8) if exempt else [],
        )
    except Exception as exc:  # 单条用例异常不应掩盖其他用例的结果
        return CaseResult(
            case_id=case.id, ok=False, recall=0.0, total_fields=0, error=f"{type(exc).__name__}: {exc}"
        )
    finally:
        store.close()


def run_eval(
    cases: Sequence[GoldCase] | None = None,
    extractor: Extractor | None = None,
    *,
    gold_dir: str | pathlib.Path | None = None,
    min_recall: float = DEFAULT_MIN_RECALL,
) -> EvalReport:
    """跑完整评测。默认用规则基线抽取器作为对照基准。"""
    selected = list(cases) if cases is not None else load_gold(gold_dir)
    engine = extractor or BaselineExtractor()
    return EvalReport(
        results=[run_case(case, engine) for case in selected],
        min_recall=min_recall,
    )


# --------------------------------------------------------------------------- 只读探针
#
# 下面这一节**不是评测口径**：它不参与 `run_eval` 的关卡判定，也不进 `summary()`。
# 它是 Phase 1 开工前的一次性探针，回答一个问题：把「一条消息一张卡」换成「按信息项归并」，
# 卡数与 token 的理论下界各是多少。Phase 1 的 I/O 契约定下来之后可以整节删掉。


@dataclass
class MergeHeadroomRow:
    """某一层级上「归并能省多少」的读数。所有列都来自已定义的口径，不引入新概念。"""

    tier: str
    cards_now: int = 0
    cards_bound: int = 0
    raw_tokens: int = 0
    card_tokens_now: int = 0
    #: 口径 A：**并集**——每个信息项取「包含它的最长句子」中 token 最多的那一次，
    #: 再把所有信息项的载体句子取并集。这是「原文里最短一段仍逐字包含全部信息项的文字」。
    item_union_tokens: int = 0
    #: 口径 A'：用户定义的字面 **求和**。共享同一句子的多个信息项会被重复计入，
    #: 因而它**不是下界**（会高于真实下界）。列出来是为了让差额可见。
    item_sum_tokens: int = 0
    #: 口径 B：**只去重、不抽象**——同一句话只写一次（跨消息与跨字段的逐字重复都去掉），
    #: 但保留全部文字。它衡量「归并 + 去重」在完全不丢内容的前提下能走多远。
    dedup_tokens: int = 0

    @property
    def ratio_now(self) -> float:
        return self.raw_tokens / self.card_tokens_now if self.card_tokens_now else 0.0

    @property
    def ratio_items(self) -> float:
        return self.raw_tokens / self.item_union_tokens if self.item_union_tokens else 0.0

    @property
    def ratio_dedup(self) -> float:
        return self.raw_tokens / self.dedup_tokens if self.dedup_tokens else 0.0


@dataclass
class MergeHeadroom:
    rows: list[MergeHeadroomRow]
    total: MergeHeadroomRow
    cases: int = 0
    #: 出现在两个及以上层级的信息项数（分层表按层级各计一次，故分层之和大于总数）。
    shared_items: int = 0
    #: 找不到承载句子的信息项数（跨句边界等）。>0 时口径 A 略偏小，必须一并声明。
    items_without_carrier: int = 0

    def row(self, tier: str) -> MergeHeadroomRow:
        for item in self.rows:
            if item.tier == tier:
                return item
        raise KeyError(tier)

    @staticmethod
    def _ratio(value: float, denominator: int) -> str:
        """分母为 0 时显示「—」：那是「不适用」，不是「压缩得很差」。"""
        return f"{value:.3f}x" if denominator else "—"

    def table(self) -> list[str]:
        lines = [
            "归并上限探针（Phase 1 开工前的只读读数；不参与关卡判定）",
            "  口径 A  = 每个信息项各留 1 处载体句子，取并集（信息项 = HARD_PATTERNS 的日期/金额/URL/代码）",
            "  口径 A' = 同样取载体，但按信息项逐个求和（共享句子会重复计入，**不是下界**）",
            "  口径 B  = 只去重不抽象：同一句话只写一次，文字不丢",
            "  合计行按 token 加权（不是逐用例均值），与分层行同口径聚合。",
            f"  {'tier':<5}{'当前卡数':>8}{'归并下界卡数':>12}{'原文token':>11}"
            f"{'当前压缩率':>11}{'A 压缩率':>10}{'B 压缩率':>10}",
        ]
        for item in [*self.rows, self.total]:
            lines.append(
                f"  {item.tier:<5}{item.cards_now:>8}{item.cards_bound:>12}"
                f"{item.raw_tokens:>11,}{self._ratio(item.ratio_now, item.card_tokens_now):>11}"
                f"{self._ratio(item.ratio_items, item.item_union_tokens):>10}"
                f"{self._ratio(item.ratio_dedup, item.dedup_tokens):>10}"
            )
        union_total = sum(i.item_union_tokens for i in self.rows)
        lines.append(
            f"  · 口径 A' 的分层求和 token：{sum(i.item_sum_tokens for i in self.rows):,}"
            f"（分层并集 {union_total:,}，差额 {sum(i.item_sum_tokens for i in self.rows) - union_total:,}"
            " 就是共享句子被重复计入的部分——所以按信息项逐个求和不是下界）"
        )
        if self.shared_items:
            lines.append(
                f"  · 出现在多个层级的信息项 {self.shared_items} 个：分层表里各计一次，"
                "故分层卡数之和大于合计卡数"
            )
        if self.items_without_carrier:
            lines.append(
                f"  · 找不到承载句子的信息项 {self.items_without_carrier} 个（跨句边界）："
                "口径 A 因此略偏小"
            )
        return lines


def merge_headroom(
    report: EvalReport | None = None,
    *,
    cases: Sequence[GoldCase] | None = None,
) -> MergeHeadroom:
    """只读探针：把「一条消息一张卡」换成「按信息项归并」后的卡数/token 下界。

    「当前」列一律**复用 `EvalReport` 的既有读数**（卡数、原文 token、卡 token），不另算一套。

    - **卡数下界** = 该层级消息里出现的**不同信息项**个数（信息项 = `verify.HARD_PATTERNS`
      抽出的日期/金额/URL/反引号代码，`normalize` 后去重）。注意它的性质：
      一张卡可以装多个信息项，所以这个数**不是卡数的数学下界**，而是
      「一个信息项一张卡」这个设计点上的卡数。它有用是因为当前基线是「一条消息一张卡」，
      两个数一比就知道「按信息项归并」相对「按消息切卡」是变多还是变少；
      它**对不含硬字段的消息（纯叙述、寒暄、噪音）没有任何说法**——
      那些层级的下界是 0，不代表可以把它们删光。
    - **口径 A**（token 下界）= 每个信息项取包含它的最长句子，再取这些句子的**并集**。
      这是「最短一段仍逐字包含全部信息项的文字」。用并集而不是求和，是因为多个信息项
      常共享同一句话，求和会把它重复计入（那样得出的数高于真实下界，不能叫下界）。
      口径 A 假设**可以丢掉不含硬字段的句子**——那是抽象，不是归并；它的读数因此
      同时包含「归并」和「抽象」两种效应，不能当成归并的功劳。
    - **口径 B**（只去重不抽象）= 该层级里不同句子各写一次。它**不删任何文字**，
      只删逐字重复。若金标准集里没有逐字重复的句子，B 恒等于原文 token，即 `1.000x`——
      这不是测出来的，是构造出来的；它说明「归并 + 去重」在这份集合上的天花板就是 `1.000x`。

    探针不写文件、不改任何状态；`cases` 只在需要自己跑基线时用。
    """
    base = report if report is not None else run_eval(cases=cases)
    selected = {case.id: case for case in (list(cases) if cases is not None else load_gold())}

    rows: dict[str, MergeHeadroomRow] = {
        tier: MergeHeadroomRow(tier=tier) for tier in TIER_ORDER
    }
    total = MergeHeadroomRow(tier="合计")
    shared_items = 0
    without_carrier = 0

    for result in base.results:
        case = selected.get(result.case_id)
        if case is None:
            continue
        # 「当前」列直接抄既有读数：卡数/原文/卡 token 全部按 tier 归集。
        for tier, stat in result.tier_stats.items():
            row = rows.setdefault(tier, MergeHeadroomRow(tier=tier))
            row.cards_now += stat.cards
            row.raw_tokens += stat.raw_tokens
            row.card_tokens_now += stat.card_tokens
        total.cards_now += result.cards
        total.raw_tokens += result.raw_tokens
        total.card_tokens_now += result.card_tokens

        tiers = [classify_tier(text) for text in case.messages]
        sentences: list[tuple[str, str, int]] = []  # (tier, 归一化句子, token)
        for tier, text in zip(tiers, case.messages):
            for sentence in SENTENCE_SPLIT_RE.split(text):
                sentence = sentence.strip()
                if sentence:
                    sentences.append((tier, normalize(sentence), estimate_tokens(sentence)))

        item_tiers: dict[str, set[str]] = {}
        for tier, text in zip(tiers, case.messages):
            for values in extract_key_fields(text).values():
                for value in values:
                    item_tiers.setdefault(normalize(value), set()).add(tier)

        carriers: dict[str, int | None] = {}
        for item in item_tiers:
            best: int | None = None
            for index, (_, normalized, tokens) in enumerate(sentences):
                if item and item in normalized and (best is None or tokens > sentences[best][2]):
                    best = index
            carriers[item] = best
            if best is None:
                without_carrier += 1
            if len(item_tiers[item]) >= 2:
                shared_items += 1

        # 口径 B：每个层级内部，不同句子各计一次（同句取 token 最多的那次）
        best_sentence: dict[tuple[str, str], int] = {}
        for tier, normalized, tokens in sentences:
            key = (tier, normalized)
            best_sentence[key] = max(best_sentence.get(key, 0), tokens)
        for (tier, _), tokens in best_sentence.items():
            rows.setdefault(tier, MergeHeadroomRow(tier=tier)).dedup_tokens += tokens
        best_sentence_case: dict[str, int] = {}
        for _, normalized, tokens in sentences:
            best_sentence_case[normalized] = max(best_sentence_case.get(normalized, 0), tokens)
        total.dedup_tokens += sum(best_sentence_case.values())

        # 口径 A / A'：按层级归集信息项的载体
        per_tier_items: dict[str, set[str]] = {}
        for item, item_tier_set in item_tiers.items():
            for tier in item_tier_set:
                per_tier_items.setdefault(tier, set()).add(item)
        for tier, items in per_tier_items.items():
            row = rows.setdefault(tier, MergeHeadroomRow(tier=tier))
            row.cards_bound += len(items)
            union: set[int] = set()
            for item in items:
                index = carriers[item]
                if index is None:
                    continue
                row.item_sum_tokens += sentences[index][2]
                union.add(index)
            row.item_union_tokens += sum(sentences[index][2] for index in union)

        total.cards_bound += len(item_tiers)
        case_union: set[int] = set()
        for item in item_tiers:
            index = carriers[item]
            if index is None:
                continue
            total.item_sum_tokens += sentences[index][2]
            case_union.add(index)
        total.item_union_tokens += sum(sentences[index][2] for index in case_union)

    return MergeHeadroom(
        rows=[rows[tier] for tier in TIER_ORDER],
        total=total,
        cases=len(base.results),
        shared_items=shared_items,
        items_without_carrier=without_carrier,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口：不达标返回非 0，可直接挂到 CI。"""
    args = list(sys.argv[1:] if argv is None else argv)
    gold_dir = args[0] if args else None
    report = run_eval(gold_dir=gold_dir)
    print(report.summary())
    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover - 需先安装包或设置 PYTHONPATH
    raise SystemExit(main())
