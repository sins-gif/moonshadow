"""参数漂移下的「反转区域」测量：加性方案 vs 有界乘性方案。

问题：全部评测结论来自 21 条人工用例。真实查询分布到来时，语料的**卡龄分布、相似度量级、
重要度分布**都会变。本模块测量：候选参数发生漂移时，两个方案的「正负例反转区域」是否同样变化。

判据（技术报告定理 3、推论 1）：设 P 为应召回的候选、N 为应排除的候选，
「反转」即 `q_N > q_P`，等价于

    r_P / r_N  <  m_N / m_P  ≤  1 / f = 1.612903

因此乘性方案有一条**与数据无关的保证**：`r_P / r_N ≥ 1/f` 时不可能反转。
加性方案没有这样的保证——它的反转条件里含有一项 `0.25 δ`，在相关性分 `r` 中没有对应项，
所以年龄差可以独立于相关性推动反转。本模块把这一点量化。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

from .clock import to_iso
from .scoring import (
    SCHEME_ADDITIVE,
    SCHEME_MULTIPLICATIVE,
    SCHEME_RELEVANCE,
    TIME_FLOOR,
    score_card,
)
from .tiers import half_life_for

#: 基准时刻（与 rank_eval 一致，保证可复现）。
REFERENCE_ISO = "2026-06-01T12:00:00+00:00"

PROBE = "E"

#: 正例（应召回）：中等相关、命中实体；层级取无卡龄窗口的 T2/T3，让卡龄自由漂移。
POSITIVE_TIERS = ("T2", "T3")
POSITIVE_IMPORTANCE = (3, 6)
POSITIVE_AGES = (5.0, 30.0, 100.0, 200.0, 400.0, 800.0)
POSITIVE_COS = (0.70, 0.85)

#: 负例（应排除）：低相似度、无实体命中、新鲜、低重要度。
NEGATIVE_TIERS = ("T3", "T5")
NEGATIVE_IMPORTANCE = (1, 2)
NEGATIVE_AGES = (0.1, 1.0, 5.0, 15.0)
NEGATIVE_COS = (0.10, 0.15, 0.20)


@dataclass(frozen=True)
class DriftPair:
    label: str
    relevance_ratio: float          # r_P / r_N
    additive_inverts: bool
    multiplicative_inverts: bool


@dataclass
class DriftReport:
    bound: float
    pairs: list[DriftPair] = field(default_factory=list)

    def _count(self, predicate) -> int:  # type: ignore[no-untyped-def]
        return sum(1 for pair in self.pairs if predicate(pair))

    @property
    def total(self) -> int:
        return len(self.pairs)

    @property
    def additive_total(self) -> int:
        return self._count(lambda p: p.additive_inverts)

    @property
    def multiplicative_total(self) -> int:
        return self._count(lambda p: p.multiplicative_inverts)

    @property
    def additive_over_bound(self) -> list[DriftPair]:
        """加性方案在「乘性方案有保证」的区域里发生的反转。"""
        return [p for p in self.pairs if p.additive_inverts and p.relevance_ratio >= self.bound]

    @property
    def multiplicative_over_bound(self) -> list[DriftPair]:
        """乘性方案越过自身上界的反转——按推论 1，此处必为空。"""
        return [p for p in self.pairs if p.multiplicative_inverts and p.relevance_ratio >= self.bound]

    @property
    def worst_additive_ratio(self) -> float:
        ratios = [p.relevance_ratio for p in self.pairs if p.additive_inverts]
        return max(ratios, default=0.0)

    @property
    def worst_multiplicative_ratio(self) -> float:
        ratios = [p.relevance_ratio for p in self.pairs if p.multiplicative_inverts]
        return max(ratios, default=0.0)


def _card(tier: str, importance: int, age_days: float, hit: bool, reference) -> dict:
    return {
        "id": "probe",
        "tier": tier,
        "importance": importance,
        "entities": [PROBE] if hit else [],
        "observed_at": to_iso(reference - timedelta(days=age_days)),
        "half_life_days": half_life_for(tier),
        "status": "active",
    }


def _score(tier, importance, age, cos, hit, scheme, reference) -> float:
    return score_card(
        _card(tier, importance, age, hit, reference),
        now=reference,
        query_cos=cos,
        query_entities=[PROBE],
        task_hit=False,
        scheme=scheme,
    )


def measure_drift(
    *,
    positive_tiers=POSITIVE_TIERS,
    positive_importance=POSITIVE_IMPORTANCE,
    positive_ages=POSITIVE_AGES,
    positive_cos=POSITIVE_COS,
    negative_tiers=NEGATIVE_TIERS,
    negative_importance=NEGATIVE_IMPORTANCE,
    negative_ages=NEGATIVE_AGES,
    negative_cos=NEGATIVE_COS,
    reference_iso: str = REFERENCE_ISO,
) -> DriftReport:
    """在给定网格上统计两个方案的反转区域。"""
    from .clock import parse_ts

    reference = parse_ts(reference_iso)
    report = DriftReport(bound=1.0 / TIME_FLOOR)

    for p_tier in positive_tiers:
        for p_i in positive_importance:
            for p_age in positive_ages:
                for p_cos in positive_cos:
                    r_p = _score(p_tier, p_i, p_age, p_cos, True, SCHEME_RELEVANCE, reference)
                    a_p = _score(p_tier, p_i, p_age, p_cos, True, SCHEME_ADDITIVE, reference)
                    q_p = _score(p_tier, p_i, p_age, p_cos, True, SCHEME_MULTIPLICATIVE, reference)
                    tag = f"P({p_tier}/i={p_i}/Δ={p_age:g}/σ={p_cos})"

                    for n_tier in negative_tiers:
                        for n_i in negative_importance:
                            for n_age in negative_ages:
                                for n_cos in negative_cos:
                                    r_n = _score(n_tier, n_i, n_age, n_cos, False, SCHEME_RELEVANCE, reference)
                                    a_n = _score(n_tier, n_i, n_age, n_cos, False, SCHEME_ADDITIVE, reference)
                                    q_n = _score(n_tier, n_i, n_age, n_cos, False, SCHEME_MULTIPLICATIVE, reference)
                                    report.pairs.append(
                                        DriftPair(
                                            label=f"{tag} vs N({n_tier}/i={n_i}/Δ={n_age:g}/σ={n_cos})",
                                            relevance_ratio=r_p / r_n,
                                            additive_inverts=a_n > a_p,
                                            multiplicative_inverts=q_n > q_p,
                                        )
                                    )
    return report


def format_drift_report(report: DriftReport, *, top: int = 5) -> str:
    lines = [
        f"配对总数：{report.total}",
        f"乘性方案的反转上界：r_P/r_N ≥ 1/f = {report.bound:.6f} 时不得反转",
        "",
        f"加性方案：反转 {report.additive_total} 组 "
        f"({report.additive_total / report.total:.2%})",
        f"  其中 r_P/r_N ≥ 1/f（乘性有保证的区域）：**{len(report.additive_over_bound)} 组**",
        f"  反转对里的最大 r_P/r_N = {report.worst_additive_ratio:.4f}",
    ]
    for pair in sorted(report.additive_over_bound, key=lambda p: -p.relevance_ratio)[:top]:
        lines.append(f"    · r_P/r_N={pair.relevance_ratio:.4f}  {pair.label}")

    lines += [
        "",
        f"有界乘性方案：反转 {report.multiplicative_total} 组 "
        f"({report.multiplicative_total / report.total:.2%})",
        f"  反转对里的最大 r_P/r_N = {report.worst_multiplicative_ratio:.4f}",
        f"  越过上界的反转：{len(report.multiplicative_over_bound)} 组"
        f"（{'违反推论 1' if report.multiplicative_over_bound else '全部落在推论 1 的上界之内'}）",
    ]
    for pair in sorted(
        (p for p in report.pairs if p.multiplicative_inverts),
        key=lambda p: -p.relevance_ratio,
    )[:3]:
        lines.append(f"    · r_P/r_N={pair.relevance_ratio:.4f}  {pair.label}")

    lines += [
        "",
        f"结论：加性方案在乘性方案**有保证**的区域里反转了 {len(report.additive_over_bound)} 组；"
        f"乘性方案违反自身上界 {len(report.multiplicative_over_bound)} 组。",
    ]
    return "\n".join(lines)
