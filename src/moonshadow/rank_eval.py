"""排序方案对照实验：加性权重 vs 乘性时间修正 vs 仅相关性。

为什么需要它：`0.45/0.25/0.20/0.10` 这组数在 v1.0 里没有任何依据，属于拍脑袋。
争论「时间该不该和相关性抢权重」是吵不出结论的，只能拿人工期望排序当基准跑数据。

方法：
- 每个用例给出**一批已经过硬门的候选**（本实验只比排序，不重复测门）与人工期望顺序；
- 三种方案都调用**同一份真实打分代码**（`scoring.score_card` 的 scheme 参数），
  而不是在实验里另抄一份公式——否则实验通过不代表线上通过；
- 指标：成对一致率（两两比较的顺序与人工期望一致的比例）与 Top-1 命中率，
  并**按查询意图分组**统计，因为差异很可能只出现在特定意图上。

基准时间固定为 ``NOW``，因此结果完全可复现。
"""

from __future__ import annotations

import json
import pathlib
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterable, Mapping, Sequence

from .scoring import (
    SCHEME_ADDITIVE,
    SCHEME_BLENDED,
    SCHEME_HYBRID,
    SCHEME_INTENT_AWARE,
    SCHEME_MULTIPLICATIVE,
    SCHEME_RELEVANCE,
    SCHEME_RRF,
    SCHEMES,
    THETA,
    THETA_HIGH,
    TIME_FLOOR,
    TIME_FLOOR_MIN,
    RELEVANCE_WEIGHTS,
    floor_for_intent,
    floor_for_sensitivity,
    match_score,
    reciprocal_rank_fusion,
    rrf_weights,
    score_card,
    sim_norm,
    time_modifier,
    time_sensitivity,
)

#: 现网在用的阈值（加性量纲下标定）。标定结果要与它对比才有意义。
CURRENT_THRESHOLDS: dict[str, float] = {"theta": THETA, "theta_high": THETA_HIGH}
from .tiers import half_life_for

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_RANK_DIR = ROOT / "eval" / "rank"
#: 留出集：与开发集物理隔离，**先写满再跑**，用于检验开发集上的结论是否为过拟合。
HOLDOUT_DIR = ROOT / "eval" / "holdout"

#: 固定基准时间：所有用例的卡龄都相对它计算，结果可逐位复现。
NOW = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)

SCHEME_LABELS: dict[str, str] = {
    SCHEME_ADDITIVE: "加性（v1.0 的 0.45/0.25/0.20/0.10；未采纳）",
    SCHEME_MULTIPLICATIVE: f"乘性修正（相关性 × [{TIME_FLOOR}, 1.0]）",
    SCHEME_RELEVANCE: "仅相关性 + 时间破同分",
    SCHEME_INTENT_AWARE: "乘性 + 按意图硬切换下限",
    SCHEME_BLENDED: "乘性 + 连续敏感度插值",
    SCHEME_RRF: "RRF 名次融合（不要求可通约）",
    SCHEME_HYBRID: "两种分数 50/50 平均（不调参）",
}

INTENT_LABELS: dict[str, str] = {
    "historical": "历史决策",
    "status": "当前状态",
    "task": "任务跟进",
    "preference": "长期偏好",
}


@dataclass(frozen=True)
class Candidate:
    """一个候选：已经过硬门的卡，加上本次查询对它的相似度。

    ``reason`` 仅对负例（``exclude`` 里的候选）有意义，说明它**为什么**应该被挡掉：

    - ``score``：卡龄在窗口内，但分数不够——只有 θ 能挡住它，因此是 θ 标定的依据；
    - ``window``：卡龄超出该层级的召回窗口，由门（而非 θ）挡住；
    - ``tier``：层级本身要求显式回溯（T8/T9），与分数无关。
    """

    id: str
    tier: str
    importance: int
    age_days: float
    cos: float
    entities: tuple[str, ...] = ()
    task_hit: bool = False
    reason: str = "score"

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "Candidate":
        return cls(
            id=str(payload["id"]),
            tier=str(payload["tier"]),
            importance=int(payload["importance"]),
            age_days=float(payload["age_days"]),
            cos=float(payload["cos"]),
            entities=tuple(str(e) for e in payload.get("entities", ())),
            task_hit=bool(payload.get("task_hit", False)),
            reason=str(payload.get("reason", "score")),
        )

    @property
    def half_life_days(self) -> float:
        return half_life_for(self.tier)

    def as_card(self) -> dict[str, Any]:
        """构造成打分器认识的卡结构，走**真实**的 delta 计算路径。"""
        observed = NOW - timedelta(days=self.age_days)
        return {
            "id": self.id,
            "tier": self.tier,
            "importance": self.importance,
            "entities": list(self.entities),
            "observed_at": observed.isoformat(timespec="seconds"),
            "half_life_days": self.half_life_days,
            "status": "active",
        }


@dataclass(frozen=True)
class RankCase:
    id: str
    intent: str
    query_text: str
    query_entities: tuple[str, ...]
    candidates: tuple[Candidate, ...]
    expect_order: tuple[str, ...]
    exclude: tuple[Candidate, ...] = ()
    note: str = ""
    #: 「答案是否依赖最新状态」——比 status/task/historical 这类意图分类**无歧义得多**：
    #: 「报价给到了吗」算任务还是状态没有客观答案，但「是否依赖最新状态」有。
    #: ``None`` 表示未提供此标签，此时回退到 ``intent == "status"``。
    needs_fresh: bool | None = None
    #: 该场景是否属于「显式回溯请求」。T8/T9 只在为真时才允许进入候选集（定义 6）。
    #: 排序实验本身不过门，这个字段用于让**用例能声明自己的门前置**，并由审计校验。
    explicit_recall: bool = False

    def freshness_label(self) -> bool:
        """评测用的真值标签：优先用 ``needs_fresh``，否则退回意图字段。"""
        if self.needs_fresh is not None:
            return self.needs_fresh
        return self.intent == "status"

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "RankCase":
        query = payload.get("query", {})
        return cls(
            id=str(payload["id"]),
            intent=str(payload.get("intent", "task")),
            query_text=str(query.get("text", "")),
            query_entities=tuple(str(e) for e in query.get("entities", ())),
            candidates=tuple(Candidate.from_json(item) for item in payload["candidates"]),
            expect_order=tuple(str(item) for item in payload["expect_order"]),
            exclude=tuple(Candidate.from_json(item) for item in payload.get("exclude", ())),
            note=str(payload.get("note", "")),
            needs_fresh=None if payload.get("needs_fresh") is None else bool(payload["needs_fresh"]),
            explicit_recall=bool(payload.get("explicit_recall", False)),
        )


def load_rank_cases(directory: str | pathlib.Path | None = None) -> list[RankCase]:
    target = pathlib.Path(directory) if directory is not None else DEFAULT_RANK_DIR
    cases = []
    for path in sorted(target.glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            cases.append(RankCase.from_json(json.load(handle)))
    return sorted(cases, key=lambda case: case.id)


def load_holdout_cases(directory: str | pathlib.Path | None = None) -> list[RankCase]:
    """加载留出集。与开发集分开加载，避免任何形式的混用。"""
    return load_rank_cases(directory if directory is not None else HOLDOUT_DIR)


def _floor_for(case: RankCase, scheme: str) -> float | None:
    """时间下限的取法——这正是各方案真正的分歧所在。

    - ``intent_aware``：按意图**硬切换**；
    - ``blended``：按查询的**连续敏感度插值**（意图不清时平滑退化）。
    """
    if scheme == SCHEME_INTENT_AWARE:
        return floor_for_intent(case.intent)
    if scheme == SCHEME_BLENDED:
        return floor_for_sensitivity(time_sensitivity(case.query_text))
    if scheme == SCHEME_HYBRID:
        # 混合方案也接上下文（否则它连「时间该有多大话语权」都不知道），
        # 用的是与 blended 相同的连续敏感度，以分离「融合算法」与「上下文来源」两个变量。
        return floor_for_sensitivity(time_sensitivity(case.query_text))
    return None


def score_of(case: RankCase, candidate: Candidate, scheme: str) -> float:
    """单卡分数。排序与 θ 标定共用同一入口，避免两处逻辑漂移。

    RRF 是集合级方案（融合名次），没有单卡分数，因此不能用于阈值标定。
    """
    if scheme == SCHEME_RRF:
        raise ValueError("RRF 没有单卡分数，无法参与阈值标定")
    return score_card(
        candidate.as_card(),
        now=NOW,
        query_cos=candidate.cos,
        query_entities=case.query_entities,
        task_hit=candidate.task_hit,
        scheme=scheme,
        time_floor=_floor_for(case, scheme),
    )


def ordering(case: RankCase, scheme: str) -> list[str]:
    """按指定方案给出排序。同分时用时间修正系数破同分，再按 id 保证确定性。

    RRF 是**集合级**方案：它融合的是「相关性名次」与「时间名次」，不是单卡分数，
    因此单独走一条分支（融合函数同样来自生产代码 `scoring.reciprocal_rank_fusion`）。
    """
    if scheme == SCHEME_RRF:
        rows = [
            (
                candidate.id,
                score_card(
                    candidate.as_card(),
                    now=NOW,
                    query_cos=candidate.cos,
                    query_entities=case.query_entities,
                    task_hit=candidate.task_hit,
                    scheme=SCHEME_RELEVANCE,
                ),
                time_modifier(candidate.age_days, candidate.half_life_days),
            )
            for candidate in case.candidates
        ]
        weights = rrf_weights(time_sensitivity(case.query_text))
        return [name for name, _ in reciprocal_rank_fusion(rows, weights=weights)]

    rows: list[tuple[float, float, str]] = []
    for candidate in case.candidates:
        rows.append(
            (
                score_of(case, candidate, scheme),
                time_modifier(candidate.age_days, candidate.half_life_days),
                candidate.id,
            )
        )
    rows.sort(key=lambda row: (-row[0], -row[1], row[2]))
    return [row[2] for row in rows]


def pairwise_agreement(expected: Sequence[str], produced: Sequence[str]) -> float:
    """成对一致率：期望顺序里每一对，在产出顺序里是否也保持先后。"""
    position = {name: index for index, name in enumerate(produced)}
    total = agree = 0
    for i in range(len(expected)):
        for j in range(i + 1, len(expected)):
            total += 1
            if position[expected[i]] < position[expected[j]]:
                agree += 1
    return agree / total if total else 1.0


def top1_hit(expected: Sequence[str], produced: Sequence[str]) -> bool:
    return bool(expected) and bool(produced) and expected[0] == produced[0]


# ---------------------------------------------------------------------------
# 意图识别评测：必须与排序一致率分开报，否则出问题无法归因
# ---------------------------------------------------------------------------

#: 判定「这条查询需要高时间敏感度」的阈值。
SENSITIVITY_THRESHOLD = 0.5


@dataclass(frozen=True)
class IntentOutcome:
    case_id: str
    expected_status: bool
    predicted_status: bool
    sensitivity: float

    @property
    def ok(self) -> bool:
        return self.expected_status == self.predicted_status


@dataclass
class IntentReport:
    """把「时间敏感度识别」当成一个二分类器来评。

    漏判（状态查询被判成不敏感）比误判更危险：它会让系统答出过时状态。
    因此单独报 recall 与 precision，而不是只报一个准确率。
    """

    outcomes: list[IntentOutcome] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return sum(1 for o in self.outcomes if o.ok) / len(self.outcomes) if self.outcomes else 1.0

    @property
    def recall(self) -> float:
        positives = [o for o in self.outcomes if o.expected_status]
        if not positives:
            return 1.0
        return sum(1 for o in positives if o.predicted_status) / len(positives)

    @property
    def precision(self) -> float:
        predicted = [o for o in self.outcomes if o.predicted_status]
        if not predicted:
            return 1.0
        return sum(1 for o in predicted if o.expected_status) / len(predicted)

    @property
    def misses(self) -> list[IntentOutcome]:
        return [o for o in self.outcomes if not o.ok]

    def summary(self) -> str:
        lines = [
            f"时间敏感度识别：准确率={self.accuracy:.3f}  "
            f"recall(状态类)={self.recall:.3f}  precision={self.precision:.3f}"
        ]
        for outcome in self.misses:
            kind = "漏判" if outcome.expected_status else "误判"
            lines.append(
                f"  {kind}  {outcome.case_id:<32} 期望状态类={outcome.expected_status}"
                f"  敏感度={outcome.sensitivity:.2f}"
            )
        return "\n".join(lines)


def evaluate_intent(cases: Sequence[RankCase]) -> IntentReport:
    report = IntentReport()
    for case in cases:
        sensitivity = time_sensitivity(case.query_text)
        report.outcomes.append(
            IntentOutcome(
                case_id=case.id,
                expected_status=case.freshness_label(),
                predicted_status=sensitivity >= SENSITIVITY_THRESHOLD,
                sensitivity=sensitivity,
            )
        )
    return report


# ---------------------------------------------------------------------------
# θ 标定：把「切方案前必须重标阈值」做成工具，而不是一条 TODO
# ---------------------------------------------------------------------------

#: 各组阈值适用的层级（与 passes_gate 的分支一致）。
THETA_TIERS = ("T2", "T3", "T4", "T5")
THETA_HIGH_TIERS = ("T6", "T7")


@dataclass(frozen=True)
class ScoredCandidate:
    case_id: str
    candidate: Candidate
    score: float


@dataclass
class ThresholdCalibration:
    """某阈值的可行区间：``θ`` 必须大于全部负例、小于全部正例。"""

    group: str
    includes: list[ScoredCandidate] = field(default_factory=list)
    excludes: list[ScoredCandidate] = field(default_factory=list)

    @property
    def lower(self) -> float | None:
        return max((item.score for item in self.excludes), default=None)

    @property
    def upper(self) -> float | None:
        return min((item.score for item in self.includes), default=None)

    @property
    def feasible(self) -> bool:
        low, high = self.lower, self.upper
        if low is None or high is None:
            return True
        return low < high

    @property
    def recommended(self) -> float | None:
        """取区间中点：两侧留最大余量，避免贴着边界、轻微扰动就翻转。

        区间不存在（正负例分数重叠）时返回 ``None``——此时**没有**可用阈值，
        给出一个「建议值」只会误导。
        """
        if not self.feasible:
            return None
        low, high = self.lower, self.upper
        if low is None and high is None:
            return None
        if low is None:
            return high
        if high is None:
            return low
        return (low + high) / 2.0

    def contains(self, value: float) -> bool:
        """现用阈值是否落在推荐区间内——不在就意味着切换会静默改变硬门行为。"""
        if not self.feasible:
            return False
        low, high = self.lower, self.upper
        return (low is None or value > low) and (high is None or value < high)

    def summary(self, current: float | None = None) -> str:
        low, high = self.lower, self.upper
        low_text = "—" if low is None else f"{low:.4f}"
        high_text = "—" if high is None else f"{high:.4f}"
        verdict = "可行" if self.feasible else "不可行（正负例分数重叠）"
        recommended = self.recommended
        rec_text = "—" if recommended is None else f"{recommended:.4f}"
        line = (
            f"  {self.group:<11} 负例上界={low_text}  正例下界={high_text}  "
            f"{verdict}  建议={rec_text}  （正例 {len(self.includes)} / 负例 {len(self.excludes)}）"
        )
        if current is not None:
            if self.feasible:
                mark = "在区间内 ✓" if self.contains(current) else "**在区间外 → 必须改**"
                line += f"\n              现用={current:.4f}：{mark}"
            else:
                line += (
                    f"\n              现用={current:.4f}：区间不存在 —— 该方案在此用例集上"
                    "无法用单一阈值分开正负例（这本身就是缺陷证据）"
                )
        return line


def calibrate_thresholds(
    cases: Sequence[RankCase], scheme: str = SCHEME_MULTIPLICATIVE
) -> dict[str, ThresholdCalibration]:
    """按层级分组标定 θ 与 θ_high。

    只有 ``reason="score"`` 的负例参与标定——被卡龄窗口或层级规则挡掉的负例与 θ 无关，
    混进来会得出错误区间。
    """
    if scheme == SCHEME_RRF:
        raise ValueError("RRF 没有单卡分数，无法标定阈值")

    groups = {
        "theta": ThresholdCalibration("theta"),
        "theta_high": ThresholdCalibration("theta_high"),
    }

    def bucket(tier: str) -> ThresholdCalibration | None:
        if tier in THETA_TIERS:
            return groups["theta"]
        if tier in THETA_HIGH_TIERS:
            return groups["theta_high"]
        return None

    for case in cases:
        for candidate in case.candidates:
            target = bucket(candidate.tier)
            if target is not None:
                target.includes.append(
                    ScoredCandidate(case.id, candidate, score_of(case, candidate, scheme))
                )
        for candidate in case.exclude:
            if candidate.reason != "score":
                continue
            target = bucket(candidate.tier)
            if target is not None:
                target.excludes.append(
                    ScoredCandidate(case.id, candidate, score_of(case, candidate, scheme))
                )
    return groups


def format_calibration(
    calibration: Mapping[str, ThresholdCalibration],
    scheme: str,
    *,
    compare_current: bool = True,
) -> str:
    lines = [f"θ 标定（方案：{SCHEME_LABELS.get(scheme, scheme)}）"]
    for group in ("theta", "theta_high"):
        if group in calibration:
            current = CURRENT_THRESHOLDS.get(group) if compare_current else None
            lines.append(calibration[group].summary(current))
    return "\n".join(lines)


@dataclass(frozen=True)
class CaseOutcome:
    case_id: str
    intent: str
    scheme: str
    produced: tuple[str, ...]
    expected: tuple[str, ...]
    agreement: float
    top1: bool

    @property
    def ok(self) -> bool:
        return self.agreement == 1.0

    def summary(self) -> str:
        mark = "PASS" if self.ok else "FAIL"
        return (
            f"  {mark}  {self.case_id:<32} 期望={' > '.join(self.expected)}"
            f"  实际={' > '.join(self.produced)}"
        )


@dataclass
class SchemeResult:
    scheme: str
    outcomes: list[CaseOutcome] = field(default_factory=list)

    @property
    def label(self) -> str:
        return SCHEME_LABELS.get(self.scheme, self.scheme)

    @property
    def agreement(self) -> float:
        return sum(o.agreement for o in self.outcomes) / len(self.outcomes) if self.outcomes else 1.0

    @property
    def top1_rate(self) -> float:
        return sum(1 for o in self.outcomes if o.top1) / len(self.outcomes) if self.outcomes else 1.0

    @property
    def failures(self) -> list[CaseOutcome]:
        return [o for o in self.outcomes if not o.ok]

    def by_intent(self) -> dict[str, float]:
        buckets: dict[str, list[float]] = {}
        for outcome in self.outcomes:
            buckets.setdefault(outcome.intent, []).append(outcome.agreement)
        return {name: sum(values) / len(values) for name, values in sorted(buckets.items())}


@dataclass
class RankReport:
    results: list[SchemeResult] = field(default_factory=list)
    intent: IntentReport | None = None
    #: 按方案分别标定，便于对比「现用阈值在哪个方案下才成立」。
    calibrations: dict[str, dict[str, ThresholdCalibration]] = field(default_factory=dict)
    floor_sweep: FloorSweep | None = None
    floor_bands: dict[str, tuple[float, float] | None] = field(default_factory=dict)
    #: 用例清单，供 band 表格按原顺序渲染。
    case_index: list[RankCase] = field(default_factory=list)

    @property
    def best(self) -> SchemeResult:
        return max(self.results, key=lambda item: (item.agreement, item.top1_rate))

    def summary(self) -> str:
        lines: list[str] = []
        lines.append(f"{'方案':<34} {'成对一致率':>10} {'Top-1':>7}  失败用例")
        lines.append("-" * 78)
        for result in self.results:
            lines.append(
                f"{result.label:<34} {result.agreement:>10.3f} {result.top1_rate:>7.3f}"
                f"  {len(result.failures)}/{len(result.outcomes)}"
            )

        intents = sorted({o.intent for r in self.results for o in r.outcomes})
        if len(intents) > 1:
            lines.append("")
            lines.append("按查询意图拆分成对一致率：")
            header = "意图".ljust(12) + "".join(f"{r.scheme:>16}" for r in self.results)
            lines.append(header)
            for intent in intents:
                cells = "".join(
                    f"{r.by_intent().get(intent, float('nan')):>16.3f}" for r in self.results
                )
                lines.append(f"{INTENT_LABELS.get(intent, intent):<12}{cells}")

        for result in self.results:
            if result.failures:
                lines.append("")
                lines.append(f"{result.label} 的失败用例：")
                lines.extend(outcome.summary() for outcome in result.failures)

        lines.append("")
        best = self.best
        tied = [
            result
            for result in self.results
            if abs(result.agreement - best.agreement) <= 1e-9
            and abs(result.top1_rate - best.top1_rate) <= 1e-9
        ]
        if len(tied) > 1:
            # 并列时**不得**挑一个说成「最高」：那会把「无区分力」读成「有优劣」。
            lines.append(
                f"结论：{len(tied)} 个方案并列最高（{best.agreement:.3f}）——"
                f"{'、'.join(r.label for r in tied)}"
            )
            lines.append(
                "      并列意味着本用例集**不能区分**这些方案，"
                "需要按 `docs/v1.2-roadmap.md` §7.1 的 R1 补边界用例。"
            )
        else:
            lines.append(f"结论：成对一致率最高的是「{best.label}」")

        if self.intent is not None:
            lines.append("")
            lines.append(self.intent.summary())

        if self.calibrations:
            for scheme, calibration in self.calibrations.items():
                lines.append("")
                lines.append(format_calibration(calibration, scheme))

        if self.floor_sweep is not None:
            lines.append("")
            lines.append(self.floor_sweep.summary())

        if self.floor_bands:
            lines.append("")
            lines.append(format_bands(self.case_index, self.floor_bands))

        return "\n".join(lines)


def run_rank_eval(
    cases: Sequence[RankCase] | None = None,
    *,
    calibration_schemes: Sequence[str] = (SCHEME_MULTIPLICATIVE, SCHEME_BLENDED, SCHEME_ADDITIVE),
) -> RankReport:
    selected = list(cases) if cases is not None else load_rank_cases()
    report = RankReport()
    for scheme in SCHEMES:
        result = SchemeResult(scheme=scheme)
        for case in selected:
            produced = ordering(case, scheme)
            result.outcomes.append(
                CaseOutcome(
                    case_id=case.id,
                    intent=case.intent,
                    scheme=scheme,
                    produced=tuple(produced),
                    expected=case.expect_order,
                    agreement=pairwise_agreement(case.expect_order, produced),
                    top1=top1_hit(case.expect_order, produced),
                )
            )
        report.results.append(result)
    report.intent = evaluate_intent(selected)
    for scheme in calibration_schemes:
        if scheme == SCHEME_RRF:
            continue  # RRF 没有单卡分数
        report.calibrations[scheme] = calibrate_thresholds(selected, scheme)
    report.floor_sweep = sweep_floor(selected)
    report.case_index = list(selected)
    report.floor_bands = case_floor_bands(selected)
    return report


def ordering_with_floor(case: RankCase, floor: float) -> list[str]:
    """用**指定**的 floor 排序（绕过按意图/按敏感度的取法）。

    这是扫描实验的入口：它把「floor 取多少」当成自变量，用来判断
    「按意图切换 floor」这套机制究竟是否必要。
    """
    rows = [
        (
            score_card(
                candidate.as_card(),
                now=NOW,
                query_cos=candidate.cos,
                query_entities=case.query_entities,
                task_hit=candidate.task_hit,
                scheme=SCHEME_MULTIPLICATIVE,
                time_floor=floor,
            ),
            time_modifier(candidate.age_days, candidate.half_life_days),
            candidate.id,
        )
        for candidate in case.candidates
    ]
    rows.sort(key=lambda row: (-row[0], -row[1], row[2]))
    return [row[2] for row in rows]


#: 扫描步长。**必须整除采纳下限 `TIME_FLOOR`**，否则网格会漏掉采纳值并给出
#: 「不可行」的假结论（`require_step_divides` 会直接抛错）。
#: `0.0025` 在 `[0,1]` 上给出 `401` 个点，且 `0.5925 / 0.0025 = 237` 为整数。
#: 注意步长会影响「窗口中点」的读数：`0.005 / 0.0025 / 0.00125 / 0.0005` 分别给出
#: `0.5925 / 0.5913 / 0.5919 / 0.5922`，因此中点只能报到步长量级，不能当精确值。
FLOOR_SWEEP_STEP = 0.0025


def _dividing_steps(value: float, limit: int = 4) -> list[float]:
    """能整除 ``value`` 的候选步长，从大到小。用于在报错时给出可用的替代值。"""
    found = []
    for power in range(1, limit + 1):
        for factor in (5, 2, 1):
            step = factor / (10**power)
            ratio = value / step
            if abs(ratio - round(ratio)) <= 1e-9 and step < value:
                found.append(round(step, 12))
    return sorted(set(found), reverse=True)[:4]


def require_step_divides(step: float, value: float, *, context: str) -> None:
    """断言扫描步长能整除被讨论的取值——否则该点不在网格上。

    这不是洁癖：网格会**静默漏掉**那个取值，从而给出「该解不在可行区内」的假结论。
    本仓库在 ``sweep_grouped_floor`` 上因此得出过两次错误结论（步长 `0.05` 不整除 `0.62`），
    所以这里必须**阻断**，不是警告。若确实要用不整除的步长，改用单点求值
    （``_agreement_grouped`` 一类的直接计算），不要依赖网格。
    """
    if step <= 0:
        raise ValueError(f"{context}：步长必须为正，收到 {step!r}")
    ratio = value / step
    if abs(ratio - round(ratio)) <= 1e-9:
        return
    suggestions = "、".join(str(candidate) for candidate in _dividing_steps(value))
    raise ValueError(
        f"{context}：步长 {step} 不能整除 {value}——该取值不在网格上，"
        f"扫描会漏掉它并给出「不可行」的假结论。"
        f"请改用能整除的步长（{suggestions}），或改用不依赖网格的单点求值。"
    )


@dataclass
class FloorSweep:
    """在全体用例上扫描全局 floor，记录每个取值的平均一致率。

    关注两点：
    - **最优值**：是否存在一个全局 floor 能让全部用例通过；
    - **窗口宽度**：满足全通过的 floor 区间有多宽。窗口越窄，说明系统对该参数越敏感，
      这本身就是「方案不稳」的证据——窄窗口上的最优值不能当作调参结论。
    """

    step: float = FLOOR_SWEEP_STEP
    points: list[tuple[float, float]] = field(default_factory=list)

    @property
    def best(self) -> tuple[float, float]:
        """一致率最高的取值。**注意：通常有大量并列**，因此它不是「该取的值」。

        该取哪个值应当看 ``recommended``（窗口中点），不要拿这个数当结论。
        """
        return max(self.points, key=lambda item: (item[1], -abs(item[0] - 0.5)))

    @property
    def max_agreement(self) -> float:
        return self.best[1]

    @property
    def recommended(self) -> float | None:
        """窗口中点，取它的理由是最小最大余量（minimax）：

        在「哪条用例代表现实」未知的前提下，让最坏情况下离最近的约束最远，
        比选并列取值里任意一个都更稳。
        """
        widest = self.widest_run
        return None if widest is None else (widest[0] + widest[1]) / 2.0

    def feasible_runs(self, target: float = 1.0) -> list[tuple[float, float]]:
        """返回所有「一致率 ≥ target」的连续区间（含端点）。"""
        tolerance = 1e-9
        runs: list[tuple[float, float]] = []
        start: float | None = None
        previous = None
        for floor, agreement in self.points:
            ok = agreement >= target - tolerance
            if ok and start is None:
                start = floor
            if not ok and start is not None:
                runs.append((start, previous if previous is not None else start))
                start = None
            previous = floor
        if start is not None:
            runs.append((start, previous if previous is not None else start))
        return runs

    @property
    def widest_run(self) -> tuple[float, float] | None:
        runs = self.feasible_runs()
        return max(runs, key=lambda run: run[1] - run[0]) if runs else None

    def summary(self) -> str:
        _best_floor, best_agreement = self.best
        ties = sum(1 for _, agreement in self.points if abs(agreement - best_agreement) < 1e-9)
        lines = [
            f"全局 floor 扫描（步长 {self.step}，{len(self.points)} 个取值，"
            f"算术固定为乘性修正）",
            f"  最高一致率={best_agreement:.3f}（{ties} 个取值并列，因此「最优值」不是结论）",
        ]
        widest = self.widest_run
        if widest is None:
            lines.append("  不存在能让全部用例通过的全局 floor（窗口为空）")
        else:
            width = widest[1] - widest[0]
            lines.append(
                f"  全通过窗口=[{widest[0]:.3f}, {widest[1]:.3f}]，宽度={width:.3f}"
                f"（共 {len(self.feasible_runs())} 段）"
            )
            recommended = self.recommended
            if recommended is not None:
                lines.append(f"  窗口中点（建议取值）={recommended:.4f}")
            if width < 0.05:
                lines.append(
                    "  **窗口过窄**：该参数上 ±0.01 的扰动就会翻转行为，"
                    "因此这个中点值不能当作精确结论，只能当作当前数据下的最稳选择。"
                )
        return "\n".join(lines)


def sweep_floor(
    cases: Sequence[RankCase],
    *,
    step: float = FLOOR_SWEEP_STEP,
    low: float = 0.0,
    high: float = 1.0,
) -> FloorSweep:
    """在全体用例上扫描全局 floor。

    用整数步进而不是浮点累加——后者会漂到 1.0000000000000007 而触发下限校验。
    """
    if low <= TIME_FLOOR <= high:
        # 采纳值会与扫描结果比较，因此它必须在网格上，否则窗口边界会被网格静默挪动。
        require_step_divides(step, TIME_FLOOR, context="全局 floor 扫描")
    sweep = FloorSweep(step=step)
    steps = int(round((high - low) / step)) + 1
    for index in range(steps):
        floor = round(min(low + index * step, high), 6)
        agreements = [
            pairwise_agreement(case.expect_order, ordering_with_floor(case, floor))
            for case in cases
        ]
        sweep.points.append((floor, sum(agreements) / len(agreements) if agreements else 1.0))
    return sweep


def case_floor_band(case: RankCase, *, step: float = FLOOR_SWEEP_STEP) -> tuple[float, float] | None:
    """单条用例的**可行 floor 区间**：落在其中的 floor 能让这条用例排序正确。

    这是比「二分类意图准确率」更有意义的中间指标：真正决定成败的是
    「选出的 floor 是否落在该用例的可行区间内」，而不是「意图标签猜得对不对」。
    """
    sweep = sweep_floor([case], step=step)
    return sweep.widest_run


def case_floor_bands(
    cases: Sequence[RankCase], *, step: float = FLOOR_SWEEP_STEP
) -> dict[str, tuple[float, float] | None]:
    return {case.id: case_floor_band(case, step=step) for case in cases}


def format_bands(
    cases: Sequence[RankCase],
    bands: Mapping[str, tuple[float, float] | None],
) -> str:
    """逐用例列出可行区间，以及当前标记词方案选出的 floor 是否落在区间内。

    这张表把「哪条用例在逼窄窗口」直接暴露出来——全局窗口的宽度由最紧的几条决定，
    因此优化目标应该是加宽这几条，而不是继续微调全局值。
    """
    lines = ["逐用例可行 floor 区间（★ = 当前标记词方案选出的 floor 落在区间外）"]
    for case in cases:
        band = bands.get(case.id)
        chosen = floor_for_sensitivity(time_sensitivity(case.query_text))
        if band is None:
            lines.append(f"  {case.id:<34} 无可行区间（任何 floor 都排不对）")
            continue
        width = band[1] - band[0]
        inside = band[0] - 1e-9 <= chosen <= band[1] + 1e-9
        mark = "" if inside else " ★"
        lines.append(
            f"  {case.id:<34} 区间=[{band[0]:.3f}, {band[1]:.3f}] 宽度={width:.3f}"
            f"  当前选择={chosen:.3f}{mark}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# floor 策略对比：把「默认值该取多少」当成可检验的假设
# ---------------------------------------------------------------------------

#: 策略 = 从用例推出 floor 的函数。策略里**不得使用 needs_fresh 标签**——
#: 那是评测标签，用它就等于作弊；策略只能看查询文本与用例本身。
FloorPolicy = Callable[["RankCase"], float]


def global_policy(floor: float) -> FloorPolicy:
    """全局固定 floor。"""
    return lambda case: floor


def _looks_status(case: "RankCase") -> bool:
    """从措辞推断是否属于「问最新状态」——这是生产环境唯一可得的信息。"""
    return time_sensitivity(case.query_text) >= SENSITIVITY_THRESHOLD


def intent_policy(default: float, status: float) -> FloorPolicy:
    """按措辞推断的意图硬切换 floor。"""
    return lambda case: status if _looks_status(case) else default


def sensitivity_policy(default: float, status: float = TIME_FLOOR_MIN) -> FloorPolicy:
    """按连续敏感度在 [status, default] 之间插值。"""
    span = default - status

    def policy(case: "RankCase") -> float:
        return default - span * time_sensitivity(case.query_text)

    return policy


#: 策略表里两个关键条目名。**用常量而不是字面量**：换采纳值时必须同时改这里，
#: 否则 `compare_policies` 会静默拿不到对照项（此前把 0.62 写死在名字里，换值即失效）。
INCUMBENT_POLICY_NAME = "全局 0.70（v1.1 旧默认）"
ADOPTED_POLICY_NAME = f"全局 {TIME_FLOOR}（现行采纳值）"


def default_policies() -> dict[str, FloorPolicy]:
    """待比较的策略集合：旧默认、现行采纳值、以及各类上下文方案。"""
    return {
        INCUMBENT_POLICY_NAME: global_policy(0.70),
        ADOPTED_POLICY_NAME: global_policy(TIME_FLOOR),
        "全局 0.55": global_policy(0.55),
        "全局 0.50": global_policy(0.50),
        "全局 0.35": global_policy(0.35),
        f"意图切换 默认0.70/状态{TIME_FLOOR_MIN}": intent_policy(0.70, TIME_FLOOR_MIN),
        f"意图切换 默认{TIME_FLOOR}/状态{TIME_FLOOR_MIN}": intent_policy(TIME_FLOOR, TIME_FLOOR_MIN),
        "连续插值 默认0.70": sensitivity_policy(0.70),
        f"连续插值 默认{TIME_FLOOR}": sensitivity_policy(TIME_FLOOR),
    }


def evaluate_policies(
    cases: Sequence[RankCase],
    policies: Mapping[str, FloorPolicy] | None = None,
) -> dict[str, float]:
    """返回每个策略在该用例集上的成对一致率。"""
    table = dict(policies or default_policies())
    return {
        name: (
            sum(pairwise_agreement(case.expect_order, ordering_with_floor(case, policy(case))) for case in cases)
            / len(cases)
            if cases
            else 1.0
        )
        for name, policy in table.items()
    }


def compare_policies(
    development: Sequence[RankCase],
    holdout: Sequence[RankCase],
    policies: Mapping[str, FloorPolicy] | None = None,
) -> str:
    """在开发集与留出集上并排比较策略——直接暴露「开发集上的优势是否为过拟合」。"""
    table = dict(policies or default_policies())
    dev = evaluate_policies(development, table)
    held = evaluate_policies(holdout, table)

    lines = [f"{'策略':<30} {'开发集':>8} {'留出集':>8} {'差值':>8}"]
    lines.append("-" * 60)
    for name in table:
        delta = held[name] - dev[name]
        lines.append(f"{name:<30} {dev[name]:>8.3f} {held[name]:>8.3f} {delta:>+8.3f}")

    best_held = max(held, key=lambda name: held[name])
    best_dev = max(dev, key=lambda name: dev[name])
    lines.append("")
    lines.append(f"开发集最优：{best_dev}（{dev[best_dev]:.3f}）")
    lines.append(f"留出集最优：{best_held}（{held[best_held]:.3f}）")
    if best_dev != best_held:
        lines.append("**两者不一致 → 开发集上的最优是过拟合的直接证据。**")

    incumbent = INCUMBENT_POLICY_NAME
    challenger = ADOPTED_POLICY_NAME
    if incumbent in held and challenger in held:
        verdict = "成立" if held[challenger] > held[incumbent] else "不成立"
        lines.append(
            f"检验「{TIME_FLOOR} 优于 0.70」：留出集 {held[challenger]:.3f} vs "
            f"{held[incumbent]:.3f} → **{verdict}**"
        )
    return "\n".join(lines)


def cross_validate_windows(
    development: Sequence[RankCase],
    holdout: Sequence[RankCase],
    *,
    step: float = FLOOR_SWEEP_STEP,
) -> str:
    """分别算开发集与留出集的可行窗口，给出交集、中点与**钉住窗口的用例**。

    这个函数存在的理由是一次具体的不严谨：只验证一个点、却宣称「留出集确认了它」。
    必须同时回答三个问题才算了事——
    1) 两边的窗口各多宽？（留出集常常远没那么有分辨力）
    2) 交集是什么？中点是哪？
    3) 谁在钉住边界？（窗口宽度往往只由极少数用例决定）
    """
    dev_sweep = sweep_floor(development, step=step)
    hold_sweep = sweep_floor(holdout, step=step)
    dev_run = dev_sweep.widest_run
    hold_run = hold_sweep.widest_run

    lines = ["开发集 / 留出集的可行 floor 窗口（分别计算）"]
    if dev_run is None or hold_run is None:
        lines.append("  至少一侧不存在全通过窗口，无法交叉验证")
        return "\n".join(lines)

    for name, run, count in (("开发集", dev_run, len(development)), ("留出集", hold_run, len(holdout))):
        lines.append(
            f"  {name}（{count} 条）窗口=[{run[0]:.3f}, {run[1]:.3f}]"
            f"  宽度={run[1] - run[0]:.3f}  中点={(run[0] + run[1]) / 2:.4f}"
        )

    low = max(dev_run[0], hold_run[0])
    high = min(dev_run[1], hold_run[1])
    if low >= high:
        lines.append("  两边窗口**无交集** → 该参数在此数据上不自洽，必须回到用例层面查原因")
        return "\n".join(lines)
    lines.append(
        f"  交集=[{low:.3f}, {high:.3f}]  宽度={high - low:.3f}  中点（建议取值）={(low + high) / 2:.4f}"
    )

    wider, narrower = (
        ("留出集", "开发集") if (hold_run[1] - hold_run[0]) > (dev_run[1] - dev_run[0]) else ("开发集", "留出集")
    )
    lines.append(
        f"  **{wider}的窗口更宽 → 起约束作用的是{narrower}。**"
        "留出集若明显更宽，它就只能否掉极端值，不能确认某个精确值。"
    )

    for name, cases, run in (("开发集", development, dev_run), ("留出集", holdout, hold_run)):
        bands = case_floor_bands(cases, step=step)
        for cid, band in bands.items():
            if not band:
                continue
            if abs(band[0] - run[0]) < 1e-9:
                lines.append(f"    {name}下界 {run[0]:.3f} ← {cid}")
            if abs(band[1] - run[1]) < 1e-9:
                lines.append(f"    {name}上界 {run[1]:.3f} ← {cid}")
    return "\n".join(lines)


#: 当前采纳的相关性权重（v1.0 去掉衰减项后重新归一化：0.45/0.75, 0.20/0.75, 0.10/0.75）。
CURRENT_RELEVANCE_WEIGHTS: tuple[float, float, float] = (0.60, 0.27, 0.13)


@dataclass
class WeightSweep:
    """在权重单纯形上扫描，看当前取值落在可行区域的何处。

    相关性权重有三个、和为 1，因此自由度是 2。扫描 `(w_sim, w_imp)`，
    `w_match = 1 − w_sim − w_imp`。
    """

    step: float
    floor: float
    #: `(w_sim, w_imp, w_match, 一致率)`
    points: list[tuple[float, float, float, float]] = field(default_factory=list)
    current: tuple[float, float, float, float] | None = None

    @property
    def total(self) -> int:
        return len(self.points)

    @property
    def feasible(self) -> list[tuple[float, float, float, float]]:
        return [point for point in self.points if point[3] >= 1.0 - 1e-9]

    @property
    def feasible_fraction(self) -> float:
        return len(self.feasible) / self.total if self.total else 1.0

    @property
    def bounding_box(self) -> dict[str, tuple[float, float]]:
        """可行区域在三个权重轴上的跨度。"""
        if not self.feasible:
            return {}
        return {
            name: (min(point[i] for point in self.feasible), max(point[i] for point in self.feasible))
            for i, name in enumerate(("sim", "importance", "match"))
        }

    @property
    def current_is_feasible(self) -> bool:
        return self.current is not None and self.current[3] >= 1.0 - 1e-9

    def summary(self) -> str:
        lines = [
            f"相关性权重扫描（步长 {self.step}，{self.total} 个点，f = {self.floor}）",
        ]
        if self.current is not None:
            w_sim, w_imp, w_match, agreement = self.current
            mark = "在可行区域内 ✓" if self.current_is_feasible else "**不可行**"
            lines.append(
                f"  当前权重 ({w_sim:.2f}, {w_imp:.2f}, {w_match:.2f}) → 一致率 {agreement:.3f}  {mark}"
            )
        lines.append(
            f"  可行点 {len(self.feasible)}/{self.total} = {self.feasible_fraction:.1%}"
        )
        box = self.bounding_box
        if box:
            lines.append(
                "  可行区域跨度："
                + "  ".join(f"{name}∈[{low:.2f}, {high:.2f}]" for name, (low, high) in box.items())
            )
        else:
            lines.append("  可行区域为空")
        return "\n".join(lines)


def _precompute_components(cases: Sequence[RankCase], floor: float):
    """把与权重无关的量（σ、重要度、μ、时间修正）预先算好，避免在单纯形上重复打分。"""
    prepared = []
    for case in cases:
        entries = []
        for candidate in case.candidates:
            entries.append(
                (
                    candidate.id,
                    sim_norm(candidate.cos),
                    candidate.importance / 10.0,
                    match_score(case.query_entities, candidate.entities, candidate.task_hit),
                    time_modifier(candidate.age_days, candidate.half_life_days, floor=floor),
                )
            )
        prepared.append((case.expect_order, entries))
    return prepared


def _agreement_at(prepared, weights: tuple[float, float, float]) -> float:
    w_sim, w_imp, w_match = weights
    scores = []
    for expected, entries in prepared:
        rows = [
            ((w_sim * sigma + w_imp * imp10 + w_match * mu) * modifier, modifier, cid)
            for cid, sigma, imp10, mu, modifier in entries
        ]
        rows.sort(key=lambda row: (-row[0], -row[1], row[2]))
        scores.append(pairwise_agreement(expected, [row[2] for row in rows]))
    return sum(scores) / len(scores) if scores else 1.0


def sweep_weights(
    cases: Sequence[RankCase] | None = None,
    *,
    step: float = 0.01,
    floor: float = TIME_FLOOR,
) -> WeightSweep:
    """在权重单纯形上扫描成对一致率。"""
    selected = list(cases) if cases is not None else load_rank_cases()
    prepared = _precompute_components(selected, floor)
    sweep = WeightSweep(step=step, floor=floor)

    steps = int(round(1.0 / step))
    for i in range(steps + 1):
        w_sim = round(i * step, 6)
        for j in range(steps + 1 - i):
            w_imp = round(j * step, 6)
            w_match = round(1.0 - w_sim - w_imp, 6)
            sweep.points.append(
                (w_sim, w_imp, w_match, _agreement_at(prepared, (w_sim, w_imp, w_match)))
            )

    w_sim, w_imp, w_match = CURRENT_RELEVANCE_WEIGHTS
    sweep.current = (w_sim, w_imp, w_match, _agreement_at(prepared, CURRENT_RELEVANCE_WEIGHTS))
    return sweep


#: v1.2-A 的层级分组：把 10 个层级压成 3 组，每组一个时间下限。
#: 分组而非逐层，是为了避免「21 条用例定 10 个参数」的过参数化。
TIER_GROUPS: dict[str, tuple[str, ...]] = {
    "high": ("T0", "T1", "T2"),
    "mid": ("T3", "T4", "T5"),
    "low": ("T6", "T7", "T8", "T9"),
}


def group_of_tier(tier: str) -> str:
    for name, tiers in TIER_GROUPS.items():
        if tier in tiers:
            return name
    raise ValueError(f"层级未分组：{tier!r}")


@dataclass
class GroupedFloorSweep:
    """按层级分组扫描时间下限：每个组一个 `f`。

    这是 `docs/v1.2-candidates.md` §5.2 所指「一族上确界」的可测形式：
    若各组的 `f` 可以不同，则反转上界变成 `1/f_{组(A)}`——**被顶掉的那张卡**所属组决定难度。
    """

    step: float
    groups: tuple[str, ...]
    #: `(f_high, f_mid, f_low, 一致率)`
    points: list[tuple[float, ...]] = field(default_factory=list)
    current: tuple[float, ...] | None = None

    @property
    def total(self) -> int:
        return len(self.points)

    @property
    def feasible(self) -> list[tuple[float, ...]]:
        return [point for point in self.points if point[-1] >= 1.0 - 1e-9]

    @property
    def feasible_fraction(self) -> float:
        return len(self.feasible) / self.total if self.total else 1.0

    @property
    def bounding_box(self) -> dict[str, tuple[float, float]]:
        if not self.feasible:
            return {}
        return {
            name: (min(p[i] for p in self.feasible), max(p[i] for p in self.feasible))
            for i, name in enumerate(self.groups)
        }

    @property
    def current_is_feasible(self) -> bool:
        return self.current is not None and self.current[-1] >= 1.0 - 1e-9

    def marginal_group(self) -> str | None:
        """可行区域最窄的那个组——它才是真正的约束。"""
        box = self.bounding_box
        if not box:
            return None
        return min(box, key=lambda name: box[name][1] - box[name][0])

    def summary(self) -> str:
        lines = [
            f"分组 floor 扫描（{len(self.groups)} 组：{', '.join(self.groups)}，"
            f"步长 {self.step}，{self.total} 个点）",
        ]
        if self.current is not None:
            values = ", ".join(f"{name}={self.current[i]:.2f}" for i, name in enumerate(self.groups))
            mark = "在可行区域内 ✓" if self.current_is_feasible else "**不可行**"
            lines.append(f"  单一 f 解（全组同值）：{values} → 一致率 {self.current[-1]:.3f}  {mark}")
        lines.append(f"  可行点 {len(self.feasible)}/{self.total} = {self.feasible_fraction:.2%}")
        box = self.bounding_box
        if box:
            lines.append(
                "  可行区间："
                + "  ".join(f"{name}∈[{low:.2f}, {high:.2f}]" for name, (low, high) in box.items())
            )
            marginal = self.marginal_group()
            if marginal:
                low, high = box[marginal]
                lines.append(f"  最窄约束：{marginal}（宽 {high - low:.2f}）")
        else:
            lines.append("  可行区域为空：不存在任何分组取值能同时满足全部用例")
        return "\n".join(lines)


def _precompute_decay_components(cases: Sequence[RankCase]):
    """预计算与 `f` 无关的量：σ、重要度、μ、δ、所属组。"""
    prepared = []
    for case in cases:
        entries = []
        for candidate in case.candidates:
            half_life = candidate.half_life_days
            delta = 1.0 if half_life <= 0 else 0.5 ** (candidate.age_days / half_life)
            entries.append(
                (
                    candidate.id,
                    sim_norm(candidate.cos),
                    candidate.importance / 10.0,
                    match_score(case.query_entities, candidate.entities, candidate.task_hit),
                    delta,
                    group_of_tier(candidate.tier),
                )
            )
        prepared.append((case.expect_order, entries))
    return prepared


def _agreement_grouped(prepared, weights, floors: Mapping[str, float]) -> float:
    w_sim, w_imp, w_match = weights
    scores = []
    for expected, entries in prepared:
        rows = []
        for cid, sigma, imp10, mu, delta, group in entries:
            relevance = w_sim * sigma + w_imp * imp10 + w_match * mu
            floor = floors[group]
            modifier = floor + (1.0 - floor) * delta
            rows.append((relevance * modifier, modifier, cid))
        rows.sort(key=lambda row: (-row[0], -row[1], row[2]))
        scores.append(pairwise_agreement(expected, [row[2] for row in rows]))
    return sum(scores) / len(scores) if scores else 1.0


def sweep_grouped_floor(
    cases: Sequence[RankCase] | None = None,
    *,
    step: float = 0.0395,
    groups: Mapping[str, tuple[str, ...]] | None = None,
    ranges: Mapping[str, tuple[float, float]] | None = None,
) -> GroupedFloorSweep:
    """按分组扫描时间下限。单一 `f` 解（全组同值）作为对照一并记录。

    ``step`` 默认 `0.0395`：它**整除采纳下限 `0.5925`**（`0.5925/0.0395 = 15`），
    且给出 `26³ = 17576` 个点的**全局**粗扫——粗到能瞬间跑完，细到能圈出可行区。
    细扫请传 ``ranges`` 把范围收到粗扫圈出的盒子里，并另选一个同样整除的细步长
    （例如 `0.0075`，`0.5925/0.0075 = 79`）。三维网格代价是 ``O(1/step³)``，全空间细扫不可行。

    ``step`` 与 ``ranges`` 都必须满足整除律：采纳值（全组同值）若落在扫描范围内
    却不在网格上，网格会静默漏掉它——本仓库为此错过两次。
    """
    selected = list(cases) if cases is not None else load_rank_cases()
    table = dict(groups or TIER_GROUPS)
    weights = (
        RELEVANCE_WEIGHTS["sim"],
        RELEVANCE_WEIGHTS["importance"],
        RELEVANCE_WEIGHTS["match"],
    )
    prepared = _precompute_decay_components(selected)
    names = tuple(table)
    windows = [
        ranges.get(name, (0.0, 1.0)) if ranges else (0.0, 1.0) for name in names
    ]
    if all(low <= TIME_FLOOR <= high for low, high in windows):
        # 单一 f 解（全组同值 = TIME_FLOOR）会与扫描结果比较，因此它必须在网格上：
        # 否则网格静默漏掉它，得出「单一解不在交集内」的假结论。
        require_step_divides(step, TIME_FLOOR, context="分组 floor 扫描")
    sweep = GroupedFloorSweep(step=step, groups=names)

    axes: list[list[float]] = []
    for name, (low, high) in zip(names, windows):
        # **必须对齐到全局步长网格**：若从 `low` 起算（`low + i*step`），当 `low` 不是
        # 步长的整数倍时，网格整体偏移，`TIME_FLOOR` 这类「步长整数倍」的取值就会漏掉——
        # 整除检查过了也没用。实测踩到过：`ranges` 细扫里采纳值从可行集中消失。
        first = -(-low // step)  # ceil
        last = high // step  # floor
        axes.append([round(index * step, 6) for index in range(int(first), int(last) + 1)])

    for first in axes[0]:
        for second in axes[1]:
            for third in axes[2]:
                triple = (first, second, third)
                sweep.points.append(
                    (*triple, _agreement_grouped(prepared, weights, dict(zip(names, triple))))
                )

    sweep.current = (
        *[TIME_FLOOR] * len(names),
        _agreement_grouped(prepared, weights, {name: TIME_FLOOR for name in names}),
    )
    return sweep


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    report = run_rank_eval(load_rank_cases(args[0] if args else None))
    print(report.summary())
    # 实验本身不设通过阈值：它产出的是证据，不是关卡。阈值由人看完数据后决定。
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
