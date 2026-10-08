"""网络赋值准确率评测。

存在理由：v1.2 提案要用 `network`（world / experience / opinion / observation）门控时间下限 `f`，
给出的区间跨度极大（observation 0.20–0.45 对 opinion 0.75–0.90）。这意味着
**`f` 的精度直接由 network 的赋值精度决定**——而在此之前，仓库里没有一条 network 真值，
这个准确率从未被测量过。

这与 v1.1 否掉「按查询意图切换 f」的理由是同一类问题：被门控的类别若是**推断出来的标签**，
它的错误率就变成 `f` 的错误率。区别在于意图要从查询文本推断，而 network 是卡的属性——
但「是卡的属性」不等于「赋值可靠」，本模块就是把后者测出来。
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from .compress import NETWORKS, baseline_network, classify_tier

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_NETWORK_PATH = ROOT / "eval" / "network" / "cases.json"


@dataclass(frozen=True)
class NetworkCase:
    id: str
    text: str
    expected: str
    split: str = "dev"
    boundary: bool = False
    #: 可辩护的备选标签。非空表示该条不止一个答案可接受（宽松指标用）。
    alternates: tuple[str, ...] = ()
    #: 四张网络里没有正确类别（schema 缺口）。这类用例任何人都答不对，用于算上限。
    unattainable: bool = False
    note: str = ""

    @property
    def acceptable(self) -> tuple[str, ...]:
        return (self.expected, *self.alternates)

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "NetworkCase":
        return cls(
            id=str(payload["id"]),
            text=str(payload["text"]),
            expected=str(payload["expected"]),
            split=str(payload.get("split", "dev")),
            boundary=bool(payload.get("boundary", False)),
            alternates=tuple(str(item) for item in payload.get("alternates", ())),
            unattainable=bool(payload.get("unattainable", False)),
            note=str(payload.get("note", "")),
        )


def load_network_cases(
    path: str | pathlib.Path | None = None, *, split: str | None = None
) -> list[NetworkCase]:
    target = pathlib.Path(path) if path is not None else DEFAULT_NETWORK_PATH
    with target.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    cases = [NetworkCase.from_json(item) for item in payload["cases"]]
    return [case for case in cases if split is None or case.split == split]


def classify(text: str, *, tier: str | None = None) -> str:
    """基线网络赋值。`tier` 缺省时按 `classify_tier` 推出，与流水线一致。"""
    resolved = classify_tier(text) if tier is None else tier
    return baseline_network(text, resolved)


@dataclass(frozen=True)
class NetworkOutcome:
    case_id: str
    text: str
    expected: str
    predicted: str
    boundary: bool
    acceptable: tuple[str, ...] = ()
    unattainable: bool = False

    @property
    def ok(self) -> bool:
        """严格判定：必须等于 `expected`。"""
        return self.expected == self.predicted

    @property
    def lenient_ok(self) -> bool:
        """宽松判定：命中任一可辩护标签即可（`alternates` 为空时等同严格）。"""
        return self.predicted in self.acceptable


@dataclass
class NetworkReport:
    outcomes: list[NetworkOutcome] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.outcomes)

    @property
    def strict_correct(self) -> int:
        return sum(1 for o in self.outcomes if o.ok)

    @property
    def lenient_correct(self) -> int:
        return sum(1 for o in self.outcomes if o.lenient_ok)

    @property
    def accuracy(self) -> float:
        return self.strict_correct / self.total if self.total else 1.0

    @property
    def lenient_accuracy(self) -> float:
        """对模糊用例给出备选宽容度后的准确率——评判 schema 难度时应看这个。"""
        return self.lenient_correct / self.total if self.total else 1.0

    @property
    def unattainable_count(self) -> int:
        return sum(1 for o in self.outcomes if o.unattainable)

    @property
    def attainable_total(self) -> int:
        """有正确答案的用例数（排除 schema 装不下的那些）。"""
        return self.total - self.unattainable_count

    @property
    def attainable_correct(self) -> int:
        """在**有答案**的用例上答对的数——这才是分类器质量的度量。

        `unattainable` 的用例必须排除：它们没有正确类别，猜中 `expected` 不算对。
        若把它们计入分子，就会出现「准确率超过上限」的自相矛盾。
        """
        return sum(1 for o in self.outcomes if o.lenient_ok and not o.unattainable)

    @property
    def attainable_accuracy(self) -> float:
        return self.attainable_correct / self.attainable_total if self.attainable_total else 1.0

    @property
    def ceiling(self) -> float:
        """schema 上限：任何分类器都不可能超过的准确率上界。

        等于「有答案的用例占比」——无正确类别的那部分谁都无法答对。
        """
        if not self.total:
            return 1.0
        return self.attainable_total / self.total

    @property
    def boundary_accuracy(self) -> float:
        subset = [o for o in self.outcomes if o.boundary]
        return sum(1 for o in subset if o.ok) / len(subset) if subset else 1.0

    @property
    def clear_accuracy(self) -> float:
        """排除边界用例后的准确率——边界用例反映的是分类法本身的模糊，不是规则的水平。"""
        subset = [o for o in self.outcomes if not o.boundary]
        return sum(1 for o in subset if o.ok) / len(subset) if subset else 1.0

    @property
    def clear_lenient_accuracy(self) -> float:
        subset = [o for o in self.outcomes if not o.boundary]
        return sum(1 for o in subset if o.lenient_ok) / len(subset) if subset else 1.0

    def confusion(self) -> dict[str, dict[str, int]]:
        """`confusion[真实][预测] = 条数`。"""
        table = {expected: {predicted: 0 for predicted in NETWORKS} for expected in NETWORKS}
        for outcome in self.outcomes:
            table[outcome.expected][outcome.predicted] += 1
        return table

    def recall(self, network: str) -> float:
        subset = [o for o in self.outcomes if o.expected == network]
        if not subset:
            return 1.0
        return sum(1 for o in subset if o.predicted == network) / len(subset)

    def precision(self, network: str) -> float:
        subset = [o for o in self.outcomes if o.predicted == network]
        if not subset:
            return 0.0
        return sum(1 for o in subset if o.expected == network) / len(subset)

    @property
    def unreachable(self) -> list[str]:
        """从未被预测出来的类别。它们对应的 `f` 配置在基线上无法生效。"""
        return [n for n in NETWORKS if self.precision(n) == 0.0 and self.recall(n) == 0.0]

    @property
    def misses(self) -> list[NetworkOutcome]:
        return [o for o in self.outcomes if not o.ok]

    def summary(self) -> str:
        lines = [
            f"网络赋值准确率：{self.accuracy:.3f}（{sum(1 for o in self.outcomes if o.ok)}/{self.total}）",
            f"  非边界用例：{self.clear_accuracy:.3f}    边界用例：{self.boundary_accuracy:.3f}",
        ]
        for network in NETWORKS:
            lines.append(
                f"  {network:<12} recall={self.recall(network):.3f}  "
                f"precision={self.precision(network):.3f}"
            )
        if self.unreachable:
            lines.append(f"  **从未被预测出的类别：{', '.join(self.unreachable)}**"
                         "——给它们配置的 `f` 在基线上不生效")
        for outcome in self.misses:
            mark = "（边界）" if outcome.boundary else ""
            lines.append(f"    ✗ {outcome.case_id} 期望 {outcome.expected} 实际 {outcome.predicted}{mark}")
        return "\n".join(lines)


def evaluate_network(
    cases: Sequence[NetworkCase] | None = None,
    classifier: Callable[[str], str] | None = None,
    *,
    split: str | None = None,
) -> NetworkReport:
    selected = list(cases) if cases is not None else load_network_cases(split=split)
    engine = classifier or classify
    return NetworkReport(
        outcomes=[
            NetworkOutcome(
                case_id=case.id,
                text=case.text,
                expected=case.expected,
                predicted=engine(case.text),
                boundary=case.boundary,
                acceptable=case.acceptable,
                unattainable=case.unattainable,
            )
            for case in selected
        ]
    )


#: 待比较的两套规则。`v1` 是基线（依赖 tier、永不输出 observation）；`v2` 在 network_rules。
def rule_variants() -> dict[str, Callable[[str], str]]:
    from .network_rules import classify_network_v2

    return {"v1 基线（3 分支，依赖 tier）": classify, "v2（4 类词汇，不依赖 tier）": classify_network_v2}


def compare_network_rules() -> str:
    """核心诊断：规则水平 vs schema 上限。

    判定：
    - 若 v2 在 holdout 上接近 `ceiling` → 瓶颈是 schema（分类法本身难）；
    - 若 v2 在 holdout 上显著低于 `ceiling` → 瓶颈是分类器（规则/模型不够强）。
    """
    lines = [
        f"{'规则':<34}{'dev 严格':>10}{'dev 可达':>10}{'holdout 严格':>13}{'holdout 可达':>13}",
        "-" * 82,
    ]
    reports: dict[tuple[str, str], NetworkReport] = {}
    for name, engine in rule_variants().items():
        for split in ("dev", "holdout"):
            reports[(name, split)] = evaluate_network(
                load_network_cases(split=split), engine
            )
        dev, held = reports[(name, "dev")], reports[(name, "holdout")]
        lines.append(
            f"{name:<34}{dev.accuracy:>10.3f}{dev.attainable_accuracy:>10.3f}"
            f"{held.accuracy:>13.3f}{held.attainable_accuracy:>13.3f}"
        )

    sample = reports[("v1 基线（3 分支，依赖 tier）", "dev")]
    held_sample = reports[("v1 基线（3 分支，依赖 tier）", "holdout")]
    lines += [
        "",
        "「严格」= 必须等于唯一标注，「可达」= 只在有正确答案的用例上统计（分类器质量）。",
        f"schema 上限：dev {sample.ceiling:.3f}（{sample.unattainable_count}/{sample.total} 条无正确类别）"
        f"    holdout {held_sample.ceiling:.3f}（{held_sample.unattainable_count}/{held_sample.total} 条）",
    ]

    v2_name = "v2（4 类词汇，不依赖 tier）"
    v2_held = reports[(v2_name, "holdout")]
    gap = 1.0 - v2_held.attainable_accuracy
    lines.append(f"v2 在 holdout 的可达用例上的失分：{gap:.3f}")
    if gap <= 0.10:
        lines.append(
            "**判定：瓶颈是 schema，不是分类器。** 有答案的用例几乎全部答对，"
            "剩余损失来自「四张网络装不下」的那部分——加规则或换模型都补不上。"
        )
    else:
        lines.append("**判定：瓶颈是分类器。** 有答案的用例仍答错不少，规则或模型可以继续改进。")
    return "\n".join(lines)


def format_confusion(report: NetworkReport) -> str:
    table = report.confusion()
    header = "真实＼预测".ljust(14) + "".join(f"{n:>12}" for n in NETWORKS)
    lines = [header, "-" * len(header)]
    for expected in NETWORKS:
        row = f"{expected:<14}" + "".join(f"{table[expected][p]:>12}" for p in NETWORKS)
        lines.append(row)
    return "\n".join(lines)
