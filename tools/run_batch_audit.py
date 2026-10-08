"""批次验收：判定一批新增用例能否合入（v1.2 路线图 §7.1 的 R1 / R2 / R3）。

    python tools/run_batch_audit.py 12 13 14 15 16 17 18 19 20 21

参数是**该批新增用例**的 id 或 id 前缀。工具不猜批次——「这一批是哪几条」
必须由调用者给出；猜错了，判定量就没有意义。

判定量（全部在**当前采纳参数**下计算：乘性修正 + 全局下限 `TIME_FLOOR`）：

- **相邻最小分差**：该用例期望次序中相邻两张候选的分数差的最小值。
  只有这个值小的用例才可能被参数改动翻转次序，因此它是「边界用例」的判据。
- **可行下限区间**：让该用例排序正确的 `f` 区间（`case_floor_band`）。
  宽度为 `1.0` 表示**任何** `f` 都通过，即该用例对 `f` 不构成约束。

三条规则：

| 编号 | 规则 | 判定 |
|---|---|---|
| R1 | 边界用例（相邻最小分差 ≤ 0.05）不少于 3 条 | 不满足 → 拒绝合入 |
| R2 | 约束产率（宽度 < 1.0 的用例占比）> 0 | 等于 0 → 拒绝合入 |
| R3 | 不得以「正例数达到 80–100」为唯一目标 | 本工具只报告正例数，不参与判定 |

**退出码**：R1、R2 均通过为 `0`，否则为 `1`（该批应被拒绝合入）。
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moonshadow.rank_eval import (  # noqa: E402
    SCHEME_MULTIPLICATIVE,
    TIME_FLOOR,
    load_holdout_cases,
    load_rank_cases,
    case_floor_band,
    score_of,
)

#: 切分冻结清单。用它判定「哪些是冻结后新增的用例」。
MANIFEST = ROOT / "eval" / "split-manifest.json"

#: 边界用例的分差上界。与 `docs/v1.2-roadmap.md` §7.1 的 R1 必须一致。
BOUNDARY_GAP = 0.05
#: 一批中最少要有的边界用例数。同上，与 R1 必须一致。
MIN_BOUNDARY = 3
#: 完全无约束的可行区间宽度（`f` 全域）。
FULL_WIDTH = 1.0


@dataclasses.dataclass(frozen=True)
class Row:
    """一条用例的判定量。"""

    case_id: str
    gap: float | None
    band: tuple[float, float] | None
    width: float | None
    #: 该用例在**当前采纳下限**下排序是否正确。
    #: 缺了它，R1 会把「排序已经错了、只是分差小」的用例算成边界用例——
    #: 分差是绝对值，不含方向。这是实测踩到的坑，见 roadmap §7.1。
    passes_at_adopted: bool = True

    @property
    def is_boundary(self) -> bool:
        """边界用例 = 分差小 **且** 当前采纳下限下排序正确。

        只满足前者的是**失败用例**，不是边界用例。
        """
        return (
            self.passes_at_adopted
            and self.gap is not None
            and self.gap <= BOUNDARY_GAP
        )

    @property
    def is_constraining(self) -> bool:
        # 宽度为 None 表示「没有任何 f 能让它通过」——它把可行区收窄成空集，当然算约束。
        return self.width is None or self.width < FULL_WIDTH - 1e-9


@dataclasses.dataclass(frozen=True)
class Verdict:
    rows: list[Row]
    failures: list[str]

    @property
    def boundary(self) -> list[str]:
        return [row.case_id for row in self.rows if row.is_boundary]

    @property
    def constrained(self) -> list[str]:
        return [row.case_id for row in self.rows if row.is_constraining]

    @property
    def yield_rate(self) -> float:
        return len(self.constrained) / len(self.rows) if self.rows else 0.0

    @property
    def passed(self) -> bool:
        return not self.failures


def adjacent_min_gap(case) -> float | None:
    """期望次序中相邻候选对的分数差的最小值。

    单张正例的用例没有相邻对，返回 ``None``——它不可能是边界用例。
    """
    scores = {c.id: score_of(case, c, SCHEME_MULTIPLICATIVE) for c in case.candidates}
    ordered = [scores[name] for name in case.expect_order if name in scores]
    if len(ordered) < 2:
        return None
    return min(abs(a - b) for a, b in zip(ordered, ordered[1:]))


def resolve(cases, keys: list[str]):
    """把 id 前缀解析成用例；未知前缀直接报错（不静默忽略）。"""
    picked, unknown = [], []
    for key in keys:
        hits = [case for case in cases if case.id == key] or [
            case for case in cases if case.id.startswith(key)
        ]
        if not hits:
            unknown.append(key)
        picked.extend(hits)
    return picked, unknown


def default_batch(dev_cases) -> tuple[list, str]:
    """不带参数时的批次：**冻结后新增的开发用例**。

    判定「这一批新用例能不能合入」时，手输 id 容易漏一条或多带一条；
    而「哪些是冻结之后加进来的」有确切答案——切分清单里的 `dev` 列表就是冻结时的快照。
    于是默认行为 = 当前开发集 − 清单里的 dev。留出集永不参与（冻结）。
    """
    if not MANIFEST.exists():
        return [], "缺少 eval/split-manifest.json，无法判定「冻结后新增」；请显式给出用例 id"
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    frozen = set(manifest.get("dev", []))
    added = [case for case in dev_cases if case.id not in frozen]
    note = f"默认批次 = 冻结后新增的开发用例（清单记录 dev {len(frozen)} 条）"
    return added, note


def evaluate(batch) -> Verdict:
    """算出一批用例的 R1 / R2 判定量。**纯函数**：不打印、不读文件。"""
    rows = []
    for case in batch:
        band = case_floor_band(case)
        passes = band is not None and band[0] - 1e-9 <= TIME_FLOOR <= band[1] + 1e-9
        rows.append(
            Row(
                case_id=case.id,
                gap=adjacent_min_gap(case),
                band=band,
                width=None if band is None else band[1] - band[0],
                passes_at_adopted=passes,
            )
        )
    failures: list[str] = []
    broken = [row.case_id for row in rows if not row.passes_at_adopted]
    if broken:
        # 排序在采纳下限下就已经错了：既不是「边界用例」也不是「无约束用例」，
        # 而是**失败用例**。必须单独报，否则会被 R1/R2 的统计掩盖。
        failures.append(
            f"R1（{len(broken)} 条用例在采纳下限下排序错误，不是边界用例：{', '.join(broken)}）"
        )
    verdict = Verdict(rows=rows, failures=failures)
    if len(verdict.boundary) < MIN_BOUNDARY:
        failures.append(f"R1（边界用例 {len(verdict.boundary)} 条 < {MIN_BOUNDARY} 条）")
    if verdict.yield_rate <= 0:
        failures.append("R2（约束产率为 0）")
    return Verdict(rows=rows, failures=failures)


def audit(keys: list[str]) -> int:
    dev = load_rank_cases()
    held = load_holdout_cases()
    split_of = {case.id: "开发集" for case in dev}
    split_of.update({case.id: "留出集" for case in held})
    note = ""
    if keys:
        batch, unknown = resolve(dev + held, keys)
    else:
        # 不带参数 = 判定「冻结后新增的开发用例」这一批（见 default_batch）。
        batch, note = default_batch(dev)
        unknown = []
    if unknown:
        print(f"未知用例 id/前缀：{', '.join(unknown)}")
        return 1
    if not batch:
        print("批次为空：没有可判定的新用例。")
        print(note or "用法：python tools/run_batch_audit.py <id 前缀> [更多…]")
        return 1
    if note:
        print(note)

    verdict = evaluate(batch)
    # 同一批用例可能被切分到两侧；两个集合上的读数**分开报**，否则会把
    # 「开发集侧不可合入」读成「整批通过」。
    dev_rows = [row for row in verdict.rows if split_of[row.case_id] == "开发集"]
    held_rows = [row for row in verdict.rows if split_of[row.case_id] == "留出集"]

    print(f"批次：{len(batch)} 条用例（开发集 {len(dev_rows)} / 留出集 {len(held_rows)}）")
    print(
        f"{'用例':38s} {'集合':6s} {'相邻最小分差':>10s} {'可行下限区间':>20s} {'宽度':>7s}"
        f"  通过  边界  约束"
    )
    for row in verdict.rows:
        gap_text = "—（无相邻对）" if row.gap is None else f"{row.gap:.4f}"
        band_text = (
            "**无可行下限**"
            if row.band is None
            else f"[{row.band[0]:.3f}, {row.band[1]:.3f}]"
        )
        width_text = "—" if row.width is None else f"{row.width:.3f}"
        print(
            f"{row.case_id:38s} {split_of[row.case_id]:6s} {gap_text:>10s} {band_text:>20s}"
            f" {width_text:>7s}   {'✓' if row.passes_at_adopted else '✗'}     "
            f"{'✓' if row.is_boundary else ' '}     {'✓' if row.is_constraining else ' '}"
        )

    positives = sum(len(case.candidates) for case in dev)
    print()
    print(
        f"R1 边界用例（分差 ≤ {BOUNDARY_GAP}）：{len(verdict.boundary)} / {len(verdict.rows)}"
        f"    需要 ≥ {MIN_BOUNDARY}    → {'通过' if len(verdict.boundary) >= MIN_BOUNDARY else '不通过'}"
    )
    print(
        f"R2 约束产率：{verdict.yield_rate:.3f}（{len(verdict.constrained)} / {len(verdict.rows)}）"
        f"                    需要 > 0      → {'通过' if verdict.yield_rate > 0 else '不通过'}"
    )
    print(f"R3 开发集正例数：{positives}（仅供记录，按规则不参与判定）")
    for label, rows in (("开发集", dev_rows), ("留出集", held_rows)):
        if not rows:
            continue
        boundary = [row for row in rows if row.is_boundary]
        constrained = [row for row in rows if row.is_constraining]
        print(
            f"  · {label}侧：边界 {len(boundary)} / {len(rows)}，"
            f"约束 {len(constrained)} / {len(rows)}"
            f"（产率 {len(constrained) / len(rows):.3f}）"
        )
    print()

    if verdict.failures:
        print(f"判定：**拒绝合入**——{'；'.join(verdict.failures)}")
        if dev_rows and not held_rows:
            return 1
        if dev_rows and held_rows:
            dev_verdict = evaluate([c for c in batch if split_of[c.id] == "开发集"])
            if not dev_verdict.passed:
                print("       注意：该批在**开发集侧**同样不达标——"
                      f"{'；'.join(dev_verdict.failures)}")
        return 1
    print("判定：可以合入（R1、R2 均通过）")
    return 0


def main(argv: list[str] | None = None) -> int:
    keys = list(sys.argv[1:] if argv is None else argv)
    return audit(keys)


if __name__ == "__main__":
    raise SystemExit(main())
