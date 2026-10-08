"""v1.2-A 实测：分组时间下限（每个层级组一个 `f`）是否可行。

    python tools/run_grouped_floor.py

对照 v1.1 的单一 `f`（见 `run_rank_eval.py` 的全通过窗口）。本工具测的是
「一族上确界」（每组一个 `1/f`）能否放宽约束，以及**是哪一组在压住可行区**。

## 为什么是两阶段

三维网格代价是 `O(1/step³)`，全空间细扫不可行（`0.005` 全域是 `201³ ≈ 810 万`点）。
而且步长**必须整除采纳下限** `0.605`（否则网格会静默漏掉采纳值，
`rank_eval.require_step_divides` 会直接抛错）。两个条件同时满足的步长里，
`0.055` 给出 `19³ = 6859` 点的**全局**粗扫（瞬间跑完，足够圈出可行区），
`0.005` 用于粗扫盒子内的细扫。两阶段都合法、都不漏点。

## 读数须知

- 交集在**同一张全局粗网格**上计算（两侧可比，是保守的内近似）；
- 逐轴范围来自粗网格，精度只有 `0.055`；细扫只在盒子内提高分辨率；
- 「非盒形占比」把外接盒里的网格点数与实际可行点数相比，**依赖细扫范围**，
  因此它是该次扫描的读数，不是可行区的固有性质。
"""

from __future__ import annotations

import itertools
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moonshadow.rank_eval import (  # noqa: E402
    TIME_FLOOR,
    load_holdout_cases,
    load_rank_cases,
    sweep_grouped_floor,
)

#: 全局粗扫步长。必须整除 `TIME_FLOOR`（`0.5925/0.0395 = 15`），给出 `26³ = 17576` 点。
COARSE_STEP = 0.0395
#: 盒内细扫步长。同样必须整除（`0.5925/0.0075 = 79`）。
FINE_STEP = 0.0075


def feasible_triples(sweep) -> set[tuple[float, ...]]:
    return {tuple(round(value, 4) for value in point[:3]) for point in sweep.feasible}


def box_of(triples, groups) -> dict[str, tuple[float, float]]:
    return {
        name: (min(p[i] for p in triples), max(p[i] for p in triples))
        for i, name in enumerate(groups)
    }


def report_set(label: str, cases, groups) -> None:
    coarse = sweep_grouped_floor(cases, step=COARSE_STEP)
    box = coarse.bounding_box
    print(f"=== {label}（{len(cases)} 条） ===")
    print(coarse.summary())
    if not box:
        print()
        return
    pad = COARSE_STEP
    ranges = {
        name: (max(0.0, low - pad), min(1.0, high + pad)) for name, (low, high) in box.items()
    }
    fine = sweep_grouped_floor(cases, step=FINE_STEP, ranges=ranges)
    triple = tuple(round(TIME_FLOOR, 4) for _ in groups)
    print(f"  细扫（步长 {FINE_STEP}，范围 "
          + "  ".join(f"{n}∈[{lo:.3f}, {hi:.3f}]" for n, (lo, hi) in ranges.items()) + "）")
    print(f"    可行点 {len(fine.feasible)}/{fine.total}；采纳值在细扫可行集内："
          f"{triple in feasible_triples(fine)}")
    print()


def main() -> int:
    dev_cases, held_cases = load_rank_cases(), load_holdout_cases()
    groups = ("high", "mid", "low")
    print(f"采纳下限 = {TIME_FLOOR}；粗扫步长 = {COARSE_STEP}（必须整除），"
          f"细扫步长 = {FINE_STEP}（必须整除）\n")
    report_set("开发集", dev_cases, groups)
    report_set("留出集", held_cases, groups)

    dev = sweep_grouped_floor(dev_cases, step=COARSE_STEP)
    held = sweep_grouped_floor(held_cases, step=COARSE_STEP)
    dev_ok, held_ok = feasible_triples(dev), feasible_triples(held)
    shared = dev_ok & held_ok
    print(f"=== 两集合交集（同一张全局粗网格 {COARSE_STEP}） ===")
    print(f"  开发集可行 {len(dev_ok)} 点 ∩ 留出集可行 {len(held_ok)} 点 = {len(shared)} 点")
    if not shared:
        print("  **交集为空**：不存在任何分组取值能同时满足两个集合")
        return 0
    box = box_of(shared, groups)
    volume = 1
    for low, high in box.values():
        volume *= int(round((high - low) / COARSE_STEP)) + 1
    axes = [sorted({p[i] for p in shared}) for i in range(len(groups))]
    inside = sum(1 for combo in itertools.product(*axes) if combo in shared)
    adopted = tuple(round(TIME_FLOOR, 4) for _ in groups)
    print("  交集的逐轴范围：" + "  ".join(
        f"{name}∈[{low:.2f}, {high:.2f}]（宽 {high - low:.2f}）" for name, (low, high) in box.items()
    ))
    print(f"  外接盒体积 {volume} 点，实际 {inside} 点 → "
          f"{'完整盒子（逐轴区间可自由组合）' if volume == inside else f'非盒形（占比 {inside / volume:.1%}）'}")
    print(f"  采纳值 (f={TIME_FLOOR} ×3) 在交集中：{adopted in shared}")
    binding = min(box, key=lambda name: box[name][1] - box[name][0])
    print(f"  最窄约束：{binding}（宽 {box[binding][1] - box[binding][0]:.2f}）")
    print("  注：交集在粗网格上计算，是保守的内近似；逐轴精度只有 "
          f"{COARSE_STEP}。细扫只提高盒内分辨率，不改变交集范围。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
