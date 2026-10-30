"""分项计时：把端到端 GPU 时间拆成「转换 / H2D / kernel / D2H」。

依赖：cupy（连带 numpy）。这是 **GPU 分支专用**工具，主分支不依赖它。

用途：检验「双态混合分布来自 pinned vs pageable 两条拷贝路径」这个假设。
判据很直接——如果 T_h2d 本身是双峰的，假设有救；如果 T_h2d 很稳而**转换**
（Python list -> ndarray）才是抖动来源，那模型里那个变量名就写错了。

    python .tmp/transfer_breakdown.py [--n 50000] [--dim 768] [--reps 12]
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from array import array
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import cupy as cp  # noqa: E402
except Exception as exc:  # pragma: no cover - 环境相关
    print(f"本工具需要 cupy（只在 GPU 分支上有意义）：{type(exc).__name__}: {exc}")
    print("装法见 requirements-gpu.txt；没有显卡时它的用途（拆 GPU 时间）本身就不成立。")
    print("对照：tools/benchmark_gpu.py 在没有 GPU 时会回退到纯 Python 并照常给出 CPU 读数。")
    raise SystemExit(2)

_LCG_A, _LCG_C, _LCG_M = 1664525, 1013904223, 2**32


def vectors(count: int, dim: int, seed: int) -> list[array]:
    state = seed
    out: list[array] = []
    for _ in range(count):
        row = array("d", bytes(8 * dim))
        for j in range(dim):
            state = (_LCG_A * state + _LCG_C) % _LCG_M
            row[j] = (state / _LCG_M) * 2.0 - 1.0
        out.append(row)
    return out


def stat(name: str, values: list[float]) -> None:
    lo, hi = min(values), max(values)
    med = statistics.median(values)
    print(f"  {name:<26}{med*1000:>8.1f} ms   min {lo*1000:7.1f}   max {hi*1000:7.1f}"
          f"   抖动 {(hi - lo) / med:6.1%}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50_000)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--reps", type=int, default=12)
    args = parser.parse_args()

    print(f"生成 {args.n:,} × {args.dim}（固定种子）")
    q_list = vectors(1, args.dim, 20260601)[0]
    cards = vectors(args.n, args.dim, 7)

    conv, h2d, kern, d2h, total_pp, total_resident = [], [], [], [], [], []

    for i in range(args.reps):
        # ① Python list -> ndarray（host 侧，可分页）
        t0 = time.perf_counter()
        host = np.asarray(cards, dtype=np.float64)
        conv.append(time.perf_counter() - t0)

        # ② 可分页 H2D
        t0 = time.perf_counter()
        dev = cp.asarray(host)
        cp.cuda.Stream.null.synchronize()
        h2d.append(time.perf_counter() - t0)

        # ③ 只算 kernel（数据已在显存）
        t0 = time.perf_counter()
        qdev = cp.asarray(q_list, dtype=cp.float64)
        dots = dev @ qdev
        norms = cp.linalg.norm(dev, axis=1)
        cos = dots / (norms * cp.linalg.norm(qdev))
        cp.cuda.Stream.null.synchronize()
        kern.append(time.perf_counter() - t0)

        # ④ D2H
        t0 = time.perf_counter()
        cp.asnumpy(cos)
        d2h.append(time.perf_counter() - t0)

        # ⑤ 端到端（可分页）：与 benchmark 里的实现同一条路径
        t0 = time.perf_counter()
        g = cp.asarray(cards)  # 注意：直接从 list 转换，这正是我们代码里的写法
        q = cp.asarray(q_list, dtype=cp.float64)
        out = (g @ q) / (cp.linalg.norm(g, axis=1) * cp.linalg.norm(q))
        cp.asnumpy(out)
        total_pp.append(time.perf_counter() - t0)

        # ⑥ 端到端（数据已驻留）：把转换与 H2D 排除在外
        t0 = time.perf_counter()
        out = (dev @ qdev) / (norms * cp.linalg.norm(qdev))
        cp.asnumpy(out)
        total_resident.append(time.perf_counter() - t0)

    print(f"\n各分项（{args.reps} 次，每次都是独立一遍）：")
    stat("① list->ndarray", conv)
    stat("② H2D（可分页）", h2d)
    stat("③ kernel（显存内）", kern)
    stat("④ D2H", d2h)
    print(f"\n  ②+③+④ 合计中位 = {(statistics.median(h2d) + statistics.median(kern) + statistics.median(d2h))*1000:.1f} ms")

    print(f"\n两种端到端口径：")
    stat("⑤ 端到端（含转换）", total_pp)
    stat("⑥ 端到端（数据驻留）", total_resident)

    median_total = statistics.median(total_pp)
    print(f"\n结论线索：")
    print(f"  · 端到端中位 {median_total*1000:.0f} ms 里，"
          f"H2D 占 {statistics.median(h2d)/median_total:.1%}，"
          f"转换占 {statistics.median(conv)/median_total:.1%}，"
          f"kernel 占 {statistics.median(kern)/median_total:.1%}")
    print(f"  · H2D 抖动 {((max(h2d)-min(h2d))/statistics.median(h2d)):.1%} vs "
          f"端到端抖动 {((max(total_pp)-min(total_pp))/median_total):.1%}")
    print(f"  · 「数据驻留」口径的抖动 {((max(total_resident)-min(total_resident))/statistics.median(total_resident)):.1%}"
          "：若它远小于端到端，抖动来自搬运而不来自计算")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
