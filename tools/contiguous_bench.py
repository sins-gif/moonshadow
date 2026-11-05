"""第三档口径：去掉 marshal 的端到端（向量已是连续缓冲）。

动机：分项实验显示端到端 84% 的时间花在 `list[array('d')] -> ndarray` 上。
如果调用方本来就把向量放在**一块连续缓冲**里（嵌入模型输出、mmap 进来的文件、
或 `array('d')` 单一对象），这一步就可以零拷贝（`np.frombuffer`）。

    python .tmp/contiguous_bench.py [--n 50000] [--dim 768] [--reps 12] [--cpu 10.56]

注意：CPU 基准默认复用 15 组实测的中位 10.56 s（同一份数据、同一台机器），
不在这里重跑 10 秒级的纯 Python 循环——想自己量就传 --cpu 0 让它跑 3 次。
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

import cupy as cp  # noqa: E402
from moonshadow import gpu_backend  # noqa: E402

_LCG_A, _LCG_C, _LCG_M = 1664525, 1013904223, 2**32


def build(count: int, dim: int, seed: int) -> array:
    """一次生成**一整块**连续缓冲（而不是 count 个独立 array）。"""
    state = seed
    buf = array("d", bytes(8 * count * dim))
    index = 0
    for _ in range(count):
        for _j in range(dim):
            state = (_LCG_A * state + _LCG_C) % _LCG_M
            buf[index] = (state / _LCG_M) * 2.0 - 1.0
            index += 1
    return buf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50_000)
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--cpu", type=float, default=10.56, help="CPU 基准秒数（0 = 现场测 3 次）")
    args = ap.parse_args()

    print(f"生成 {args.n:,} × {args.dim}，**一整块连续缓冲**（array('d')）")
    t0 = time.perf_counter()
    flat = build(args.n, args.dim, 7)
    q_flat = build(1, args.dim, 20260601)
    print(f"  生成耗时 {time.perf_counter() - t0:.1f} s；{len(flat) * 8 / 1024**2:.0f} MiB")

    matrix = np.frombuffer(flat, dtype=np.float64).reshape(args.n, args.dim)
    query = np.frombuffer(q_flat, dtype=np.float64)

    if args.cpu:
        cpu_seconds = args.cpu
        print(f"\nCPU 基准：复用 15 组实测中位 {cpu_seconds:.2f} s（同一份数据、同一台机器）")
    else:
        cards = [matrix[i] for i in range(args.n)]
        t0 = time.perf_counter()
        gpu_backend.batch_query_cos_cpu(query, cards)
        cpu_seconds = time.perf_counter() - t0
        print(f"\nCPU 基准：现场测 {cpu_seconds:.2f} s")

    times: list[float] = []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        dev = cp.asarray(matrix)          # 连续缓冲 → 一次 H2D（无 marshal）
        qdev = cp.asarray(query)
        out = (dev @ qdev) / (cp.linalg.norm(dev, axis=1) * cp.linalg.norm(qdev))
        cp.asnumpy(out)
        times.append(time.perf_counter() - t0)

    lo, hi = min(times), max(times)
    med = statistics.median(times)
    print(f"GPU 端到端（连续缓冲，{args.reps} 次）：中位 {med*1000:.1f} ms"
          f"  min {lo*1000:.1f}  max {hi*1000:.1f}  抖动 {(hi-lo)/med:.1%}")
    print(f"加速比：中位 {cpu_seconds/med:.0f}×  区间 {cpu_seconds/hi:.0f}–{cpu_seconds/lo:.0f}×")
    print("\n对照：含 marshal 的口径是 24.3×（15 组中位 24.31×，区间 20.7–27.0×）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
