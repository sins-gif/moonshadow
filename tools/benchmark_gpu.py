"""GPU 加速基准测试：同一件事的两种实现，同一批假向量。

    python tools/benchmark_gpu.py                    # 50,000 × 768（默认，按指令）
    python tools/benchmark_gpu.py --n 2000 --dim 128  # 快速冒烟
    python tools/benchmark_gpu.py --self-check        # 只做两侧一致性比对，不做计时

**设计要点（这几条决定了读数可不可信）**：

1. **同一份数据、同一个函数**：CPU 侧调用 `gpu_backend.batch_query_cos_cpu`，GPU 侧调用
   `batch_query_cos_gpu`；两者语义相同，差异只可能来自后端，不可能来自「两份代码写得不一样」。
2. **定值输入**：向量由固定种子的线性同余生成器产生，不用 `random` 的浮点抽样——
   同一份脚本在任何机器上生成同一批数据，否则耗时对比没有意义。
3. **没有 GPU 也照跑**：打印「不可用 + 原因」，CPU 列照常输出，GPU 列写 `—`。
   不假装测过，也不因为缺显卡就整体失败（退出码仍为 `0`）。
4. **计时口径**：取 `--repeat` 次的**最小值**（排除调度抖动），并给出加速比。
   第一次调用会包含内存分配与 JIT 预热，因此 GPU 侧另有 `--warmup` 次预热不计入。
"""

from __future__ import annotations

import argparse
import pathlib
import statistics
import sys
import time
from array import array

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moonshadow import gpu_backend  # noqa: E402

#: 线性同余参数（Numerical Recipes 的经典取值）：只为可复现，不追求统计质量。
_LCG_A, _LCG_C, _LCG_M = 1664525, 1013904223, 2**32


def _vectors(count: int, dim: int, seed: int = 20260601) -> list[array]:
    """生成 `count` 个 `dim` 维向量，元素落在 [-1, 1]。

    用 `array('d')` 而不是 `list[float]`：50,000 × 768 若用 Python float 对象，
    光对象头就要 ~0.9 GB；`array` 是紧凑的 8 字节双精度，约 0.3 GB。
    """
    state = seed
    data: list[array] = []
    for _ in range(count):
        row = array("d", bytes(8 * dim))
        for j in range(dim):
            state = (_LCG_A * state + _LCG_C) % _LCG_M
            row[j] = (state / _LCG_M) * 2.0 - 1.0
        data.append(row)
    return data


def _time(fn, query, cards, repeat: int) -> float:
    """跑 `repeat` 次取最小值（秒）。"""
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        fn(query, cards)
        best = min(best, time.perf_counter() - start)
    return best


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="批量余弦相似度：纯 Python vs cupy 基准")
    parser.add_argument("--n", type=int, default=50_000, help="卡片向量条数（默认 50,000）")
    parser.add_argument("--dim", type=int, default=768, help="向量维度（默认 768）")
    parser.add_argument("--repeat", type=int, default=3, help="CPU 侧重复次数，取最小值")
    parser.add_argument("--warmup", type=int, default=1, help="GPU 侧预热次数（不计入）")
    parser.add_argument("--runs", type=int, default=1, help="重复分组数（>1 时给出加速比区间）")
    parser.add_argument("--self-check", action="store_true", help="只做两侧一致性比对")
    args = parser.parse_args(argv)

    if args.self_check:
        report = gpu_backend.self_check()
        print("两侧一致性自检（定值输入，不用随机数）：")
        for key, value in report.items():
            print(f"  {key:<16}{value}")
        if report["gpu_available"]:
            return 0 if report["consistent"] else 1
        print("  → GPU 不可用，仅验证了 CPU 参考实现可跑；一致性检查留待有显卡的机器。")
        return 0

    print(f"生成假向量：{args.n:,} 条 × {args.dim} 维（固定种子 20260601，可复现）")
    started = time.perf_counter()
    query = _vectors(1, args.dim)[0]
    cards = _vectors(args.n, args.dim, seed=7)
    print(f"  生成耗时 {time.perf_counter() - started:.2f} s；"
          f"约 {args.n * args.dim * 8 / 1024 / 1024:.0f} MiB（array('d') 紧凑存储）")

    # 注意顺序：**先探测、再打印原因**。反过来的话打出来的是 import 期的旧状态，
    # 真正失败的原因（设备 / JIT）会被掩盖——这条是实测踩出来的，不是风格问题。
    gpu_ok = gpu_backend.probe()
    print(f"\ncupy 后端：{gpu_backend.PROBE_REASON}")
    if gpu_ok:
        for _ in range(args.warmup):
            gpu_backend.batch_query_cos_gpu(query, cards)  # 预热：分配 + JIT 不计入

    # 分组重复：组内各自 repeat 次取最小，组间独立。为什么要分组而不是只加深 repeat——
    # 组内取最小只是「这一段时间里最好的一次」，无法反映跨时间的漂移（笔记本功耗墙、
    # 其它程序占用 GPU、温度）。要让读者看到一个区间，必须让**组**成为统计单位。
    groups: list[tuple[float, float | None]] = []
    for index in range(args.runs):
        cpu_s = _time(gpu_backend.batch_query_cos_cpu, query, cards, args.repeat)
        gpu_s = (
            _time(gpu_backend.batch_query_cos_gpu, query, cards, args.repeat) if gpu_ok else None
        )
        groups.append((cpu_s, gpu_s))
        tail = f" / gpu {gpu_s:.3f} s → {cpu_s / gpu_s:.2f}x" if gpu_s else " / gpu —"
        print(f"  第 {index + 1}/{args.runs} 组：cpu {cpu_s:.3f} s{tail}")

    cpu_seconds = statistics.median(g[0] for g in groups)
    gpu_seconds = (
        statistics.median(g[1] for g in groups) if gpu_ok and groups[0][1] is not None else None
    )
    rate = args.n / cpu_seconds
    ratios = [c / g for c, g in groups if g]
    print(f"\n取中位数（{args.runs} 组 × {args.repeat} 次取最小）："
          f"cpu {cpu_seconds:.3f} s" + (f" / gpu {gpu_seconds:.3f} s" if gpu_seconds else ""))

    print()
    print(f"{'后端':<22}{'耗时(s)':>10}{'条/秒':>14}{'加速比':>10}")
    print(f"{'cpu(pure-python)':<22}{cpu_seconds:>10.3f}{rate:>14,.0f}{'1.00x':>10}")
    if gpu_seconds is None:
        print(f"{'gpu(cupy)':<22}{'—':>10}{'—':>14}{'—':>10}")
        print("\n注：GPU 列为空不代表失败，只代表本机没有算力；CPU 侧读数与回退路径一致。")
    else:
        print(f"{'gpu(cupy)':<22}{gpu_seconds:>10.3f}"
              f"{args.n / gpu_seconds:>14,.0f}{cpu_seconds / gpu_seconds:>9.1f}x")
        if len(ratios) > 1:
            so = sorted(ratios)
            print(f"\n加速比区间（{len(ratios)} 组）：最小 {so[0]:.2f}x · 中位 {statistics.median(so):.2f}x"
                  f" · 最大 {so[-1]:.2f}x（抖动 {(so[-1] - so[0]) / statistics.median(so):.1%}）")
            print("  汇报时请给区间并说明是「组间漂移」，不要只取最好的那一次。")
    print(f"\n小输入自检（8 条 × 16 维，只验算法形状）：{gpu_backend.self_check()}")

    # 全量一致性：用**本次实际输入**把两侧输出逐条比一遍。
    # 为什么必须单独做：小输入自检只能证明「算法写得一样」，不能证明「这一次算得一样」；
    # 而简历/汇报里那句「误差 2.2e-16」如果不说清是在哪种输入上量的，就是含糊其辞。
    if gpu_seconds is not None:
        cpu_all = gpu_backend.batch_query_cos_cpu(query, cards)
        gpu_all = gpu_backend.batch_query_cos_gpu(query, cards)
        diffs = [abs(a - b) for a, b in zip(cpu_all, gpu_all)]
        worst = max(diffs)
        print(
            f"全量输出一致性（真正的 {args.n:,} 条）：max_abs_diff = {worst:.3e}"
            f"（最差在第 {diffs.index(worst)} 条）· 容差 {gpu_backend.TOLERANCE:g} → "
            f"{'PASS' if worst <= gpu_backend.TOLERANCE else 'FAIL'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
