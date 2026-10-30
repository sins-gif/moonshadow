"""GPU 加速后端（**可选**，不在默认依赖里）。

本文件存在的理由只有一个：**把可选的加速层与主路径彻底隔离**。

隔离契约（三条，都是可验证的）：

1. **主路径不导入本模块**：`import moonshadow` 不会碰到它，更不会碰到 cupy。
   只有调用方显式要求（`use_gpu=True` 或传入向量）时才在本模块内部惰性导入。
2. **本模块自带 guard**：`import cupy` 失败、或装了 cupy 但没有可用 CUDA 设备时，
   `AVAILABLE=False`、`PROBE_REASON` 说明原因；模块本身仍可导入，调用方拿到的是
   「不可用」这一状态，而不是一个 ImportError。
3. **两个实现必须逐值一致**：`self_check()` 在同一个小输入上比对 CPU/GPU 两侧，
   不一致就说明加速层有 bug——那时宁可用 CPU，而不是接受「结果差不多」。

为什么把 CPU 参考实现也放在这里：基准测试要比的必须是「同一件事的两种实现」。
如果 CPU 侧在别处另写一份，测出来的差异就分不清是「后端不同」还是「代码不同」。
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

#: GPU 侧（cupy）是否可用。缺依赖 / 缺设备时为 False，**不是异常**。
AVAILABLE = False
#: 不可用（或可用）的原因，直接打进日志与警告里，避免「为什么没走 GPU」这种猜测。
PROBE_REASON = "尚未探测"
#: 容差：两侧都是 IEEE-754 双精度，但归约顺序不同，允许 1e-6 的相对误差。
TOLERANCE = 1e-6

try:  # pragma: no cover - 环境相关
    import cupy as _cp  # type: ignore[import-not-found]

    AVAILABLE = True
    PROBE_REASON = f"cupy {getattr(_cp, '__version__', '?')} 已导入"
except Exception as exc:  # pragma: no cover - 环境相关
    _cp = None  # type: ignore[assignment]
    AVAILABLE = False
    PROBE_REASON = f"未安装 cupy（{type(exc).__name__}）"


def probe() -> bool:
    """真正探一次设备：cupy 可以导入成功但机器上没有可用显卡。

    结果会被缓存（cupy 的设备初始化很贵，而且重复探测没有意义）。
    """
    global AVAILABLE, PROBE_REASON
    if not AVAILABLE or _cp is None:
        return False
    if PROBE_REASON.endswith("（已探测）"):
        return True
    try:  # pragma: no cover - 环境相关
        _cp.zeros(1)
    except Exception as exc:  # pragma: no cover - 环境相关
        AVAILABLE = False
        PROBE_REASON = f"cupy 已导入但没有可用设备（{type(exc).__name__}: {exc}）"
        return False
    # 再触发一次**极小的 JIT 编译**。实测踩到的坑：cupy 能导入、能分配显存，
    # 但第一次编译内核就失败——Windows 中文用户名会让 nvrtc 读不到临时 .cu 与 cupy 自带的头文件
    # （报 cannot open source file "cupy/complex.cuh"）。这种失败必须在 probe() 里撞出来，
    # 而不是等业务调用跑到一半才炸：回退路径的存在意义就是「探测不到就别用」。
    try:  # pragma: no cover - 环境相关
        (_cp.arange(4, dtype=_cp.float64) * 2.0).sum()
    except Exception as exc:  # pragma: no cover - 环境相关
        AVAILABLE = False
        PROBE_REASON = (
            f"cupy 可用但 JIT 编译失败（{type(exc).__name__}）。"
            "常见原因：Windows 中文用户名路径让 nvrtc 读不到 .cu 或 cupy 头文件。"
            "两个已实测的解法：① 把 TEMP/TMP 与 CUPY_CACHE_DIR 指到纯 ASCII 目录；"
            "② 用目录联接把 site-packages 映到 ASCII 路径"
            "（mklink /J C:\\cu-site <site-packages>，再把 C:\\cu-site 放进 PYTHONPATH）。"
        )
        return False
    PROBE_REASON = f"{PROBE_REASON}，设备与 JIT 探测通过（已探测）"
    return True


def batch_query_cos_cpu(
    query_vec: Sequence[float], card_vecs: Sequence[Sequence[float]]
) -> list[float]:
    """纯 Python 参考实现：批量余弦相似度。

    这是**基准**也是**回退路径**：不依赖任何第三方库，所以它在任何机器上都能跑，
    从而保证「没有显卡时仍然得到同一批数字」。空向量返回 0.0（不做除零）。
    """
    q = list(query_vec)
    q_norm = math.sqrt(math.fsum(x * x for x in q))
    out: list[float] = []
    for vec in card_vecs:
        norm = math.sqrt(math.fsum(x * x for x in vec))
        if q_norm == 0.0 or norm == 0.0:
            out.append(0.0)
            continue
        dot = math.fsum(a * b for a, b in zip(q, vec))
        out.append(dot / (q_norm * norm))
    return out


def batch_query_cos_gpu(
    query_vec: Sequence[float], card_vecs: Sequence[Sequence[float]]
) -> list[float]:
    """cupy 实现：批量余弦相似度。

    不可用时抛 `RuntimeError`（而不是静默回退）——回退是 `resolve()` 的职责，
    直接调用本函数的人应该知道自己拿到的是 GPU 结果，还是一个问题。
    """
    if not probe():
        raise RuntimeError(f"GPU 后端不可用：{PROBE_REASON}")
    import cupy as cp  # pragma: no cover - 环境相关

    q = cp.asarray(query_vec, dtype=cp.float64)
    m = cp.asarray(card_vecs, dtype=cp.float64)
    q_norm = cp.linalg.norm(q)
    norms = cp.linalg.norm(m, axis=1)
    denom = q_norm * norms
    dots = m @ q
    safe = cp.where(denom == 0, 1.0, denom)
    cos = cp.where(denom == 0, 0.0, dots / safe)
    return [float(value) for value in cp.asnumpy(cos)]


def resolve(use_gpu: bool) -> tuple[Callable[..., list[float]], str]:
    """选一个后端：返回 `(函数, 名字)`。

    `use_gpu=True` 但仍然不可用时，**回退并 warn**（这是本分支最关键的一条行为：
    调用方不该因为「没装 cupy」而拿到一个异常，也不该在不知情的情况下跑在 CPU 上）。
    """
    if use_gpu and probe():
        return batch_query_cos_gpu, "gpu(cupy)"
    if use_gpu:
        import warnings

        warnings.warn(
            f"请求了 GPU 但后端不可用（{PROBE_REASON}），已回退到纯 Python 实现；"
            "读数仍然可复现，但耗时不代表 GPU 性能。",
            RuntimeWarning,
            stacklevel=3,
        )
    return batch_query_cos_cpu, "cpu(pure-python)"


def self_check(card_count: int = 8, dim: int = 16) -> dict[str, Any]:
    """在固定小输入上比对两侧，返回可直接打印的报告。

    不用随机数：定值输入才能让「两侧不一致」这件事可复现。
    """
    query = [((i * 7) % 11) / 11.0 for i in range(dim)]
    cards = [
        [(((i + j) * 13) % 17) / 17.0 for j in range(dim)] for i in range(card_count)
    ]
    cpu = batch_query_cos_cpu(query, cards)
    report: dict[str, Any] = {
        "dim": dim,
        "cards": card_count,
        "cpu": cpu[:3],
        "gpu_available": probe(),
        "gpu": None,
        "max_abs_diff": None,
        "consistent": None,
        "reason": PROBE_REASON,
    }
    if report["gpu_available"]:
        gpu = batch_query_cos_gpu(query, cards)
        diff = max(abs(a - b) for a, b in zip(cpu, gpu))
        report.update(gpu=gpu[:3], max_abs_diff=diff, consistent=diff <= TOLERANCE)
    return report


__all__ = [
    "AVAILABLE",
    "PROBE_REASON",
    "TOLERANCE",
    "batch_query_cos_cpu",
    "batch_query_cos_gpu",
    "probe",
    "resolve",
    "self_check",
]
