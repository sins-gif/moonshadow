"""召回的规模压力测试：卡数 → 延迟与预算，**带退出码**，可进 CI。

为什么要有这个工具
------------------
`run_*` 那一批验证的是**正确性**（排序、门、冻结、批次），没有一个回答
「库长到一万张卡还能不能在两秒内召回」。这两件事会分叉：正确性可以一直绿，
延迟却随卡数增长——实测大区边际约 `230–280 ms/千卡`，一万卡时一次召回要 `2.6 s`。

门限与区间是怎么来的（**这一段是本文档存在的理由**）
--------------------------------------------------
### 1. 成本不是比例的，是仿射的

实测（同机、每档 5 次取中位、`realistic` 层级比例）：

| 卡数 | 合计 ms | 每千卡 ms | 相邻档边际 ms/千卡 |
|---|---|---|---|
| 10 | 5.0 | 500 | — |
| 100 | 12.5 | 125 | 82 |
| 500 | 46.1 | 92 | 86 |
| 1000 | 91.8 | 92 | 91 |
| 1500 | 333.4 | 222 | **483** |
| 2000 | 517.7 | 259 | 369 |
| 3000 | 754.2 | 251 | 237 |
| 5000 | 1307.4 | 261 | 277 |
| 10000 | 2682.5 | 268 | 275 |
| **20000** | **6068.1** | **303** | **339** |

`10 → 1000` 这一段是线性的：相邻边际稳定在 `79–91 ms/千卡`。用两点差分求截距：

```
b = (T(1000) − T(10)) / 990 = 0.0877 ms/卡   ≈ 87.7 ms/千卡
a = T(10) − b·10 = 5.0 − 0.877 ≈ 4.1 ms
```

即 **`T(n) ≈ 4.1 ms + 0.0877·n`**。所以：

- **纯比例门在小库必然误杀**：`n = 10` 时比例门给出 `0.88 ms`，实测 `5.0 ms`——
  差的正是那 `4.1 ms` 固定开销。**这就是「下限」存在的唯一数学理由：它必须等于截距 `a`。**
- 早期版本把下限写成 `50 ms`（拍出来的），意味着「10 张卡和 1000 张卡共用同一门槛」，
  门等于失效。**已废弃。**

### 2. 门限取值：`limit(n) = A + B·n`

| 常数 | 取值 | 出处 |
|---|---|---|
| `A`（固定开销） | `15 ms` | 实测截距 `4.1 ms` × `3.7`，向上取整。**倍数 `3.7` 是假设**：本机没测过 CI 容器/沙箱的解释器与 IO 额外开销；验证方法：在 CI 上跑一次 `--cards 10,100`，看截距是否仍 ≈ `4 ms` |
| `B`（边际成本） | `300 ms/千卡` | 在 `1500–20000` 区间上仿射拟合得到的**等效斜率 `306`**，取整到 `300`。**它不是物理边际**：同一批数据幂律拟合给 `k = 1.0588`（边际 `295`）、二次拟合给 `301→372`。形状证据与残差表见 `docs/v1.2-perf-model.md` §4–§5 |

**余量必须远小于最想抓的故障**：`15 ms` 对 `1000→1500` 那个 `240 ms` 的台阶只占 `6%`，
所以它不会掩盖真问题——这一条可判定，不是「留点余量」的托词。

### 3. 绝对门抓不到台阶，必须再加边际门

同一个 `1500` 卡读数，两把刀结论相反：

```
绝对门：T = 333 ≤ 15 + 300×1.5 = 465   → PASS（漏了）
边际门：ΔT/Δn = 483 > 300              → FAIL（抓到）
```

`1000 → 1500` 之间读库、相似度、装箱三者同时跳 `2.4–4.5` 倍，是**非线性台阶**，
不是平滑变慢。**最可能是 SQLite 默认页缓存（约 2 MB）在这一区间装不下 —— 这是假设，
没做 profiling**；验证方法见 `docs/backlog.md` 的对应条目。所以就两条门都要。

### 4. 为什么用区间，而不是一个点

同配置重复 9 次的实测：

| 卡数 | n | min | p25 | 中位 | p75 | max | max/min | IQR/中位 |
|---|---|---|---|---|---|---|---|---|
| 1000 | 9 | 88.0 | 89.0 | 91.1 | 92.8 | 95.6 | 1.09 | 4.2% |
| 3000 | 9 | 742.0 | 754.4 | 774.8 | 812.3 | 815.5 | 1.10 | 7.5% |

结论分三层：

1. `[min, max]` 只是**观测范围**，不是置信区间：`n` 这么小时 min/max 是极端统计量，
   一次 GC 就能抬高 max。**不许叫它 CI。**
2. `IQR/中位` 是**稳健离散度**（只依赖中间 50% 数据），所以 **判门用它**；
   `max/min` 对尾部敏感，只用来**诊断**，不用来判门。
3. **跨机器/跨时间的差异完全没测过**（可能整体差 2–3 倍），只能标注为未知。

由此定下**两把刀各判一种病**：

| 门 | 统计量 | 判什么病 | 阈值 | 硬/软 |
|---|---|---|---|---|
| G2a 速度 | 中位总耗时 | 变慢了（算法/IO 退化） | `A + B·n` | **硬** |
| G2b 边际 | 相邻档 Δ中位/Δn | 出现台阶（非线性跳变） | `B` | 软（今天 `483 > 300`） |
| G5 稳定 | IQR/中位 | 抖了（GC/竞争/冷热混跑） | `25%` | 软 |

`G2b`/`G5` 默认只报警：它们今天都超标，而按本仓库既有纪律「规则要能被满足」，
把已知缺陷做成硬门只会让工具常年红灯。修好之后用 `--strict-marginal` / `--strict-stability`
把它们升级成硬门。

### 5. 比较两次运行的可判定规则

> **只有当 B 的 `max` < A 的 `min` 时，才能说「B 比 A 快」。**

区间一重叠就不能宣称变快——那是把噪声当结论。这是「不许 hedging」的量化版本。

### 6. 模型的有效范围（超过就必须标「外推」）

`A + B·n` 只在**有实测的区间**里成立：本工具的数据覆盖 `10 – 20000` 卡。
超过 `20000`，把那两个常数继续往右延长**不是结论**，理由是可量的：

| 模型 | `T(50000)`（外推，无实测） |
|---|---|
| 仿射 `A+B·n` | 15,149 ms |
| 幂律 `c·n^1.0588` | 15,330 ms |
| 二次 `a+b·n+c·n²` | **20,433 ms** |

三种拟合在 `1500–20000` 上 `R² ≥ 0.9967`（看着都很好），外推 2.5 倍规模却差 **35%**。
所以：**拟合好 ≠ 能外推**。

由此定了 `--verified-up-to`（默认 `20000`）：
`cards > 该值` 时速度门**降级为警告**并标 `[外推]`——门限本身没有实测支撑，
不能拿它判红；要按外推硬判，加 `--strict-extrapolated`。

还有一条方法论上的教训要记在这里：**`R²` 高不等于模型对**。
仿射模型在 `1500–20000` 上 `R² = 0.9969`，但截距是 `−165 ms`（物理上不可能），
1500 卡处逐点误差 **23.1%**。判定标准必须是**逐点残差**，不是 `R²`。
详细的残差表、导数表与两段截距（`+4.1` / `−165`）见 `docs/v1.2-perf-model.md`。

其他判据
--------
- G1 **确定性**（硬）：同输入跑两次，装箱条目 id 序列必须逐位相同。
- G3 **无静默超预算**（硬）：`tokens_used > budget` 时 `over_budget` 必须为 `True`。
- G4 **常驻区占比**（软）：`T0/T1` 的 token 占预算比。今天必然爆——常驻区不受配额，
  是已知缺陷（待办 #19），修好后同样可升级为硬门。

稳定性措施（为了让判定本身稳）
------------------------------
- **预热**：默认丢弃首个样本（`--warmup 1`），避免把冷页缓存算进来；
- **测量前强制一次 `gc.collect()`**，避免把上一档残留的垃圾回收算进这一档；
- **对称分位**：`n ≥ 7` 时 IQR 才有意义（默认 `--repeats 7`）；
- 输出**同时报 `n` 与 `warmup`**——只报中位数不报样本量等于隐瞒不确定性。

用法
----
    python tools/stress_recall.py                          # 默认 1000 + 10000 两档
    python tools/stress_recall.py --cards 1000,1500,3000    # 三档 → 会算出两个边际
    python tools/stress_recall.py --mix adversarial         # T0+T1 占 2/3 的极端比例
    python tools/stress_recall.py --json                    # 机器可读
    python tools/stress_recall.py --strict-marginal         # 把台阶警告升级成硬门

退出码：全部硬门通过为 `0`，任一失败为 `1`。软门不影响退出码（除非加 `--strict-*`）。
"""

from __future__ import annotations

import argparse
import gc
import json
import pathlib
import random
import shutil
import statistics
import sys
import time
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from moonshadow import FixedClock, Store  # noqa: E402
from moonshadow.cards import make_card  # noqa: E402
from moonshadow.recall import recall  # noqa: E402
from moonshadow.verify import card_text  # noqa: E402
from moon_cli import _lexical_sim  # noqa: E402

CLOCK = FixedClock(datetime(2026, 11, 5, 9, 0, tzinfo=timezone.utc))
WORDS = "预算 截止 交付 架构 迁移 限流 扩容 报价 评审 口径 分片 缓存 回滚 灰度 探针 索引 队列 事务 快照 对账".split()

#: 层级比例。`realistic` 对应「一个正常项目聊了一两个月」；
#: `adversarial` 逼近常驻区打爆预算的极端（T0+T1 占 2/3）。
MIXES: dict[str, tuple[tuple[str, int], ...]] = {
    "realistic": (("T0", 1), ("T1", 1), ("T2", 10), ("T3", 18), ("T5", 45), ("T6", 25)),
    "adversarial": (("T0", 33), ("T1", 33), ("T3", 34)),
}

DEFAULT_QUERY = "GPU 加速的三档数字 和 预算 截止 交付"

#: 门限常数。取值依据见模块 docstring §2 —— 那两个数字不是拍的，`A` 由实测截距推出。
FLOOR_MS = 15.0             # A：固定开销上限（实测截距 4.1 ms × 3.7，倍数为假设）
BUDGET_MS_PER_1K = 300.0    # B：边际成本上限（实测 236–280 ms/千卡，取整 300）
IQR_RATIO_LIMIT = 0.25      # G5：IQR/中位上限（实测 4.2–7.5%，取 25% 留约 3 倍余量）

#: 模型**有数据支撑**的上界（卡数）。超过它，`A + B·n` 只是把曲线延长，不是结论。
#:
#: 为什么是 20000：实测覆盖到 20000，且那一档已经**超 SLO**（6068 > 6015）。
#: 再往上没有实测；而三种同样"拟合得很好"的函数形式（仿射 / 幂律 k=1.0588 / 二次）
#: 外推到 50000 时分 别给出 15149 / 15330 / 20433 ms——**分歧 35%**，
#: 所以外推值只能报，不能判。详见 `docs/v1.2-perf-model.md` §7。
VERIFIED_UP_TO = 20000


def tier_for(index: int, mix: str) -> str:
    """按权重把下标映射到层级（确定性，不依赖随机数）。"""
    weights = MIXES[mix]
    total = sum(weight for _, weight in weights)
    point = index % total
    cursor = 0
    for tier, weight in weights:
        cursor += weight
        if point < cursor:
            return tier
    return weights[-1][0]


def build_store(root: pathlib.Path, cards: int, mix: str, seed: int) -> None:
    """直接造卡（不走抽取管线），这样测的是**召回**的复杂度。"""
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    store = Store(root, clock=CLOCK)
    rng = random.Random(seed)
    try:
        for index in range(cards):
            tier = tier_for(index, mix)
            body = "，".join(rng.choice(WORDS) for _ in range(rng.randint(4, 18)))
            card = make_card(
                clock=CLOCK,
                observed_at=f"2026-11-05T09:00:{index % 60:02d}+00:00",
                session_id="stress",
                tier=tier,
                importance=8 if tier in ("T0", "T1", "T2") else 5,
                network="experience",
                summary=f"第 {index} 条：{body}",
                facts=[f"{rng.choice(WORDS)} {rng.randint(1, 999)}", body],
                decisions=[f"决定采用{body} 方案 {index}"] if tier == "T2" else [],
                todos=[f"{body} 负责跟进"] if tier == "T3" else [],
                constraints=[f"{body} 不得越过"] if tier == "T0" else [],
                claims=[],
                confidence=0.8,
                source_ids=[f"src_{index:06d}"],
                entities=[],
                keywords=[],
            )
            store.put_card(card)
    finally:
        store.close()


def summarize(samples: list[float]) -> dict:
    """把原始样本折成「观测区间 + 稳健离散度」。纯函数，便于单测。

    口径：
    - `median` 用于**判速度**（对尾部离群稳健）；
    - `iqr_ratio` 用于**判稳定性**（只依赖中间 50%，同样稳健）；
    - `max_ratio` 只做诊断——`n` 小时它是极端统计量，不能用来判门。
    """
    ordered = sorted(samples)
    n = len(ordered)
    median = statistics.median(ordered)
    if n >= 4:
        q1, _, q3 = statistics.quantiles(ordered, n=4, method="inclusive")
    else:
        q1, q3 = ordered[0], ordered[-1]
    return {
        "n": n,
        "min_ms": round(ordered[0], 1),
        "p25_ms": round(q1, 1),
        "median_ms": round(median, 1),
        "p75_ms": round(q3, 1),
        "max_ms": round(ordered[-1], 1),
        "max_ratio": round(ordered[-1] / ordered[0], 3) if ordered[0] > 0 else float("inf"),
        "iqr_ratio": round((q3 - q1) / median, 3) if median > 0 else float("inf"),
    }


def limit_ms(cards: int, floor_ms: float = FLOOR_MS, per_1k: float = BUDGET_MS_PER_1K) -> float:
    """仿射门限：`A + B·n`。**不是比例门**——比例门在小库会误杀（见 docstring §1）。"""
    return floor_ms + per_1k * cards / 1000.0


def measure(root: pathlib.Path, query: str, budget_tokens: int, repeats: int, warmup: int) -> dict:
    """跑 `warmup + repeats` 次召回，丢掉前 `warmup` 次，返回区间统计与单次读数。"""
    samples: list[float] = []
    first_ids: list[str] | None = None
    deterministic = True
    cards = packed = tokens_used = reserved_tokens = 0
    over_budget = False

    for iteration in range(max(1, warmup) + max(1, repeats)):
        gc.collect()                                   # 别把上一档的垃圾回收算进这一档
        t0 = time.perf_counter()
        store = Store(root, clock=CLOCK)
        rows = store.cards()
        cos = {str(card["id"]): _lexical_sim(query, card_text(card)) for card in rows}
        result = recall(store, now=CLOCK.now(), budget_tokens=budget_tokens, query_cos=cos)
        elapsed = (time.perf_counter() - t0) * 1000
        store.close()

        ids = [placement.item.id for placement in result["pack_result"].placements]
        if first_ids is None:
            first_ids = ids
        elif ids != first_ids:
            deterministic = False

        cards = len(rows)
        packed = result["stats"].packed
        tokens_used = result["stats"].tokens_used
        reserved_tokens = result["pack_result"].reserved_tokens
        over_budget = result["pack_result"].over_budget
        if iteration >= warmup:
            samples.append(elapsed)

    stats = summarize(samples)
    stats.update(
        {
            "cards": cards,
            "packed": packed,
            "tokens_used": tokens_used,
            "budget_tokens": budget_tokens,
            "reserved_tokens": reserved_tokens,
            "reserved_share": round(reserved_tokens / max(budget_tokens, 1), 3),
            "over_budget": over_budget,
            "deterministic": deterministic,
        }
    )
    return stats


def verdict(row: dict, *, floor_ms: float = FLOOR_MS, per_1k: float = BUDGET_MS_PER_1K,
            reserved_limit: float = 0.30, iqr_limit: float = IQR_RATIO_LIMIT,
            verified_up_to: int = VERIFIED_UP_TO, strict_extrapolated: bool = False) -> dict:
    """三硬三软。硬门全过才 `passed`；软门只影响 `warnings`。

    `cards > verified_up_to` 时，速度门**降级为警告**并标 `[外推]`：
    门限本身是把曲线延长出来的，没有实测支撑，不能拿它当结论判红。
    想按外推值硬判，传 `strict_extrapolated=True`。
    """
    cap = limit_ms(row["cards"], floor_ms, per_1k)
    speed_ok = row["median_ms"] <= cap
    extrapolated = row["cards"] > verified_up_to

    hard = {
        "G1_deterministic": bool(row["deterministic"]),
        "G3_no_silent_overrun": (row["tokens_used"] <= row["budget_tokens"]) or bool(row["over_budget"]),
    }
    soft = {
        "G4_reserved_share": row["reserved_share"] <= reserved_limit,
        "G5_stability": row["iqr_ratio"] <= iqr_limit,
    }
    if extrapolated and not strict_extrapolated:
        soft["G2a_speed_extrapolated"] = speed_ok
    else:
        hard["G2a_speed"] = speed_ok

    return {
        "limit_ms": round(cap, 1),
        "extrapolated": extrapolated,
        "hard": hard,
        "soft": soft,
        "warnings": [name for name, ok in soft.items() if not ok],
        "passed": all(hard.values()),
    }


def marginal_rate(lower: dict, upper: dict) -> float | None:
    """相邻档的边际成本（ms/千卡）。只用中位数——它是这里最稳的统计量。"""
    delta_cards = upper["cards"] - lower["cards"]
    if delta_cards <= 0:
        return None
    return (upper["median_ms"] - lower["median_ms"]) / (delta_cards / 1000.0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="召回的规模压力测试（带退出码，可进 CI）")
    parser.add_argument("--cards", default="1000,10000", help="卡数档位，逗号分隔（默认 1000,10000）")
    parser.add_argument("--mix", choices=sorted(MIXES), default="realistic", help="层级比例（默认 realistic）")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="召回查询")
    parser.add_argument("--budget-tokens", type=int, default=4000, help="召回 token 预算（默认 4000）")
    parser.add_argument("--repeats", type=int, default=7, help="每档有效样本数（默认 7，n≥7 时 IQR 才有意义）")
    parser.add_argument("--warmup", type=int, default=1, help="预热次数，丢弃（默认 1，排除冷页缓存）")
    parser.add_argument("--floor-ms", type=float, default=FLOOR_MS, help=f"绝对门限的固定开销项 A（默认 {FLOOR_MS}）")
    parser.add_argument("--budget-ms-per-1k", type=float, default=BUDGET_MS_PER_1K,
                        help=f"边际门限 B，ms/千卡（默认 {BUDGET_MS_PER_1K}）")
    parser.add_argument("--reserved-limit", type=float, default=0.30, help="常驻区占比警告线（默认 0.30）")
    parser.add_argument("--iqr-limit", type=float, default=IQR_RATIO_LIMIT,
                        help=f"稳定性警告线 IQR/中位（默认 {IQR_RATIO_LIMIT}）")
    parser.add_argument("--verified-up-to", type=int, default=VERIFIED_UP_TO,
                        help=f"模型有数据支撑的卡数上界（默认 {VERIFIED_UP_TO}）；超过即标「外推」")
    parser.add_argument("--strict-extrapolated", action="store_true",
                        help="外推区间也按硬门判（默认只报警，因为门限本身没有实测支撑）")
    parser.add_argument("--strict-marginal", action="store_true", help="把边际台阶警告升级成硬门")
    parser.add_argument("--strict-stability", action="store_true", help="把稳定性警告升级成硬门")
    parser.add_argument("--seed", type=int, default=20261105, help="合成库随机种子")
    parser.add_argument("--json", action="store_true", help="只输出 JSON")
    parser.add_argument("--keep", action="store_true", help="保留合成库（默认删除）")
    args = parser.parse_args(argv)

    sizes = [int(part) for part in args.cards.split(",") if part.strip()]
    if not sizes or any(size <= 0 for size in sizes):
        parser.error("--cards 必须是正整数，逗号分隔")
    if sizes != sorted(sizes):
        parser.error("--cards 必须升序（边际门依赖相邻档）")
    if args.repeats < 4:
        parser.error("--repeats 至少 4（否则 IQR 无意义）")

    base = ROOT / ".tmp" / "stress-recall"
    rows = []
    for size in sizes:
        root = base / f"{args.mix}-{size}"
        build_store(root, size, args.mix, args.seed)
        row = measure(root, args.query, args.budget_tokens, args.repeats, args.warmup)
        row["mix"] = args.mix
        row.update(verdict(row, floor_ms=args.floor_ms, per_1k=args.budget_ms_per_1k,
                          reserved_limit=args.reserved_limit, iqr_limit=args.iqr_limit,
                          verified_up_to=args.verified_up_to,
                          strict_extrapolated=args.strict_extrapolated))
        rows.append(row)
        if not args.keep:
            shutil.rmtree(root, ignore_errors=True)

    # 边际门：相邻档的 Δ中位 / Δ卡数
    for index in range(1, len(rows)):
        rate = marginal_rate(rows[index - 1], rows[index])
        rows[index]["marginal_ms_per_1k"] = None if rate is None else round(rate, 1)
        rows[index]["G2b_marginal"] = rate is None or rate <= args.budget_ms_per_1k

    hard_failed = [row for row in rows if not row["passed"]]
    marginal_failed = [row for row in rows[1:] if not row.get("G2b_marginal", True)]
    stability_failed = [row for row in rows if not row["soft"]["G5_stability"]]
    if args.strict_marginal and marginal_failed:
        hard_failed = hard_failed + marginal_failed
    if args.strict_stability and stability_failed:
        hard_failed = hard_failed + stability_failed

    if args.json:
        print(json.dumps({"reports": rows, "failed": len(hard_failed)}, ensure_ascii=False, indent=1))
    else:
        print("召回压力测试")
        print(
            f"  层级比例 {args.mix} · 预算 {args.budget_tokens} tok · "
            f"n={args.repeats}（预热丢弃 {args.warmup}）· 门限 A={args.floor_ms:g} ms + "
            f"B={args.budget_ms_per_1k:g} ms/千卡"
        )
        print()
        header = (
            f"  {'卡数':>6}{'n':>3}{'min':>8}{'中位':>8}{'max':>8}{'max/min':>9}{'IQR/中位':>9}"
            f"{'门限':>8}{'装进':>6}{'tok':>8}{'常驻':>8}{'边际':>7}{'状态':>11}"
        )
        print(header)
        print("  " + "-" * (len(header) - 2))
        for row in rows:
            marginal = row.get("marginal_ms_per_1k")
            state = "PASS" if row["passed"] else "FAIL"
            if row.get("extrapolated"):
                state = f"{state}·外推"
            print(
                f"  {row['cards']:>6}{row['n']:>3}{row['min_ms']:>8.1f}{row['median_ms']:>8.1f}"
                f"{row['max_ms']:>8.1f}{row['max_ratio']:>9.2f}{row['iqr_ratio']:>8.1%}"
                f"{row['limit_ms']:>8.0f}{row['packed']:>6}{row['tokens_used']:>8}"
                f"{row['reserved_share']:>8.0%}"
                f"{('—' if marginal is None else f'{marginal:.0f}'):>7}{state:>11}"
            )
        print()
        for row in rows:
            hard = row["hard"]
            flags = [
                f"G1 确定性 {'✓' if hard['G1_deterministic'] else '✗'}",
            ]
            if "G2a_speed" in hard:
                flags.append(f"G2a 速度 {'✓' if hard['G2a_speed'] else '✗'}")
            else:
                ok = row["soft"].get("G2a_speed_extrapolated", True)
                flags.append(f"G2a 速度 {'✓' if ok else '✗'}（外推，不作判据）")
            flags.append(f"G3 无静默超预算 {'✓' if hard['G3_no_silent_overrun'] else '✗'}")
            if "G2b_marginal" in row:
                flags.append(f"G2b 边际 {'✓' if row['G2b_marginal'] else '✗ 台阶'}")
            flags.append(f"G5 稳定 {'✓' if row['soft']['G5_stability'] else '⚠ 抖动'}（max/min {row['max_ratio']:.2f}）")
            flags.append(f"G4 常驻 {'✓' if row['soft']['G4_reserved_share'] else '⚠ 超线（待办 #19）'}")
            print(f"  {row['cards']:>6} 卡： " + " · ".join(flags))
        print()
        if hard_failed:
            print(f"  ✗ 硬门未通过：{[row['cards'] for row in hard_failed]} 档")
        else:
            print("  ✓ 硬门全部通过；软门（G2b 台阶 / G5 抖动 / G4 常驻）为已知缺陷，只报警")

    return 1 if hard_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
