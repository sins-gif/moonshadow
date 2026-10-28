"""召回编排器：把 store / scoring / pack 串成一次召回（`v1.2-spec.md` §5）。

    顺序（与 §5 的标题一致）：**硬门 → 打分 → 装箱**，而硬门分两段——
    定性段（卡龄窗口 + `T8/T9` 载体禁令）在打分**前**，定量段（`q ≥ θ`）在打分**后**。
    这不是实现口味：`θ` 是 `q` 的阈值，它不可能排在算出 `q` 之前。

本模块**只做编排，不定义任何规则**：

- 卡龄窗口、`T8/T9` 的准入、阈值比较 → `scoring.passes_qualitative_gate` /
  `scoring.passes_quantitative_gate`（`passes_gate` 仍是两者的合取）；
- 分区配额、`full → lite` 降级、丢弃理由 → `pack.pack`；
- 打分与排序 → `scoring.select`。

这里唯一的判断是「卡进哪个分区」，见 `_section_for` 的说明（那是一个**接口未定项**的落点，
不是对契约的解释）。

零副作用：不写库、不改卡、不打印（只有 `verbose=True` 时打印诊断）。因此它可被
`eval`、`demo`、测试反复调用而不改变任何状态。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .pack import Item, PackResult, estimate_tokens, pack
from .scoring import (
    ADOPTED_SCHEME,
    ALWAYS_ON_TIERS,
    THETA,
    THETA_HIGH,
    card_delta_days,
    passes_qualitative_gate,
    passes_quantitative_gate,
    score_card,
    select,
)
from .store import Store
from .verify import card_text


@dataclass(frozen=True)
class RecallStats:
    """每一步砍掉了多少张卡。字段名即诊断输出的行序。"""

    cards: int = 0
    qualitative_blocked: int = 0
    quantitative_blocked: int = 0
    packed: int = 0
    degraded: int = 0
    dropped: int = 0
    tokens_used: int = 0
    budget: int = 0
    over_budget: bool = False

    @property
    def gated(self) -> int:
        """过硬门的卡数。

        两段互斥（先定性、后定量，且定量只对定性通过者计算），所以可以直接相减：
        这个等式是**计数口径**，不是近似。`recall()` 里还有一条断言把
        「两段合取」与 `scoring.select()` 的结果对齐，防止规则被复制成两份。
        """
        return self.cards - self.qualitative_blocked - self.quantitative_blocked

    @property
    def remaining_budget(self) -> int:
        """剩余预算。超支（常驻区独自撑爆预算）时按 0 报，`over_budget` 才是那个信号。"""
        return max(0, self.budget - self.tokens_used)


def _section_for(tier: str) -> str:
    """卡进哪个分区（`v1.2-spec.md` §6）。

    `T0/T1` 进常驻区 `long_term`（不受配额限制，且 `reserved=True`）；其余进「相关记忆卡」
    `cards`。

    **这是接口未定项**：§6 的四个分区按**内容种类**切（原文 / 卡片 / 决策待办 / 冲突提示），
    而卡上没有「种类」标签。`todo` 与 `conflict` 两个分区留给原生的待办条目与冲突提示，
    本轮没有它们的产出者，所以这里**不猜**：只做「常驻区 vs 卡片区」的二分。
    等 §6 的分区路由被契约定义时，改这一个函数即可，`recall()` 不需要动。
    """
    return "long_term" if tier in ALWAYS_ON_TIERS else "cards"


def _to_item(row: Mapping[str, Any]) -> Item:
    """`scoring.select` 的一行 → `pack.Item`。

    `text_lite` 取 `summary`：§6 规定 lite = 只留 `summary`。
    `text_full` 用 `verify.card_text`（按 `CARD_TEXT_FIELDS` 拼，**不含 `raw_quote`**），
    所以 full/lite 的差别就是「整张卡」与「只有摘要」——与契约口径一致，不另造一种渲染。
    """
    card = row["card"]
    tier = str(card["tier"])
    return Item(
        id=str(card["id"]),
        section=_section_for(tier),
        tier=tier,
        score=float(row["score"]),
        text_full=card_text(card),
        text_lite=str(card.get("summary") or ""),
        reserved=tier in ALWAYS_ON_TIERS,
    )


def _print_stats(stats: RecallStats) -> None:
    """诊断输出。**只由 `verbose=True` 触发**——召回本身不允许有副作用。"""
    print(f"  库里 active 卡       {stats.cards:>5}")
    print(f"  定性硬门挡掉         {stats.qualitative_blocked:>5}  （卡龄窗口 + T8/T9 载体禁令）")
    print(f"  定量硬门挡掉         {stats.quantitative_blocked:>5}  （q ≤ θ）")
    print(f"  过硬门               {stats.gated:>5}")
    print(f"  装箱降级 full→lite   {stats.degraded:>5}")
    print(f"  装箱丢弃             {stats.dropped:>5}  （no_space / quota_exhausted）")
    print(
        f"  预算                 {stats.tokens_used:>5} / {stats.budget} tok"
        + ("  ⚠ over_budget" if stats.over_budget else "")
    )


def recall(
    store: Store,
    *,
    now: datetime,
    budget_tokens: int,
    query_cos: Mapping[str, float] | None = None,
    query_entities: Iterable[str] = (),
    task_hit_ids: Iterable[str] = (),
    open_task_ids: Iterable[str] = (),
    explicit_recall: bool = False,
    theta: float = THETA,
    theta_high: float = THETA_HIGH,
    scheme: str = ADOPTED_SCHEME,
    session_id: str | None = None,
    tier: str | None = None,
    quotas: Mapping[str, float] | None = None,
    estimator: Callable[[str], int] = estimate_tokens,
    verbose: bool = False,
) -> dict[str, Any]:
    """一次召回：读卡 → 定性硬门 → 打分 → 定量硬门 → 装箱。

    参数默认值**就是生产配置**（`ADOPTED_SCHEME` / `THETA` / `THETA_HIGH` / `TIME_FLOOR`）：
    默认调用即 v1.2 的定值，不需要再传参。要复现历史行为请显式传旧值。

    返回字典（键集合是契约的一部分，测试钉住）：
    - `"items"`：装箱后的 `list[Item]`，按分降序；
    - `"remaining_budget"`：`budget_tokens − tokens_used`（超支时为 `0`，看 `over_budget`）；
    - `"stats"`：`RecallStats`，每一步砍了多少张；
    - `"blocks"`：`[(section, text)]`，可直接拼进提示词；
    - `"pack_result"`：`pack` 的原始结果，含丢弃理由与降级明细（审计用）。
    """
    # 先把可迭代参数实体化：既要传给 select()，又要在这里重算一遍做计数，
    # 生成器只消费一次——不实体化就会在第二遍悄悄变成空序列。
    entities = list(query_entities)
    task_hits = set(task_hit_ids)
    open_tasks = set(open_task_ids)
    cos_map = dict(query_cos or {})

    cards = store.cards(session_id=session_id, tier=tier)
    ranked = select(
        cards,
        now=now,
        query_entities=entities,
        query_cos=cos_map,
        task_hit_ids=task_hits,
        open_task_ids=open_tasks,
        explicit_recall=explicit_recall,
        theta=theta,
        theta_high=theta_high,
        scheme=scheme,
    )

    # 两段硬门的**计数**。规则来自 scoring 的两个谓词，这里只负责数；
    # 数完还要与 select() 的结果对齐——否则说明规则被复制成了两份。
    qualitative_blocked = 0
    quantitative_blocked = 0
    survivors: set[str] = set()
    for card in cards:
        card_id = str(card["id"])
        delta = card_delta_days(card, now)
        if not passes_qualitative_gate(
            str(card["tier"]), delta_days=delta, explicit_recall=explicit_recall
        ):
            qualitative_blocked += 1
            continue
        value = score_card(
            card,
            now=now,
            query_cos=cos_map.get(card_id, 0.0),
            query_entities=entities,
            task_hit=card_id in task_hits,
            delta_days=delta,
            scheme=scheme,
        )
        if not passes_quantitative_gate(
            str(card["tier"]),
            score=value,
            has_open_task=card_id in open_tasks,
            theta=theta,
            theta_high=theta_high,
        ):
            quantitative_blocked += 1
            continue
        survivors.add(card_id)
    gated_ids = {str(row["card"]["id"]) for row in ranked}
    if survivors != gated_ids:
        raise AssertionError(
            "两段硬门的合取与 scoring.select() 的结果不一致，差集为 "
            f"{sorted(survivors ^ gated_ids)}：规则被复制成了两份。"
        )

    items = [_to_item(row) for row in ranked]
    result: PackResult = pack(items, budget_tokens, estimator=estimator, quotas=quotas)
    stats = RecallStats(
        cards=len(cards),
        qualitative_blocked=qualitative_blocked,
        quantitative_blocked=quantitative_blocked,
        packed=len(result.placements),
        degraded=len(result.degraded()),
        dropped=len(result.dropped),
        tokens_used=result.tokens_used,
        budget=budget_tokens,
        over_budget=result.over_budget,
    )
    if verbose:
        _print_stats(stats)
    return {
        "items": items,
        "remaining_budget": stats.remaining_budget,
        "stats": stats,
        "blocks": result.blocks(),
        "pack_result": result,
    }


__all__ = ["RecallStats", "recall"]
