"""Token 预算装箱。

v1.0 给了 ``Token预算`` 变量，却没说什么装不下时丢谁。这里定成确定性算法：

1. 常驻区（T0/T1，或显式 reserved 的条目）**无条件进入**；若它单独超预算，
   标记 ``over_budget=True`` 并照常输出——身份与硬约束不允许被静默裁掉。
2. 其余四个分区按配额分配剩余预算，区内按 score 降序贪心填充。
3. 装不下先降级渲染 full → lite；lite 也装不下才丢弃，并记录原因。
4. 某分区的余额滚入下一分区，避免预算浪费。

同输入必得同输出（同分按 id 排序），因此可写确定性测试。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Sequence

#: 分区顺序即装箱顺序，也是最终拼装上下文时的呈现顺序。
SECTIONS: tuple[str, ...] = ("long_term", "recent_raw", "cards", "todo", "conflict")

#: 非常驻分区的配额，合计必须为 1.0。
QUOTAS: dict[str, float] = {
    "recent_raw": 0.30,
    "cards": 0.40,
    "todo": 0.20,
    "conflict": 0.10,
}

RESERVED_SECTIONS = ("long_term",)


def estimate_tokens(text: str) -> int:
    """保守估算：CJK 每字 1 token，其余每 4 字符 1 token（向下取整后取 1 为下限）。"""
    if not text:
        return 0
    cjk = 0
    for ch in text:
        code = ord(ch)
        if 0x4E00 <= code <= 0x9FFF or 0x3000 <= code <= 0x303F or 0xFF00 <= code <= 0xFFEF:
            cjk += 1
    other = len(text) - cjk
    return max(1, int(math.ceil(cjk + other / 4.0)))


@dataclass(frozen=True)
class Item:
    """一个候选片段。``text_lite`` 为空表示不可降级。"""

    id: str
    section: str
    tier: str
    score: float
    text_full: str
    text_lite: str = ""
    reserved: bool = False


@dataclass(frozen=True)
class Placement:
    item: Item
    render: str
    text: str
    tokens: int


@dataclass(frozen=True)
class Dropped:
    item: Item
    reason: str


@dataclass
class PackResult:
    budget: int
    placements: list[Placement] = field(default_factory=list)
    dropped: list[Dropped] = field(default_factory=list)
    tokens_used: int = 0
    reserved_tokens: int = 0
    over_budget: bool = False

    def blocks(self) -> list[tuple[str, str]]:
        """按分区拼装，返回 [(section, text)]，供召回器渲染上下文。"""
        buckets: dict[str, list[str]] = {name: [] for name in SECTIONS}
        for placement in self.placements:
            buckets.setdefault(placement.item.section, []).append(placement.text)
        return [(name, "\n".join(buckets[name])) for name in SECTIONS if buckets.get(name)]

    def degraded(self) -> list[Placement]:
        return [p for p in self.placements if p.render == "lite"]


def _validate_quotas(quotas: Mapping[str, float], sections: Sequence[str]) -> None:
    total = sum(quotas.get(name, 0.0) for name in sections if name not in RESERVED_SECTIONS)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"非常驻分区配额之和必须为 1，当前为 {total}")


def pack(
    items: Iterable[Item],
    budget_tokens: int,
    *,
    estimator: Callable[[str], int] = estimate_tokens,
    quotas: Mapping[str, float] | None = None,
    sections: Sequence[str] = SECTIONS,
) -> PackResult:
    """把候选片段装进 token 预算。"""
    if budget_tokens <= 0:
        raise ValueError("budget_tokens 必须为正整数")
    table = dict(quotas or QUOTAS)
    _validate_quotas(table, sections)

    result = PackResult(budget=budget_tokens)
    grouped: dict[str, list[Item]] = {name: [] for name in sections}
    for item in items:
        if item.section not in grouped:
            raise ValueError(f"未知分区：{item.section!r}")
        grouped[item.section].append(item)

    def take(item: Item, allowed: int) -> bool:
        """尝试装入；先 full 后 lite，都不够则放弃。"""
        for render, text in (("full", item.text_full), ("lite", item.text_lite)):
            if render == "lite" and not text:
                continue
            cost = estimator(text)
            if cost <= allowed:
                result.placements.append(Placement(item=item, render=render, text=text, tokens=cost))
                return True
        return False

    # 1) 常驻区：无条件进入，不受配额限制。
    reserved_ids: set[str] = set()

    def take_reserved(item: Item) -> None:
        for render, text in (("full", item.text_full), ("lite", item.text_lite)):
            if render == "lite" and not text:
                continue
            cost = estimator(text)
            result.placements.append(Placement(item=item, render=render, text=text, tokens=cost))
            return
        raise ValueError(f"常驻条目 {item.id} 两种渲染都为空")

    for item in sorted(grouped.pop("long_term", []), key=lambda i: (-i.score, i.id)):
        take_reserved(item)
        reserved_ids.add(item.id)
    for name in sections:
        for item in sorted(grouped.get(name, []), key=lambda i: (-i.score, i.id)):
            if item.reserved and item.id not in reserved_ids:
                take_reserved(item)
                reserved_ids.add(item.id)

    result.reserved_tokens = sum(p.tokens for p in result.placements)
    remaining = budget_tokens - result.reserved_tokens
    if remaining < 0:
        result.over_budget = True
        remaining = 0
    result.tokens_used = result.reserved_tokens

    # 2) 其余分区：配额 + 滚存，区内贪心。
    rollover = 0
    for name in sections:
        if name in RESERVED_SECTIONS:
            continue
        candidates = sorted(grouped.get(name, []), key=lambda i: (-i.score, i.id))
        if not candidates:
            continue
        allowed = int(remaining * table.get(name, 0.0)) + rollover
        if allowed <= 0:
            result.dropped.extend(Dropped(item, "quota_exhausted") for item in candidates)
            rollover = 0
            continue
        spent = 0
        for item in candidates:
            if take(item, allowed - spent):
                spent += result.placements[-1].tokens
            else:
                result.dropped.append(Dropped(item, "no_space"))
        result.tokens_used += spent
        rollover = max(0, allowed - spent)

    result.over_budget = result.tokens_used > budget_tokens
    return result
