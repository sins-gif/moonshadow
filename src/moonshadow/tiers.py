"""T0-T9 层级定义表（唯一权威来源）。

与 v1.0 的两处修正：
1. T1 的半衰期明确为 365 天（原文档 §7 注释写「T0/T1 都不衰减」，与 §4 表格冲突）。
2. 半衰期是「衰减到一半所需天数」，因此 decay = 0.5 ** (delta / half_life)，
   而不是原文档的 exp(-delta / half_life)（后者是时间常数 τ，在 delta=half_life 时得 0.368）。
"""

from __future__ import annotations

TIER_ORDER: tuple[str, ...] = (
    "T0", "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9",
)

#: 单位：天。0 表示不衰减（仅 T0）。T9 为小时级 => 1/24 天。
TIER_HALF_LIFE: dict[str, float] = {
    "T0": 0.0,
    "T1": 365.0,
    "T2": 180.0,
    "T3": 90.0,
    "T4": 30.0,
    "T5": 14.0,
    "T6": 7.0,
    "T7": 3.0,
    "T8": 1.0,
    "T9": 1.0 / 24.0,
}

TIER_LABEL: dict[str, str] = {
    "T0": "永久核心：身份、硬约束、安全、法律",
    "T1": "长期关键：长期偏好、核心目标、关系",
    "T2": "季度关键：项目决策、重要承诺",
    "T3": "月度重要：近期待办、截止",
    "T4": "周度重要：一般方案、讨论结论",
    "T5": "日常事实：一般信息、偏好片段",
    "T6": "短期上下文：当前任务细节",
    "T7": "临时细节：中间推理、草稿",
    "T8": "低价值：寒暄、确认",
    "T9": "噪音：无意义内容",
}

#: 规则引擎能触及的区间：T0/T1 只能由抽取阶段显式声明，规则绝不升降；
#: 规则降级下限为 T8——绝不把内容「自动」降成可丢弃的 T9。
RULE_TIER_MIN = 2   # T2
RULE_TIER_MAX = 8   # T8


def tier_index(tier: str) -> int:
    try:
        return TIER_ORDER.index(tier)
    except ValueError as exc:  # pragma: no cover - 由 schema 校验兜住
        raise ValueError(f"未知层级：{tier!r}") from exc


def tier_from_index(index: int) -> str:
    if not 0 <= index < len(TIER_ORDER):
        raise ValueError(f"层级下标越界：{index}")
    return TIER_ORDER[index]


def half_life_for(tier: str) -> float:
    """取层级的默认半衰期（天）。0 表示不衰减。"""
    return TIER_HALF_LIFE[tier]


def clamp_rule_tier(index: int) -> int:
    """把规则引擎算出的下标夹到 T2..T8，防止自动升降碰到 T0/T1 或 T9。"""
    return max(RULE_TIER_MIN, min(RULE_TIER_MAX, index))
