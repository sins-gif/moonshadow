"""统一时钟注入点。

设计约束：衰减、升降级、有效期判定全部依赖「现在」，因此核心代码禁止直接调用
``datetime.now()``——一律通过 ``Clock`` 注入，否则无法写确定性测试。
内部统一存 UTC（aware），展示时保留原始偏移。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Protocol

_TS_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
)


class Clock(Protocol):
    """时钟协议。任何实现都必须返回 aware datetime。"""

    def now(self) -> datetime:  # pragma: no cover - protocol
        ...


def to_utc(moment: datetime) -> datetime:
    """归一到 aware UTC。naive 时间按 UTC 解释（不猜本地时区）。"""
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def parse_ts(text: str) -> datetime:
    """解析 ISO-8601 时间戳，返回 aware UTC datetime。"""
    if not isinstance(text, str) or not _TS_RE.match(text):
        raise ValueError(f"非法时间戳：{text!r}")
    value = text.replace(" ", "T")
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return to_utc(datetime.fromisoformat(value))


def to_iso(moment: datetime) -> str:
    """序列化为 ISO-8601（秒精度，带偏移）。"""
    return to_utc(moment).isoformat(timespec="seconds")


def days_between(earlier: datetime, later: datetime) -> float:
    """两个时刻之间的天数（可正可负，不做绝对值）。"""
    return (to_utc(later) - to_utc(earlier)).total_seconds() / 86400.0


@dataclass(frozen=True)
class SystemClock:
    """生产用时钟。"""

    def now(self) -> datetime:
        return datetime.now(timezone.utc)


@dataclass
class FixedClock:
    """测试用时钟：可显式推进，保证衰减/升降级结果可复现。"""

    _moment: datetime = field(default_factory=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc))

    def __post_init__(self) -> None:
        self._moment = to_utc(self._moment)

    def now(self) -> datetime:
        return self._moment

    def set(self, moment: datetime) -> datetime:
        self._moment = to_utc(moment)
        return self._moment

    def advance(self, days: float = 0.0, seconds: float = 0.0) -> datetime:
        self._moment = self._moment + timedelta(days=days, seconds=seconds)
        return self._moment
