"""Moonshadow v1.1 —— T0-T9 时间分层上下文压缩与记忆协议（可执行骨架）。

分层：SQLite 为唯一真相源，raw/ 为不可变追加日志，chunks/ 为冷存分帧。
凡涉及时间的地方一律注入 Clock，核心代码不出现 datetime.now()。
"""

from .clock import Clock, FixedClock, SystemClock, days_between, parse_ts, to_iso
from .cards import make_card
from .compress import (
    BaselineExtractor,
    CompileReport,
    Extractor,
    build_prompt,
    classify_fast_path,
    classify_tier,
    compile_session,
)
from .eval import EvalReport, GoldCase, load_gold, run_eval
from .ids import card_id, content_id, dedup_key, message_id
from .schema import SchemaError, load_schema, validate_card
from .store import CardWriteResult, RawIntegrityError, Store
from .scoring import (
    THETA,
    THETA_HIGH,
    WEIGHTS,
    decay,
    next_tier,
    passes_gate,
    score_card,
    select,
)
from .tiers import TIER_HALF_LIFE, TIER_ORDER, half_life_for, tier_index
from .verify import Report, extract_key_fields, verify_no_silent_loss
from .pack import Item, PackResult, estimate_tokens, pack

__all__ = [
    "Clock",
    "FixedClock",
    "SystemClock",
    "days_between",
    "parse_ts",
    "to_iso",
    "make_card",
    "BaselineExtractor",
    "CompileReport",
    "Extractor",
    "build_prompt",
    "classify_fast_path",
    "classify_tier",
    "compile_session",
    "EvalReport",
    "GoldCase",
    "load_gold",
    "run_eval",
    "card_id",
    "content_id",
    "dedup_key",
    "message_id",
    "SchemaError",
    "load_schema",
    "validate_card",
    "Store",
    "CardWriteResult",
    "RawIntegrityError",
    "THETA",
    "THETA_HIGH",
    "WEIGHTS",
    "decay",
    "next_tier",
    "passes_gate",
    "score_card",
    "select",
    "TIER_HALF_LIFE",
    "TIER_ORDER",
    "half_life_for",
    "tier_index",
    "Report",
    "extract_key_fields",
    "verify_no_silent_loss",
    "Item",
    "PackResult",
    "estimate_tokens",
    "pack",
]

__version__ = "1.1.0"
SCHEMA_VERSION = 1
