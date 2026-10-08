"""JSON Schema 校验器（标准库实现，零依赖）。

只实现 schema.json 用到的关键字子集：type / const / enum / minimum / maximum /
minLength / maxLength / pattern / minItems / uniqueItems / items / required /
properties / additionalProperties。

之所以不引 jsonschema：Phase 0 的契约必须在任何环境都能强制生效，
不能让「装不上依赖」成为跳过校验的理由。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

DEFAULT_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schema.json"

_TYPE_CHECKS = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}


class SchemaError(ValueError):
    """校验失败。message 中逐条列出所有违反项及其 JSON 路径。"""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("记忆卡校验失败：\n  - " + "\n  - ".join(errors))


def _type_ok(value: Any, expected: str | list[str]) -> bool:
    names = [expected] if isinstance(expected, str) else list(expected)
    return any(_TYPE_CHECKS[name](value) for name in names if name in _TYPE_CHECKS)


def _type_label(expected: str | list[str]) -> str:
    return expected if isinstance(expected, str) else " | ".join(expected)


def _collect(instance: Any, schema: dict, path: str, errors: list[str]) -> None:
    if not isinstance(schema, dict):
        return

    if "type" in schema and not _type_ok(instance, schema["type"]):
        errors.append(f"{path}: 期望类型 {_type_label(schema['type'])}，实际 {type(instance).__name__}")
        return  # 类型都不对，后续约束无意义

    if "const" in schema and instance != schema["const"]:
        errors.append(f"{path}: 期望常量 {schema['const']!r}，实际 {instance!r}")

    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: 取值 {instance!r} 不在允许集合 {schema['enum']}")

    if isinstance(instance, str):
        if "minLength" in schema and len(instance) < schema["minLength"]:
            errors.append(f"{path}: 长度 {len(instance)} < minLength {schema['minLength']}")
        if "maxLength" in schema and len(instance) > schema["maxLength"]:
            errors.append(f"{path}: 长度 {len(instance)} > maxLength {schema['maxLength']}")
        if "pattern" in schema and not re.search(schema["pattern"], instance):
            errors.append(f"{path}: 值 {instance!r} 不匹配 pattern {schema['pattern']}")

    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            errors.append(f"{path}: 值 {instance} < minimum {schema['minimum']}")
        if "maximum" in schema and instance > schema["maximum"]:
            errors.append(f"{path}: 值 {instance} > maximum {schema['maximum']}")

    if isinstance(instance, list):
        if "minItems" in schema and len(instance) < schema["minItems"]:
            errors.append(f"{path}: 元素数 {len(instance)} < minItems {schema['minItems']}")
        if schema.get("uniqueItems"):
            seen: set[str] = set()
            for i, item in enumerate(instance):
                key = json.dumps(item, sort_keys=True, ensure_ascii=False)
                if key in seen:
                    errors.append(f"{path}[{i}]: 元素重复 {item!r}")
                seen.add(key)
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for i, item in enumerate(instance):
                _collect(item, item_schema, f"{path}[{i}]", errors)

    if isinstance(instance, dict):
        for name in schema.get("required", []):
            if name not in instance:
                errors.append(f"{path}: 缺少必填字段 {name!r}")
        props = schema.get("properties", {})
        for name, value in instance.items():
            if name in props:
                _collect(value, props[name], f"{path}.{name}", errors)
            else:
                extra = schema.get("additionalProperties", True)
                if extra is False:
                    errors.append(f"{path}: 出现未定义字段 {name!r}")
                elif isinstance(extra, dict):
                    _collect(value, extra, f"{path}.{name}", errors)


def validate(instance: Any, schema: dict) -> None:
    """就地校验，失败抛 SchemaError（一次性列出全部问题）。"""
    errors: list[str] = []
    _collect(instance, schema, "$", errors)
    if errors:
        raise SchemaError(errors)


_SCHEMA_CACHE: dict[Path, dict] = {}


def load_schema(path: str | Path | None = None) -> dict:
    """加载 schema.json（带缓存）。"""
    target = Path(path) if path is not None else DEFAULT_SCHEMA_PATH
    if target not in _SCHEMA_CACHE:
        with target.open(encoding="utf-8") as handle:
            _SCHEMA_CACHE[target] = json.load(handle)
    return _SCHEMA_CACHE[target]


def validate_card(card: dict, schema: dict | None = None) -> dict:
    """校验一张记忆卡；通过则原样返回，便于链式调用。"""
    validate(card, schema if schema is not None else load_schema())
    return card
