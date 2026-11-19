"""Минимальный валидатор JSON Schema (подмножество draft 2020-12) на stdlib.

Поддержка: type, properties, required, additionalProperties(bool|schema), items,
enum, const, pattern, minimum, maximum, minLength, maxLength, minItems, maxItems,
uniqueItems, anyOf, oneOf, $ref (#/$defs/...). "integer" исключает bool и float,
"number" — только int (float в проекте запрещён).
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schemas"


def _type_ok(value, t: str) -> bool:
    if t == "null":
        return value is None
    if t == "boolean":
        return isinstance(value, bool)
    if t == "integer" or t == "number":
        return isinstance(value, int) and not isinstance(value, bool)
    if t == "string":
        return isinstance(value, str)
    if t == "array":
        return isinstance(value, list)
    if t == "object":
        return isinstance(value, dict)
    raise ValueError(f"unsupported type {t}")


@lru_cache(maxsize=None)
def _compile(p: str) -> re.Pattern:
    return re.compile(p)


def validate(instance, schema: dict, root: dict | None = None, path: str = "$", max_errors: int = 20) -> list[str]:
    errors: list[str] = []
    _validate(instance, schema, root or schema, path, errors, max_errors)
    return errors


def _validate(v, s: dict, root: dict, path: str, errors: list[str], max_errors: int) -> None:
    if len(errors) >= max_errors:
        return
    if "$ref" in s:
        ref = s["$ref"]
        if not ref.startswith("#/$defs/"):
            raise ValueError("only local $defs refs supported")
        _validate(v, root["$defs"][ref[len("#/$defs/"):]], root, path, errors, max_errors)
        return
    if "anyOf" in s:
        if not any(not validate(v, sub, root, path) for sub in s["anyOf"]):
            errors.append(f"{path}: does not match anyOf")
        return
    if "oneOf" in s:
        n = sum(1 for sub in s["oneOf"] if not validate(v, sub, root, path))
        if n != 1:
            errors.append(f"{path}: must match exactly one schema in oneOf")
        return
    t = s.get("type")
    if t is not None:
        types = t if isinstance(t, list) else [t]
        if not any(_type_ok(v, tt) for tt in types):
            errors.append(f"{path}: expected type {t}")
            return
    if "const" in s and v != s["const"]:
        errors.append(f"{path}: must equal const")
    if "enum" in s and v not in s["enum"]:
        errors.append(f"{path}: not in enum")
    if isinstance(v, str):
        if "minLength" in s and len(v) < s["minLength"]:
            errors.append(f"{path}: string too short")
        if "maxLength" in s and len(v) > s["maxLength"]:
            errors.append(f"{path}: string too long")
        if "pattern" in s and not _compile(s["pattern"]).search(v):
            errors.append(f"{path}: pattern mismatch")
    if isinstance(v, int) and not isinstance(v, bool):
        if "minimum" in s and v < s["minimum"]:
            errors.append(f"{path}: below minimum")
        if "maximum" in s and v > s["maximum"]:
            errors.append(f"{path}: above maximum")
    if isinstance(v, list):
        if "minItems" in s and len(v) < s["minItems"]:
            errors.append(f"{path}: too few items")
        if "maxItems" in s and len(v) > s["maxItems"]:
            errors.append(f"{path}: too many items")
            return
        if s.get("uniqueItems"):
            seen = set()
            for item in v:
                k = json.dumps(item, sort_keys=True)
                if k in seen:
                    errors.append(f"{path}: items not unique")
                    break
                seen.add(k)
        if "items" in s:
            for i, item in enumerate(v):
                _validate(item, s["items"], root, f"{path}[{i}]", errors, max_errors)
    if isinstance(v, dict):
        props = s.get("properties", {})
        for req in s.get("required", []):
            if req not in v:
                errors.append(f"{path}: missing required '{req}'")
        addl = s.get("additionalProperties", True)
        for k, val in v.items():
            if k in props:
                _validate(val, props[k], root, f"{path}.{k}", errors, max_errors)
            elif addl is False:
                errors.append(f"{path}: unexpected property")
            elif isinstance(addl, dict):
                _validate(val, addl, root, f"{path}.{k}", errors, max_errors)


@lru_cache(maxsize=None)
def load_schema(name: str) -> dict:
    with open(SCHEMA_DIR / f"{name}.schema.json", "r", encoding="utf-8") as f:
        return json.load(f)


def check(instance, name: str) -> list[str]:
    return validate(instance, load_schema(name))
