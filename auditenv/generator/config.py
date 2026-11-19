"""Конфиг генератора: дефолт, слияние частичного конфига, понятная валидация.

reset(seed) без config -> configs/default.json; частичный config мержится поверх дефолта.
Ошибка валидации называет поле и ожидание, ничего не раскрывая о внутреннем состоянии.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from ..common.money import CURRENCY_EXPONENTS

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.json"
ERROR_TYPES = ("duplicate_invoice", "missing_or_partial_allocation", "fx_wrong_date_or_rate", "period_cutoff_error")
SCENARIOS = ("partial_fx_boundary", "duplicate_cutoff")
TOP_KEYS = {"config_version", "schema_version", "size", "currencies", "distractors", "errors", "interactions", "clean"}
# Верхние границы выбраны так, чтобы эпизод оставался в диапазоне раздела 03 (≤2 000 проводок,
# ≤200 документов) и честная политика (competent_reconciler) укладывалась в 100 шагов с запасом:
# худший замеренный случай n_invoices=75, n_expenses=700 -> см. REPORT.md (solvability).
SIZE_KEYS = {"n_customers": (1, 30), "n_vendors": (1, 20), "n_invoices": (10, 75), "n_prior_invoices": (0, 10),
             "n_next_invoices": (0, 6), "n_expenses": (0, 700)}
DISTRACTOR_RANGES = {"recurring_groups": (0, 6), "same_total_diff_lines": (0, 6), "tz_boundary_invoices": (0, 8),
                     "tz_boundary_next_invoices": (0, 6), "tz_boundary_receipts": (0, 6), "prepayments": (0, 8),
                     "next_prepayments": (0, 6)}
DISTRACTOR_PCT = ("refund_pct", "batch_payment_pct")
PAYMENT_MIX = ("none", "full", "split", "partial", "overpay")


class ConfigError(ValueError):
    pass


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_default() -> dict:
    return json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))


def _int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _range(path: str, v, lo: int, hi: int) -> None:
    """int или [lo, hi] в допустимых границах."""
    if _int(v):
        if not lo <= v <= hi:
            raise ConfigError(f"config.{path}: ожидается целое от {lo} до {hi}, получено {v}")
        return
    if isinstance(v, list) and len(v) == 2 and all(_int(x) for x in v):
        if not (lo <= v[0] <= v[1] <= hi):
            raise ConfigError(f"config.{path}: ожидается [min, max] с {lo} ≤ min ≤ max ≤ {hi}, получено {v}")
        return
    raise ConfigError(f"config.{path}: ожидается целое или [min, max], получено {type(v).__name__}")


def _hi(v) -> int:
    return v if _int(v) else v[1]


def validate_config(cfg) -> None:
    if not isinstance(cfg, dict):
        raise ConfigError("config: ожидается JSON-объект")
    extra = sorted(set(cfg) - TOP_KEYS)
    if extra:
        raise ConfigError(f"config: неизвестные поля {extra}; допустимы {sorted(TOP_KEYS)}")
    if not isinstance(cfg.get("config_version"), str) or not cfg["config_version"]:
        raise ConfigError("config.config_version: ожидается непустая строка")
    if not isinstance(cfg.get("clean", False), bool):
        raise ConfigError("config.clean: ожидается true/false")
    for sect in ("size", "currencies", "distractors", "errors", "interactions"):
        if not isinstance(cfg.get(sect), dict):
            raise ConfigError(f"config.{sect}: ожидается объект")
    for k, v in cfg["size"].items():
        if k not in SIZE_KEYS:
            raise ConfigError(f"config.size.{k}: неизвестное поле; допустимы {sorted(SIZE_KEYS)}")
        _range(f"size.{k}", v, *SIZE_KEYS[k])
    cur = cfg["currencies"]
    for k in ("base", "foreign"):
        v = cur.get(k)
        if not isinstance(v, list) or not v or any(c not in CURRENCY_EXPONENTS for c in v):
            raise ConfigError(f"config.currencies.{k}: ожидается непустой список из {sorted(CURRENCY_EXPONENTS)}")
    share = cur.get("foreign_customer_share_pct")
    if not _int(share) or not 0 <= share <= 100:
        raise ConfigError("config.currencies.foreign_customer_share_pct: ожидается целое 0..100")
    if set(cur) - {"base", "foreign", "foreign_customer_share_pct"}:
        raise ConfigError("config.currencies: неизвестные поля")
    d = cfg["distractors"]
    for k, v in d.items():
        if k in DISTRACTOR_RANGES:
            _range(f"distractors.{k}", v, *DISTRACTOR_RANGES[k])
        elif k in DISTRACTOR_PCT:
            if not _int(v) or not 0 <= v <= 100:
                raise ConfigError(f"config.distractors.{k}: ожидается целое 0..100")
        elif k == "payment_mix_pct":
            if not isinstance(v, dict) or set(v) != set(PAYMENT_MIX) or not all(_int(x) and x >= 0 for x in v.values()) \
                    or sum(v.values()) <= 0:
                raise ConfigError(f"config.distractors.payment_mix_pct: ожидаются неотрицательные веса {list(PAYMENT_MIX)}")
        else:
            raise ConfigError(f"config.distractors.{k}: неизвестное поле")
    for k, v in cfg["errors"].items():
        if k not in ERROR_TYPES:
            raise ConfigError(f"config.errors.{k}: неизвестный тип ошибки; допустимы {list(ERROR_TYPES)}")
        _range(f"errors.{k}", v, 0, 12)
    for k, v in cfg["interactions"].items():
        if k not in SCENARIOS:
            raise ConfigError(f"config.interactions.{k}: неизвестный сценарий; допустимы {list(SCENARIOS)}")
        _range(f"interactions.{k}", v, 0, 3)
    if not cfg.get("clean") and 2 * sum(_hi(v) for v in cfg["interactions"].values()) > 12:
        raise ConfigError("config.interactions: сценарии дают больше 12 ошибок (максимум эпизода)")


def resolve_config(config) -> dict:
    """None -> дефолт; dict -> дефолт ⊕ config (глубокое слияние). Затем валидация."""
    base = load_default()
    if config is None:
        cfg = base
    elif isinstance(config, dict):
        cfg = deep_merge(base, config)
    else:
        raise ConfigError("config: ожидается JSON-объект или None")
    validate_config(cfg)
    return cfg
