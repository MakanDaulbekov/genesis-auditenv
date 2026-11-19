"""Денежная арифметика: только int (minor units) и Decimal. Никаких float."""
from __future__ import annotations

import decimal
from decimal import Decimal

# Показатели разрядности (ISO 4217). Все валюты вымышленного мира берутся отсюда.
CURRENCY_EXPONENTS: dict[str, int] = {
    "USD": 2,
    "EUR": 2,
    "GBP": 2,
    "CHF": 2,
    "KZT": 2,
    "JPY": 0,
}

ROUNDING_MODES = {
    "ROUND_HALF_EVEN": decimal.ROUND_HALF_EVEN,
    "ROUND_HALF_UP": decimal.ROUND_HALF_UP,
}

_CTX = decimal.Context(prec=60, rounding=decimal.ROUND_HALF_EVEN, Emax=10**6, Emin=-(10**6),
                       traps=[decimal.InvalidOperation, decimal.DivisionByZero, decimal.Overflow])


def exponent(currency: str) -> int:
    return CURRENCY_EXPONENTS[currency]


def parse_rate(rate: str) -> Decimal:
    """FX-курс хранится как десятичная строка; float запрещён."""
    if not isinstance(rate, str):
        raise TypeError("rate must be a decimal string")
    d = Decimal(rate)
    if not d.is_finite() or d <= 0:
        raise ValueError("rate must be positive and finite")
    return d


def convert_minor(amount_minor: int, src: str, tgt: str, rate: str, rounding: str = "ROUND_HALF_EVEN") -> int:
    """Перевод суммы в minor units валюты src в minor units валюты tgt.

    rate = сколько единиц tgt стоит одна единица src (в major units).
    result = round(amount_minor * rate * 10^(exp_tgt - exp_src)) по правилу rounding.
    """
    if isinstance(amount_minor, bool) or not isinstance(amount_minor, int):
        raise TypeError("amount_minor must be int")
    if src == tgt:
        return amount_minor
    r = parse_rate(rate)
    shift = exponent(tgt) - exponent(src)
    with decimal.localcontext(_CTX):
        value = Decimal(amount_minor) * r * (Decimal(10) ** shift)
        q = value.quantize(Decimal(1), rounding=ROUNDING_MODES[rounding])
    return int(q)


def format_major(amount_minor: int, currency: str) -> str:
    """Человекочитаемая сумма (только для текстов документов)."""
    e = exponent(currency)
    sign = "-" if amount_minor < 0 else ""
    a = abs(amount_minor)
    if e == 0:
        return f"{sign}{a}"
    return f"{sign}{a // 10**e}.{a % 10**e:0{e}d}"
