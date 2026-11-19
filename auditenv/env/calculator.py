"""Безопасный калькулятор: разбор через ast, белый список узлов, только Decimal.

Разрешено: десятичные литералы вида 123 или 123.45, бинарные + - * /, унарные + -, скобки.
Запрещено всё остальное: имена, вызовы, атрибуты, степень, %, //, экспонента в литерале,
подчёркивания, строки, сравнения. Лимиты: длина, число узлов, глубина, размер литерала,
величина результата. eval/exec не используются.
"""
from __future__ import annotations

import ast
import decimal
import re
from decimal import Decimal

MAX_LEN = 256
MAX_NODES = 64
MAX_DEPTH = 24
MAX_LITERAL_DIGITS = 30
MAX_ABS = Decimal(10) ** 30
_LITERAL_RE = re.compile(r"^\d{1,30}(\.\d{1,30})?$")
_ALLOWED_CHARS = re.compile(r"^[0-9+\-*/(). \t]*$")

_CTX = decimal.Context(prec=40, rounding=decimal.ROUND_HALF_EVEN, Emax=200, Emin=-200,
                       traps=[decimal.InvalidOperation, decimal.DivisionByZero, decimal.Overflow])


class CalcError(Exception):
    pass


def evaluate(expression) -> str:
    if not isinstance(expression, str):
        raise CalcError("expression must be a string")
    if len(expression) == 0 or len(expression) > MAX_LEN:
        raise CalcError("bad length")
    if not _ALLOWED_CHARS.match(expression):
        raise CalcError("forbidden characters")
    if "**" in expression or "//" in expression:
        raise CalcError("forbidden operator")
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        raise CalcError("syntax error") from None
    count = 0
    for _ in ast.walk(tree):
        count += 1
        if count > MAX_NODES:
            raise CalcError("expression too large")
    with decimal.localcontext(_CTX):
        try:
            value = _eval(tree.body, expression, 0)
        except (decimal.InvalidOperation, decimal.DivisionByZero, decimal.Overflow):
            raise CalcError("arithmetic error") from None
        if abs(value) > MAX_ABS:
            raise CalcError("result too large")
        value = (+value).normalize()
        s = format(value, "f")
    if s == "-0":
        s = "0"
    return s


def _eval(node, src: str, depth: int) -> Decimal:
    if depth > MAX_DEPTH:
        raise CalcError("expression too deep")
    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            raise CalcError("forbidden operator")
        a = _eval(node.left, src, depth + 1)
        b = _eval(node.right, src, depth + 1)
        if isinstance(node.op, ast.Add):
            r = a + b
        elif isinstance(node.op, ast.Sub):
            r = a - b
        elif isinstance(node.op, ast.Mult):
            r = a * b
        else:
            if b == 0:
                raise CalcError("division by zero")
            r = a / b
        if abs(r) > MAX_ABS:
            raise CalcError("intermediate too large")
        return r
    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, (ast.USub, ast.UAdd)):
            raise CalcError("forbidden operator")
        v = _eval(node.operand, src, depth + 1)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise CalcError("forbidden literal")
        # Берём исходный текст литерала, а не float-значение ast: точность Decimal сохраняется.
        text = ast.get_source_segment(src, node)
        if text is None or not _LITERAL_RE.match(text):
            raise CalcError("forbidden literal")
        return Decimal(text)
    raise CalcError("forbidden construct")
