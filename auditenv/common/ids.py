"""Непрозрачные ID из детерминированного потока. Не содержат тип ошибки и не идут подряд."""
from __future__ import annotations

from .drbg import DRBG

ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # 32 символа, без 0/O/1/I

PREFIX_LEN = {
    "R": 9,     # ledger row
    "INV": 7,   # invoice
    "PAY": 7,   # payment
    "D": 9,     # document
    "E": 6,     # entity
    "ISS": 8,   # issue (только у оценщика)
}


class IdGen:
    def __init__(self, rng: DRBG):
        self._rng = rng
        self._used: set[str] = set()

    def new(self, prefix: str) -> str:
        n = PREFIX_LEN[prefix]
        while True:
            body = "".join(self._rng.choice(ALPHABET) for _ in range(n))
            ident = f"{prefix}-{body}"
            if ident not in self._used:
                self._used.add(ident)
                return ident

    def reserve(self, ident: str) -> None:
        self._used.add(ident)

    def __repr__(self) -> str:
        return "<IdGen>"
