"""Детерминированный криптографический ГПСЧ на SHA-256 (counter mode).

Почему не голый random.Random(seed):
  * Mersenne Twister линеен над GF(2): по достаточному числу наблюдаемых выходов
    (ID, суммы, времена в открытых записях) состояние восстанавливается, после чего
    открытый генератор позволяет воспроизвести «чистый» мир и найти мутации диффом.
  * SHA-256 в режиме счётчика такой атаки не допускает, при этом полностью
    детерминирован и не зависит от PYTHONHASHSEED.
Подпотоки (world / mutations / ids / shuffle) получают независимые ключи.
float-методы запрещены: random() бросает исключение.
"""
from __future__ import annotations

import hashlib
import random


def derive_key(*parts: object) -> bytes:
    h = hashlib.sha256()
    for p in parts:
        b = p if isinstance(p, bytes) else str(p).encode("utf-8")
        h.update(len(b).to_bytes(8, "big"))
        h.update(b)
    return h.digest()


class DRBG(random.Random):
    """random.Random с источником битов SHA-256(key || counter)."""

    def __init__(self, key: bytes):  # noqa: D401
        self._key = bytes(key)
        self._counter = 0
        self._buf = b""
        super().__init__(0)

    # random.Random вызывает seed() из __init__; игнорируем — источник битов свой.
    def seed(self, a=None, version=2):  # type: ignore[override]
        return None

    def _bytes(self, n: int) -> bytes:
        while len(self._buf) < n:
            self._buf += hashlib.sha256(self._key + self._counter.to_bytes(8, "big")).digest()
            self._counter += 1
        out, self._buf = self._buf[:n], self._buf[n:]
        return out

    def getrandbits(self, k: int) -> int:  # type: ignore[override]
        if k < 0:
            raise ValueError("k must be non-negative")
        if k == 0:
            return 0
        nbytes = (k + 7) // 8
        v = int.from_bytes(self._bytes(nbytes), "big")
        return v >> (nbytes * 8 - k)

    def random(self):  # type: ignore[override]
        raise RuntimeError("float randomness is forbidden in auditenv")

    def getstate(self):  # type: ignore[override]
        raise RuntimeError("state export disabled")

    def setstate(self, state):  # type: ignore[override]
        raise RuntimeError("state import disabled")

    def child(self, label: str) -> "DRBG":
        return DRBG(derive_key(self._key, "child", label))

    def __repr__(self) -> str:  # не раскрываем ключ
        return "<DRBG>"
