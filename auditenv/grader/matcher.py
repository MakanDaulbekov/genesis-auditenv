"""Общий matcher (раздел 05). Код нормализации, допусков и tie-break опубликован здесь.

Нормализация вывода (finding):
  * структура проверяется схемой findings.schema.json (строго: лишние ключи, float/str/bool
    в суммах, отрицательные суммы, превышение лимитов -> неверная структура);
  * episode_id обязан совпадать с эпизодом;
  * type — из закрытого enum;
  * affected_ids и evidence_ids: doc_id заменяется на invoice_id/payment_id того же
    документа (алиас, известный из открытых записей); затем множество -> отсортированный список;
  * currency -> верхний регистр.
TP требует ВСЕГО сразу: тот же type, те же канонические affected_ids, |claimed - expected| <=
tolerance_minor, та же валюта, evidence_ids ⊇ хотя бы одного допустимого набора.
Сопоставление взаимно-однозначное: максимальное паросочетание (алгоритм Куна). Порядок
перебора: выводы — по индексу в submit, эталон — по issue_id. Размер максимального
паросочетания от порядка не зависит, поэтому TP/FP/FN инвариантны к перестановкам.
Текст explanation на сопоставление не влияет.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..common.jsonschema_lite import check

# Синтаксис ID: буква/цифра, затем [A-Za-z0-9_.:-], до 128 символов (с запасом для переименованных
# ID жюри, напр. UUID). Пути, пробелы и управляющие символы — неверная структура.
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
MAX_FINDINGS_BYTES = 256 * 1024


class InvalidFindings(Exception):
    pass


@dataclass(frozen=True)
class NormFinding:
    index: int
    type: str
    affected: tuple[str, ...]
    amount: int
    currency: str
    evidence: frozenset


def alias_map_from_tables(invoices: list[dict], payments: list[dict]) -> dict[str, str]:
    m = {}
    for r in invoices:
        m[r["doc_id"]] = r["invoice_id"]
    for r in payments:
        m[r["doc_id"]] = r["payment_id"]
    return m


def odd_known_ids(ids) -> frozenset:
    """ID из открытых записей эпизода, не проходящие ID_RE (бывает только после смены ID жюри в свой
    формат: «INV/2026/0001», «{uuid}», пробел, кириллица). У эпизодов генератора множество пустое."""
    return frozenset(x for x in ids if isinstance(x, str) and x and not ID_RE.match(x))


def normalize(raw, episode_id: str, alias: dict[str, str], known_ids: frozenset = frozenset()) -> list[NormFinding]:
    """-> список нормализованных выводов или InvalidFindings (награда -1).

    ID допустим, если проходит ID_RE или реально есть в открытых записях эпизода (known_ids, см. odd_known_ids):
    честный ответ на эпизод с переименованными жюри ID не должен становиться «неверной структурой»."""
    try:
        size = len(json.dumps(raw, ensure_ascii=False))
    except (TypeError, ValueError):
        raise InvalidFindings("not JSON-serializable") from None
    if size > MAX_FINDINGS_BYTES:
        raise InvalidFindings("findings too large")
    errs = check(raw, "findings")
    if errs:
        raise InvalidFindings("schema: " + "; ".join(errs[:3]))
    if raw["episode_id"] != episode_id:
        raise InvalidFindings("episode_id mismatch")
    out = []
    for i, f in enumerate(raw["findings"]):
        for x in f["affected_ids"] + f["evidence_ids"]:
            if not ID_RE.match(x) and x not in known_ids:
                raise InvalidFindings("malformed id")
        affected = tuple(sorted({alias.get(x, x) for x in f["affected_ids"]}))
        evidence = frozenset(alias.get(x, x) for x in f["evidence_ids"])
        out.append(NormFinding(i, f["type"], affected, f["claimed_amount_minor"], f["currency"].upper(), evidence))
    return out


def is_match(f: NormFinding, t: dict, tolerance: int) -> bool:
    return (f.type == t["type"]
            and f.affected == tuple(t["affected_ids"])
            and abs(f.amount - t["expected_amount_minor"]) <= tolerance
            and f.currency == t["currency"]
            and any(set(es) <= f.evidence for es in t["evidence_sets"]))


def max_matching(findings: list[NormFinding], truth: list[dict], tolerance: int) -> list[tuple[int, str]]:
    issues = sorted(truth, key=lambda t: t["issue_id"])
    adj = {f.index: [j for j, t in enumerate(issues) if is_match(f, t, tolerance)] for f in findings}
    match_t: dict[int, int] = {}

    def try_assign(fi: int, seen: set) -> bool:
        for j in adj[fi]:
            if j in seen:
                continue
            seen.add(j)
            if j not in match_t or try_assign(match_t[j], seen):
                match_t[j] = fi
                return True
        return False

    for f in sorted(findings, key=lambda f: f.index):
        try_assign(f.index, set())
    return sorted((fi, issues[j]["issue_id"]) for j, fi in match_t.items())
