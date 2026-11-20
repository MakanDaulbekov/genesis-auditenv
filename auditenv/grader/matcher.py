"""Общий matcher (раздел 05). Код нормализации, допусков и tie-break опубликован здесь.

Соглашения совпадают с «Genesis FinTech: определения проверок» (20.11.2026, раздел «Точные
соглашения matcher»):
  * структура: схема findings.schema.json — обязательные поля и их JSON-типы, лимиты кейса
    (≤30 выводов, ≤10 evidence_ids в выводе ДО удаления повторов, explanation ≤1000 code points),
    ID — строки 1..128 символов; claimed_amount_minor — JSON-целое (не bool/строка/дробь).
    Нарушение -> неверная структура всего submit, награда -1. Лишние ключи игнорируются;
  * episode_id обязан совпадать с эпизодом (иначе неверная структура);
  * ID сравниваются ТОЧНО (регистр, пробелы); единственное документированное правило — doc_id
    счёта/платежа заменяется на invoice_id/payment_id того же документа (алиас из открытых записей);
  * affected_ids и evidence_ids — множества (порядок и повторы не важны);
  * evidence_ids: если хоть один ID не существует в открытых записях эпизода, вывод не
    сопоставляется (FP), а не раскрывает закрытую запись;
  * type и currency — точное совпадение строк (без смены регистра и обрезки пробелов);
    неизвестный тип, неверная валюта, сумма или доказательства — FP, а не авария эпизода.
TP требует ВСЕГО сразу: тот же type, те же канонические affected_ids, |claimed - expected| <=
tolerance_minor, та же валюта, все evidence_ids существуют и включают хотя бы один допустимый набор.
Сопоставление взаимно-однозначное: максимальное паросочетание. Среди паросочетаний максимальной
мощности выбирается лексикографически наименьшая последовательность пар (issue_id, индекс вывода),
отсортированная по issue_id (сравнение строк по Unicode code points). TP/FP/FN от порядка выводов
не зависят. Текст explanation на сопоставление не влияет.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..common.jsonschema_lite import check

# Наш собственный формат ID (генератор выдаёт только такие). Для структуры submit НЕ используется:
# по определениям жюри ID — любая строка 1..128, а несуществующий ID даёт FP.
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
    evidence_exists: bool = True


def alias_map_from_tables(invoices: list[dict], payments: list[dict]) -> dict[str, str]:
    m = {}
    for r in invoices:
        m[r["doc_id"]] = r["invoice_id"]
    for r in payments:
        m[r["doc_id"]] = r["payment_id"]
    return m


def public_ids(tables: dict) -> frozenset:
    """Все ID открытых записей эпизода: строковые поля *_id таблиц ledger/invoices/payments,
    fx_id строк курсов и doc_id документов. Строится из тех же открытых файлов, что видит агент."""
    from ..env.store import fx_id
    out = set()
    for t in ("ledger", "invoices", "payments"):
        for r in tables.get(t, ()):
            for k, v in r.items():
                if k.endswith("_id") and isinstance(v, str) and v:
                    out.add(v)
    for r in tables.get("fx", ()):
        out.add(fx_id(r))
    for d in tables.get("documents", ()):
        if isinstance(d.get("doc_id"), str):
            out.add(d["doc_id"])
    return frozenset(out)


def public_ids_from_files(files: dict) -> frozenset:
    from ..env.store import parse_fx_csv, parse_jsonl, parse_ledger_csv
    return public_ids({"ledger": parse_ledger_csv(files["ledger.csv"]),
                       "invoices": parse_jsonl(files["invoices.jsonl"]),
                       "payments": parse_jsonl(files["payments.jsonl"]),
                       "fx": parse_fx_csv(files["fx.csv"]),
                       "documents": parse_jsonl(files["documents.jsonl"])})


def tolerance_from_files(files: dict) -> int:
    """Опубликованный допуск суммы τ (целые минимальные единицы): manifest.rounding.tolerance_minor и
    policy.matching.tolerance_minor. В наших эпизодах оба равны 0; если жюри задало τ в одном из мест,
    действует большее из опубликованных значений (τ не выбирается по ответу агента)."""
    vals = []
    for name, path in (("manifest.json", ("rounding", "tolerance_minor")), ("policy.json", ("matching", "tolerance_minor"))):
        try:
            v = json.loads(files[name])
            for k in path:
                v = v[k]
        except (KeyError, TypeError, ValueError):
            continue
        if isinstance(v, int) and not isinstance(v, bool) and v >= 0:
            vals.append(v)
    return max(vals) if vals else 0


def structure_error(raw, episode_id: str) -> str | None:
    """Проверка СТРУКТУРЫ submit без эталона (её можно сообщить политике). None — структура верна."""
    try:
        size = len(json.dumps(raw, ensure_ascii=False))
    except (TypeError, ValueError, RecursionError):
        return "not JSON-serializable"
    if size > MAX_FINDINGS_BYTES:
        return "findings too large"
    errs = check(raw, "findings")
    if errs:
        return "schema: " + "; ".join(errs[:3])
    if raw["episode_id"] != episode_id:
        return "episode_id mismatch"
    return None


def normalize(raw, episode_id: str, alias: dict[str, str], known_ids: frozenset | None = None) -> list[NormFinding]:
    """-> список нормализованных выводов или InvalidFindings (награда -1).

    known_ids — все ID открытых записей эпизода (public_ids). Если задано, вывод с несуществующим
    evidence_id не сопоставляется (FP). None/пусто — проверка существования не выполняется
    (unit-тесты matcher без эпизода)."""
    e = structure_error(raw, episode_id)
    if e:
        raise InvalidFindings(e)
    out = []
    for i, f in enumerate(raw["findings"]):
        affected = tuple(sorted({alias.get(x, x) for x in f["affected_ids"]}))
        evidence = frozenset(alias.get(x, x) for x in f["evidence_ids"])
        exists = not known_ids or all(x in known_ids for x in f["evidence_ids"])
        out.append(NormFinding(i, f["type"], affected, f["claimed_amount_minor"], f["currency"], evidence, exists))
    return out


def is_match(f: NormFinding, t: dict, tolerance: int) -> bool:
    return (f.evidence_exists
            and f.type == t["type"]
            and f.affected == tuple(t["affected_ids"])
            and abs(f.amount - t["expected_amount_minor"]) <= tolerance
            and f.currency == t["currency"]
            and any(set(es) <= f.evidence for es in t["evidence_sets"]))


def _max_card(adj: dict[int, list[int]], issues: list[int], banned: set[int]) -> int:
    """Мощность максимального паросочетания issues -> выводы (Кун), без выводов из banned."""
    match_f: dict[int, int] = {}

    def aug(j: int, seen: set) -> bool:
        for fi in adj[j]:
            if fi in banned or fi in seen:
                continue
            seen.add(fi)
            if fi not in match_f or aug(match_f[fi], seen):
                match_f[fi] = j
                return True
        return False

    return sum(1 for j in issues if aug(j, set()))


def max_matching(findings: list[NormFinding], truth: list[dict], tolerance: int) -> list[tuple[int, str]]:
    """Максимальное паросочетание с детерминированным выбором (см. docstring модуля).
    -> [(индекс вывода, issue_id)], отсортировано по индексу вывода."""
    issues = sorted(truth, key=lambda t: t["issue_id"])
    adj = {j: sorted(f.index for f in findings if is_match(f, t, tolerance)) for j, t in enumerate(issues)}
    order = list(range(len(issues)))
    target = _max_card(adj, order, set())
    used: set[int] = set()
    pairs: list[tuple[int, str]] = []
    for pos, j in enumerate(order):
        rest = order[pos + 1:]
        for fi in adj[j]:
            if fi in used:
                continue
            if len(pairs) + 1 + _max_card(adj, rest, used | {fi}) == target:
                used.add(fi)
                pairs.append((fi, issues[j]["issue_id"]))
                break
    return sorted(pairs)
