"""НЕЗАВИСИМЫЙ валидатор эпизода.

Не импортирует код генератора (world/mutations/render/episode/truth): бухгалтерские
правила, конвертация и границы периода переписаны здесь заново по тексту policy.json.
Вход: открытые файлы эпизода + truth.json + provenance.json (только diff «было» и список
отвлекающих случаев). Проверки:
  1. схемы всех записей, manifest = фактические счётчики и хеши файлов;
  2. ДО мутаций (состояние, восстановленное из provenance.diff) все равенства выполняются,
     детекторы не находят ни одной ошибки;
  3. ПОСЛЕ мутаций детекторы находят РОВНО эталон: тот же тип, affected_ids, сумма, валюта;
  4. каждый допустимый набор доказательств сам по себе (только его записи + policy)
     пересчитывает то же расхождение; все ID набора доступны через инструменты;
  5. отвлекающие случаи не дают расхождений;
  6. нет противоречивых/неразрешимых данных (баланс записей, суммы строк, ссылки, ID).
"""
from __future__ import annotations

import csv
import decimal
import hashlib
import io
import json
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from ..common.jsonschema_lite import check

PUBLIC = ("manifest.json", "policy.json", "ledger.csv", "invoices.jsonl", "payments.jsonl", "fx.csv", "documents.jsonl")


# ----------------------------------------------------------------------------- независимые примитивы

def _tz(offset: str) -> timezone:
    sign = 1 if offset[0] == "+" else -1
    return timezone(sign * timedelta(hours=int(offset[1:3]), minutes=int(offset[4:6])))


def _instant(ts: str) -> datetime:
    return datetime.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=_tz(ts[19:]))


def _local_date(ts: str, offset: str) -> date:
    return _instant(ts).astimezone(_tz(offset)).date()


def _in_period(ts: str, pol: dict) -> bool:
    tz = _tz(pol["timezone"])
    s = date.fromisoformat(pol["period"]["start"])
    e = date.fromisoformat(pol["period"]["end"]) + timedelta(days=1)
    t = _instant(ts)
    return datetime(s.year, s.month, s.day, tzinfo=tz) <= t < datetime(e.year, e.month, e.day, tzinfo=tz)


def _convert(amount: int, src: str, tgt: str, rate: str, exps: dict, mode: str) -> int:
    if src == tgt:
        return amount
    ctx = decimal.Context(prec=80, rounding=getattr(decimal, mode))
    v = ctx.multiply(ctx.multiply(Decimal(amount), Decimal(rate)), Decimal(10) ** (exps[tgt] - exps[src]))
    return int(v.quantize(Decimal(1), rounding=getattr(decimal, mode)))


def _canon_lines(lines: list[dict]) -> str:
    return json.dumps(sorted(json.dumps(li, sort_keys=True) for li in lines))


# ----------------------------------------------------------------------------- загрузка

def parse_public(files: dict[str, bytes]) -> dict:
    ledger = []
    for r in csv.DictReader(io.StringIO(files["ledger.csv"].decode("utf-8"))):
        r = dict(r)
        r["debit_minor"] = int(r["debit_minor"])
        r["credit_minor"] = int(r["credit_minor"])
        for k in ("entity_id", "invoice_id", "payment_id"):
            r[k] = r[k] if r[k] != "" else None
        ledger.append(r)
    jl = lambda b: [json.loads(x) for x in b.decode("utf-8").splitlines() if x.strip()]  # noqa: E731
    return {
        "manifest": json.loads(files["manifest.json"]),
        "policy": json.loads(files["policy.json"]),
        "ledger": ledger,
        "invoices": jl(files["invoices.jsonl"]),
        "payments": jl(files["payments.jsonl"]),
        "fx": [dict(r) for r in csv.DictReader(io.StringIO(files["fx.csv"].decode("utf-8")))],
        "documents": jl(files["documents.jsonl"]),
    }


KEYS = {"ledger": "row_id", "invoices": "invoice_id", "payments": "payment_id", "documents": "doc_id"}


def reconstruct_before(t: dict, diff: dict) -> dict:
    out = dict(t)
    for table, k in KEYS.items():
        d = diff[table]
        ins = set(d["inserted"])
        upd = {u["id"]: u["before"] for u in d["updated"]}
        rows = [upd.get(r[k], r) for r in t[table] if r[k] not in ins]
        rows += d["deleted"]
        out[table] = rows
    return out


# ----------------------------------------------------------------------------- аудит (детекторы)

class Book:
    def __init__(self, t: dict):
        self.t = t
        self.pol = t["policy"]
        self.base = self.pol["base_currency"]
        self.tzs = self.pol["timezone"]
        self.exps = self.pol["currency_exponents"]
        self.mode = self.pol["rounding"]["mode"]
        self.inv = {r["invoice_id"]: r for r in t["invoices"]}
        self.pay = {r["payment_id"]: r for r in t["payments"]}
        self.docs = {d["doc_id"]: d for d in t["documents"]}
        self.pay_doc = {p["payment_id"]: self.docs.get(p["doc_id"]) for p in t["payments"]}
        self.fx = {(r["date"], r["source_currency"], r["target_currency"]): r["rate"] for r in t["fx"]}
        self.rows = {r["row_id"]: r for r in t["ledger"]}
        self.by_inv = defaultdict(list)
        self.by_pay = defaultdict(list)
        for r in t["ledger"]:
            if r["payment_id"]:
                self.by_pay[r["payment_id"]].append(r)
            elif r["invoice_id"]:
                self.by_inv[r["invoice_id"]].append(r)

    def to_base(self, amount: int, ccy: str, d: date) -> int | None:
        if ccy == self.base:
            return amount
        rate = self.fx.get((d.isoformat(), ccy, self.base))
        if rate is None:
            return None
        return _convert(amount, ccy, self.base, rate, self.exps, self.mode)

    def invoice_posting(self, iid: str) -> tuple[str | None, dict | None]:
        rows = self.by_inv.get(iid, [])
        ar = [r for r in rows if r["account"] == "accounts_receivable" and r["debit_minor"] > 0]
        rev = [r for r in rows if r["account"] == "revenue"]
        obe = [r for r in rows if r["account"] == "opening_balance_equity"]
        if not ar:
            return None, None
        kind = "issue" if rev else ("opening" if obe else "unknown")
        return kind, ar[0]


def audit(t: dict) -> tuple[list[dict], list[str]]:
    """-> (найденные ошибки, противоречия данных). Полностью независимая реализация правил."""
    b = Book(t)
    issues: list[dict] = []
    bad: list[str] = []

    # --- целостность и равенства
    ids = [r["row_id"] for r in t["ledger"]]
    if len(ids) != len(set(ids)):
        bad.append("duplicate row_id")
    for table, k in (("invoices", "invoice_id"), ("payments", "payment_id"), ("documents", "doc_id")):
        v = [r[k] for r in t[table]]
        if len(v) != len(set(v)):
            bad.append(f"duplicate {k}")
    groups = defaultdict(list)
    for r in t["ledger"]:
        if r["currency"] != b.base:
            bad.append("ledger row not in base currency")
        if not _in_period(r["timestamp"], b.pol):
            bad.append("ledger timestamp outside period")
        if r["payment_id"]:
            g = ("p", r["payment_id"])
        elif r["invoice_id"]:
            g = ("i", r["invoice_id"], r["timestamp"])
        else:
            g = ("e", r["entity_id"], r["timestamp"])
        groups[g].append(r)
        if (r["debit_minor"] > 0) == (r["credit_minor"] > 0):
            bad.append("row must have exactly one side")
        if r["invoice_id"] and r["invoice_id"] not in b.inv:
            bad.append("ledger refers to unknown invoice")
        if r["payment_id"] and r["payment_id"] not in b.pay:
            bad.append("ledger refers to unknown payment")
    for g, rows in groups.items():
        if sum(r["debit_minor"] for r in rows) != sum(r["credit_minor"] for r in rows):
            bad.append("unbalanced entry")
    for inv in t["invoices"]:
        for li in inv["line_items"]:
            if li["quantity"] * li["unit_price_minor"] != li["amount_minor"]:
                bad.append("line amount != qty*unit")
        if sum(li["amount_minor"] for li in inv["line_items"]) != inv["total_minor"]:
            bad.append("line items do not sum to total")
        if _instant(inv["due_at"]) < _instant(inv["issued_at"]):
            bad.append("due before issue")
    for d in t["documents"]:
        rec = d["record"]
        if d["doc_type"] == "invoice":
            if b.inv.get(rec["invoice_id"]) != rec:
                bad.append("invoice document != record")
        else:
            if b.pay.get(rec["payment_id"]) != rec:
                bad.append("payment document != record")
    if len(b.docs) != len(t["invoices"]) + len(t["payments"]):
        bad.append("documents do not cover records")

    # --- A. дубли: одинаковая суть, проведены более одного раза
    posted = {iid for iid in b.inv if b.invoice_posting(iid)[0] is not None}
    dup_groups = defaultdict(list)
    for iid in sorted(posted):
        inv = b.inv[iid]
        dup_groups[(inv["entity_id"], inv["currency"], inv["total_minor"], _canon_lines(inv["line_items"]))].append(iid)
    for key, members in dup_groups.items():
        if len(members) == 2:
            inv = b.inv[members[0]]
            issues.append({"type": "duplicate_invoice", "affected_ids": sorted(members),
                           "amount": inv["total_minor"], "currency": inv["currency"]})
        elif len(members) > 2:
            bad.append("ambiguous duplicate group (>2)")

    # --- B. распределение: invoice_refs должны совпадать с remittance advice
    alloc_by_inv = defaultdict(int)
    for p in t["payments"]:
        doc = b.pay_doc.get(p["payment_id"])
        if doc is None:
            bad.append("payment without document")
            continue
        lines = {l["invoice_id"]: l["amount_minor"] for l in doc["remittance"]["lines"]}
        booked = {r["invoice_id"]: r["amount_minor"] for r in (p["invoice_refs"] or [])}
        if p["invoice_refs"] is not None and len(booked) != len(p["invoice_refs"]):
            bad.append("repeated invoice in refs")
        for iid, a in booked.items():
            alloc_by_inv[iid] += a
            if iid not in b.inv:
                bad.append("ref to unknown invoice")
            elif b.inv[iid]["currency"] != p["currency"]:
                bad.append("cross-currency allocation")
            if iid not in lines or a > lines[iid]:
                bad.append("allocation not supported by remittance")
        for iid, r in lines.items():
            a = booked.get(iid, 0)
            if a < r:
                issues.append({"type": "missing_or_partial_allocation", "affected_ids": sorted([iid, p["payment_id"]]),
                               "amount": r - a, "currency": p["currency"]})
        if p["amount_minor"] >= 0 and sum(booked.values()) > p["amount_minor"]:
            bad.append("allocations exceed payment")
        if p["amount_minor"] < 0 and (p["invoice_refs"] is not None or not doc.get("refund_of")):
            bad.append("refund must have null refs and refund_of")
        if doc.get("refund_of") and doc["refund_of"] not in b.pay:
            bad.append("refund_of unknown payment")
    for iid, a in alloc_by_inv.items():
        if iid in b.inv and a > b.inv[iid]["total_minor"]:
            bad.append("invoice over-allocated")

    # --- C. FX: проведённая сумма = конвертация по курсу локальной даты документа
    for iid, inv in b.inv.items():
        kind, ar = b.invoice_posting(iid)
        if kind is None or inv["currency"] == b.base:
            if kind is not None and ar["debit_minor"] != inv["total_minor"]:
                bad.append("base-currency invoice posted with different amount")
            continue
        exp = b.to_base(inv["total_minor"], inv["currency"], _local_date(inv["issued_at"], b.tzs))
        if exp is None:
            bad.append("missing fx rate for invoice")
            continue
        if ar["debit_minor"] != exp:
            issues.append({"type": "fx_wrong_date_or_rate", "affected_ids": [iid],
                           "amount": abs(ar["debit_minor"] - exp), "currency": b.base})
    for pid, p in b.pay.items():
        rows = b.by_pay.get(pid, [])
        bank = [r for r in rows if r["account"] == "bank"]
        if not rows:
            if _in_period(p["timestamp"], b.pol):
                bad.append("in-period payment not booked")
            continue
        if not _in_period(p["timestamp"], b.pol):
            issues.append({"type": "period_cutoff_error", "affected_ids": [pid],
                           "amount": (bank[0]["debit_minor"] or bank[0]["credit_minor"]) if bank else 0,
                           "currency": b.base})
        if len(bank) != 1:
            bad.append("payment entry must have one bank row")
            continue
        bank_amt = bank[0]["debit_minor"] if p["amount_minor"] >= 0 else bank[0]["credit_minor"]
        own = _local_date(p["timestamp"], b.tzs)
        exp = b.to_base(abs(p["amount_minor"]), p["currency"], own)
        if exp is None:
            bad.append("missing fx rate for payment")
            continue
        if bank_amt != exp:
            issues.append({"type": "fx_wrong_date_or_rate", "affected_ids": [pid],
                           "amount": abs(bank_amt - exp), "currency": b.base})
        # customer_advances по курсу собственной даты платежа
        adv = [r for r in rows if r["account"] == "customer_advances"]
        if p["amount_minor"] >= 0:
            unalloc = p["amount_minor"] - sum(r["amount_minor"] for r in (p["invoice_refs"] or []))
            adv_amt = sum(r["credit_minor"] for r in adv)
            if (unalloc > 0) != bool(adv) or (unalloc > 0 and adv_amt != b.to_base(unalloc, p["currency"], own)):
                bad.append("customer_advances inconsistent with unallocated amount")
            ar_rows = [r for r in rows if r["account"] == "accounts_receivable"]
            if {r["invoice_id"] for r in ar_rows} != {r["invoice_id"] for r in (p["invoice_refs"] or [])}:
                bad.append("AR relief rows do not match invoice_refs")

    # --- D. cutoff: проводка выпуска счёта, момент которого вне периода (по времени компании)
    for iid, inv in b.inv.items():
        kind, ar = b.invoice_posting(iid)
        inside = _in_period(inv["issued_at"], b.pol)
        if kind == "issue":
            if not inside:
                issues.append({"type": "period_cutoff_error", "affected_ids": [iid],
                               "amount": ar["debit_minor"], "currency": b.base})
            elif _instant(ar["timestamp"]) != _instant(inv["issued_at"]):
                bad.append("posting timestamp differs from document instant")
        elif kind == "opening":
            if _instant(inv["issued_at"]) >= _instant(ar["timestamp"]):
                bad.append("opening balance for invoice not issued before period")
        elif kind == "unknown":
            bad.append("unrecognised invoice posting")
        elif kind is None and inside:
            bad.append("in-period invoice not booked")

    # --- E. погашение AR: исторический курс, полное погашение обнуляет AR
    for iid, inv in b.inv.items():
        kind, ar = b.invoice_posting(iid)
        if kind is None:
            if alloc_by_inv.get(iid):
                bad.append("allocation to unbooked invoice")
            continue
        hist = _local_date(inv["issued_at"], b.tzs)
        reliefs = []
        for p in t["payments"]:
            for ref in p["invoice_refs"] or []:
                if ref["invoice_id"] == iid:
                    row = [r for r in b.by_pay.get(p["payment_id"], [])
                           if r["account"] == "accounts_receivable" and r["invoice_id"] == iid]
                    reliefs.append((_instant(p["timestamp"]), p["payment_id"], ref["amount_minor"],
                                    row[0]["credit_minor"] if row else None))
        reliefs.sort()
        cum = 0
        relieved = 0
        hist_base = b.to_base(inv["total_minor"], inv["currency"], hist)
        for _, _, amt, credit in reliefs:
            cum += amt
            if credit is None:
                bad.append("missing AR relief row")
                continue
            exp = hist_base - relieved if cum == inv["total_minor"] else b.to_base(amt, inv["currency"], hist)
            if credit != exp:
                bad.append("AR relief not at historical rate")
            relieved += credit
        if cum == inv["total_minor"] and ar["debit_minor"] == hist_base:
            bal = sum(r["debit_minor"] - r["credit_minor"] for r in b.t["ledger"]
                      if r["account"] == "accounts_receivable" and r["invoice_id"] == iid)
            if bal != 0:
                bad.append("settled invoice has AR balance")

    issues.sort(key=lambda i: (i["type"], i["affected_ids"]))
    return issues, sorted(set(bad))


# ----------------------------------------------------------------------------- достаточность доказательств

def recompute_from_evidence(t: dict, issue: dict, evidence: list[str]) -> tuple[dict | None, str]:
    """Пересчитывает расхождение ТОЛЬКО по записям набора + policy. -> (issue_sig | None, причина)."""
    pol = t["policy"]
    base = pol["base_currency"]
    full = Book(t)
    # видимость: каждый ID набора должен быть получаем через инструменты
    recs = {}
    for x in evidence:
        if x in full.rows:
            recs[x] = ("ledger", full.rows[x])
        elif x in full.inv:
            recs[x] = ("invoice", full.inv[x])
        elif x in full.pay:
            recs[x] = ("payment", full.pay[x])
        elif x.startswith("fx:"):
            _, d, s, tg = x.split(":")
            if (d, s, tg) not in full.fx:
                return None, "fx id not retrievable"
            recs[x] = ("fx", {"date": d, "source_currency": s, "target_currency": tg, "rate": full.fx[(d, s, tg)]})
        else:
            return None, f"evidence id not retrievable"
    kinds = defaultdict(list)
    for x, (k, r) in recs.items():
        kinds[k].append(r)
    typ = issue["type"]
    if typ == "duplicate_invoice":
        invs = kinds["invoice"]
        if len(invs) != 2:
            return None, "need exactly two invoices"
        a, c = invs
        if (a["entity_id"], a["currency"], a["total_minor"], _canon_lines(a["line_items"])) != \
           (c["entity_id"], c["currency"], c["total_minor"], _canon_lines(c["line_items"])):
            return None, "invoices differ"
        ok_row = any(r["invoice_id"] in (a["invoice_id"], c["invoice_id"]) and r["payment_id"] is None
                     and r["account"] in ("accounts_receivable", "revenue") for r in kinds["ledger"])
        if not ok_row:
            return None, "no posting row proving the double booking"
        return {"type": typ, "affected_ids": sorted([a["invoice_id"], c["invoice_id"]]),
                "amount": a["total_minor"], "currency": a["currency"]}, ""
    if typ == "missing_or_partial_allocation":
        if len(kinds["payment"]) != 1 or len(kinds["invoice"]) != 1:
            return None, "need payment and invoice"
        p, inv = kinds["payment"][0], kinds["invoice"][0]
        doc = full.docs.get(p["doc_id"])  # документ платежа доступен через get_document(doc_id платежа)
        line = [l for l in doc["remittance"]["lines"] if l["invoice_id"] == inv["invoice_id"]]
        if not line:
            return None, "remittance does not name invoice"
        booked = sum(r["amount_minor"] for r in (p["invoice_refs"] or []) if r["invoice_id"] == inv["invoice_id"])
        adv = [r for r in kinds["ledger"] if r["payment_id"] == p["payment_id"] and r["account"] == "customer_advances"]
        if not adv:
            return None, "no row showing unapplied cash"
        return {"type": typ, "affected_ids": sorted([inv["invoice_id"], p["payment_id"]]),
                "amount": line[0]["amount_minor"] - booked, "currency": p["currency"]}, ""
    if typ == "fx_wrong_date_or_rate":
        docs = kinds["invoice"] + kinds["payment"]
        if len(docs) != 1 or len(kinds["fx"]) != 1:
            return None, "need one document and one fx row"
        d, fx = docs[0], kinds["fx"][0]
        is_inv = "invoice_id" in d and "issued_at" in d
        ts = d["issued_at"] if is_inv else d["timestamp"]
        amt = d["total_minor"] if is_inv else abs(d["amount_minor"])
        if fx["date"] != _local_date(ts, pol["timezone"]).isoformat() or fx["source_currency"] != d["currency"] \
           or fx["target_currency"] != base:
            return None, "fx row is not the policy rate for this document"
        if is_inv:
            rows = [r for r in kinds["ledger"] if r["invoice_id"] == d["invoice_id"] and r["payment_id"] is None
                    and r["account"] in ("accounts_receivable", "revenue")]
            ident = d["invoice_id"]
        else:
            rows = [r for r in kinds["ledger"] if r["payment_id"] == d["payment_id"] and r["account"] == "bank"]
            ident = d["payment_id"]
        if len(rows) != 1:
            return None, "need the posting row"
        booked = rows[0]["debit_minor"] or rows[0]["credit_minor"]
        exp = _convert(amt, d["currency"], base, fx["rate"], pol["currency_exponents"], pol["rounding"]["mode"])
        return {"type": typ, "affected_ids": [ident], "amount": abs(booked - exp), "currency": base}, ""
    if typ == "period_cutoff_error":
        if len(kinds["invoice"]) != 1:
            return None, "need one invoice"
        inv = kinds["invoice"][0]
        if _in_period(inv["issued_at"], pol):
            return None, "invoice instant is inside the period"
        rows = [r for r in kinds["ledger"] if r["invoice_id"] == inv["invoice_id"] and r["payment_id"] is None
                and r["account"] in ("accounts_receivable", "revenue")]
        if len(rows) != 1:
            return None, "need the posting row"
        return {"type": typ, "affected_ids": [inv["invoice_id"]],
                "amount": rows[0]["debit_minor"] or rows[0]["credit_minor"], "currency": base}, ""
    return None, "unknown type"


# ----------------------------------------------------------------------------- главный вход

def _sig(i: dict) -> tuple:
    return (i["type"], tuple(i["affected_ids"]), i["amount"], i["currency"])


def validate_episode(files: dict[str, bytes], truth: dict, provenance: dict) -> dict:
    errors: list[str] = []
    n = 0

    def req(cond: bool, msg: str):
        nonlocal n
        n += 1
        if not cond:
            errors.append(msg)

    t = parse_public(files)
    # 1. схемы и manifest
    for name, recs in (("ledger_row", t["ledger"]), ("invoice", t["invoices"]), ("payment", t["payments"]),
                       ("fx_row", t["fx"]), ("document", t["documents"])):
        bad = sum(1 for r in recs if check(r, name))
        req(bad == 0, f"schema: {bad} invalid {name} records")
    req(check(t["manifest"], "manifest") == [], "schema: manifest")
    req(check(t["policy"], "policy") == [], "schema: policy")
    req(check(truth, "truth") == [], "schema: truth")
    m = t["manifest"]
    req(m["counts"] == {"ledger_rows": len(t["ledger"]), "invoices": len(t["invoices"]),
                        "payments": len(t["payments"]), "fx_rows": len(t["fx"]), "documents": len(t["documents"])},
        "manifest counts mismatch")
    for fname, h in m["files"].items():
        req(hashlib.sha256(files[fname]).hexdigest() == h, f"manifest hash mismatch: {fname}")
    req(m["episode_id"] == truth["episode_id"] == provenance["episode_id"], "episode_id mismatch")
    req(m["base_currency"] == t["policy"]["base_currency"] and m["timezone"] == t["policy"]["timezone"],
        "manifest/policy mismatch")
    req(len(t["ledger"]) >= 1 and len(t["invoices"]) + len(t["payments"]) >= 1, "empty episode")

    # 2. ДО мутаций: ни ошибок, ни противоречий
    before = reconstruct_before(t, provenance["diff"])
    b_issues, b_bad = audit(before)
    req(b_issues == [], f"before-state has issues: {b_issues[:3]}")
    req(b_bad == [], f"before-state inconsistent: {b_bad[:5]}")

    # 3. ПОСЛЕ мутаций: ровно эталон
    a_issues, a_bad = audit(t)
    req(a_bad == [], f"after-state inconsistent: {a_bad[:5]}")
    truth_sigs = sorted(_sig({"type": i["type"], "affected_ids": i["affected_ids"], "amount": i["expected_amount_minor"],
                              "currency": i["currency"]}) for i in truth["issues"])
    found = sorted(_sig(i) for i in a_issues)
    req(found == truth_sigs, f"detected != truth: missing={sorted(set(truth_sigs) - set(found))[:3]} "
                             f"extra={sorted(set(found) - set(truth_sigs))[:3]}")
    req(len(set(truth_sigs)) == len(truth_sigs), "truth has duplicate issues")
    req(len(truth["issues"]) <= 12, "more than 12 issues")

    # 4. достаточность доказательств
    for i in truth["issues"]:
        want = (i["type"], tuple(i["affected_ids"]), i["expected_amount_minor"], i["currency"])
        for es in i["evidence_sets"]:
            got, why = recompute_from_evidence(t, i, es)
            req(got is not None and _sig(got) == want, f"evidence set insufficient for {i['issue_id']}: {why or got}")
            req(len(es) <= 10, "evidence set longer than findings limit")

    # 5. отвлекающие случаи: ни одной найденной ошибки, кроме внедрённых сценариями
    truth_ids = {x for i in truth["issues"] for x in i["affected_ids"]}
    flagged = {x for i in a_issues for x in i["affected_ids"]}
    for d in provenance["distractors"]:
        ids = set(d["ids"])
        req(not ((ids & flagged) - truth_ids), f"distractor flagged: {d['kind']}")

    # 6. взаимодействия: явно заданы и не съедают друг друга
    for g in truth["interactions"]:
        members = [i for i in truth["issues"] if i["interaction_group"] == g["group"]]
        req(len(members) == g["n_issues"] >= 2, "interaction group size mismatch")
        req(all(x["expected_amount_minor"] > 0 for x in members), "interaction amount is zero")
    return {"ok": not errors, "errors": errors, "n_checks": n, "n_issues": len(truth["issues"]),
            "n_distractors": len(provenance["distractors"])}


def validate_episode_dir(pub_dir: Path, grader_dir: Path) -> dict:
    files = {n: (pub_dir / n).read_bytes() for n in PUBLIC}
    truth = json.loads((grader_dir / "truth.json").read_text(encoding="utf-8"))
    prov = json.loads((grader_dir / "provenance.json").read_text(encoding="utf-8"))
    return validate_episode(files, truth, prov)
