"""Внедрение ошибок как явных причинных мутаций.

Каждая мутация: mutate(world, rng, params) -> provenance_entry.
  * Меняет только «бухгалтерские решения» (bookings) или добавляет ошибочно
    проведённый документ (дубль). Факты (remittance advice, даты документов,
    курсы) не трогаются — именно по ним расхождение доказуемо.
  * Возвращает запись с причинным изменением и эталонным issue (тип, канонические
    affected_ids, сумма в minor units, валюта, допустимые наборы доказательств).
  * Дифф открытых записей до/после фиксируется в provenance вызывающим кодом
    (generator.episode) путём сравнения рендеров.
truth.json строится ТОЛЬКО из этих записей.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime, timedelta

from ..common.drbg import DRBG
from ..common.timeutil import format_ts, local_date, parse_ts, period_bounds
from .world import (GenerationError, _hid, booked_base, invoice_posting_kind, rate_for, to_base)

ERROR_TYPES = ["duplicate_invoice", "missing_or_partial_allocation", "fx_wrong_date_or_rate", "period_cutoff_error"]


def rid(world: dict, source: str, role: str) -> str:
    return _hid(bytes.fromhex(world["key_hex"]), "R", 9, source, role)


def fx_key(d: date, src: str, tgt: str) -> str:
    return f"fx:{d.isoformat()}:{src}:{tgt}"


def _touched(world: dict) -> set:
    return world.setdefault("_touched", set())


def _distractor_ids(world: dict) -> set:
    out = set()
    for d in world["distractors"]:
        out.update(d["ids"])
    return out


def _payments_by_invoice(world: dict) -> dict[str, list[str]]:
    m: dict[str, list[str]] = {}
    for pid, p in world["payments"].items():
        for ref in p["invoice_refs"] or []:
            m.setdefault(ref["invoice_id"], []).append(pid)
        for line in p["remittance"]["lines"]:
            m.setdefault(line["invoice_id"], [])
            if pid not in m[line["invoice_id"]]:
                m[line["invoice_id"]].append(pid)
    return m


def _period(world: dict):
    return period_bounds(world["period"]["start"], world["period"]["end"], world["company"]["timezone"])


def _issue(world: dict, typ: str, affected: list[str], amount: int, ccy: str, evidence: list[list[str]],
           group: str | None = None) -> dict:
    if amount <= 0:
        raise GenerationError("zero discrepancy")
    return {"type": typ, "affected_ids": sorted(set(affected)), "expected_amount_minor": amount,
            "currency": ccy, "evidence_sets": sorted(sorted(set(e)) for e in evidence),
            "interaction_group": group}


# ----------------------------------------------------------------------------- duplicate_invoice

def duplicate_invoice(world: dict, rng: DRBG, params: dict) -> dict:
    """Повторно проведённый счёт: копия с новым doc_id/invoice_id/датой, та же суть."""
    start, end = _period(world)
    touched = _touched(world)
    excl = touched | _distractor_ids(world)
    want_out_of_period = params.get("copy_out_of_period", False)
    cands = sorted(iid for iid, inv in world["invoices"].items()
                   if inv["_kind"] == "regular" and iid not in excl
                   and parse_ts(inv["issued_at"]) < end - timedelta(days=2))
    if not cands:
        raise GenerationError("no duplicate candidates")
    x = rng.choice(cands)
    inv = world["invoices"][x]
    ids = world["_idgen"]
    tz = world["company"]["timezone"]
    offset = inv["issued_at"][-6:]
    lo = parse_ts(inv["issued_at"]) + timedelta(hours=3)
    if want_out_of_period:
        # копия выпущена уже в следующем периоде (по времени компании), но проведена в этом
        for _ in range(200):
            t = end + timedelta(seconds=rng.randrange(1, 3 * 86400))
            if int(t.timestamp()) not in world["_used_instants"]:
                break
    else:
        span = int((end - timedelta(minutes=1) - lo).total_seconds())
        for _ in range(200):
            t = lo + timedelta(seconds=rng.randrange(max(1, span)))
            if int(t.timestamp()) not in world["_used_instants"]:
                break
    world["_used_instants"].add(int(t.timestamp()))
    delta = t - parse_ts(inv["issued_at"])
    dup = deepcopy(inv)
    dup["invoice_id"] = ids.new("INV")
    dup["doc_id"] = ids.new("D")
    dup["issued_at"] = format_ts(t, offset)
    dup["due_at"] = format_ts(parse_ts(inv["due_at"]) + delta, offset)
    dup["_kind"] = "duplicate"
    world["invoices"][dup["invoice_id"]] = dup
    touched.update([x, dup["invoice_id"]])
    xd = dup["invoice_id"]
    rows = [rid(world, x, "ar"), rid(world, x, "rev"), rid(world, xd, "ar"), rid(world, xd, "rev")]
    issue = _issue(world, "duplicate_invoice", [x, xd], inv["total_minor"], inv["currency"],
                   [[x, xd, r] for r in rows], params.get("group"))
    return {"type": "duplicate_invoice", "variant": "out_of_period_copy" if want_out_of_period else "in_period_copy",
            "cause": {"original_invoice_id": x, "duplicate_invoice_id": xd,
                      "before": None, "after": {"invoice_id": xd, "issued_at": dup["issued_at"]}},
            "issue": issue}


# ----------------------------------------------------------------------------- allocation

def missing_or_partial_allocation(world: dict, rng: DRBG, params: dict) -> dict:
    """Платёж по remittance advice предназначен счёту I, но в книге не распределён или
    распределён частично; остаток ошибочно отнесён на customer_advances."""
    touched = _touched(world)
    excl = touched | (_distractor_ids(world) - set(params.get("allow_ids", [])))
    by_inv = _payments_by_invoice(world)
    only = params.get("only_payment_id")
    cands = []
    for pid in sorted(world["payments"]):
        p = world["payments"][pid]
        if only and pid != only:
            continue
        if p["_kind"] != "receipt" or pid in excl or p["invoice_refs"] is None:
            continue
        lines = p["remittance"]["lines"]
        if len(lines) != 1 or len(p["invoice_refs"]) != 1:
            continue
        iid = lines[0]["invoice_id"]
        if iid in excl or by_inv.get(iid) != [pid]:
            continue
        r = lines[0]["amount_minor"]
        if p["invoice_refs"][0] != lines[0] or p["amount_minor"] != r or r < 10:
            continue
        cands.append(pid)
    if not cands:
        raise GenerationError("no allocation candidates")
    pid = rng.choice(cands)
    p = world["payments"][pid]
    iid = p["remittance"]["lines"][0]["invoice_id"]
    r = p["remittance"]["lines"][0]["amount_minor"]
    variant = params.get("variant") or rng.choice(["missing", "partial"])
    before = deepcopy(p["invoice_refs"])
    if variant == "missing":
        p["invoice_refs"] = None
        amount = r
    else:
        a = rng.randint(max(1, r // 10), r - max(1, r // 10))
        p["invoice_refs"] = [{"invoice_id": iid, "amount_minor": a}]
        amount = r - a
    touched.update([pid, iid])
    issue = _issue(world, "missing_or_partial_allocation", [iid, pid], amount, p["currency"],
                   [[iid, pid, rid(world, pid, "adv")]], params.get("group"))
    return {"type": "missing_or_partial_allocation", "variant": variant,
            "cause": {"payment_id": pid, "invoice_id": iid, "field": "invoice_refs",
                      "before": before, "after": deepcopy(p["invoice_refs"])},
            "issue": issue}


# ----------------------------------------------------------------------------- fx

def _fx_wrong_choice(world: dict, rng: DRBG, ccy: str, ts: str, due: str | None, variants: list[str]):
    """Подбирает ошибочное решение о курсе с ненулевым расхождением. -> (variant, bk) | None."""
    tz = world["company"]["timezone"]
    correct = local_date(ts, tz)
    opts = []
    for v in variants:
        if v == "naive_date":
            nd = date.fromisoformat(ts[:10])
            if nd != correct:
                opts.append((v, {"rate_date": nd.isoformat()}))
        elif v == "shifted_date":
            k = rng.choice([-3, -2, -1, 1, 2])
            opts.append((v, {"rate_date": (correct + timedelta(days=k)).isoformat()}))
        elif v == "due_date" and due:
            opts.append((v, {"rate_date": local_date(due, tz).isoformat()}))
        elif v == "wrong_pair":
            others = [c for c in world["foreign_currencies"] if c != ccy]
            if others:
                opts.append((v, {"rate_ccy": rng.choice(others)}))
    rng.shuffle(opts)
    return opts


def fx_wrong_date_or_rate(world: dict, rng: DRBG, params: dict) -> dict:
    """Конвертация по курсу не той даты (нарушение policy.fx.rate_date) или по чужому курсу."""
    tz = world["company"]["timezone"]
    base = world["company"]["base_currency"]
    touched = _touched(world)
    excl = touched | (_distractor_ids(world) - set(params.get("allow_ids", [])))
    by_inv = _payments_by_invoice(world)
    targets = []
    only = params.get("only_payment_id")
    if not only:
        for iid in sorted(world["invoices"]):
            inv = world["invoices"][iid]
            if inv["currency"] == base or iid in excl or inv["_kind"] != "regular":
                continue
            if invoice_posting_kind(world, inv) != "issue" or by_inv.get(iid):
                continue
            targets.append(("invoice", iid))
    for pid in sorted(world["payments"]):
        p = world["payments"][pid]
        if only and pid != only:
            continue
        if p["currency"] == base or pid in excl or p["_kind"] != "receipt":
            continue
        targets.append(("payment", pid))
    rng.shuffle(targets)
    variants = params.get("variants") or ["naive_date", "shifted_date", "due_date", "wrong_pair"]
    for kind, ident in targets:
        if kind == "invoice":
            doc = world["invoices"][ident]
            ts, due, amt = doc["issued_at"], doc["due_at"], doc["total_minor"]
        else:
            doc = world["payments"][ident]
            ts, due, amt = doc["timestamp"], None, doc["amount_minor"]
        ccy = doc["currency"]
        correct_d = local_date(ts, tz)
        correct = to_base(world, amt, ccy, correct_d)
        for variant, bk in _fx_wrong_choice(world, rng, ccy, ts, due, variants):
            try:
                wrong = booked_base(world, amt, ccy, bk, correct_d)
            except GenerationError:
                continue
            diff = abs(wrong - correct)
            if diff == 0:
                continue
            world["bookings"][kind][ident] = dict(world["bookings"][kind].get(ident, {}), **bk)
            touched.add(ident)
            fk = fx_key(correct_d, ccy, base)
            if kind == "invoice":
                ev = [[ident, rid(world, ident, "ar"), fk], [ident, rid(world, ident, "rev"), fk]]
            else:
                ev = [[ident, rid(world, ident, "bank"), fk]]
            issue = _issue(world, "fx_wrong_date_or_rate", [ident], diff, base, ev, params.get("group"))
            return {"type": "fx_wrong_date_or_rate", "variant": variant,
                    "cause": {"document_id": ident, "document_kind": kind,
                              "before": {"rate_date": correct_d.isoformat(), "rate_currency": ccy,
                                         "rate": rate_for(world, ccy, correct_d), "base_amount_minor": correct},
                              "after": dict(bk, base_amount_minor=wrong)},
                    "issue": issue}
    raise GenerationError("no fx candidates")


# ----------------------------------------------------------------------------- cutoff

def period_cutoff_error(world: dict, rng: DRBG, params: dict) -> dict:
    """Документ другого периода проведён в этом (момент документа вне периода по времени компании)."""
    tz = world["company"]["timezone"]
    base = world["company"]["base_currency"]
    start, end = _period(world)
    touched = _touched(world)
    excl = touched | (_distractor_ids(world) - set(params.get("allow_ids", [])))
    by_inv = _payments_by_invoice(world)
    only = params.get("only_invoice_id")
    variants = params.get("variants") or ["next_period_booked", "prior_period_booked"]
    cands = []
    for iid in sorted(world["invoices"]):
        inv = world["invoices"][iid]
        if only and iid != only:
            continue
        if iid in excl or by_inv.get(iid):
            continue
        if inv["_kind"] in ("next", "duplicate") and "next_period_booked" in variants:
            if parse_ts(inv["issued_at"]) >= end:
                cands.append(("next_period_booked", iid))
        elif inv["_kind"] == "prior" and "prior_period_booked" in variants:
            cands.append(("prior_period_booked", iid))
    if not cands:
        raise GenerationError("no cutoff candidates")
    # в половине случаев предпочитаем ловушку часового пояса: наивная дата документа внутри периода
    tz_trap = [c for c in cands if start <= datetime.fromisoformat(world["invoices"][c[1]]["issued_at"][:19] + tz) < end]
    pool = tz_trap if tz_trap and rng.randint(0, 1) == 1 else cands
    variant, iid = rng.choice(pool)
    inv = world["invoices"][iid]
    t = parse_ts(inv["issued_at"])
    # «наивная» интерпретация: настенное время документа считается временем компании
    naive = datetime.fromisoformat(inv["issued_at"][:19] + tz)
    if start <= naive < end:
        ledger_t = naive
        how = "timezone_ignored"
    elif variant == "next_period_booked":
        ledger_t = end - timedelta(seconds=rng.randint(60, 6 * 3600))
        how = "accrued_at_period_close"
    else:
        ledger_t = start + timedelta(seconds=rng.randint(60, 6 * 3600))
        how = "booked_at_period_open"
    ledger_ts = format_ts(ledger_t, tz)
    before = deepcopy(world["bookings"]["invoice"].get(iid, {}))
    world["bookings"]["invoice"][iid] = dict(before, post_as="issue", ledger_ts=ledger_ts)
    amount = booked_base(world, inv["total_minor"], inv["currency"], world["bookings"]["invoice"][iid],
                         local_date(inv["issued_at"], tz))
    touched.add(iid)
    issue = _issue(world, "period_cutoff_error", [iid], amount, base,
                   [[iid, rid(world, iid, "ar")], [iid, rid(world, iid, "rev")]], params.get("group"))
    return {"type": "period_cutoff_error", "variant": f"{variant}:{how}",
            "cause": {"invoice_id": iid, "document_instant": inv["issued_at"],
                      "before": {"posting": invoice_posting_kind(dict(world, bookings={"invoice": {iid: before}, "payment": {}}), inv)},
                      "after": {"posting": "issue", "ledger_timestamp": ledger_ts}},
            "issue": issue}


MUTATIONS = {
    "duplicate_invoice": duplicate_invoice,
    "missing_or_partial_allocation": missing_or_partial_allocation,
    "fx_wrong_date_or_rate": fx_wrong_date_or_rate,
    "period_cutoff_error": period_cutoff_error,
}


# ----------------------------------------------------------------------------- interactions

def scenario_partial_fx_boundary(world: dict, rng: DRBG, group: str) -> list[dict]:
    """Частичный платёж в иностранной валюте у границы периода: allocation + fx (наивная дата)."""
    cands = sorted(d["ids"] for d in world["distractors"] if d["kind"] == "tz_boundary_foreign_receipt")
    cands = [c for c in cands if not (set(c) & _touched(world))]
    if not cands:
        raise GenerationError("no boundary receipt for scenario")
    ids = rng.choice(cands)
    pid = next(i for i in ids if i.startswith("PAY-"))
    out = [missing_or_partial_allocation(world, rng, {"variant": "partial", "only_payment_id": pid,
                                                      "allow_ids": ids, "group": group})]
    # платёж уже помечен touched — разрешаем повторное касание в рамках сценария
    _touched(world).discard(pid)
    out.append(fx_wrong_date_or_rate(world, rng, {"variants": ["naive_date"], "only_payment_id": pid,
                                                  "allow_ids": ids, "group": group}))
    return out


def scenario_duplicate_cutoff(world: dict, rng: DRBG, group: str) -> list[dict]:
    """Дубль счёта, копия которого выпущена в следующем периоде, но проведена в этом."""
    d = duplicate_invoice(world, rng, {"copy_out_of_period": True, "group": group})
    dup_id = d["cause"]["duplicate_invoice_id"]
    _touched(world).discard(dup_id)
    c = period_cutoff_error(world, rng, {"variants": ["next_period_booked"], "only_invoice_id": dup_id,
                                         "group": group})
    return [d, c]


SCENARIOS = {
    "partial_fx_boundary": scenario_partial_fx_boundary,
    "duplicate_cutoff": scenario_duplicate_cutoff,
}
