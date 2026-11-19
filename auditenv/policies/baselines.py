"""Четыре базовые политики (раздел 06). Политика видит ТОЛЬКО наблюдения.

Каждая политика — генератор: получает первое наблюдение, yield-ит действия, получает
следующие наблюдения через send(). oracle здесь — лишь маркер: его ответ строит процесс
оценщика (op "oracle"), эталон никогда не попадает в процесс политики.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date

from ..common.drbg import DRBG, derive_key
from ..common.money import convert_minor

TYPES = ["duplicate_invoice", "missing_or_partial_allocation", "fx_wrong_date_or_rate", "period_cutoff_error"]


def act(name, **args):
    return {"action": name, "args": args}


def _read_table(table: str, budget_floor: int = 3):
    """Подпрограмма: читает всю таблицу постранично. -> список записей."""
    recs, cur = [], None
    while True:
        obs = yield act("list_records", table=table, cursor=cur, limit=50)
        if obs["error"] is not None:
            return recs
        recs += obs["result"]["records"]
        cur = obs["result"]["next_cursor"]
        if cur is None or obs["steps_remaining"] <= budget_floor:
            return recs


# ----------------------------------------------------------------------------- random_explorer

def random_explorer(first: dict, seed: int = 20261122):
    """Случайный исследователь с фиксированным seed: случайные чтения, затем случайный ответ."""
    man = first["result"]["manifest"]
    rng = DRBG(derive_key("random_explorer", seed, man["episode_id"]))
    seen_ids: dict[str, list[str]] = defaultdict(list)
    docs: list[str] = []
    cursors: dict[str, list] = {t: [None] for t in ("ledger", "invoices", "payments", "fx")}
    n = rng.randint(5, 40)
    obs = first
    for _ in range(n):
        k = rng.randint(0, 9)
        if k <= 4 or not docs:
            t = rng.choice(["ledger", "invoices", "payments", "fx"])
            obs = yield act("list_records", table=t, cursor=rng.choice(cursors[t]), limit=rng.randint(1, 50))
            if obs["error"] is None:
                if obs["result"]["next_cursor"]:
                    cursors[t].append(obs["result"]["next_cursor"])
                for r in obs["result"]["records"]:
                    if t == "ledger":
                        seen_ids["ledger"].append(r["row_id"])
                    elif t == "invoices":
                        seen_ids["invoices"].append(r["invoice_id"])
                        docs.append(r["doc_id"])
                    elif t == "payments":
                        seen_ids["payments"].append(r["payment_id"])
                        docs.append(r["doc_id"])
        elif k <= 6:
            obs = yield act("get_document", doc_id=rng.choice(docs))
        elif k == 7:
            t = rng.choice(["ledger", "invoices", "payments"])
            ident = rng.choice(seen_ids[t]) if seen_ids[t] else "R-UNKNOWN"
            obs = yield act("get_record", table=t, id=ident)
        else:
            obs = yield act("calculator", expression=f"{rng.randint(1, 10**6)} * {rng.randint(1, 999)} / 100")
    pool = [x for v in seen_ids.values() for x in v]
    findings = []
    if pool:
        for _ in range(rng.randint(0, 3)):
            findings.append({"type": rng.choice(TYPES), "affected_ids": rng.sample(pool, min(len(pool), rng.randint(1, 2))),
                             "claimed_amount_minor": rng.randint(1, 10**7), "currency": man["base_currency"],
                             "evidence_ids": rng.sample(pool, min(len(pool), rng.randint(1, 6))),
                             "explanation": "random"})
    yield act("submit", findings={"episode_id": man["episode_id"], "findings": findings})


# ----------------------------------------------------------------------------- simple_reconciler

def simple_reconciler(first: dict):
    """Честная скриптовая сверка «как в Excel».

    Сознательные упрощения (измеряются в REPORT.md):
      * дата берётся из строки времени как написано (YYYY-MM-DD), без перевода в часовой
        пояс компании — и для курса FX, и для границы периода;
      * дубли ищутся по (контрагент, валюта, сумма, описания/кол-во/цены строк) без учёта
        периодов оказания услуг;
      * распределение проверяется по remittance advice только для платежей с
        нераспределённым остатком (get_document по каждому).
    """
    res = first["result"]
    man, pol = res["manifest"], res["policy"]
    base, mode = pol["base_currency"], pol["rounding"]["mode"]
    p_start, p_end = date.fromisoformat(pol["period"]["start"]), date.fromisoformat(pol["period"]["end"])
    invoices = yield from _read_table("invoices")
    payments = yield from _read_table("payments")
    ledger = yield from _read_table("ledger")
    fx_rows = yield from _read_table("fx", budget_floor=25)
    fx = {(r["date"], r["source_currency"]): r["rate"] for r in fx_rows}

    def naive(ts: str) -> date:
        return date.fromisoformat(ts[:10])

    def conv(amount: int, ccy: str, d: date):
        if ccy == base:
            return amount
        rate = fx.get((d.isoformat(), ccy))
        return None if rate is None else convert_minor(amount, ccy, base, rate, mode)

    rows_inv = defaultdict(list)
    rows_pay = defaultdict(list)
    for r in ledger:
        if r["payment_id"]:
            rows_pay[r["payment_id"]].append(r)
        elif r["invoice_id"]:
            rows_inv[r["invoice_id"]].append(r)

    def ar_row(iid):
        x = [r for r in rows_inv[iid] if r["account"] == "accounts_receivable" and r["debit_minor"] > 0]
        return x[0] if x else None

    findings = []

    # дубли
    groups = defaultdict(list)
    for inv in invoices:
        if ar_row(inv["invoice_id"]) is None:
            continue
        key = (inv["entity_id"], inv["currency"], inv["total_minor"],
               tuple(sorted((li["description"], li["quantity"], li["unit_price_minor"]) for li in inv["line_items"])))
        groups[key].append(inv)
    for key, g in groups.items():
        if len(g) >= 2:
            a, b = sorted(g, key=lambda x: x["issued_at"])[:2]
            findings.append({"type": "duplicate_invoice", "affected_ids": [a["invoice_id"], b["invoice_id"]],
                             "claimed_amount_minor": a["total_minor"], "currency": a["currency"],
                             "evidence_ids": [a["invoice_id"], b["invoice_id"], ar_row(b["invoice_id"])["row_id"]],
                             "explanation": "same customer, amount and lines posted twice"})

    # FX и cutoff (наивная дата)
    for inv in invoices:
        ar = ar_row(inv["invoice_id"])
        if ar is None:
            continue
        is_issue = any(r["account"] == "revenue" for r in rows_inv[inv["invoice_id"]])
        d = naive(inv["issued_at"])
        if is_issue and not (p_start <= d <= p_end):
            findings.append({"type": "period_cutoff_error", "affected_ids": [inv["invoice_id"]],
                             "claimed_amount_minor": ar["debit_minor"], "currency": base,
                             "evidence_ids": [inv["invoice_id"], ar["row_id"]],
                             "explanation": "invoice dated outside the period but booked in it"})
            continue
        if inv["currency"] != base:
            exp = conv(inv["total_minor"], inv["currency"], d)
            if exp is not None and exp != ar["debit_minor"]:
                findings.append({"type": "fx_wrong_date_or_rate", "affected_ids": [inv["invoice_id"]],
                                 "claimed_amount_minor": abs(exp - ar["debit_minor"]), "currency": base,
                                 "evidence_ids": [inv["invoice_id"], ar["row_id"], f"fx:{d}:{inv['currency']}:{base}"],
                                 "explanation": "posting differs from conversion at document date"})
    for p in payments:
        bank = [r for r in rows_pay[p["payment_id"]] if r["account"] == "bank"]
        if not bank or p["currency"] == base:
            continue
        d = naive(p["timestamp"])
        exp = conv(abs(p["amount_minor"]), p["currency"], d)
        got = bank[0]["debit_minor"] or bank[0]["credit_minor"]
        if exp is not None and exp != got:
            findings.append({"type": "fx_wrong_date_or_rate", "affected_ids": [p["payment_id"]],
                             "claimed_amount_minor": abs(exp - got), "currency": base,
                             "evidence_ids": [p["payment_id"], bank[0]["row_id"], f"fx:{d}:{p['currency']}:{base}"],
                             "explanation": "bank posting differs from conversion at payment date"})

    # распределение: платежи с нераспределённым остатком -> remittance advice
    for p in payments:
        if p["amount_minor"] <= 0:
            continue
        booked = {r["invoice_id"]: r["amount_minor"] for r in (p["invoice_refs"] or [])}
        if sum(booked.values()) >= p["amount_minor"]:
            continue
        o = yield act("get_document", doc_id=p["doc_id"])
        if o["error"] is not None:
            continue
        adv = [r for r in rows_pay[p["payment_id"]] if r["account"] == "customer_advances"]
        for line in o["result"]["document"]["remittance"]["lines"]:
            gap = line["amount_minor"] - booked.get(line["invoice_id"], 0)
            if gap > 0 and adv:
                findings.append({"type": "missing_or_partial_allocation",
                                 "affected_ids": [line["invoice_id"], p["payment_id"]],
                                 "claimed_amount_minor": gap, "currency": p["currency"],
                                 "evidence_ids": [line["invoice_id"], p["payment_id"], adv[0]["row_id"]],
                                 "explanation": "remittance advice names the invoice but cash went to advances"})
        if o["steps_remaining"] <= 2:
            break
    yield act("submit", findings={"episode_id": man["episode_id"], "findings": findings[:30]})


# ----------------------------------------------------------------------------- flag_all (атака)

def flag_all(first: dict):
    """Атака «пометить всё»: 30 выводов всех типов по крупнейшим документам."""
    man = first["result"]["manifest"]
    invoices = yield from _read_table("invoices")
    o = yield act("list_records", table="ledger", limit=50)
    rows = [r["row_id"] for r in o["result"]["records"]] if o["error"] is None else []
    invoices.sort(key=lambda r: -r["total_minor"])
    findings = []
    for i in range(30):
        inv = invoices[i % len(invoices)]
        typ = TYPES[i % 4]
        aff = [inv["invoice_id"]]
        if typ == "duplicate_invoice":
            aff.append(invoices[(i + 1) % len(invoices)]["invoice_id"])
        if typ == "missing_or_partial_allocation":
            aff.append(invoices[(i + 2) % len(invoices)]["invoice_id"])
        findings.append({"type": typ, "affected_ids": aff, "claimed_amount_minor": inv["total_minor"],
                         "currency": inv["currency"], "evidence_ids": (aff + rows)[:10], "explanation": "flag all"})
    yield act("submit", findings={"episode_id": man["episode_id"], "findings": findings})




# ----------------------------------------------------------------------------- competent_reconciler

def competent_reconciler(first: dict):
    """Компетентная скриптовая сверка — доказательство решаемости по открытым данным за ≤100 шагов.

    Честно читает данные через инструменты и применяет правила policy:
      * время переводится в часовой пояс компании (граница периода и дата курса);
      * дубли — по полной сути счёта, включая периоды услуг в line_items;
      * remittance advice (get_document) — только для платежей с нераспределённым остатком;
      * курсы: дешевле из «прочитать fx постранично» и «get_record по нужным fx-ключам».
    Эталон не используется.
    """
    from ..common.timeutil import in_period, local_date

    res = first["result"]
    man, pol = res["manifest"], res["policy"]
    base, mode, tz = pol["base_currency"], pol["rounding"]["mode"], pol["timezone"]
    ps, pe = pol["period"]["start"], pol["period"]["end"]
    invoices = yield from _read_table("invoices", budget_floor=0)
    payments = yield from _read_table("payments", budget_floor=0)
    ledger = yield from _read_table("ledger", budget_floor=0)

    rows_inv, rows_pay = defaultdict(list), defaultdict(list)
    for r in ledger:
        if r["payment_id"]:
            rows_pay[r["payment_id"]].append(r)
        elif r["invoice_id"]:
            rows_inv[r["invoice_id"]].append(r)

    def ar_row(iid):
        x = [r for r in rows_inv[iid] if r["account"] == "accounts_receivable" and r["debit_minor"] > 0]
        return x[0] if x else None

    def bank_row(pid):
        x = [r for r in rows_pay[pid] if r["account"] == "bank"]
        return x[0] if len(x) == 1 else None

    # --- какие курсы нужны
    need = set()
    for inv in invoices:
        if inv["currency"] != base and ar_row(inv["invoice_id"]):
            need.add((local_date(inv["issued_at"], tz).isoformat(), inv["currency"]))
    for p in payments:
        if p["currency"] != base and bank_row(p["payment_id"]):
            need.add((local_date(p["timestamp"], tz).isoformat(), p["currency"]))
    fx = {}
    pages = -(-man["counts"]["fx_rows"] // 50)
    if len(need) <= pages:
        for d, c in sorted(need):
            o = yield act("get_record", table="fx", id=f"fx:{d}:{c}:{base}")
            if o["error"] is None:
                fx[(d, c)] = o["result"]["record"]["rate"]
    else:
        for r in (yield from _read_table("fx", budget_floor=0)):
            fx[(r["date"], r["source_currency"])] = r["rate"]

    def conv(amount, ccy, d):
        if ccy == base:
            return amount
        rate = fx.get((d.isoformat(), ccy))
        return None if rate is None else convert_minor(amount, ccy, base, rate, mode)

    findings = []
    # --- дубли: полная суть счёта (контрагент, валюта, сумма, все поля строк вкл. периоды услуг)
    groups = defaultdict(list)
    for inv in invoices:
        if ar_row(inv["invoice_id"]):
            key = (inv["entity_id"], inv["currency"], inv["total_minor"],
                   tuple(sorted(tuple(sorted(li.items())) for li in inv["line_items"])))
            groups[key].append(inv)
    for g in groups.values():
        if len(g) == 2:
            a, b = sorted(g, key=lambda x: x["invoice_id"])
            findings.append({"type": "duplicate_invoice", "affected_ids": [a["invoice_id"], b["invoice_id"]],
                             "claimed_amount_minor": a["total_minor"], "currency": a["currency"],
                             "evidence_ids": [a["invoice_id"], b["invoice_id"], ar_row(b["invoice_id"])["row_id"]],
                             "explanation": "identical invoice (entity, lines, service periods) posted twice"})

    # --- cutoff и FX по счетам (время компании)
    for inv in invoices:
        ar = ar_row(inv["invoice_id"])
        if ar is None:
            continue
        is_issue = any(r["account"] == "revenue" for r in rows_inv[inv["invoice_id"]])
        if is_issue and not in_period(inv["issued_at"], ps, pe, tz):
            findings.append({"type": "period_cutoff_error", "affected_ids": [inv["invoice_id"]],
                             "claimed_amount_minor": ar["debit_minor"], "currency": base,
                             "evidence_ids": [inv["invoice_id"], ar["row_id"]],
                             "explanation": "invoice instant is outside the period in company time but booked in it"})
        if inv["currency"] != base:
            d = local_date(inv["issued_at"], tz)
            exp = conv(inv["total_minor"], inv["currency"], d)
            if exp is not None and exp != ar["debit_minor"]:
                findings.append({"type": "fx_wrong_date_or_rate", "affected_ids": [inv["invoice_id"]],
                                 "claimed_amount_minor": abs(exp - ar["debit_minor"]), "currency": base,
                                 "evidence_ids": [inv["invoice_id"], ar["row_id"], f"fx:{d}:{inv['currency']}:{base}"],
                                 "explanation": "posting differs from conversion at the policy rate date"})
    # --- платежи: cutoff, FX
    for p in payments:
        bank = bank_row(p["payment_id"])
        if bank is None:
            continue
        got = bank["debit_minor"] or bank["credit_minor"]
        if not in_period(p["timestamp"], ps, pe, tz):
            findings.append({"type": "period_cutoff_error", "affected_ids": [p["payment_id"]],
                             "claimed_amount_minor": got, "currency": base,
                             "evidence_ids": [p["payment_id"], bank["row_id"]],
                             "explanation": "payment instant is outside the period but booked in it"})
        if p["currency"] != base:
            d = local_date(p["timestamp"], tz)
            exp = conv(abs(p["amount_minor"]), p["currency"], d)
            if exp is not None and exp != got:
                findings.append({"type": "fx_wrong_date_or_rate", "affected_ids": [p["payment_id"]],
                                 "claimed_amount_minor": abs(exp - got), "currency": base,
                                 "evidence_ids": [p["payment_id"], bank["row_id"], f"fx:{d}:{p['currency']}:{base}"],
                                 "explanation": "bank posting differs from conversion at the policy rate date"})
    # --- распределение по remittance advice
    for p in payments:
        if p["amount_minor"] <= 0:
            continue
        booked = defaultdict(int)
        for r in p["invoice_refs"] or []:
            booked[r["invoice_id"]] += r["amount_minor"]
        if sum(booked.values()) >= p["amount_minor"]:
            continue
        o = yield act("get_document", doc_id=p["doc_id"])
        if o["error"] is not None:
            continue
        adv = [r for r in rows_pay[p["payment_id"]] if r["account"] == "customer_advances"]
        for line in o["result"]["document"]["remittance"]["lines"]:
            gap = line["amount_minor"] - booked[line["invoice_id"]]
            if gap > 0 and adv:
                findings.append({"type": "missing_or_partial_allocation",
                                 "affected_ids": [line["invoice_id"], p["payment_id"]],
                                 "claimed_amount_minor": gap, "currency": p["currency"],
                                 "evidence_ids": [line["invoice_id"], p["payment_id"], adv[0]["row_id"]],
                                 "explanation": "remittance advice names the invoice but the amount went to advances"})
        if o["steps_remaining"] <= 1:
            break
    yield act("submit", findings={"episode_id": man["episode_id"], "findings": findings[:30]})


POLICIES = {
    "random_explorer": random_explorer,
    "simple_reconciler": simple_reconciler,
    "competent_reconciler": competent_reconciler,
    "flag_all": flag_all,
    "oracle": None,  # исполняется процессом оценщика
}
