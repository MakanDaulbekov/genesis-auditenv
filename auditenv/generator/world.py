"""Скрытая согласованная история компании (шаг 1 генератора).

Модель:
  * Компания выставляет счета клиентам (AR) и платит поставщикам (расходы).
  * Главная книга ведётся в базовой валюте (functional currency). Каждая проводка
    счёта/платежа в иностранной валюте конвертируется по курсу на ЛОКАЛЬНУЮ дату
    момента документа в часовом поясе компании (policy.fx.rate_date).
  * Погашение AR — по историческому курсу счёта; платёж, полностью закрывающий
    счёт, снимает весь остаток AR в базовой валюте. Разница -> fx_gain_loss.
  * Мир хранит документы (факты) и «бухгалтерские решения» (bookings). Проводки
    всегда строятся функцией post_all() из документов и решений, поэтому мутация =
    изменение решения/документа + перепроводка; дифф фиксируется в provenance.
  * row_id проводок детерминированно выводится из (ключ эпизода, документ, роль),
    поэтому неизменённые строки сохраняют ID после перепроводки.
"""
from __future__ import annotations

import calendar
import hashlib
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, localcontext

from ..common.drbg import DRBG, derive_key
from ..common.ids import ALPHABET, IdGen
from ..common.money import CURRENCY_EXPONENTS, convert_minor
from ..common.timeutil import format_ts, local_date, parse_offset, parse_ts, period_bounds

GENERATOR_VERSION = "1.0.0"
SCHEMA_VERSION = "1.0"
POLICY_VERSION = "1.0"

USD_VALUE = {"USD": "1", "EUR": "1.085", "GBP": "1.27", "CHF": "1.12", "KZT": "0.0021", "JPY": "0.0067"}
PRICE_MULT = {"USD": 1, "EUR": 1, "GBP": 1, "CHF": 1, "KZT": 480, "JPY": 150}
OFFSETS = ["+05:00", "+06:00", "+03:00", "+01:00", "+00:00", "-03:00", "-05:00", "+09:00", "+04:00", "-08:00"]
EXPENSE_ACCOUNTS = ["expense:rent", "expense:salaries", "expense:utilities", "expense:software",
                    "expense:travel", "expense:office"]
LINE_DESCRIPTIONS = ["Consulting hours", "Cloud hosting plan", "Support subscription", "Hardware lease",
                     "Training session", "Data processing", "Maintenance service", "License seat",
                     "Logistics fee", "Design work", "Integration services", "Audit preparation",
                     "Translation services", "Storage capacity", "Field engineering"]
NAME_A = ["Altyn", "Nord", "Vela", "Kestrel", "Saryn", "Orbita", "Tumar", "Lumen", "Qazyna", "Breeze",
          "Granit", "Miras", "Helio", "Zhibek", "Corvid", "Aster", "Polar", "Steppe", "Ember", "Riviera"]
NAME_B = ["Trade", "Systems", "Logistics", "Foods", "Analytics", "Works", "Labs", "Textiles", "Energy",
          "Motors", "Retail", "Media", "Agro", "Build", "Pharma", "Data"]
NAME_C = ["LLP", "Ltd", "JSC", "GmbH", "Inc", "S.A.", "Co"]
REMIT_NOTES = ["Payment for invoice", "Settlement", "Per remittance advice", "Invoice payment", ""]


class GenerationError(Exception):
    pass


# ----------------------------------------------------------------------------- helpers

def _rand_range(rng: DRBG, spec) -> int:
    if isinstance(spec, int):
        return spec
    lo, hi = spec
    return rng.randint(lo, hi)


def _rate_str(d: Decimal) -> str:
    with localcontext() as ctx:
        ctx.prec = 7
        d = +d
    s = format(d, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def _hid(key: bytes, prefix: str, n: int, *parts: object) -> str:
    """ID из ключевого хеша — для строк книги (стабильны при перепроводке)."""
    h = hashlib.sha256(derive_key(key, "hid", prefix, *parts)).digest()
    v = int.from_bytes(h, "big")
    out = []
    for _ in range(n):
        out.append(ALPHABET[v & 31])
        v >>= 5
    return f"{prefix}-{''.join(out)}"


def weighted_choice(rng: DRBG, items: list, weights: list[int]):
    """Взвешенный выбор на целых числах (random.choices использует float)."""
    total = sum(weights)
    if total <= 0:
        raise GenerationError("bad weights")
    x = rng.randrange(total)
    for it, w in zip(items, weights):
        if x < w:
            return it
        x -= w
    raise AssertionError


def _company_name(rng: DRBG) -> str:
    return f"{rng.choice(NAME_A)} {rng.choice(NAME_B)} {rng.choice(NAME_C)}"


# ----------------------------------------------------------------------------- world

def build_world(key: bytes, cfg: dict) -> dict:
    """Строит согласованную историю. Отвлекающие случаи создаются ЗДЕСЬ как корректные данные."""
    rng = DRBG(derive_key(key, "world"))
    ids = IdGen(DRBG(derive_key(key, "ids")))
    size = cfg["size"]
    cur = cfg["currencies"]

    base = rng.choice(cur["base"])
    tz = rng.choice(OFFSETS)
    year = rng.randint(2030, 2034)
    month = rng.randint(1, 12)
    p_start = date(year, month, 1)
    p_end = date(year, month, calendar.monthrange(year, month)[1])
    foreign = sorted(c for c in cur["foreign"] if c != base)

    world: dict = {
        "key_hex": key.hex(),
        "episode_id": "ep_" + hashlib.sha256(derive_key(key, "episode_id")).hexdigest()[:16],
        "company": {"name": _company_name(rng), "base_currency": base, "timezone": tz},
        "period": {"start": p_start.isoformat(), "end": p_end.isoformat()},
        "foreign_currencies": foreign,
        "fx": {},
        "entities": {},
        "invoices": {},
        "payments": {},
        "expenses": [],
        "bookings": {"invoice": {}, "payment": {}},
        "distractors": [],
        "config_version": cfg["config_version"],
    }

    # --- FX: каждый календарный день, foreign -> base, десятичные строки
    fx_start = p_start - timedelta(days=75)
    fx_end = p_end + timedelta(days=10)
    for c in foreign:
        with localcontext() as ctx:
            ctx.prec = 30
            r = Decimal(USD_VALUE[c]) / Decimal(USD_VALUE[base])
        d = fx_start
        while d <= fx_end:
            k = rng.randint(-45, 45)
            with localcontext() as ctx:
                ctx.prec = 30
                r = r * (Decimal(1) + Decimal(k) / Decimal(10000))
            world["fx"][f"{d.isoformat()}|{c}"] = _rate_str(r)
            d += timedelta(days=1)

    # --- контрагенты
    used_names = {world["company"]["name"]}

    def new_entity(kind: str, currency: str, offset: str) -> str:
        while True:
            name = _company_name(rng)
            if name not in used_names:
                used_names.add(name)
                break
        eid = ids.new("E")
        world["entities"][eid] = {"entity_id": eid, "name": name, "kind": kind,
                                  "currency": currency, "offset": offset}
        return eid

    n_cust = _rand_range(rng, size["n_customers"])
    share = cur["foreign_customer_share_pct"]
    customers = []
    for i in range(n_cust):
        is_foreign = bool(foreign) and rng.randint(1, 100) <= share
        ccy = rng.choice(foreign) if is_foreign else base
        customers.append(new_entity("customer", ccy, rng.choice(OFFSETS)))
    # гарантируем хотя бы одного клиента в иностранной валюте и одного в базовой
    if foreign and all(world["entities"][e]["currency"] == base for e in customers):
        customers.append(new_entity("customer", rng.choice(foreign), rng.choice(OFFSETS)))
    if all(world["entities"][e]["currency"] != base for e in customers):
        customers.append(new_entity("customer", base, rng.choice(OFFSETS)))
    vendors = [new_entity("vendor", base, tz) for _ in range(_rand_range(rng, size["n_vendors"]))]
    world["customers"] = customers
    world["vendors"] = vendors

    start_dt, end_dt = period_bounds(world["period"]["start"], world["period"]["end"], tz)
    period_secs = int((end_dt - start_dt).total_seconds())
    used_instants: set[int] = set()

    def rand_instant(lo: datetime, hi: datetime) -> datetime:
        span = int((hi - lo).total_seconds())
        if span <= 1:
            raise GenerationError("empty time window")
        for _ in range(1000):
            t = lo + timedelta(seconds=rng.randrange(span))
            ep = int(t.timestamp())
            if ep not in used_instants:
                used_instants.add(ep)
                return t
        raise GenerationError("cannot find unique instant")

    def make_lines(ccy: str, svc_start: date, svc_end: date, n: int) -> list[dict]:
        e = CURRENCY_EXPONENTS[ccy]
        mult = PRICE_MULT[ccy]
        lines = []
        descs = rng.sample(LINE_DESCRIPTIONS, n)
        for dsc in descs:
            qty = rng.randint(1, 12)
            if rng.randint(1, 100) <= 15:  # круглые цены встречаются у всех записей, не только у ошибок
                unit = rng.randint(1, 40) * 100 * mult * 10**e
            else:
                unit = rng.randint(15 * mult * 10**e, 4000 * mult * 10**e)
            lines.append({"description": dsc, "quantity": qty, "unit_price_minor": unit,
                          "amount_minor": qty * unit,
                          "service_period_start": svc_start.isoformat(),
                          "service_period_end": svc_end.isoformat()})
        return lines

    def new_invoice(eid: str, issued: datetime, lines: list[dict], kind: str) -> str:
        ent = world["entities"][eid]
        iid = ids.new("INV")
        did = ids.new("D")
        due = issued + timedelta(days=rng.choice([14, 30, 45]))
        world["invoices"][iid] = {
            "doc_id": did, "invoice_id": iid, "entity_id": eid,
            "issued_at": format_ts(issued, ent["offset"]),
            "due_at": format_ts(due, ent["offset"]),
            "currency": ent["currency"],
            "total_minor": sum(li["amount_minor"] for li in lines),
            "line_items": lines,
            "po_number": f"PO-{rng.randint(100000, 999999)}",
            "memo": rng.choice(["", "Net terms per contract", "Thank you for your business",
                                "Reference contract on file"]),
            "_kind": kind,
        }
        return iid

    # --- счета текущего периода
    n_inv = _rand_range(rng, size["n_invoices"])
    for _ in range(n_inv):
        eid = rng.choice(customers)
        issued = rand_instant(start_dt + timedelta(hours=1), end_dt - timedelta(minutes=1))
        svc_s = p_start + timedelta(days=rng.randint(0, 10))
        svc_e = p_end - timedelta(days=rng.randint(0, 10))
        new_invoice(eid, issued, make_lines(world["entities"][eid]["currency"], svc_s, svc_e,
                                            rng.randint(1, 4)), "regular")

    # --- отвлекающее: законные повторяющиеся суммы (еженедельная подписка, разные периоды услуг)
    for _ in range(_rand_range(rng, cfg["distractors"]["recurring_groups"])):
        eid = rng.choice(customers)
        ccy = world["entities"][eid]["currency"]
        proto = make_lines(ccy, p_start, p_start, rng.randint(1, 2))
        group = []
        for w in range(rng.randint(2, 4)):
            ws = p_start + timedelta(days=7 * w)
            we = ws + timedelta(days=6)
            if we > p_end:
                break
            lines = [dict(li, service_period_start=ws.isoformat(), service_period_end=we.isoformat())
                     for li in proto]
            lo = datetime(we.year, we.month, we.day, tzinfo=parse_offset(tz))
            issued = rand_instant(lo, min(lo + timedelta(hours=20), end_dt - timedelta(minutes=1)))
            group.append(new_invoice(eid, issued, lines, "recurring"))
        if len(group) >= 2:
            world["distractors"].append({"kind": "recurring_same_amount", "ids": sorted(group)})

    # --- отвлекающее: одинаковая сумма, другой состав строк у того же клиента
    for _ in range(_rand_range(rng, cfg["distractors"]["same_total_diff_lines"])):
        eid = rng.choice(customers)
        ccy = world["entities"][eid]["currency"]
        e = CURRENCY_EXPONENTS[ccy]
        unit = rng.randint(50 * PRICE_MULT[ccy] * 10**e, 900 * PRICE_MULT[ccy] * 10**e)
        d1, d2 = rng.sample(LINE_DESCRIPTIONS, 2)
        a = [{"description": d1, "quantity": 2, "unit_price_minor": unit, "amount_minor": 2 * unit,
              "service_period_start": p_start.isoformat(), "service_period_end": p_end.isoformat()}]
        b = [{"description": d2, "quantity": 1, "unit_price_minor": 2 * unit, "amount_minor": 2 * unit,
              "service_period_start": p_start.isoformat(), "service_period_end": p_end.isoformat()}]
        i1 = new_invoice(eid, rand_instant(start_dt + timedelta(hours=1), end_dt - timedelta(minutes=1)), a, "regular")
        i2 = new_invoice(eid, rand_instant(start_dt + timedelta(hours=1), end_dt - timedelta(minutes=1)), b, "regular")
        world["distractors"].append({"kind": "same_total_different_lines", "ids": sorted([i1, i2])})

    # --- отвлекающее: счета у границы периода, у которых «наивная» дата (в смещении клиента)
    #     вне периода, а локальная дата компании — внутри (проведены верно)
    def naive_date(ts: str) -> date:
        return date.fromisoformat(ts[:10])

    def boundary_instant(near_end: bool) -> datetime:
        if near_end:
            return rand_instant(end_dt - timedelta(hours=10), end_dt - timedelta(minutes=1))
        return rand_instant(start_dt + timedelta(minutes=1), start_dt + timedelta(hours=10))

    for _ in range(_rand_range(rng, cfg["distractors"]["tz_boundary_invoices"])):
        for _try in range(40):
            eid = rng.choice(customers)
            near_end = rng.randint(0, 1) == 1
            t = boundary_instant(near_end)
            rendered = format_ts(t, world["entities"][eid]["offset"])
            nd = naive_date(rendered)
            if nd < p_start or nd > p_end:
                iid = new_invoice(eid, t, make_lines(world["entities"][eid]["currency"], p_start, p_end,
                                                     rng.randint(1, 3)), "regular")
                world["distractors"].append({"kind": "tz_boundary_invoice_in_period", "ids": [iid]})
                break

    # --- счета прошлого периода, открытые на начало (входящее сальдо)
    for _ in range(_rand_range(rng, size["n_prior_invoices"])):
        eid = rng.choice(customers)
        issued = rand_instant(start_dt - timedelta(days=60), start_dt - timedelta(hours=2))
        prev_end = p_start - timedelta(days=1)
        prev_start = prev_end.replace(day=1)
        new_invoice(eid, issued, make_lines(world["entities"][eid]["currency"], prev_start, prev_end,
                                            rng.randint(1, 3)), "prior")

    # --- счета следующего периода (видны в документах, в книгу этого периода не попадают)
    for _ in range(_rand_range(rng, size["n_next_invoices"])):
        eid = rng.choice(customers)
        issued = rand_instant(end_dt + timedelta(minutes=5), end_dt + timedelta(days=3))
        nxt_s = p_end + timedelta(days=1)
        new_invoice(eid, issued, make_lines(world["entities"][eid]["currency"], nxt_s,
                                            nxt_s + timedelta(days=27), rng.randint(1, 2)), "next")

    # --- отвлекающее: счета СЛЕДУЮЩЕГО периода, чья наивная дата выглядит как дата периода
    for _ in range(_rand_range(rng, cfg["distractors"]["tz_boundary_next_invoices"])):
        for _try in range(40):
            eid = rng.choice(customers)
            t = rand_instant(end_dt + timedelta(minutes=1), end_dt + timedelta(hours=12))
            rendered = format_ts(t, world["entities"][eid]["offset"])
            if naive_date(rendered) <= p_end:
                nxt_s = p_end + timedelta(days=1)
                iid = new_invoice(eid, t, make_lines(world["entities"][eid]["currency"], nxt_s,
                                                     nxt_s + timedelta(days=27), rng.randint(1, 2)), "next")
                world["distractors"].append({"kind": "tz_boundary_next_invoice_not_booked", "ids": [iid]})
                break

    # --- платежи по счетам
    def new_payment(eid: str, ts: datetime, ccy: str, amount: int, refs: list[dict] | None,
                    remit_lines: list[dict], note: str, kind: str, refund_of: str | None = None) -> str:
        pid = ids.new("PAY")
        did = ids.new("D")
        ent = world["entities"][eid]
        world["payments"][pid] = {
            "doc_id": did, "payment_id": pid,
            "timestamp": format_ts(ts, ent["offset"]),
            "currency": ccy, "amount_minor": amount,
            "invoice_refs": deepcopy(refs),
            "payer_entity_id": eid,
            "remittance": {"lines": deepcopy(remit_lines), "note": note},
            "bank_reference": f"BR{rng.randint(10**9, 10**10 - 1)}",
            "refund_of": refund_of,
            "_kind": kind,
        }
        return pid

    def split_amount(total: int, k: int) -> list[int]:
        if total < k * 2:
            return [total]
        cuts = sorted(rng.sample(range(1, total), k - 1))
        parts = [b - a for a, b in zip([0] + cuts, cuts + [total])]
        return parts

    payable = [iid for iid, inv in world["invoices"].items() if inv["_kind"] in ("regular", "recurring", "prior")]
    payable.sort()
    w = cfg["distractors"]["payment_mix_pct"]
    patterns = ["none", "full", "split", "partial", "overpay"]
    weights = [w[p] for p in patterns]
    single_full: dict[tuple, list[str]] = {}
    for iid in payable:
        inv = world["invoices"][iid]
        pat = weighted_choice(rng, patterns, weights)
        eid = inv["entity_id"]
        ccy = inv["currency"]
        lo = max(parse_ts(inv["issued_at"]) + timedelta(hours=2), start_dt + timedelta(minutes=30))
        hi = end_dt - timedelta(minutes=1)
        if lo >= hi - timedelta(hours=1):
            pat = "none"
        total = inv["total_minor"]
        if pat == "full":
            ref = [{"invoice_id": iid, "amount_minor": total}]
            pid = new_payment(eid, rand_instant(lo, hi), ccy, total, ref, ref, rng.choice(REMIT_NOTES), "receipt")
            single_full.setdefault((eid, ccy), []).append(pid)
        elif pat == "split":
            parts = split_amount(total, rng.randint(2, 3))
            times = sorted(rand_instant(lo, hi) for _ in parts)
            pids = []
            for amt, ts in zip(parts, times):
                ref = [{"invoice_id": iid, "amount_minor": amt}]
                pids.append(new_payment(eid, ts, ccy, amt, ref, ref, "Partial settlement", "receipt"))
            if len(pids) >= 2:
                world["distractors"].append({"kind": "split_payment", "ids": sorted(pids + [iid])})
        elif pat == "partial":
            amt = rng.randint(max(1, total // 5), max(1, (total * 4) // 5))
            ref = [{"invoice_id": iid, "amount_minor": amt}]
            pid = new_payment(eid, rand_instant(lo, hi), ccy, amt, ref, ref, "Partial payment", "receipt")
            world["distractors"].append({"kind": "partial_payment_open_invoice", "ids": sorted([pid, iid])})
        elif pat == "overpay":
            extra = rng.randint(1, max(1, total // 10))
            ref = [{"invoice_id": iid, "amount_minor": total}]
            pid = new_payment(eid, rand_instant(lo, hi), ccy, total + extra, ref, ref,
                              "Includes advance for next order", "receipt")
            world["distractors"].append({"kind": "overpayment_to_advances", "ids": sorted([pid, iid])})

    # --- отвлекающее: платёж в иностранной валюте у конца периода, наивная дата которого
    #     уже в следующем периоде (верно проведён по курсу локальной даты компании)
    foreign_customers = [e for e in customers if world["entities"][e]["currency"] != base]
    for _ in range(_rand_range(rng, cfg["distractors"]["tz_boundary_receipts"])):
        if not foreign_customers:
            break
        for _try in range(60):
            eid = rng.choice(foreign_customers)
            t = rand_instant(end_dt - timedelta(hours=10), end_dt - timedelta(minutes=1))
            ent = world["entities"][eid]
            rendered = format_ts(t, ent["offset"])
            if naive_date(rendered) != local_date(rendered, tz):
                issued = rand_instant(start_dt + timedelta(hours=1), start_dt + timedelta(days=12))
                iid = new_invoice(eid, issued, make_lines(ent["currency"], p_start, p_end, rng.randint(1, 3)),
                                  "regular")
                inv = world["invoices"][iid]
                ref = [{"invoice_id": iid, "amount_minor": inv["total_minor"]}]
                pid = new_payment(eid, t, ent["currency"], inv["total_minor"], ref, ref,
                                  rng.choice(REMIT_NOTES), "receipt")
                world["distractors"].append({"kind": "tz_boundary_foreign_receipt", "ids": sorted([iid, pid])})
                break

    # --- отвлекающее: один платёж на два счёта клиента (batch)
    for (eid, ccy), pids in sorted(single_full.items()):
        if len(pids) >= 3 and rng.randint(1, 100) <= cfg["distractors"]["batch_payment_pct"]:
            p1, p2 = pids[0], pids[1]
            a, b = world["payments"][p1], world["payments"][p2]
            refs = a["invoice_refs"] + b["invoice_refs"]
            later = max(parse_ts(a["timestamp"]), parse_ts(b["timestamp"]))
            del world["payments"][p1]
            del world["payments"][p2]
            pid = new_payment(eid, later + timedelta(seconds=rng.randint(1, 3000)), ccy,
                              a["amount_minor"] + b["amount_minor"], refs, refs, "Batch payment", "receipt")
            if parse_ts(world["payments"][pid]["timestamp"]) >= end_dt:
                # не выходим за период — откатываем объединение
                del world["payments"][pid]
                world["payments"][p1] = a
                world["payments"][p2] = b
            else:
                world["distractors"].append({"kind": "batch_payment", "ids": sorted([pid] + [r["invoice_id"] for r in refs])})

    # --- отвлекающее: предоплаты (без счёта) и возвраты предоплат
    for _ in range(_rand_range(rng, cfg["distractors"]["prepayments"])):
        eid = rng.choice(customers)
        ccy = world["entities"][eid]["currency"]
        e = CURRENCY_EXPONENTS[ccy]
        amt = rng.randint(100 * PRICE_MULT[ccy] * 10**e, 6000 * PRICE_MULT[ccy] * 10**e)
        ts = rand_instant(start_dt + timedelta(hours=1), end_dt - timedelta(days=4))
        pid = new_payment(eid, ts, ccy, amt, None, [], "Advance payment under contract", "prepayment")
        ids_ = [pid]
        if rng.randint(1, 100) <= cfg["distractors"]["refund_pct"]:
            rts = rand_instant(ts + timedelta(days=1), end_dt - timedelta(minutes=1))
            rid = new_payment(eid, rts, ccy, -amt, None, [], "Refund of advance payment", "refund", refund_of=pid)
            ids_.append(rid)
        world["distractors"].append({"kind": "prepayment" if len(ids_) == 1 else "prepayment_refunded",
                                     "ids": sorted(ids_)})

    # --- расходы (шум книги)
    for n in range(_rand_range(rng, size["n_expenses"])):
        vid = rng.choice(vendors)
        e = CURRENCY_EXPONENTS[base]
        amt = rng.randint(5 * PRICE_MULT[base] * 10**e, 3000 * PRICE_MULT[base] * 10**e)
        ts = rand_instant(start_dt + timedelta(minutes=1), end_dt - timedelta(minutes=1))
        world["expenses"].append({"exp_key": f"X{n}", "vendor_id": vid, "account": rng.choice(EXPENSE_ACCOUNTS),
                                  "amount_minor": amt, "timestamp": format_ts(ts, tz)})

    world["_idgen"] = ids
    world["_used_instants"] = used_instants
    return world


# ----------------------------------------------------------------------------- posting

def rate_for(world: dict, ccy: str, d: date) -> str:
    k = f"{d.isoformat()}|{ccy}"
    if k not in world["fx"]:
        raise GenerationError("fx rate missing")
    return world["fx"][k]


def to_base(world: dict, amount: int, ccy: str, d: date) -> int:
    base = world["company"]["base_currency"]
    if ccy == base:
        return amount
    return convert_minor(amount, ccy, base, rate_for(world, ccy, d), "ROUND_HALF_EVEN")


def booked_base(world: dict, amount: int, ccy: str, bk: dict, default_date: date) -> int:
    """Сумма в base так, как её провёл бухгалтер (с учётом ошибочных решений bk)."""
    base = world["company"]["base_currency"]
    if ccy == base:
        return amount
    d = date.fromisoformat(bk["rate_date"]) if "rate_date" in bk else default_date
    rate_ccy = bk.get("rate_ccy", ccy)
    return convert_minor(amount, ccy, base, rate_for(world, rate_ccy, d), "ROUND_HALF_EVEN")


def invoice_posting_kind(world: dict, inv: dict) -> str | None:
    """'issue' — проводка в периоде, 'opening' — входящее сальдо, None — не проводится."""
    bk = world["bookings"]["invoice"].get(inv["invoice_id"], {})
    if "post_as" in bk:
        return bk["post_as"]
    k = inv["_kind"]
    if k in ("regular", "recurring", "duplicate"):
        return "issue"
    if k == "prior":
        return "opening"
    return None


def post_all(world: dict) -> list[dict]:
    """Строит все журнальные записи из документов + решений. Детерминированно."""
    key = bytes.fromhex(world["key_hex"])
    tz = world["company"]["timezone"]
    base = world["company"]["base_currency"]
    entries: list[dict] = []
    p_start = world["period"]["start"]

    def row(source: str, role: str, ts: str, account: str, debit: int, credit: int,
            entity_id, invoice_id, payment_id) -> dict:
        return {"row_id": _hid(key, "R", 9, source, role), "timestamp": ts, "account": account,
                "debit_minor": debit, "credit_minor": credit, "currency": base,
                "entity_id": entity_id, "invoice_id": invoice_id, "payment_id": payment_id}

    ar_state: dict[str, dict] = {}

    # счета: выпуск и входящее сальдо
    for iid in sorted(world["invoices"]):
        inv = world["invoices"][iid]
        kind = invoice_posting_kind(world, inv)
        if kind is None:
            continue
        bk = world["bookings"]["invoice"].get(iid, {})
        hist_d = local_date(inv["issued_at"], tz)
        amount_base = booked_base(world, inv["total_minor"], inv["currency"], bk, hist_d)
        if kind == "issue":
            ts = bk.get("ledger_ts") or format_ts(parse_ts(inv["issued_at"]), tz)
            rows = [row(iid, "ar", ts, "accounts_receivable", amount_base, 0, inv["entity_id"], iid, None),
                    row(iid, "rev", ts, "revenue", 0, amount_base, inv["entity_id"], iid, None)]
        else:
            ts = format_ts(datetime.fromisoformat(p_start + "T00:00:00" + tz), tz)
            rows = [row(iid, "ar", ts, "accounts_receivable", amount_base, 0, inv["entity_id"], iid, None),
                    row(iid, "obe", ts, "opening_balance_equity", 0, amount_base, inv["entity_id"], iid, None)]
        entries.append({"source": iid, "kind": f"invoice_{kind}", "timestamp": ts, "rows": rows})
        # AR погашается по ИСТОРИЧЕСКОМУ (верному) курсу; ошибка курса при выпуске
        # проявляется в строках выпуска, а не каскадом в погашениях.
        ar_state[iid] = {"base_total": to_base(world, inv["total_minor"], inv["currency"], hist_d),
                         "hist_date": hist_d, "allocated": 0, "relieved": 0, "currency": inv["currency"],
                         "total": inv["total_minor"]}

    adv_state: dict[str, date] = {}
    pays = sorted(world["payments"].values(), key=lambda p: (parse_ts(p["timestamp"]), p["payment_id"]))
    for p in pays:
        pid = p["payment_id"]
        bk = world["bookings"]["payment"].get(pid, {})
        if bk.get("post_as", "post") == "skip":
            continue
        ts = bk.get("ledger_ts") or format_ts(parse_ts(p["timestamp"]), tz)
        own_d = local_date(p["timestamp"], tz)
        eid = p["payer_entity_id"]
        ccy = p["currency"]
        rows = []
        if p["amount_minor"] >= 0:
            bank = booked_base(world, p["amount_minor"], ccy, bk, own_d)
            rows.append(row(pid, "bank", ts, "bank", bank, 0, eid, None, pid))
            credits = 0
            allocated = 0
            for ref in p["invoice_refs"] or []:
                iid = ref["invoice_id"]
                st = ar_state[iid]
                st["allocated"] += ref["amount_minor"]
                if st["allocated"] > st["total"]:
                    raise GenerationError("over-allocation")
                if st["allocated"] == st["total"]:
                    relief = st["base_total"] - st["relieved"]
                else:
                    relief = to_base(world, ref["amount_minor"], ccy, st["hist_date"])
                st["relieved"] += relief
                credits += relief
                allocated += ref["amount_minor"]
                rows.append(row(pid, f"ar:{iid}", ts, "accounts_receivable", 0, relief, eid, iid, pid))
            unalloc = p["amount_minor"] - allocated
            if unalloc < 0:
                raise GenerationError("allocation exceeds payment")
            if unalloc > 0:
                adv = to_base(world, unalloc, ccy, own_d)
                credits += adv
                rows.append(row(pid, "adv", ts, "customer_advances", 0, adv, eid, None, pid))
                adv_state[pid] = own_d
            diff = bank - credits
        else:
            amt = -p["amount_minor"]
            src = p["refund_of"]
            hist = adv_state.get(src, own_d)
            adv = to_base(world, amt, ccy, hist)
            bank = booked_base(world, amt, ccy, bk, own_d)
            rows.append(row(pid, "adv", ts, "customer_advances", adv, 0, eid, None, pid))
            rows.append(row(pid, "bank", ts, "bank", 0, bank, eid, None, pid))
            diff = adv - bank  # дебет минус кредит
        if diff > 0:
            rows.append(row(pid, "fxgl", ts, "fx_gain_loss", 0, diff, eid, None, pid))
        elif diff < 0:
            rows.append(row(pid, "fxgl", ts, "fx_gain_loss", -diff, 0, eid, None, pid))
        entries.append({"source": pid, "kind": "payment", "timestamp": ts, "rows": rows})

    for x in world["expenses"]:
        rows = [row(x["exp_key"], "exp", x["timestamp"], x["account"], x["amount_minor"], 0, x["vendor_id"], None, None),
                row(x["exp_key"], "bank", x["timestamp"], "bank", 0, x["amount_minor"], x["vendor_id"], None, None)]
        entries.append({"source": x["exp_key"], "kind": "expense", "timestamp": x["timestamp"], "rows": rows})

    # самопроверка генератора: каждая запись сбалансирована
    for en in entries:
        d = sum(r["debit_minor"] for r in en["rows"])
        c = sum(r["credit_minor"] for r in en["rows"])
        if d != c:
            raise GenerationError("unbalanced entry")
    seen = set()
    for en in entries:
        for r in en["rows"]:
            if r["row_id"] in seen:
                raise GenerationError("row id collision")
            seen.add(r["row_id"])
    return entries


def clone_world(world: dict) -> dict:
    idgen = world.get("_idgen")
    used = world.get("_used_instants")
    w = {k: v for k, v in world.items() if k not in ("_idgen", "_used_instants")}
    w = deepcopy(w)
    w["_idgen"] = idgen
    w["_used_instants"] = used
    return w
