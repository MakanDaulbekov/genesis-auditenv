"""История -> открытые таблицы (раздел 03) + документы + policy + manifest."""
from __future__ import annotations

import csv
import hashlib
import io
import json

from ..common.drbg import derive_key
from ..common.money import CURRENCY_EXPONENTS
from .world import GENERATOR_VERSION, POLICY_VERSION, SCHEMA_VERSION, post_all

LEDGER_COLUMNS = ["row_id", "timestamp", "account", "debit_minor", "credit_minor", "currency",
                  "entity_id", "invoice_id", "payment_id"]
FX_COLUMNS = ["date", "source_currency", "target_currency", "rate"]
INVOICE_FIELDS = ["doc_id", "invoice_id", "entity_id", "issued_at", "due_at", "currency", "total_minor", "line_items"]
PAYMENT_FIELDS = ["doc_id", "payment_id", "timestamp", "currency", "amount_minor", "invoice_refs"]
ACCOUNTS = ["accounts_receivable", "revenue", "bank", "customer_advances", "fx_gain_loss",
            "opening_balance_equity", "expense:rent", "expense:salaries", "expense:utilities",
            "expense:software", "expense:travel", "expense:office"]


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _order_key(key: bytes, ident: str) -> str:
    return hashlib.sha256(derive_key(key, "order", ident)).hexdigest()


def invoice_record(inv: dict) -> dict:
    return {k: inv[k] for k in INVOICE_FIELDS}


def payment_record(p: dict) -> dict:
    return {k: p[k] for k in PAYMENT_FIELDS}


def build_policy(world: dict) -> dict:
    base = world["company"]["base_currency"]
    return {
        "policy_version": POLICY_VERSION,
        "base_currency": base,
        "timezone": world["company"]["timezone"],
        "currency_exponents": {c: CURRENCY_EXPONENTS[c] for c in sorted(set(world["foreign_currencies"]) | {base})},
        "fx": {
            "rate_definition": "1 unit of source_currency = rate units of target_currency (major units)",
            "rate_date": "local_date_of_document_instant_in_company_timezone",
            "conversion": "base_minor = round(amount_minor * rate * 10^(exp_target - exp_source))",
            "ledger_currency": base,
            "ar_relief": "invoice historical rate; the payment that fully settles an invoice clears its remaining AR",
            "advances": "converted at the receipt's own rate_date; refunds relieve advances at the original receipt rate",
            "gain_loss_account": "fx_gain_loss",
        },
        "period": {
            "start": world["period"]["start"],
            "end": world["period"]["end"],
            "membership": "document instant in [start 00:00, end+1day 00:00) in company timezone",
            "ledger_timestamp_rule": "posting timestamp equals the source document instant",
            "opening_balances": "invoices issued before start and open at start are posted as opening_balance_equity at start 00:00",
        },
        "rounding": {"mode": "ROUND_HALF_EVEN", "scope": "each conversion separately, to integer minor units"},
        "allocation": {
            "rule": "booked invoice_refs of a receipt must equal the remittance advice lines of its document",
            "excess": "payment amount beyond remittance lines is posted to customer_advances",
            "prepayments": "receipts without remittance lines are advances (customer_advances)",
            "refunds": "negative amount_minor; refund_of names the refunded receipt",
        },
        "duplicates": {
            "rule": "same entity, currency, total_minor and identical line_items (incl. service periods) posted more than once",
        },
        "matching": {
            "tolerance_minor": 0,
            "amount_semantics": "claimed_amount_minor is a non-negative magnitude in the stated currency",
            "canonical_affected_ids": {
                "duplicate_invoice": "both invoice_id (order irrelevant)",
                "missing_or_partial_allocation": "[invoice_id, payment_id]",
                "fx_wrong_date_or_rate": "[invoice_id or payment_id of the mis-converted document]",
                "period_cutoff_error": "[invoice_id of the document booked in the wrong period]",
            },
            "amounts": {
                "duplicate_invoice": "total_minor of the duplicate, invoice currency",
                "missing_or_partial_allocation": "remittance amount not allocated to the invoice, payment currency",
                "fx_wrong_date_or_rate": "|booked base amount - correctly converted base amount|, base currency",
                "period_cutoff_error": "base amount of the wrongly included posting, base currency",
            },
            "evidence_id_formats": {
                "ledger": "row_id", "invoices": "invoice_id or doc_id", "payments": "payment_id or doc_id",
                "fx": "fx:<date>:<source_currency>:<target_currency>",
            },
        },
        "ledger": {
            "accounts": ACCOUNTS,
            "entry_grouping": "rows with the same payment_id form one entry; otherwise same (invoice_id, timestamp); otherwise same (entity_id, timestamp)",
        },
    }


def render(world: dict, entries: list[dict] | None = None) -> dict:
    """Возвращает dict таблиц (python-объекты) — открытая часть эпизода."""
    key = bytes.fromhex(world["key_hex"])
    if entries is None:
        entries = post_all(world)
    ordered = sorted(entries, key=lambda e: _order_key(key, e["source"]))
    ledger = [r for e in ordered for r in e["rows"]]
    invoices = sorted((invoice_record(v) for v in world["invoices"].values()),
                      key=lambda r: _order_key(key, r["doc_id"]))
    payments = sorted((payment_record(v) for v in world["payments"].values()),
                      key=lambda r: _order_key(key, r["doc_id"]))
    base = world["company"]["base_currency"]
    fx = []
    for k in sorted(world["fx"]):
        d, c = k.split("|")
        fx.append({"date": d, "source_currency": c, "target_currency": base, "rate": world["fx"][k]})
    docs = []
    for inv in world["invoices"].values():
        ent = world["entities"][inv["entity_id"]]
        docs.append({"doc_type": "invoice", "doc_id": inv["doc_id"], "record": invoice_record(inv),
                     "customer_name": ent["name"], "po_number": inv["po_number"], "memo": inv["memo"]})
    for p in world["payments"].values():
        ent = world["entities"][p["payer_entity_id"]]
        docs.append({"doc_type": "payment", "doc_id": p["doc_id"], "record": payment_record(p),
                     "payer_entity_id": p["payer_entity_id"], "payer_name": ent["name"],
                     "bank_reference": p["bank_reference"], "remittance": p["remittance"],
                     "refund_of": p["refund_of"]})
    docs.sort(key=lambda d: _order_key(key, d["doc_id"]))
    return {"ledger": ledger, "invoices": invoices, "payments": payments, "fx": fx,
            "documents": docs, "policy": build_policy(world)}


# ----------------------------------------------------------------------------- serialization

def ledger_csv(rows: list[dict]) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(LEDGER_COLUMNS)
    for r in rows:
        w.writerow(["" if r[c] is None else r[c] for c in LEDGER_COLUMNS])
    return buf.getvalue().encode("utf-8")


def fx_csv(rows: list[dict]) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(FX_COLUMNS)
    for r in rows:
        w.writerow([r[c] for c in FX_COLUMNS])
    return buf.getvalue().encode("utf-8")


def jsonl(rows: list[dict]) -> bytes:
    return "".join(canonical_json(r) + "\n" for r in rows).encode("utf-8")


def json_bytes(obj) -> bytes:
    return (json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def public_files(world: dict, tables: dict, config_version: str) -> dict[str, bytes]:
    files = {
        "ledger.csv": ledger_csv(tables["ledger"]),
        "invoices.jsonl": jsonl(tables["invoices"]),
        "payments.jsonl": jsonl(tables["payments"]),
        "fx.csv": fx_csv(tables["fx"]),
        "documents.jsonl": jsonl(tables["documents"]),
        "policy.json": json_bytes(tables["policy"]),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "config_version": config_version,
        "generator_version": GENERATOR_VERSION,
        "episode_id": world["episode_id"],
        "company_name": world["company"]["name"],
        "base_currency": world["company"]["base_currency"],
        "period_start": world["period"]["start"],
        "period_end": world["period"]["end"],
        "timezone": world["company"]["timezone"],
        "rounding": {"mode": "ROUND_HALF_EVEN", "fx_rate_date": "local_date_of_document_instant_in_company_timezone",
                     "tolerance_minor": 0},
        "currency_exponents": tables["policy"]["currency_exponents"],
        "counts": {"ledger_rows": len(tables["ledger"]), "invoices": len(tables["invoices"]),
                   "payments": len(tables["payments"]), "fx_rows": len(tables["fx"]),
                   "documents": len(tables["documents"])},
        "files": {name: hashlib.sha256(b).hexdigest() for name, b in sorted(files.items())},
    }
    files["manifest.json"] = json_bytes(manifest)
    return files
