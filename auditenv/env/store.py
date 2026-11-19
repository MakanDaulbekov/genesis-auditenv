"""Read-only хранилище открытых записей эпизода + стабильные непрозрачные курсоры.

Хранилище строится ТОЛЬКО из открытых файлов (manifest, policy, ledger.csv, invoices.jsonl,
payments.jsonl, fx.csv, documents.jsonl). Записи хранятся как канонические JSON-строки и
десериализуются при каждой выдаче: агент не может изменить хранилище через ссылки.
"""
from __future__ import annotations

import base64
import csv
import hashlib
import hmac
import io
import json

TABLES = ("ledger", "invoices", "payments", "fx")
PRIMARY_KEY = {"ledger": "row_id", "invoices": "invoice_id", "payments": "payment_id", "fx": None}
MAX_PAGE = 50
_LEDGER_INT = ("debit_minor", "credit_minor")
_LEDGER_NULLABLE = ("entity_id", "invoice_id", "payment_id")


def fx_id(r: dict) -> str:
    return f"fx:{r['date']}:{r['source_currency']}:{r['target_currency']}"


def _dumps(o) -> str:
    return json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def parse_ledger_csv(data: bytes) -> list[dict]:
    rows = []
    for r in csv.DictReader(io.StringIO(data.decode("utf-8"))):
        out = dict(r)
        for k in _LEDGER_INT:
            out[k] = int(out[k])
        for k in _LEDGER_NULLABLE:
            out[k] = out[k] or None
        rows.append(out)
    return rows


def parse_fx_csv(data: bytes) -> list[dict]:
    return [dict(r) for r in csv.DictReader(io.StringIO(data.decode("utf-8")))]


def parse_jsonl(data: bytes) -> list[dict]:
    return [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]


class ReadOnlyStore:
    __slots__ = ("_tables", "_index", "_docs", "_doc_alias", "_mac_key", "manifest", "policy")

    def __init__(self, files: dict[str, bytes]):
        manifest = json.loads(files["manifest.json"])
        policy = json.loads(files["policy.json"])
        tables = {
            "ledger": parse_ledger_csv(files["ledger.csv"]),
            "invoices": parse_jsonl(files["invoices.jsonl"]),
            "payments": parse_jsonl(files["payments.jsonl"]),
            "fx": parse_fx_csv(files["fx.csv"]),
        }
        # вычисляемое поле fx_id: агенту не нужно конструировать ID доказательства вслепую
        # (сам fx.csv не меняется — схема раздела 03 буквальная)
        tables["fx"] = [dict(r, fx_id=fx_id(r)) for r in tables["fx"]]
        docs = parse_jsonl(files["documents.jsonl"])
        self._tables = {t: tuple(_dumps(r) for r in rows) for t, rows in tables.items()}
        self._index = {}
        for t, rows in tables.items():
            idx = {}
            for i, r in enumerate(rows):
                k = fx_id(r) if t == "fx" else r[PRIMARY_KEY[t]]
                idx[k] = i
            self._index[t] = idx
        self._doc_alias = {}
        for t in ("invoices", "payments"):
            for i, r in enumerate(tables[t]):
                self._doc_alias[(t, r["doc_id"])] = i
        self._docs = {d["doc_id"]: _dumps(d) for d in docs}
        # Ключ MAC курсоров выводится из открытого содержимого (не секрет и не seed):
        # курсоры стабильны между reset одного эпизода и не раскрывают внутреннего состояния.
        self._mac_key = hashlib.sha256(b"cursor" + files["manifest.json"]).digest()
        self.manifest = _dumps(manifest)
        self.policy = _dumps(policy)

    # ------------------------------------------------------------------ queries
    def has_table(self, table) -> bool:
        return isinstance(table, str) and table in self._tables

    def table_size(self, table: str) -> int:
        return len(self._tables[table])

    def page(self, table: str, offset: int, limit: int) -> tuple[list[dict], int | None]:
        rows = self._tables[table]
        chunk = [json.loads(s) for s in rows[offset:offset + limit]]
        nxt = offset + limit if offset + limit < len(rows) else None
        return chunk, nxt

    def get(self, table: str, ident: str) -> dict | None:
        i = self._index[table].get(ident)
        if i is None and table in ("invoices", "payments"):
            i = self._doc_alias.get((table, ident))
        return None if i is None else json.loads(self._tables[table][i])

    def document(self, doc_id: str) -> dict | None:
        s = self._docs.get(doc_id)
        return None if s is None else json.loads(s)

    # ------------------------------------------------------------------ cursors
    def make_cursor(self, table: str, offset: int) -> str:
        msg = f"{table}:{offset}".encode()
        mac = hmac.new(self._mac_key, msg, hashlib.sha256).hexdigest()[:16]
        raw = f"{table}:{offset}:{mac}".encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    def parse_cursor(self, table: str, cursor: str) -> int | None:
        if not isinstance(cursor, str) or len(cursor) > 128:
            return None
        try:
            pad = "=" * (-len(cursor) % 4)
            raw = base64.urlsafe_b64decode(cursor + pad).decode()
            t, off_s, mac = raw.split(":")
            off = int(off_s)
        except Exception:
            return None
        if t != table or off < 0 or off >= len(self._tables[table]) or not off_s.isdigit():
            return None
        good = hmac.new(self._mac_key, f"{table}:{off}".encode(), hashlib.sha256).hexdigest()[:16]
        if not hmac.compare_digest(good, mac):
            return None
        return off

    def __repr__(self) -> str:
        return "<ReadOnlyStore>"
