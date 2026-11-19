"""Безвредные преобразования эпизода: перестановка строк и биективное переименование ID.

Применяются к открытым файлам И к truth/provenance согласованно (одна биекция). Суть
эпизода не меняется: ответ oracle (после той же биекции) и награда обязаны сохраниться.
fx-ключи (fx:<date>:<src>:<tgt>) — это содержимое, а не ID; они не переименовываются.
"""
from __future__ import annotations

import copy
import hashlib
import json

from .common.drbg import DRBG, derive_key
from .common.ids import IdGen
from .env.store import parse_fx_csv, parse_jsonl, parse_ledger_csv
from .generator.render import fx_csv, json_bytes, jsonl, ledger_csv

PREFIXES = ("R-", "INV-", "PAY-", "D-", "E-")


class Bijection:
    def __init__(self, rng: DRBG, used: set[str]):
        self._gen = IdGen(rng)
        for u in used:
            self._gen.reserve(u)
        self.map: dict[str, str] = {}

    def __call__(self, x):
        if x is None:
            return None
        if not isinstance(x, str) or not x.startswith(PREFIXES):
            return x
        if x not in self.map:
            prefix = x.split("-", 1)[0]
            self.map[x] = self._gen.new(prefix)
        return self.map[x]

    def ep(self, eid: str, rng: DRBG) -> str:
        if eid not in self.map:
            self.map[eid] = "ep_" + "".join(rng.choice("0123456789abcdef") for _ in range(16))
        return self.map[eid]


def _ren_ledger(r, f):
    return dict(r, row_id=f(r["row_id"]), entity_id=f(r["entity_id"]), invoice_id=f(r["invoice_id"]),
                payment_id=f(r["payment_id"]))


def _ren_invoice(r, f):
    return dict(r, doc_id=f(r["doc_id"]), invoice_id=f(r["invoice_id"]), entity_id=f(r["entity_id"]))


def _ren_payment(r, f):
    refs = None if r["invoice_refs"] is None else [dict(x, invoice_id=f(x["invoice_id"])) for x in r["invoice_refs"]]
    return dict(r, doc_id=f(r["doc_id"]), payment_id=f(r["payment_id"]), invoice_refs=refs)


def _ren_doc(d, f):
    out = dict(d, doc_id=f(d["doc_id"]))
    if d["doc_type"] == "invoice":
        out["record"] = _ren_invoice(d["record"], f)
    else:
        out["record"] = _ren_payment(d["record"], f)
        out["payer_entity_id"] = f(d["payer_entity_id"])
        out["refund_of"] = f(d["refund_of"])
        out["remittance"] = dict(d["remittance"], lines=[dict(l, invoice_id=f(l["invoice_id"]))
                                                           for l in d["remittance"]["lines"]])
    return out


def transform_episode(public: dict[str, bytes], truth: dict, provenance: dict, seed: int,
                      permute: bool = True, rename: bool = True) -> tuple[dict[str, bytes], dict, dict, dict]:
    """-> (public', truth', provenance', mapping). mapping: старый ID -> новый ID."""
    rng = DRBG(derive_key("transform", seed))
    ledger = parse_ledger_csv(public["ledger.csv"])
    invoices = parse_jsonl(public["invoices.jsonl"])
    payments = parse_jsonl(public["payments.jsonl"])
    fx = parse_fx_csv(public["fx.csv"])
    docs = parse_jsonl(public["documents.jsonl"])
    manifest = json.loads(public["manifest.json"])
    policy = json.loads(public["policy.json"])

    used = {r["row_id"] for r in ledger} | {r["invoice_id"] for r in invoices} | {r["doc_id"] for r in invoices} \
        | {r["payment_id"] for r in payments} | {r["doc_id"] for r in payments} \
        | {r["entity_id"] for r in ledger if r["entity_id"]} | {r["entity_id"] for r in invoices}
    bij = Bijection(rng.child("ids"), used)
    f = bij if rename else (lambda x: x)
    new_eid = bij.ep(manifest["episode_id"], rng.child("ep")) if rename else manifest["episode_id"]

    ledger = [_ren_ledger(r, f) for r in ledger]
    invoices = [_ren_invoice(r, f) for r in invoices]
    payments = [_ren_payment(r, f) for r in payments]
    docs = [_ren_doc(d, f) for d in docs]
    if permute:
        prng = rng.child("perm")
        for lst in (ledger, invoices, payments, fx, docs):
            prng.shuffle(lst)

    files = {"ledger.csv": ledger_csv(ledger), "invoices.jsonl": jsonl(invoices), "payments.jsonl": jsonl(payments),
             "fx.csv": fx_csv(fx), "documents.jsonl": jsonl(docs), "policy.json": json_bytes(policy)}
    manifest = dict(manifest, episode_id=new_eid,
                    files={n: hashlib.sha256(b).hexdigest() for n, b in sorted(files.items())})
    files["manifest.json"] = json_bytes(manifest)

    t2 = copy.deepcopy(truth)
    t2["episode_id"] = new_eid
    for i in t2["issues"]:
        i["affected_ids"] = sorted(f(x) for x in i["affected_ids"])
        i["evidence_sets"] = sorted(sorted(f(x) for x in es) for es in i["evidence_sets"])

    p2 = copy.deepcopy(provenance)
    p2["episode_id"] = new_eid
    p2["transform"] = {"seed": seed, "permute": permute, "rename": rename}
    ren = {"ledger": _ren_ledger, "invoices": _ren_invoice, "payments": _ren_payment, "documents": _ren_doc}
    for table, d in p2["diff"].items():
        d["inserted"] = sorted(f(x) for x in d["inserted"])
        d["deleted"] = [ren[table](r, f) for r in d["deleted"]]
        d["updated"] = [{"id": f(u["id"]), "before": ren[table](u["before"], f)} for u in d["updated"]]
    for dd in p2["distractors"]:
        dd["ids"] = sorted(f(x) for x in dd["ids"])
    p2.pop("mutations", None)  # причинные записи генератора не нужны после преобразования
    mapping = dict(bij.map) if rename else {}
    return files, t2, p2, mapping


def rename_findings(findings: dict, mapping: dict) -> dict:
    out = copy.deepcopy(findings)
    out["episode_id"] = mapping.get(out["episode_id"], out["episode_id"])
    for x in out["findings"]:
        x["affected_ids"] = [mapping.get(i, i) for i in x["affected_ids"]]
        x["evidence_ids"] = [mapping.get(i, i) for i in x["evidence_ids"]]
    return out
