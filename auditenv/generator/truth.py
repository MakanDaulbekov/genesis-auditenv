"""truth.json и provenance.json строятся ТОЛЬКО из записей мутаций и диффа рендеров."""
from __future__ import annotations

from ..common.drbg import DRBG, derive_key
from ..common.ids import IdGen
from .world import SCHEMA_VERSION

TABLE_KEYS = {"ledger": "row_id", "invoices": "invoice_id", "payments": "payment_id", "documents": "doc_id"}


def diff_tables(before: dict, after: dict) -> dict:
    """Дифф открытых записей: inserted (ids), deleted (records), updated (id + before)."""
    out = {}
    for table, k in TABLE_KEYS.items():
        b = {r[k]: r for r in before[table]}
        a = {r[k]: r for r in after[table]}
        out[table] = {
            "inserted": sorted(set(a) - set(b)),
            "deleted": [b[i] for i in sorted(set(b) - set(a))],
            "updated": [{"id": i, "before": b[i]} for i in sorted(set(a) & set(b)) if a[i] != b[i]],
        }
    return out


def build_truth(key: bytes, episode_id: str, mutation_log: list[dict], scenarios: list[dict]) -> dict:
    gen = IdGen(DRBG(derive_key(key, "issue_ids")))
    issues = []
    for m in mutation_log:
        iss = dict(m["issue"])
        iss["issue_id"] = gen.new("ISS")
        m["issue_id"] = iss["issue_id"]
        issues.append(iss)
    seen = set()
    for iss in issues:
        sig = (iss["type"], tuple(iss["affected_ids"]))
        if sig in seen:
            raise ValueError("ambiguous truth: two issues with same type and affected_ids")
        seen.add(sig)
    interactions = []
    for sc in scenarios:
        members = [m["issue_id"] for m in mutation_log if m["issue"]["interaction_group"] == sc["group"]]
        interactions.append({"group": sc["group"], "scenario": sc["scenario"], "issue_ids": sorted(members),
                             "n_issues": len(members)})
    issues.sort(key=lambda i: i["issue_id"])
    return {"schema_version": SCHEMA_VERSION, "episode_id": episode_id, "issues": issues,
            "interactions": interactions}
