"""Доработки после ревью: доказательства дублей, полнота policy, крайние config, снимок «до», имена эпизодов."""
import copy
import json
import shutil
from pathlib import Path

import pytest

from auditenv.cli import deep_merge, export_before
from auditenv.env.core import AuditEnv
from auditenv.env.store import parse_jsonl
from auditenv.generator.episode import MAX_ISSUES, _plan, generate_episode
from auditenv.generator.validate import audit, parse_public, recompute_from_evidence, validate_episode
from auditenv.grader.matcher import alias_map_from_tables
from auditenv.grader.reward import grade, oracle_findings
from auditenv.grader.service import REF_RE, GraderClient
from auditenv.common.drbg import DRBG

ROOT = Path(__file__).resolve().parents[1]
TYPES = ["duplicate_invoice", "missing_or_partial_allocation", "fx_wrong_date_or_rate", "period_cutoff_error"]


def only(cfg, **counts):
    c = copy.deepcopy(cfg)
    c["errors"] = {t: counts.get(t, 0) for t in TYPES}
    c["interactions"] = {"partial_fx_boundary": counts.get("partial_fx_boundary", 0),
                         "duplicate_cutoff": counts.get("duplicate_cutoff", 0)}
    return c


def _alias(ep):
    return alias_map_from_tables(parse_jsonl(ep.public["invoices.jsonl"]), parse_jsonl(ep.public["payments.jsonl"]))


# ----------------------------------------------------------------------------- дубли: проводка КАЖДОГО счёта

def test_duplicate_evidence_needs_posting_of_both_invoices(cfg):
    ep = generate_episode(5, only(cfg, duplicate_invoice=1))
    (iss,) = ep.truth["issues"]
    a, b = iss["affected_ids"]
    for es in iss["evidence_sets"]:
        rows = [x for x in es if x.startswith("R-")]
        assert len(es) == 4 and len(rows) == 2
    t = parse_public(ep.public)
    by_inv = {}
    for r in t["ledger"]:
        if r["payment_id"] is None and r["invoice_id"] in (a, b):
            by_inv.setdefault(r["invoice_id"], set()).add(r["row_id"])
    for es in iss["evidence_sets"]:
        assert set(es) & by_inv[a] and set(es) & by_inv[b]
    # одной проводки недостаточно ни для независимого валидатора, ни для matcher
    one_row = [a, b, sorted(by_inv[b])[0]]
    got, why = recompute_from_evidence(t, iss, one_row)
    assert got is None and "each" in why
    f = oracle_findings(ep.truth)
    f["findings"][0]["evidence_ids"] = one_row
    assert grade(f, ep.truth, _alias(ep), 0, 0)["tp"] == 0
    f["findings"][0]["evidence_ids"] = one_row + [sorted(by_inv[a])[0]]
    assert grade(f, ep.truth, _alias(ep), 0, 0)["tp"] == 1


def test_policy_states_required_evidence_and_cutoff_ids(cfg):
    ep = generate_episode(1, cfg)
    pol = json.loads(ep.public["policy.json"])
    req = pol["matching"]["required_evidence"]
    assert set(TYPES) <= set(req) and "EACH" in req["duplicate_invoice"]
    assert "payment_id" in pol["matching"]["canonical_affected_ids"]["period_cutoff_error"]
    blob = json.dumps(pol)
    for marker in ("evidence_sets", "issue_id", "expected_amount_minor", "ISS-"):
        assert marker not in blob


# ----------------------------------------------------------------------------- PO подписок

def test_recurring_invoices_share_po_like_duplicates(cfg):
    """Совпадение PO не должно отличать законную подписку от дубля (оба делят PO)."""
    seen = 0
    for seed in range(12):
        ep = generate_episode(30_000 + seed, cfg)
        docs = {d["doc_id"]: d for d in parse_jsonl(ep.public["documents.jsonl"])}
        po = {d["record"]["invoice_id"]: d["po_number"] for d in docs.values() if d["doc_type"] == "invoice"}
        for d in ep.provenance["distractors"]:
            if d["kind"] == "recurring_same_amount":
                assert len({po[i] for i in d["ids"]}) == 1
                seen += 1
        for m in ep.provenance["mutations"]:
            if m["type"] == "duplicate_invoice":
                c = m["cause"]
                assert po[c["original_invoice_id"]] == po[c["duplicate_invoice_id"]]
    assert seen


# ----------------------------------------------------------------------------- крайние config

@pytest.mark.parametrize("over", [
    {"interactions": {"partial_fx_boundary": 3, "duplicate_cutoff": 0}},
    {"interactions": {"partial_fx_boundary": 3, "duplicate_cutoff": 3}},
    {"interactions": {"partial_fx_boundary": 2, "duplicate_cutoff": 0}, "distractors": {"tz_boundary_receipts": 0}},
    {"errors": {"duplicate_invoice": 0, "missing_or_partial_allocation": 0, "fx_wrong_date_or_rate": 0,
                "period_cutoff_error": 10}, "interactions": {"partial_fx_boundary": 0, "duplicate_cutoff": 0}},
    {"errors": {"period_cutoff_error": 4}, "size": {"n_prior_invoices": 0, "n_next_invoices": 0},
     "distractors": {"tz_boundary_next_invoices": 0, "next_prepayments": 0}},
])
def test_extreme_configs_generate_and_validate(cfg, over):
    c = deep_merge(cfg, over)
    for seed in range(4):
        ep = generate_episode(77_000 + seed, c)
        rep = validate_episode(ep.public, ep.truth, ep.provenance)
        assert rep["ok"], (over, seed, rep["errors"][:3])


def test_committed_episodes_match_generator(cfg):
    """Закоммиченные открытые эпизоды побайтово совпадают с тем, что генерирует текущий код по сплитам."""
    from auditenv.cli import split_configs
    for split in ("train", "dev"):
        _, items = split_configs(str(ROOT / "configs" / "splits" / f"{split}.json"))
        idx = json.loads((ROOT / "episodes" / split / "index.json").read_text(encoding="utf-8"))
        for (seed, c), e in zip(items, idx):
            ep = generate_episode(seed, c)
            for n, b in ep.public.items():
                assert (ROOT / "episodes" / e["ref"] / n).read_bytes() == b, (e["ref"], n)


def test_over_max_issues_is_balanced_across_types(cfg):
    c = deep_merge(cfg, {"errors": {t: 3 for t in TYPES}, "interactions": {"partial_fx_boundary": 1, "duplicate_cutoff": 1}})
    scen, singles, dropped = _plan(DRBG(b"x"), c)
    assert len(scen) == 2 and len(singles) == MAX_ISSUES - 4 and dropped == 4
    assert {t: singles.count(t) for t in TYPES} == {t: 2 for t in TYPES}
    ep = generate_episode(3, c)
    assert len(ep.truth["issues"]) == MAX_ISSUES and ep.provenance["plan"]["dropped_over_max_issues"] == 4
    assert {i["type"] for i in ep.truth["issues"]} == set(TYPES)


# ----------------------------------------------------------------------------- снимок «до»

def test_export_before_is_clean_and_balanced(tmp_path):
    idx = json.loads((ROOT / "episodes" / "train" / "index.json").read_text(encoding="utf-8"))
    for e in idx[:4]:
        out = tmp_path / e["ref"]
        export_before(ROOT / "episodes" / e["ref"], ROOT / "grader_data" / e["ref"], out)
        files = {n: (out / n).read_bytes() for n in ("ledger.csv", "invoices.jsonl", "payments.jsonl",
                                                       "documents.jsonl", "fx.csv", "policy.json")}
        files["manifest.json"] = (ROOT / "episodes" / e["ref"] / "manifest.json").read_bytes()
        t = parse_public(files)
        issues, bad = audit(t)
        assert issues == [] and bad == [], (e["ref"], issues[:2], bad[:2])
        assert sum(r["debit_minor"] for r in t["ledger"]) == sum(r["credit_minor"] for r in t["ledger"])


def test_grader_stats_op_reveals_only_memory(cfg):
    with GraderClient() as g:
        env = AuditEnv(g, "eval")
        env.reset(seed=4, config=cfg)
        out = g.call("stats")
    assert set(out) == {"peak_rss_mb"} and (out["peak_rss_mb"] is None or out["peak_rss_mb"] > 0)


# ----------------------------------------------------------------------------- имена эпизодов на диске

def test_ref_names_relaxed_but_traversal_rejected():
    for ok in ("dev/ep_053daf06ab62f121", "hidden/ep_test_01", "jury-pairs/case.3-permuted", "X/1"):
        assert REF_RE.match(ok), ok
    for bad in ("../grader_data/dev", "/etc/passwd", "dev/..", "dev/../x", "dev/.hidden", "dev", "a/b/c", "dev/ep x"):
        assert not REF_RE.match(bad), bad


def test_renamed_episode_dir_opens_and_oracle_is_max(tmp_path):
    """Жюри может положить преобразованный эпизод под своим именем каталога — он открывается по ref."""
    src = json.loads((ROOT / "episodes" / "dev" / "index.json").read_text(encoding="utf-8"))[4]["ref"]
    (tmp_path / "ep" / "jury").mkdir(parents=True)
    (tmp_path / "gr" / "jury").mkdir(parents=True)
    shutil.copytree(ROOT / "episodes" / src, tmp_path / "ep" / "jury" / "case_01")
    shutil.copytree(ROOT / "grader_data" / src, tmp_path / "gr" / "jury" / "case_01")
    with GraderClient(str(tmp_path / "ep"), str(tmp_path / "gr"), allow_oracle=True) as g:
        env = AuditEnv(g, "train")
        env.reset(config={"episode_ref": "jury/case_01"})
        _, r, te, tr, _ = env.oracle_step()
        assert r == 1.0 and te
        (tmp_path / "ep" / "jury" / "case_01" / "documents.jsonl").unlink()
        out = g.call("open", mode="train", ref="jury/case_01")
        assert out["error"] == "missing_files" and "documents.jsonl" in out["detail"]
