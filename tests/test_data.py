"""Данные и эталон: независимый валидатор, настраиваемость, чистые случаи, анти-трюки."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from auditenv.cli import deep_merge
from auditenv.generator.episode import generate_episode
from auditenv.generator.render import json_bytes
from auditenv.generator.validate import audit, parse_public, validate_episode, validate_episode_dir

ROOT = Path(__file__).resolve().parents[1]
TYPES = ["duplicate_invoice", "missing_or_partial_allocation", "fx_wrong_date_or_rate", "period_cutoff_error"]


def only(cfg, **counts):
    c = copy.deepcopy(cfg)
    c["errors"] = {t: counts.get(t, 0) for t in TYPES}
    c["interactions"] = {"partial_fx_boundary": counts.get("partial_fx_boundary", 0),
                         "duplicate_cutoff": counts.get("duplicate_cutoff", 0)}
    return c


def test_committed_episodes_validate():
    for split in ("train", "dev"):
        idx = json.loads((ROOT / "episodes" / split / "index.json").read_text())
        assert len(idx) == 10
        for e in idx:
            rep = validate_episode_dir(ROOT / "episodes" / e["ref"], ROOT / "grader_data" / e["ref"])
            assert rep["ok"], (e["ref"], rep["errors"])


@pytest.mark.parametrize("block", range(4))
def test_validator_on_many_random_seeds(cfg, block):
    for seed in range(block * 30, block * 30 + 30):
        ep = generate_episode(10_000 + seed, cfg)
        rep = validate_episode(ep.public, ep.truth, ep.provenance)
        assert rep["ok"], (seed, rep["errors"])


@pytest.mark.parametrize("typ", TYPES)
def test_each_error_type_independently_configurable(cfg, typ):
    c = only(cfg, **{typ: 2})
    for seed in range(5):
        ep = generate_episode(seed, c)
        assert [i["type"] for i in ep.truth["issues"]] == [typ, typ]
        assert validate_episode(ep.public, ep.truth, ep.provenance)["ok"]


def test_variants_present_across_seeds(cfg):
    seen = set()
    for seed in range(40):
        ep = generate_episode(seed, cfg)
        seen |= {(m["type"], m["variant"].split(":")[0]) for m in ep.provenance["mutations"]}
    for need in [("missing_or_partial_allocation", "missing"), ("missing_or_partial_allocation", "partial"),
                 ("fx_wrong_date_or_rate", "naive_date"), ("fx_wrong_date_or_rate", "shifted_date"),
                 ("period_cutoff_error", "next_period_booked"), ("period_cutoff_error", "prior_period_booked"),
                 ("duplicate_invoice", "in_period_copy"), ("period_cutoff_error", "next_period_payment_booked")]:
        assert need in seen, need


def test_clean_companies_have_distractors(clean_cfg):
    kinds = set()
    for seed in range(15):
        ep = generate_episode(seed, clean_cfg)
        assert ep.truth["issues"] == [] and ep.provenance["mutations"] == []
        assert all(not d["inserted"] and not d["deleted"] and not d["updated"] for d in ep.provenance["diff"].values())
        kinds |= {d["kind"] for d in ep.provenance["distractors"]}
        assert len(ep.provenance["distractors"]) >= 3
        assert validate_episode(ep.public, ep.truth, ep.provenance)["ok"]
    for k in ["recurring_same_amount", "split_payment", "partial_payment_open_invoice", "prepayment_refunded",
              "tz_boundary_invoice_in_period", "tz_boundary_next_invoice_not_booked", "tz_boundary_foreign_receipt",
              "overpayment_to_advances", "batch_payment"]:
        assert k in kinds, k


@pytest.mark.parametrize("scenario,types", [
    ("partial_fx_boundary", {"missing_or_partial_allocation", "fx_wrong_date_or_rate"}),
    ("duplicate_cutoff", {"duplicate_invoice", "period_cutoff_error"}),
])
def test_interaction_scenarios(cfg, scenario, types):
    c = only(cfg, **{scenario: 1})
    for seed in range(8):
        ep = generate_episode(seed, c)
        assert len(ep.truth["interactions"]) == 1
        g = ep.truth["interactions"][0]
        members = [i for i in ep.truth["issues"] if i["issue_id"] in g["issue_ids"]]
        assert g["scenario"] == scenario and g["n_issues"] == 2 and {m["type"] for m in members} == types
        assert all(m["expected_amount_minor"] > 0 for m in members)
        # оба issue касаются одного документа, но не совпадают по сигнатуре
        shared = set(members[0]["affected_ids"]) & set(members[1]["affected_ids"])
        assert shared
        if scenario == "partial_fx_boundary":
            assert members[0]["currency"] != members[1]["currency"]  # валюта платежа vs base
        assert validate_episode(ep.public, ep.truth, ep.provenance)["ok"]


def test_sizes_within_spec(cfg):
    for seed in range(40):
        ep = generate_episode(seed, cfg)
        m = json.loads(ep.public["manifest.json"])["counts"]
        assert 200 <= m["ledger_rows"] <= 2000
        assert 20 <= m["documents"] <= 200
        assert 0 <= len(ep.truth["issues"]) <= 12


def test_no_floats_and_formats(cfg):
    ep = generate_episode(3, cfg)
    t = parse_public(ep.public)

    def walk(o):
        if isinstance(o, float):
            raise AssertionError("float in public data")
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        if isinstance(o, list):
            for v in o:
                walk(v)
    for name in ("invoices", "payments", "documents", "policy", "manifest"):
        walk(t[name])
    for r in t["fx"]:
        assert "e" not in r["rate"].lower() and r["rate"].replace(".", "", 1).isdigit()
    offsets = {i["issued_at"][-6:] for i in t["invoices"]} | {p["timestamp"][-6:] for p in t["payments"]}
    assert len(offsets) >= 3  # несколько часовых поясов — граница периода реальная ловушка
    # явные null вместо угаданных связей
    assert any(r["invoice_id"] is None for r in t["ledger"])
    assert all(p["invoice_refs"] is None or len(p["invoice_refs"]) > 0 for p in t["payments"])


def _rehash(files):
    m = json.loads(files["manifest.json"])
    for n in m["files"]:
        m["files"][n] = hashlib.sha256(files[n]).hexdigest()
    files["manifest.json"] = json_bytes(m)
    return files


def test_validator_detects_corruption(cfg):
    s = next(s for s in range(50) if any(len(i["affected_ids"]) == 2 for i in generate_episode(s, cfg).truth["issues"]))
    ep = generate_episode(s, cfg)
    good = validate_episode(ep.public, ep.truth, ep.provenance)
    assert good["ok"]
    cases = []
    t = copy.deepcopy(ep.truth); t["issues"][0]["expected_amount_minor"] += 1; cases.append(("amount", ep.public, t))
    t = copy.deepcopy(ep.truth); t["issues"].pop(); cases.append(("missing", ep.public, t))
    t = copy.deepcopy(ep.truth); t["issues"][0]["evidence_sets"][0] = t["issues"][0]["evidence_sets"][0][:1]
    cases.append(("evidence", ep.public, t))
    t = copy.deepcopy(ep.truth)
    two = next(i for i in t["issues"] if len(i["affected_ids"]) == 2); two["affected_ids"] = two["affected_ids"][:1]
    cases.append(("affected", ep.public, t))
    t = copy.deepcopy(ep.truth); t["issues"][0]["currency"] = "CHF" if t["issues"][0]["currency"] != "CHF" else "USD"
    cases.append(("currency", ep.public, t))
    # испорченная проводка (manifest пересчитан, чтобы ловил именно бухгалтерский контроль)
    f = dict(ep.public)
    lines = f["ledger.csv"].decode().splitlines()
    for i, l in enumerate(lines[1:], 1):
        c = l.split(",")
        if c[2] == "revenue":
            c[4] = str(int(c[4]) + 7); lines[i] = ",".join(c); break
    f["ledger.csv"] = ("\n".join(lines) + "\n").encode()
    cases.append(("unbalanced", _rehash(f), ep.truth))
    f2 = dict(ep.public); f2["fx.csv"] = ep.public["fx.csv"] + b"2099-01-01,EUR,USD,1\n"
    cases.append(("hash", f2, ep.truth))
    for name, files, truth in cases:
        assert not validate_episode(files, truth, ep.provenance)["ok"], name


def test_before_state_is_clean_and_after_matches(cfg):
    from auditenv.generator.validate import reconstruct_before
    ep = generate_episode(21, cfg)
    t = parse_public(ep.public)
    issues, bad = audit(reconstruct_before(t, ep.provenance["diff"]))
    assert issues == [] and bad == []
    issues, bad = audit(t)
    assert len(issues) == len(ep.truth["issues"]) and bad == []


# ----------------------------------------------------------------------------- анти-трюки (п.7 валидатора)

def _population(cfg, n=60):
    eps = [generate_episode(20_000 + s, cfg) for s in range(n)]
    return [(ep, parse_public(ep.public)) for ep in eps]


@pytest.fixture(scope="module")
def population(cfg):
    return _population(cfg)


def test_position_heuristic_no_better_than_chance(population):
    """«Последние 10% строк/документов» не находят внедрённые записи чаще случайного."""
    hit_inv = tot_inv = hit_row = tot_row = 0
    for ep, t in population:
        aff = {x for i in ep.truth["issues"] for x in i["affected_ids"]}
        ev_rows = {x for i in ep.truth["issues"] for es in i["evidence_sets"] for x in es if x.startswith("R-")}
        n = len(t["invoices"])
        tail = {r["invoice_id"] for r in t["invoices"][int(n * 0.9):]}
        hit_inv += len(aff & tail); tot_inv += len(aff & {r["invoice_id"] for r in t["invoices"]})
        m = len(t["ledger"])
        tail_r = {r["row_id"] for r in t["ledger"][int(m * 0.9):]}
        hit_row += len(ev_rows & tail_r); tot_row += len(ev_rows)
    assert 0.02 <= hit_inv / tot_inv <= 0.22
    assert 0.02 <= hit_row / tot_row <= 0.22


def test_id_order_carries_no_signal(population):
    ranks = []
    for ep, t in population:
        ids = sorted(r["invoice_id"] for r in t["invoices"])
        pos = {x: i / (len(ids) - 1) for i, x in enumerate(ids)}
        for i in ep.truth["issues"]:
            ranks += [pos[x] for x in i["affected_ids"] if x in pos]
    mean = sum(ranks) / len(ranks)
    assert 0.38 <= mean <= 0.62


def test_round_amounts_not_a_label(population):
    def is_round(inv):
        e = 0 if inv["currency"] == "JPY" else 2
        return inv["total_minor"] % (100 * 10 ** e) == 0
    mut = base = n_mut = n_all = 0
    for ep, t in population:
        aff = {x for i in ep.truth["issues"] for x in i["affected_ids"]}
        for inv in t["invoices"]:
            n_all += 1; base += is_round(inv)
            if inv["invoice_id"] in aff:
                n_mut += 1; mut += is_round(inv)
    assert mut / n_mut <= 2 * base / n_all + 0.05


def test_mutated_records_same_formats(population):
    """Формат времени/ID у внедрённых документов не отличается от остальных."""
    for ep, t in population[:20]:
        dup_ids = {m["cause"]["duplicate_invoice_id"] for m in ep.provenance["mutations"] if m["type"] == "duplicate_invoice"}
        for inv in t["invoices"]:
            assert len(inv["invoice_id"]) == 11 and len(inv["doc_id"]) == 11
            assert inv["issued_at"][19] in "+-"
        for d in dup_ids:
            assert d in {i["invoice_id"] for i in t["invoices"]}


# ----------------------------------------------------------------------------- P1.2: категориальные поля документов

def _mi_bits(counter):
    import math
    n = sum(counter.values())
    px, py = {}, {}
    for (x, y), v in counter.items():
        px[x] = px.get(x, 0) + v
        py[y] = py.get(y, 0) + v
    return sum(v / n * math.log2((v / n) / (px[x] / n * py[y] / n)) for (x, y), v in counter.items() if v)


def test_categorical_fields_carry_no_label_signal(cfg):
    """Взаимная информация «поле документа ↔ затронут ошибкой» мала на 200 seed, и ни одно частое
    значение заметки не является «безопасным» (доля ошибок < 1/5 средней при ≥200 появлениях).
    Структурные исключения законны: у предоплаты нет remittance-строк, поэтому allocation-ошибки
    у неё невозможны по определению; это учитывается только через общий порог MI."""
    from collections import Counter
    from auditenv.common.drbg import DRBG
    rng = DRBG(b"mi-test")
    fields = {k: Counter() for k in ("note", "note_fx", "refund_of", "n_remit_lines", "memo", "has_po")}
    for _ in range(200):
        ep = generate_episode(rng.randint(0, 2**62), cfg)
        aff = {x for t in ep.truth["issues"] for x in t["affected_ids"]}
        fxaff = {x for t in ep.truth["issues"] if t["type"] == "fx_wrong_date_or_rate" for x in t["affected_ids"]}
        for line in ep.public["documents.jsonl"].decode().splitlines():
            d = json.loads(line)
            r = d["record"]
            if d["doc_type"] == "payment":
                lab = r["payment_id"] in aff
                fields["note"][(d["remittance"]["note"], lab)] += 1
                fields["note_fx"][(d["remittance"]["note"], r["payment_id"] in fxaff)] += 1
                fields["refund_of"][(d["refund_of"] is not None, lab)] += 1
                fields["n_remit_lines"][(len(d["remittance"]["lines"]), lab)] += 1
            else:
                lab = r["invoice_id"] in aff
                fields["memo"][(d["memo"], lab)] += 1
                fields["has_po"][(bool(d["po_number"]), lab)] += 1
    for name, c in fields.items():
        assert _mi_bits(c) < 0.01, (name, _mi_bits(c))
        if name == "note":
            overall = sum(v for (x, y), v in c.items() if y) / sum(c.values())
            for v in {x for x, _ in c}:
                n = c[(v, True)] + c[(v, False)]
                if n >= 200:  # нет «безопасной» заметки: доля ошибок не ниже 1/5 средней
                    assert c[(v, True)] / n >= 0.2 * overall, (name, v, c[(v, True)], n)


def test_minute_offsets_and_jpy_base(cfg):
    """P2: компании со смещением с минутами (+05:30/+05:45/-09:30) и base=JPY (экспонента 0) корректны."""
    seen = set()
    for seed in range(120):
        ep = generate_episode(40_000 + seed, cfg)
        tz = json.loads(ep.public["manifest.json"])["timezone"]
        if tz[-2:] != "00":
            seen.add(tz)
            assert validate_episode(ep.public, ep.truth, ep.provenance)["ok"]
    assert seen, "нет ни одной компании со смещением с минутами"
    dev = json.loads((ROOT / "configs" / "splits" / "dev.json").read_text())
    assert any(e["overrides"].get("currencies", {}).get("base") == ["JPY"] for e in dev["episodes"])
    jp = [e for e in json.loads((ROOT / "episodes" / "dev" / "index.json").read_text())
          if json.loads((ROOT / "episodes" / e["ref"] / "manifest.json").read_text())["base_currency"] == "JPY"]
    assert jp


def test_payment_cutoff_variant_validates(cfg):
    c = only(cfg, period_cutoff_error=3)
    hits = 0
    for seed in range(30):
        ep = generate_episode(seed, c)
        pays = [m for m in ep.provenance["mutations"] if m["variant"].startswith("next_period_payment_booked")]
        hits += len(pays)
        for m in pays:
            assert m["issue"]["affected_ids"][0].startswith("PAY-")
        assert validate_episode(ep.public, ep.truth, ep.provenance)["ok"]
    assert hits > 0
