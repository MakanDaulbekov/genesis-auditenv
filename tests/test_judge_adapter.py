"""judge_adapter.py против «Genesis FinTech: определения проверок» (20.11.2026).

Каждый тест назван по пункту документа: C01–C20 (контракт), T01–T20 (matcher и награда, точные числа
из таблицы), M01–M12 (открытые примеры), P (парные проверки через episode_dir), D (рецепты D01–D20).
"""
import copy
import csv
import io
import json
import random
import time
from fractions import Fraction
from pathlib import Path

import pytest

import judge_adapter as ja
from auditenv.grader.reward import grade

ZERO_ERR = {"duplicate_invoice": 0, "missing_or_partial_allocation": 0, "fx_wrong_date_or_rate": 0,
            "period_cutoff_error": 0}
ZERO_INT = {"partial_fx_boundary": 0, "duplicate_cutoff": 0}


def _one(t, **extra):
    return dict({"errors": dict(ZERO_ERR, **{t: 1}), "interactions": dict(ZERO_INT)}, **extra)


# рецепты D01–D20 (те же, что в README «Адаптер жюри»)
RECIPES = {
    "D01": {"clean": True},
    "D02": {"clean": True, "distractors": {"payment_mix_pct": {"none": 10, "full": 30, "split": 60, "partial": 0,
                                                                "overpay": 0}}},
    "D03": {"clean": True, "distractors": {"prepayments": [2, 3], "refund_pct": 100}},
    "D04": {"clean": True, "distractors": {"recurring_groups": [3, 4], "same_total_diff_lines": [1, 2]}},
    "D05": {"clean": True, "currencies": {"foreign_customer_share_pct": 70},
            "distractors": {"tz_boundary_invoices": [3, 5], "tz_boundary_next_invoices": [2, 3],
                            "tz_boundary_receipts": [2, 3]}},
    "D06": _one("duplicate_invoice"),
    "D09": _one("missing_or_partial_allocation", error_variants={"missing_or_partial_allocation": ["missing"]}),
    "D10": _one("missing_or_partial_allocation", error_variants={"missing_or_partial_allocation": ["partial"]}),
    "D12": _one("fx_wrong_date_or_rate"),
    "D14": _one("period_cutoff_error"),
    "D16": {"errors": dict(ZERO_ERR), "interactions": dict(ZERO_INT, partial_fx_boundary=1)},
    "D19": {"errors": dict(ZERO_ERR), "interactions": dict(ZERO_INT, duplicate_cutoff=1)},
}
EXPECTED_TYPES = {
    "D06": ["duplicate_invoice"], "D09": ["missing_or_partial_allocation"], "D10": ["missing_or_partial_allocation"],
    "D12": ["fx_wrong_date_or_rate"], "D14": ["period_cutoff_error"],
    "D16": ["fx_wrong_date_or_rate", "missing_or_partial_allocation"],
    "D19": ["duplicate_invoice", "period_cutoff_error"],
}


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    return tmp_path_factory.mktemp("judge")


@pytest.fixture(scope="module", autouse=True)
def _log(root, monkeypatch_module):
    monkeypatch_module.setenv("AUDITENV_JUDGE_LOG", str(root / "grader_log.jsonl"))


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture(scope="module")
def C(root):
    d = root / "C"
    ja.generate_episode(101, {"clean": True}, str(d))
    return d


@pytest.fixture(scope="module")
def E(root):
    d = root / "E"
    ja.generate_episode(202, _one("duplicate_invoice"), str(d))
    assert len(json.loads((d / "truth.json").read_text(encoding="utf-8"))["issues"]) == 1
    return d


@pytest.fixture(scope="module")
def env():
    e = ja.make_env()
    yield e
    e.close()


def _closed(d):
    return {"mode": "closed", "episode_dir": str(d)}


def _rows(d, table):
    if table in ("ledger", "fx"):
        return list(csv.DictReader(io.StringIO((d / f"{table}.csv").read_text(encoding="utf-8"))))
    return [json.loads(x) for x in (d / f"{table}.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]


def _truth(d):
    return json.loads((d / "truth.json").read_text(encoding="utf-8"))


def _oracle(d, truth=None):
    t = truth or _truth(d)
    return {"episode_id": t["episode_id"], "findings": [
        {"type": i["type"], "affected_ids": i["affected_ids"], "claimed_amount_minor": i["expected_amount_minor"],
         "currency": i["currency"], "evidence_ids": i["evidence_sets"][0], "explanation": "oracle"}
        for i in t["issues"]]}


def A(name, **args):
    return {"name": name, "args": args}


# ============================================================================= Блок 1: C01–C20

def test_C01_C02_C03_reset_and_first_observation(env, C):
    obs, info = env.reset(seed=0, config=_closed(C))
    json.dumps(obs), json.dumps(info)
    assert set(obs) == {"task", "manifest", "policy", "remaining_steps"} and obs["remaining_steps"] == 100
    assert isinstance(obs["task"], str) and isinstance(obs["policy"], dict) and info == {}
    m = obs["manifest"]
    for k in ("schema_version", "episode_id", "base_currency", "period_start", "period_end", "timezone", "counts",
              "rounding"):
        assert k in m, k
    assert m["counts"]["ledger_rows"] == len(_rows(C, "ledger"))
    assert m["counts"]["invoices"] == len(_rows(C, "invoices"))
    assert m["counts"]["payments"] == len(_rows(C, "payments"))
    assert m["counts"]["documents"] == len(_rows(C, "documents"))
    s = json.dumps(obs)
    assert "truth" not in s.lower() and "provenance" not in s.lower() and "issue_id" not in s


def test_C04_C05_C06_pagination(env, C):
    env.reset(config=_closed(C))
    o, r, te, tr, info = env.step(A("list_records", table="ledger", cursor=None, limit=50))
    assert set(o) == {"result", "error", "remaining_steps"} and o["error"] is None
    assert type(te) is bool and type(tr) is bool and r == 0.0 and info == {}
    json.dumps(o)
    total = len(_rows(C, "ledger"))
    assert total >= 101
    seen, cur, pages = [], None, []
    first_cursor = o["result"]["next_cursor"]
    while True:
        o, *_ = env.step(A("list_records", table="ledger", cursor=cur, limit=50))
        assert len(o["result"]["records"]) <= 50
        pages.append(cur)
        seen += [x["row_id"] for x in o["result"]["records"]]
        cur = o["result"]["next_cursor"]
        if cur is None:
            break
    assert len(seen) == total == len(set(seen)) and sorted(seen) == sorted(x["row_id"] for x in _rows(C, "ledger"))
    a, *_ = env.step(A("list_records", table="ledger", cursor=first_cursor, limit=50))
    b, *_ = env.step(A("list_records", table="ledger", cursor=first_cursor, limit=50))
    assert a["result"] == b["result"] and b["remaining_steps"] == a["remaining_steps"] - 1


def test_C07_C08_records_and_documents(env, C):
    env.reset(config=_closed(C))
    r1 = _rows(C, "ledger")[3]
    o, *_ = env.step(A("get_record", table="ledger", id=r1["row_id"]))
    assert o["error"] is None and o["result"]["row_id"] == r1["row_id"] and o["result"]["account"] == r1["account"]
    d1 = _rows(C, "documents")[0]
    o, *_ = env.step(A("get_document", doc_id=d1["doc_id"]))
    assert o["error"] is None and o["result"] == d1


def test_C09_C10_calculator(env, C):
    env.reset(config=_closed(C))
    o, *_ = env.step(A("calculator", expression="0.1+0.2"))
    assert o["error"] is None and o["result"] == "0.3"
    for bad in ("open('/etc/passwd').read()", "__import__('os').system('id')", "2**99999"):
        o, *_ = env.step(A("calculator", expression=bad))
        assert o["result"] is None and o["error"]["code"] == "calc_error"


def test_C11_C12_C13_C14_bounded_errors_consume_steps(env, C):
    obs, _ = env.reset(config=_closed(C))
    known = {x["row_id"] for x in _rows(C, "ledger")} | {x["doc_id"] for x in _rows(C, "documents")}
    U = next(f"U-{i}" for i in range(10 ** 6) if f"U-{i}" not in known)
    left = obs["remaining_steps"]
    for act in (A("get_record", table="ledger", id=U), A("get_document", doc_id=U), A("shell", cmd="ls")):
        o, r, te, tr, _ = env.step(act)
        assert o["result"] is None and set(o["error"]) == {"code", "message"} and not te and not tr
        assert "/" not in o["error"]["message"]
        left -= 1
        assert o["remaining_steps"] == left
    o, *_ = env.step(A("get_record", table="ledger", id=_rows(C, "ledger")[0]["row_id"]))
    assert o["error"] is None and o["remaining_steps"] == left - 1


def test_C15_submit_is_final(env, E):
    env.reset(config=_closed(E))
    o, r, te, tr, info = env.step(A("submit", findings=_oracle(E)))
    assert te and not tr and o["result"] == {"accepted": True} and r == 0.0 and info == {}
    o2, r2, te2, *_ = env.step(A("submit", findings={"episode_id": _truth(E)["episode_id"], "findings": []}))
    assert te2 and r2 == 0.0 and o2["error"]["code"] == "episode_finished"


def test_C16_submit_as_action_100_is_accepted(C):
    e = ja.make_env()
    try:
        e.reset(config={"mode": "train", "episode_dir": str(C)})
        for _ in range(99):
            o, r, te, tr, _ = e.step(A("calculator", expression="1+1"))
            assert not te and not tr
        o, r, te, tr, _ = e.step(A("submit", findings={"episode_id": _truth(C)["episode_id"], "findings": []}))
        assert te and not tr and o["result"] == {"accepted": True}
        assert r == pytest.approx(1 - 0.002 * 99, abs=1e-9)     # submit не входит в inspection_steps
    finally:
        e.close()


def test_C17_100_actions_without_submit_truncate(E):
    e = ja.make_env()
    try:
        e.reset(config={"mode": "train", "episode_dir": str(E)})
        for i in range(100):
            o, r, te, tr, _ = e.step(A("calculator", expression="1+1"))
        assert tr and not te and r == -1.0                       # пустые findings на эпизоде с ошибкой
        o, r, te, tr, _ = e.step(A("submit", findings=_oracle(E)))
        assert o["error"]["code"] == "episode_finished" and r == 0.0
    finally:
        e.close()


def test_C18_time_limit_truncates(C):
    e = ja.JudgeEnv(time_limit_s=0.3)
    try:
        e.reset(config={"mode": "train", "episode_dir": str(C)})
        e.step(A("calculator", expression="1+1"))
        time.sleep(0.5)
        o, r, te, tr, _ = e.step(A("submit", findings=_oracle(C)))
        assert tr and not te
        assert r == pytest.approx(1 - 0.002 * 1, abs=1e-9)       # чистый эпизод, пустые findings
    finally:
        e.close()


def test_C19_closed_mode_hides_reward(root, E):
    log = root / "grader_log.jsonl"
    n0 = len(log.read_text(encoding="utf-8").splitlines()) if log.exists() else 0
    e = ja.make_env()
    try:
        e.reset(config=_closed(E))
        for _ in range(3):
            o, r, te, tr, info = e.step(A("list_records", table="invoices"))
            assert r == 0.0 and info == {}
        o, r, te, tr, info = e.step(A("submit", findings=_oracle(E)))
        assert te and r == 0.0 and type(r) is float and info == {} and o["result"] == {"accepted": True}
    finally:
        e.close()
    recs = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()[n0:]]
    graded = [x for x in recs if x.get("event") == "graded"]
    assert graded and graded[-1]["reward"] == pytest.approx(1 - 0.002 * 3, abs=1e-9) and graded[-1]["mode"] == "eval"


def test_C20_no_leak_anywhere(env, E):
    t = _truth(E)
    secrets = {i["issue_id"] for i in t["issues"]} | {"truth.json", "provenance", "issue_id", "evidence_sets",
                                                       "expected_amount_minor", str(E)}
    seen = []
    obs, info = env.reset(config=_closed(E))
    seen += [obs, info]
    for table in ("ledger", "invoices", "payments", "fx"):
        cur = None
        while True:
            o, r, te, tr, info = env.step(A("list_records", table=table, cursor=cur))
            seen += [o, info]
            cur = o["result"]["next_cursor"] if o["result"] else None
            if cur is None or o["remaining_steps"] < 10:
                break
    for d in _rows(E, "documents")[:5]:
        seen.append(env.step(A("get_document", doc_id=d["doc_id"]))[0])
    for bad in (A("get_document", doc_id="../truth.json"), A("get_record", table="../truth", id="x"),
                A("calculator", expression="open('truth.json')"), {"name": 1}, "not json", [1]):
        seen += list(env.step(bad))
    blob = json.dumps(seen, ensure_ascii=False)
    for s in secrets:
        assert s not in blob, s


# ============================================================================= Блок 3: T01–T20 (точные числа)

@pytest.fixture(scope="module")
def M(root, C):
    """Фикстура блока 3: копия чистого эпизода C с эталоном из одной ошибки A, сумма 10000 KZT,
    допустимое доказательство {D1, D2}, τ = 0 (manifest.rounding.tolerance_minor)."""
    d = root / "M"
    d.mkdir()
    for f in C.iterdir():
        (d / f.name).write_bytes(f.read_bytes())
    inv = [x["invoice_id"] for x in _rows(C, "invoices")]
    led = [x["row_id"] for x in _rows(C, "ledger")]
    ids = {"R": inv[:5], "D": led[:4]}
    eid = _truth(C)["episode_id"]
    A_ = {"issue_id": "I-A", "type": "duplicate_invoice", "affected_ids": [inv[0]], "expected_amount_minor": 10000,
          "currency": "KZT", "evidence_sets": [[led[0], led[1]]]}
    B_ = {"issue_id": "I-B", "type": "period_cutoff_error", "affected_ids": [inv[1]], "expected_amount_minor": 20000,
          "currency": "KZT", "evidence_sets": [[led[2], led[3]]]}
    assert json.loads((d / "manifest.json").read_text(encoding="utf-8"))["rounding"]["tolerance_minor"] == 0
    return {"dir": d, "eid": eid, "A": A_, "B": B_, **ids}


def _set_truth(m, issues):
    (m["dir"] / "truth.json").write_text(json.dumps({"episode_id": m["eid"], "issues": issues, "interactions": []}),
                                         encoding="utf-8")


def _F(m, issue, **over):
    f = {"type": issue["type"], "affected_ids": list(issue["affected_ids"]),
         "claimed_amount_minor": issue["expected_amount_minor"], "currency": issue["currency"],
         "evidence_ids": list(issue["evidence_sets"][0]), "explanation": "short"}
    f.update(over)
    return f


def _S(m, findings, steps=10):
    return ja.score_episode(str(m["dir"]), {"episode_id": m["eid"], "findings": findings}, steps)


def _check(res, tp, fp, fn, p, r, f1, ret):
    assert (res["tp"], res["fp"], res["fn"]) == (tp, fp, fn)
    for k, v in (("precision", p), ("recall", r), ("f1", f1), ("return", ret)):
        assert res[k] == pytest.approx(float(Fraction(v)), abs=1e-9), k


def test_T01_to_T17_table(M):
    a, b = M["A"], M["B"]
    _set_truth(M, [a])
    _check(_S(M, [_F(M, a)]), 1, 0, 0, 1, 1, 1, "0.98")                                         # 1
    _check(_S(M, []), 0, 0, 1, 0, 0, 0, -1)                                                    # 2
    _check(_S(M, [_F(M, a), _F(M, a)]), 1, 1, 0, "1/2", 1, "2/3", "47/150")                    # 5
    _check(_S(M, [_F(M, a, type="period_cutoff_error")]), 0, 1, 1, 0, 0, 0, -1)                # 6
    _check(_S(M, [_F(M, a, affected_ids=[M["R"][2]])]), 0, 1, 1, 0, 0, 0, -1)                  # 7
    _check(_S(M, [_F(M, a, claimed_amount_minor=10001)]), 0, 1, 1, 0, 0, 0, -1)                # 8
    _check(_S(M, [_F(M, a, currency="USD")]), 0, 1, 1, 0, 0, 0, -1)                            # 9
    _check(_S(M, [_F(M, a, evidence_ids=[a["evidence_sets"][0][0]])]), 0, 1, 1, 0, 0, 0, -1)   # 10
    _check(_S(M, [_F(M, a, evidence_ids=[a["evidence_sets"][0][0], "NO-SUCH-ID"])]), 0, 1, 1, 0, 0, 0, -1)  # 11
    flood = [_F(M, a, type=t, affected_ids=[r]) for r in M["R"] for t in ja.ERROR_TYPES]          # 12
    assert len(flood) == 20
    _check(_S(M, flood), 1, 19, 0, "1/20", 1, "2/21", "-871/1050")
    _check(_S(M, [_F(M, a)], 0), 1, 0, 0, 1, 1, 1, 1)                                          # 15
    _check(_S(M, [_F(M, a)], 50), 1, 0, 0, 1, 1, 1, "0.90")                                    # 16
    _check(_S(M, [_F(M, a)], 100), 1, 0, 0, 1, 1, 1, "0.80")                                   # 17
    _set_truth(M, [])
    _check(_S(M, []), 0, 0, 0, 0, 0, 1, "0.98")                                                # 3
    _check(_S(M, [_F(M, a)]), 0, 1, 0, 0, 0, 0, -1)                                            # 4
    _set_truth(M, [a, b])
    _check(_S(M, [_F(M, a)]), 1, 0, 1, 1, "1/2", "2/3", "47/150")                              # 13
    _check(_S(M, [_F(M, a), _F(M, a, claimed_amount_minor=1)]), 1, 1, 1, "1/2", "1/2", "1/2", "-0.02")  # 14


@pytest.mark.parametrize("bad", ["31_findings", "11_evidence", "1001_explanation"])
def test_T18_T19_T20_limits_invalid_structure(M, bad):
    a = M["A"]
    _set_truth(M, [a])
    f = _F(M, a)
    if bad == "31_findings":
        fs = [_F(M, a) for _ in range(31)]
    elif bad == "11_evidence":
        fs = [_F(M, a, evidence_ids=M["D"] + M["R"] + [_rows(M["dir"], "ledger")[10]["row_id"],
                                                         _rows(M["dir"], "ledger")[11]["row_id"]])]
        assert len(set(fs[0]["evidence_ids"])) == 11
    else:
        fs = [_F(M, a, explanation="x" * 1001)]
    res = _S(M, fs)
    assert res == {"tp": None, "fp": None, "fn": None, "precision": None, "recall": None, "f1": None, "return": -1.0}
    assert _S(M, [_F(M, a, explanation="x" * 1000)])["tp"] == 1


# ============================================================================= M01–M12

@pytest.fixture()
def M2(M):
    a = dict(M["A"], affected_ids=[M["R"][0], M["R"][1]])
    _set_truth(M, [a])
    return M, a


def test_M01_to_M09(M2):
    m, a = M2
    r1, r2, r3 = m["R"][0], m["R"][1], m["R"][2]
    d1, d2 = a["evidence_sets"][0]
    d3 = m["D"][2]
    assert _S(m, [_F(m, a, affected_ids=[r2, r1, r1])])["tp"] == 1                       # M01
    assert _S(m, [_F(m, a, affected_ids=[r1, r2, r3])])["fp"] == 1                       # M02
    assert _S(m, [_F(m, a, evidence_ids=[d2, d1, d3])])["tp"] == 1                       # M03
    res = _S(m, [_F(m, a, evidence_ids=[d1, d2, "U-NOT-THERE"])])                        # M04
    assert (res["tp"], res["fp"], res["fn"]) == (0, 1, 1)
    assert _S(m, [_F(m, a, evidence_ids=[d1] * 10 + [d2])])["return"] == -1.0            # M05 (до удаления повторов)
    assert _S(m, [_F(m, a, evidence_ids=[d1] * 9 + [d2])])["tp"] == 1
    for c in ("kzt", " KZT", "KZT "):                                                    # M06
        res = _S(m, [_F(m, a, currency=c)])
        assert (res["tp"], res["fp"], res["fn"]) == (0, 1, 1), c
    assert _S(m, [_F(m, a, claimed_amount_minor=10001)])["fp"] == 1                      # M07
    for v in ("10000", True, 10000.0):                                                   # M09
        assert _S(m, [_F(m, a, claimed_amount_minor=v)])["return"] == -1.0, v


def test_M08_published_tolerance(M2):
    m, a = M2
    mf = m["dir"] / "manifest.json"
    orig = mf.read_bytes()
    try:
        man = json.loads(orig)
        man["rounding"]["tolerance_minor"] = 1
        mf.write_text(json.dumps(man), encoding="utf-8")
        assert _S(m, [_F(m, a, claimed_amount_minor=10001)])["tp"] == 1
        assert _S(m, [_F(m, a, claimed_amount_minor=9999)])["tp"] == 1
        assert _S(m, [_F(m, a, claimed_amount_minor=10002)])["fp"] == 1
    finally:
        mf.write_bytes(orig)


def _matches(m, issues, findings):
    _set_truth(m, issues)
    _S(m, findings)
    from auditenv.grader.matcher import public_ids_from_files
    from auditenv.grader.service import PUBLIC_FILES
    pub = {n: (m["dir"] / n).read_bytes() for n in PUBLIC_FILES}
    return grade({"episode_id": m["eid"], "findings": findings}, {"episode_id": m["eid"], "issues": issues}, {}, 0, 10,
                 public_ids_from_files(pub))


def test_M10_M11_maximum_matching_and_tie_break(M):
    a = M["A"]
    d1, d2 = a["evidence_sets"][0]
    A_ = dict(a, issue_id="A", evidence_sets=[[d1]])            # F0 {d1,d2} подходит A и B; F1 {d1} — только A
    B_ = dict(a, issue_id="B", evidence_sets=[[d1, d2]])
    res = _matches(M, [A_, B_], [_F(M, a, evidence_ids=[d1, d2]), _F(M, a, evidence_ids=[d1])])
    assert (res["tp"], res["fp"], res["fn"]) == (2, 0, 0)
    assert {(x["issue_id"], x["finding_index"]) for x in res["matches"]} == {("A", 1), ("B", 0)}      # M10
    B2 = dict(a, issue_id="B", evidence_sets=[[d1]])
    fs = [_F(M, a), _F(M, a)]
    res = _matches(M, [A_, B2], fs)
    assert {(x["issue_id"], x["finding_index"]) for x in res["matches"]} == {("A", 0), ("B", 1)}      # M11
    assert res["tp"] == 2 and _matches(M, [B2, A_], list(reversed(fs)))["reward"] == res["reward"]


def test_M12_duplicate_findings_not_merged(M):
    _set_truth(M, [M["A"]])
    res = _S(M, [_F(M, M["A"]), _F(M, M["A"])])
    assert (res["tp"], res["fp"], res["fn"]) == (1, 1, 0) and res["return"] == pytest.approx(47 / 150, abs=1e-9)


# ============================================================================= Блок 4: парные проверки

def _transform(src: Path, dst: Path, rename: bool, permute: bool, seed: int = 7):
    """Как у жюри: копия эпизода, перестановка записей и/или согласованная биективная замена ID
    (row_id, doc_id, invoice_id, payment_id, entity_id и все ссылки); fx_id не меняется."""
    from auditenv.grader.matcher import public_ids_from_files
    from auditenv.grader.service import PUBLIC_FILES
    rng = random.Random(seed)
    pub = {n: (src / n).read_bytes() for n in PUBLIC_FILES}
    ids = sorted(x for x in public_ids_from_files(pub) if not x.startswith("fx:"))
    mp = {x: f"J/{i:05d} {x[::-1].lower()}" for i, x in enumerate(ids)} if rename else {}

    def sub(o):
        if isinstance(o, dict):
            return {k: sub(v) for k, v in o.items()}
        if isinstance(o, list):
            return [sub(v) for v in o]
        return mp.get(o, o) if isinstance(o, str) else o

    dst.mkdir(parents=True)
    for n in PUBLIC_FILES + ("truth.json",):
        text = (src / n).read_text(encoding="utf-8")
        if n.endswith(".csv"):
            rows = list(csv.reader(io.StringIO(text)))
            head, body = rows[0], [[mp.get(c, c) for c in r] for r in rows[1:]]
            if permute:
                rng.shuffle(body)
            buf = io.StringIO()
            w = csv.writer(buf, lineterminator="\n")
            w.writerow(head)
            w.writerows(body)
            out = buf.getvalue()
        elif n.endswith(".jsonl"):
            body = [json.dumps(sub(json.loads(x)), ensure_ascii=False) for x in text.splitlines() if x.strip()]
            if permute:
                rng.shuffle(body)
            out = "\n".join(body) + "\n"
        else:
            out = json.dumps(sub(json.loads(text)), ensure_ascii=False)
        (dst / n).write_text(out, encoding="utf-8", newline="\n")
    return mp


def _run(d, actions):
    e = ja.make_env()
    try:
        trace = [e.reset(config=_closed(d))]
        trace += [e.step(a) for a in actions]
        return trace
    finally:
        e.close()


@pytest.mark.parametrize("rename,permute", [(False, True), (True, False), (True, True)])
def test_P_pairs_oracle_reward_invariant(root, rename, permute):
    src = root / f"P_src_{rename}_{permute}"
    ja.generate_episode(303, {"errors": {k: 1 for k in ZERO_ERR}, "interactions": dict(ZERO_INT, duplicate_cutoff=1)},
                        str(src))
    dst = root / f"P_dst_{rename}_{permute}"
    mp = _transform(src, dst, rename, permute)
    t1, t2 = _truth(src), _truth(dst)
    assert len(t1["issues"]) >= 5
    # 1) повтор на исходном варианте точно воспроизводит записи, переходы и награды
    acts = [A("list_records", table="invoices"), A("get_record", table="fx", id=_rows(src, "fx")[0]["date"] and
              f"fx:{_rows(src, 'fx')[0]['date']}:{_rows(src, 'fx')[0]['source_currency']}:"
              f"{_rows(src, 'fx')[0]['target_currency']}"), A("submit", findings=_oracle(src))]
    assert _run(src, acts) == _run(src, acts)
    # 2–3) oracle с переназначенными ID: TP/FP/FN, P, R, F1 и награда совпадают при равном inspection_steps
    o1, o2 = _oracle(src, t1), _oracle(dst, t2)
    inv = {v: k for k, v in mp.items()}
    back = [{**f, "affected_ids": sorted(inv.get(x, x) for x in f["affected_ids"]),
             "evidence_ids": sorted(inv.get(x, x) for x in f["evidence_ids"])} for f in o2["findings"]]
    assert back == [{**f, "affected_ids": sorted(f["affected_ids"]), "evidence_ids": sorted(f["evidence_ids"])}
                    for f in o1["findings"]]
    s1, s2 = ja.score_episode(str(src), o1, 7), ja.score_episode(str(dst), o2, 7)
    assert s1 == s2 and s1["tp"] == len(t1["issues"]) and s1["fp"] == 0
    # тот же путь через среду в закрытом режиме: submit принят, награда политике 0
    tr = _run(dst, [A("list_records", table="ledger"), A("submit", findings=o2)])
    assert tr[-1][0]["result"] == {"accepted": True} and tr[-1][1] == 0.0
    # 4) сломанный набор доказательств не даёт ту же награду
    broken = copy.deepcopy(o2)
    broken["findings"][0]["evidence_ids"] = broken["findings"][0]["evidence_ids"][:1]
    assert ja.score_episode(str(dst), broken, 7)["return"] < s2["return"]


# ============================================================================= Блок 2: рецепты D01–D20

@pytest.mark.parametrize("name", sorted(RECIPES))
def test_D_recipes_generate_valid_episodes(root, name):
    for seed in (11, 12):
        d = root / f"{name}_{seed}"
        ja.generate_episode(seed, RECIPES[name], str(d))
        rep = ja.validate_episode(str(d))
        assert rep["ok"], (name, seed, rep["errors"][:3])
        t = _truth(d)
        if name <= "D05":
            assert t["issues"] == []
        else:
            assert sorted(i["type"] for i in t["issues"]) == EXPECTED_TYPES[name], (name, seed)
        prov = json.loads((d / "provenance.json").read_text(encoding="utf-8"))
        if name in ("D09", "D10"):
            assert prov["mutations"][0]["variant"] == ("missing" if name == "D09" else "partial")
        if name == "D03":
            assert any(p["amount_minor"] < 0 for p in _rows(d, "payments"))           # возврат = отрицательный платёж
        if name == "D02":
            assert any(len([x for x in _rows(d, "payments") if x.get("invoice_refs") and
                            any(r["invoice_id"] == i for r in x["invoice_refs"])]) >= 2
                       for i in {x["invoice_id"] for x in _rows(d, "invoices")})


def test_generate_episode_is_deterministic_and_accepts_wrapper(root):
    a, b, c = root / "det_a", root / "det_b", root / "det_c"
    ja.generate_episode(5, {"errors": {"duplicate_invoice": 1}}, str(a))
    ja.generate_episode(5, {"errors": {"duplicate_invoice": 1}}, str(b))
    ja.generate_episode(5, {"team_config": {"errors": {"duplicate_invoice": 1}}}, str(c))
    for f in a.iterdir():
        assert f.read_bytes() == (b / f.name).read_bytes() == (c / f.name).read_bytes(), f.name
