"""Изоляция оценщика и отсутствие утечек: «злой агент» атакует среду."""
import base64
import builtins
import gc
import io
import json
import os
from pathlib import Path

import pytest

from auditenv.env.core import AuditEnv
from auditenv.env.tools import ERRORS
from auditenv.generator.episode import generate_episode
from auditenv.grader.service import GraderClient
from conftest import act

ROOT = Path(__file__).resolve().parents[1]
SECRET_MARKERS = ("expected_amount_minor", "evidence_sets", "issue_id", "interaction_group", "provenance",
                  "ISS-", "key_hex", "_kind", "bookings", "distractor")


def dev_refs():
    return [e["ref"] for e in json.loads((ROOT / "episodes" / "dev" / "index.json").read_text())]


def truth_of(ref):
    return json.loads((ROOT / "grader_data" / ref / "truth.json").read_text())


def _walk_strings(obj, seen=None, depth=0):
    """Обходит граф объектов среды (dict/list/tuple/slots/__dict__) и выдаёт строки."""
    if seen is None:
        seen = set()
    if id(obj) in seen or depth > 12:
        return
    seen.add(id(obj))
    if isinstance(obj, str):
        yield obj
        return
    if isinstance(obj, bytes):
        yield obj.decode("utf-8", "replace")
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk_strings(k, seen, depth + 1)
            yield from _walk_strings(v, seen, depth + 1)
        return
    if isinstance(obj, (list, tuple, set, frozenset)):
        for v in obj:
            yield from _walk_strings(v, seen, depth + 1)
        return
    if hasattr(obj, "__dict__"):
        yield from _walk_strings(vars(obj), seen, depth + 1)
    for s in getattr(type(obj), "__slots__", ()):
        if hasattr(obj, s):
            yield from _walk_strings(getattr(obj, s), seen, depth + 1)


def test_env_never_opens_private_files(grader, monkeypatch):
    opened = []
    real_open = builtins.open
    real_io_open = io.open

    def spy(file, *a, **k):
        opened.append(str(file))
        return real_open(file, *a, **k)
    monkeypatch.setattr(builtins, "open", spy)
    monkeypatch.setattr(io, "open", spy)
    for m in ("read_bytes", "read_text"):
        orig = getattr(Path, m)
        monkeypatch.setattr(Path, m, lambda self, *a, _o=orig, **k: (opened.append(str(self)), _o(self, *a, **k))[1])
    env = AuditEnv(grader, "train")
    obs, _ = env.reset(config={"episode_ref": dev_refs()[0]})
    env.step(act("list_records", table="ledger"))
    env.step(act("submit", findings={"episode_id": obs["result"]["manifest"]["episode_id"], "findings": []}))
    monkeypatch.setattr(builtins, "open", real_open)
    assert not any(x in p for p in opened for x in ("grader_data", "truth", "provenance")), opened


def test_env_object_graph_holds_no_truth(grader):
    ref = dev_refs()[3]
    t = truth_of(ref)
    env = AuditEnv(grader, "eval")
    obs, _ = env.reset(config={"episode_ref": ref})
    for _ in range(3):
        env.step(act("list_records", table="payments"))
    strings = list(_walk_strings(env))
    blob = "\n".join(strings)
    for marker in SECRET_MARKERS:
        assert marker not in blob, marker
    for i in t["issues"]:
        assert i["issue_id"] not in blob
    assert "seed" not in json.dumps(obs)


def test_grader_is_separate_os_process(grader):
    assert grader.pid != os.getpid()
    env = AuditEnv(grader, "train")
    assert "Grader" not in repr(env) and repr(grader) == "<GraderClient>"


def test_seed_not_exposed_and_not_bruteforceable(cfg, tmp_path):
    seed = 7
    secret = tmp_path / "secret.txt"
    secret.write_text("organizers-only-secret")
    with GraderClient(secret_file=str(secret)) as g:
        env = AuditEnv(g, "eval")
        obs, info = env.reset(seed=seed, config=cfg)
    text = json.dumps([obs, info])
    assert '"seed"' not in text
    eid = obs["result"]["manifest"]["episode_id"]
    # атакующий перебирает маленькие seed открытым генератором без секрета: совпадений нет
    for s in range(0, 40):
        assert generate_episode(s, cfg).episode_id != eid


def test_cursor_reveals_nothing(env, cfg):
    env.reset(seed=5, config=cfg)
    o, *_ = env.step(act("list_records", table="ledger", limit=10))
    raw = base64.urlsafe_b64decode(o["result"]["next_cursor"] + "==").decode()
    table, offset, mac = raw.split(":")
    assert table == "ledger" and offset == "10" and len(mac) == 16  # только публичная позиция


def test_error_messages_are_fixed_and_anonymous():
    for code, msg in ERRORS.items():
        assert len(msg) <= 120
        for bad in ("/", "\\", "Traceback", "File ", "grader", "truth", "INV-", "R-", "line "):
            if bad == "/" and "{" in msg:
                continue
            assert bad not in msg, (code, msg)


def test_info_never_contains_hints(grader):
    for ref in dev_refs()[:3]:
        env = AuditEnv(grader, "train")
        obs, info = env.reset(config={"episode_ref": ref})
        assert info == {}
        for a in [act("list_records", table="invoices"), act("get_document", doc_id="D-NOPE"),
                  act("calculator", expression="1+1")]:
            o, r, te, tr, info = env.step(a)
            assert info == {} and r == 0.0
        o, r, te, tr, info = env.step(act("submit", findings={"episode_id": obs["result"]["manifest"]["episode_id"],
                                                             "findings": []}))
        assert set(info) == {"metrics"}
        assert set(info["metrics"]) == {"valid", "precision", "recall", "f1", "tp", "fp", "fn", "inspection_steps"}


def test_grader_rejects_malicious_ops(grader):
    bad = [("open", {"mode": "train", "ref": "../grader_data/dev"}), ("open", {"mode": "train", "ref": "/etc/passwd"}),
           ("open", {"mode": "god"}), ("grade", {"token": "t999", "findings": {}, "inspection_steps": 0, "reason": "x"}),
           ("truth", {}), ("results", {}), ("open", {"mode": "train", "seed": True, "config": {}}),
           ("open", {"mode": "train", "evil": 1})]
    for op, kw in bad:
        out = grader.call(op, **kw)
        assert "error" in out, (op, out)
        assert "files" not in out and "issues" not in json.dumps(out)


def test_grader_survives_garbage_and_pickle(cfg):
    import pickle
    with GraderClient() as g:
        p = g._proc
        for junk in (b"not json\n", pickle.dumps({"op": "results"}) + b"\n", b"[]\n", b'{"op": 1}\n',
                     b'{"op":"open","kw":[1]}\n'):
            p.stdin.write(junk)
            p.stdin.flush()
            out = json.loads(p.stdout.readline())
            assert "error" in out
        assert "token" in g.call("open", mode="train", seed=1, config=cfg)


def test_eval_results_only_after_finalize(cfg):
    with GraderClient() as g:
        env = AuditEnv(g, "eval")
        obs, _ = env.reset(seed=2, config=cfg)
        env.step(act("submit", findings={"episode_id": obs["result"]["manifest"]["episode_id"], "findings": []}))
        assert g.call("results")["error"] == "not_finalized"
        assert g.call("finalize")["ok"]
        res = g.call("results")["results"]
        assert len(res) == 1 and "matches" not in res[0]
        # после finalize новые эпизоды не открываются (нельзя пробовать дальше)
        assert g.call("open", mode="eval", seed=3, config=cfg)["error"] == "run_finalized"


def test_hedging_attack_is_not_profitable(grader):
    """Злой агент знает тип и ID, но перебирает суммы/валюты — повторы штрафуются как FP."""
    from auditenv.env.store import parse_jsonl
    from auditenv.grader.matcher import alias_map_from_tables
    from auditenv.grader.reward import grade, oracle_findings
    ref = dev_refs()[5]
    t = truth_of(ref)
    pub = ROOT / "episodes" / ref
    alias = alias_map_from_tables(parse_jsonl((pub / "invoices.jsonl").read_bytes()),
                                  parse_jsonl((pub / "payments.jsonl").read_bytes()))
    exact = grade(oracle_findings(t), t, alias, 0, 0)["reward"]
    o = oracle_findings(t)
    hedged = []
    for f in o["findings"]:
        for d in (-1, 0, 1):
            hedged.append(dict(f, claimed_amount_minor=max(0, f["claimed_amount_minor"] + d)))
    r = grade(dict(o, findings=hedged[:30]), t, alias, 0, 0)
    assert r["reward"] < exact - 0.5


def test_huge_and_nested_actions_are_bounded(env, cfg):
    obs, _ = env.reset(seed=5, config=cfg)
    deep = {}
    cur = deep
    for _ in range(5000):
        cur["a"] = {}
        cur = cur["a"]
    o, r, te, tr, info = env.step(act("submit", findings=deep))
    assert te and r == -1.0
    obs, _ = env.reset(seed=5, config=cfg)
    o, *_ = env.step("x" * (600 * 1024))
    assert o["error"]["code"] == "invalid_json"
    o, *_ = env.step("[" * 100000)
    assert o["error"]["code"] in ("invalid_json", "invalid_action_format")


def test_store_is_read_only_for_agent(env, cfg):
    env.reset(seed=5, config=cfg)
    o, *_ = env.step(act("get_document", doc_id="D-NOPE"))
    o, *_ = env.step(act("list_records", table="invoices", limit=1))
    rec = o["result"]["records"][0]
    rec["line_items"].clear()
    o2, *_ = env.step(act("get_record", table="invoices", id=rec["invoice_id"]))
    assert o2["result"]["record"]["line_items"]


def test_oracle_disabled_for_agent_runs(cfg):
    """Политика с доступом к клиенту среды не может попросить у оценщика ответ oracle."""
    with GraderClient() as g:
        env = AuditEnv(g, "eval")
        env.reset(seed=1, config=cfg)
        o, r, te, tr, info = env.oracle_step()
        assert r == 0.0
        assert g.call("oracle", token="t1")["error"] in ("oracle_disabled", "already_graded")
        g.call("finalize")
        res = g.call("results")["results"]
        assert all(x["policy"] != "oracle" for x in res)


def test_external_agent_process_on_closed_episodes(cfg, tmp_path):
    """Закрытый прогон: seed+config только у оценщика, эталон не касается диска, агент — отдельный процесс."""
    import sys
    from auditenv.agent_host import run_agent
    closed = tmp_path / "closed.json"
    closed.write_text(json.dumps([{"seed": 123456789, "config": cfg}, {"seed": 987654321, "config": dict(cfg, clean=True)}]))
    secret = tmp_path / "s.txt"
    secret.write_text("x")
    agent = f"{sys.executable} {ROOT / 'examples' / 'stdin_agent.py'}"
    traj = tmp_path / "t.jsonl"
    with GraderClient(closed_episodes_file=str(closed), secret_file=str(secret)) as g:
        res = run_agent(agent, [{"closed_index": 0}, {"closed_index": 1}], g, "eval", str(traj))
        g.call("finalize")
        graded = g.call("results")["results"]
    assert [r["return"] for r in res] == [0.0, 0.0]
    assert graded[1]["reward"] == pytest.approx(1 - 0.002)  # чистая компания, 1 шаг осмотра
    text = traj.read_text()
    assert "123456789" not in text and "987654321" not in text


def test_hanging_agent_is_truncated(cfg, tmp_path):
    import sys
    from auditenv.agent_host import run_agent
    hang = tmp_path / "hang.py"
    hang.write_text("import time\nwhile True: time.sleep(1)\n")
    closed = tmp_path / "c.json"
    closed.write_text(json.dumps([{"seed": 3, "config": cfg}]))
    with GraderClient(closed_episodes_file=str(closed)) as g:
        res = run_agent(f"{sys.executable} {hang}", [{"closed_index": 0}], g, "eval", time_limit_s=1.0)
        g.call("finalize")
        graded = g.call("results")["results"]
    assert graded[0]["reason"] == "truncated_time" and graded[0]["n_findings"] == 0
    assert res[0]["wall_s"] < 5


def test_closed_episodes_eval_only_and_single_open(cfg, tmp_path):
    """Атака: переоткрыть закрытый эпизод в train-режиме или повторно, чтобы прощупать эталон наградой."""
    closed = tmp_path / "c.json"
    closed.write_text(json.dumps([{"seed": 11, "config": cfg}]))
    with GraderClient(closed_episodes_file=str(closed)) as g:
        assert g.call("open", mode="train", closed_index=0)["error"] == "closed_requires_eval"
        assert "token" in g.call("open", mode="eval", closed_index=0)
        assert g.call("open", mode="eval", closed_index=0)["error"] == "closed_already_opened"
        assert g.call("open", mode="eval", closed_index=True)["error"] == "bad_index"
