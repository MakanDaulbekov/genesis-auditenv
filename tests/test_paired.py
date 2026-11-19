"""Воспроизводимость и обобщение: повтор seed/действий; перестановка и смена ID (20+ пар)."""
import hashlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from auditenv.env.core import AuditEnv
from auditenv.env.store import parse_jsonl
from auditenv.generator.episode import generate_episode
from auditenv.generator.validate import audit, parse_public, validate_episode
from auditenv.grader.matcher import alias_map_from_tables
from auditenv.grader.reward import grade, oracle_findings
from auditenv.grader.service import GraderClient
from auditenv.runner import run_episode
from auditenv.transforms import rename_findings, transform_episode

ROOT = Path(__file__).resolve().parents[1]
N_PAIRS = 24


def _alias(files):
    return alias_map_from_tables(parse_jsonl(files["invoices.jsonl"]), parse_jsonl(files["payments.jsonl"]))


@pytest.fixture(scope="module")
def pairs(cfg):
    out = []
    for i in range(N_PAIRS):
        c = json.loads(json.dumps(cfg))
        if i % 6 == 0:
            c["clean"] = True
        if i % 6 == 1:
            c["interactions"] = {"partial_fx_boundary": 1, "duplicate_cutoff": 1}
        ep = generate_episode(30_000 + i, c)
        files, t2, p2, mp = transform_episode(ep.public, ep.truth, ep.provenance, seed=777 + i)
        out.append((ep, files, t2, p2, mp))
    return out


def test_transform_changes_ids_and_order(pairs):
    for ep, files, t2, p2, mp in pairs:
        assert files["ledger.csv"] != ep.public["ledger.csv"]
        a = {r["invoice_id"] for r in parse_jsonl(ep.public["invoices.jsonl"])}
        b = {r["invoice_id"] for r in parse_jsonl(files["invoices.jsonl"])}
        assert len(a) == len(b) and not (a & b)  # ни один ID не сохранился
        assert len(set(mp.values())) == len(mp)  # биекция


def test_transformed_episodes_validate(pairs):
    for ep, files, t2, p2, mp in pairs:
        rep = validate_episode(files, t2, p2)
        assert rep["ok"], rep["errors"]


def test_truth_renamed_consistently(pairs):
    for ep, files, t2, p2, mp in pairs:
        issues, bad = audit(parse_public(files))
        sigs = sorted((i["type"], tuple(i["affected_ids"]), i["amount"], i["currency"]) for i in issues)
        want = sorted((i["type"], tuple(sorted(mp.get(x, x) for x in i["affected_ids"])), i["expected_amount_minor"],
                       i["currency"]) for i in ep.truth["issues"])
        assert sigs == want and bad == []


def test_oracle_answer_and_reward_preserved(pairs):
    for ep, files, t2, p2, mp in pairs:
        r1 = grade(oracle_findings(ep.truth), ep.truth, _alias(ep.public), 0, 13)
        # ответ oracle исходного эпизода, переведённый той же биекцией
        r2 = grade(rename_findings(oracle_findings(ep.truth), mp), t2, _alias(files), 0, 13)
        # oracle, построенный заново по преобразованному эталону
        r3 = grade(oracle_findings(t2), t2, _alias(files), 0, 13)
        assert r1["reward_exact"] == r2["reward_exact"] == r3["reward_exact"]
        assert (r1["tp"], r1["fp"], r1["fn"]) == (r2["tp"], r2["fp"], r2["fn"])


def test_partial_answers_preserved_under_transform(pairs):
    """Не только oracle: частично верный ответ даёт ту же награду после переименования."""
    for ep, files, t2, p2, mp in pairs:
        o = oracle_findings(ep.truth)
        o["findings"] = o["findings"][::2] + [dict(o["findings"][0], claimed_amount_minor=1)] if o["findings"] else []
        r1 = grade(o, ep.truth, _alias(ep.public), 0, 40)
        r2 = grade(rename_findings(o, mp), t2, _alias(files), 0, 40)
        assert r1["reward_exact"] == r2["reward_exact"]


def _write(root: Path, split: str, files, truth):
    eid = truth["episode_id"]
    (root / "episodes" / split / eid).mkdir(parents=True)
    (root / "grader" / split / eid).mkdir(parents=True)
    for n, b in files.items():
        (root / "episodes" / split / eid / n).write_bytes(b)
    (root / "grader" / split / eid / "truth.json").write_text(json.dumps(truth))
    return f"{split}/{eid}"


def test_env_level_pairs_oracle_and_reconciler(pairs, tmp_path):
    refs = []
    for k, (ep, files, t2, p2, mp) in enumerate(pairs[:12]):
        refs.append((_write(tmp_path, "orig", ep.public, ep.truth), _write(tmp_path, "perm", files, t2)))
    with GraderClient(str(tmp_path / "episodes"), str(tmp_path / "grader"), allow_oracle=True) as g:
        for a, b in refs:
            ra = run_episode(AuditEnv(g, "train"), "oracle", {"episode_ref": a})
            rb = run_episode(AuditEnv(g, "train"), "oracle", {"episode_ref": b})
            assert ra["return"] == rb["return"] == 1.0
            sa = run_episode(AuditEnv(g, "train"), "simple_reconciler", {"episode_ref": a})
            sb = run_episode(AuditEnv(g, "train"), "simple_reconciler", {"episode_ref": b})
            # честная сверка опирается на содержание, а не на ID/порядок
            assert (sa["metrics"]["tp"], sa["metrics"]["fp"], sa["metrics"]["fn"]) == \
                   (sb["metrics"]["tp"], sb["metrics"]["fp"], sb["metrics"]["fn"])


def _replay(cfg, seed, policy):
    with GraderClient() as g:
        env = AuditEnv(g, "train")
        obs, _ = env.reset(seed=seed, config=cfg)
        from auditenv.policies.baselines import POLICIES
        gen = POLICIES[policy](obs)
        a = next(gen)
        trace = [json.dumps(obs, sort_keys=True)]
        while True:
            o, rew, te, tr, info = env.step(a)
            trace.append(json.dumps([a, o, rew, te, tr, info], sort_keys=True))
            if te or tr:
                break
            a = gen.send(o)
    return hashlib.sha256("\n".join(trace).encode()).hexdigest()


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
@pytest.mark.parametrize("policy", ["simple_reconciler", "random_explorer"])
def test_replay_same_seed_same_actions(cfg, seed, policy):
    assert _replay(cfg, seed, policy) == _replay(cfg, seed, policy)


def test_replay_across_processes_and_hashseed(cfg):
    code = ("import sys,json; sys.path.insert(0, %r); sys.path.insert(0, %r);"
            "from test_paired import _replay; cfg=json.load(open(%r)); print(_replay(cfg, 5, 'simple_reconciler'))"
            % (str(ROOT), str(ROOT / "tests"), str(ROOT / "configs" / "default.json")))
    outs = []
    for hs in ("0", "4242"):
        env = dict(os.environ, PYTHONHASHSEED=hs, PYTHONPATH=str(ROOT))
        outs.append(subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env,
                                   cwd=ROOT / "tests", check=True).stdout.strip())
    assert outs[0] == outs[1] and len(outs[0]) == 64
    assert outs[0] == _replay(cfg, 5, "simple_reconciler")


def test_generation_bytes_identical(cfg):
    a = generate_episode(99, cfg)
    b = generate_episode(99, cfg)
    assert a.public == b.public and a.private == b.private
    c = dict(cfg, config_version="other-1.0.0")
    assert generate_episode(99, c).public != a.public  # версия конфига входит в ключ
