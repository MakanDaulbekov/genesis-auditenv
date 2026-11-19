"""Базовые политики, траектории, отсутствие эталона в наблюдениях."""
import io
import json
from pathlib import Path

import pytest

from auditenv.env.core import AuditEnv
from auditenv.runner import run_episode

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = ["ISS-", "issue_id", "expected_amount_minor", "evidence_sets", "interaction_group", "provenance",
             "truth", "grader_data", "\"seed\"", "mutation", "distractor", "_kind", "key_hex", "secret"]


def refs(split="dev"):
    return [e["ref"] for e in json.loads((ROOT / "episodes" / split / "index.json").read_text())]


@pytest.fixture(scope="module")
def runs(grader):
    out = {}
    buf = io.StringIO()
    for pol in ("random_explorer", "simple_reconciler", "flag_all", "oracle"):
        out[pol] = [run_episode(AuditEnv(grader, "train", policy_name=pol), pol, {"episode_ref": r}, buf)
                    for r in refs("dev")[:6]]
    return out, buf.getvalue()


def test_oracle_reaches_max(runs):
    res, _ = runs
    assert all(r["return"] == 1.0 for r in res["oracle"])


def test_flag_all_is_punished(runs):
    res, _ = runs
    assert all(r["return"] <= -0.9 for r in res["flag_all"])


def test_policies_valid_and_bounded(runs):
    res, _ = runs
    for pol, rs in res.items():
        for r in rs:
            assert -1.0 <= r["return"] <= 1.0 and r["steps"] <= 100
            if pol != "oracle":
                assert r["metrics"]["valid"] is True


def test_simple_reconciler_finds_something(runs):
    res, _ = runs
    assert sum(r["metrics"]["tp"] for r in res["simple_reconciler"]) > 0


def test_oracle_beats_everyone(runs):
    res, _ = runs
    for pol in ("random_explorer", "simple_reconciler", "flag_all"):
        for a, b in zip(res[pol], res["oracle"]):
            assert a["return"] <= b["return"]


def test_trajectory_fields_and_no_truth(runs):
    _, text = runs
    lines = [json.loads(l) for l in text.splitlines()]
    assert lines
    need = {"episode_id", "step", "action", "observation", "reward", "terminated", "truncated"}
    for l in lines:
        assert need <= set(l)
        assert set(l["observation"]) == {"result", "error", "steps_remaining"}
        if l["step"] > 0 and not (l["terminated"] or l["truncated"]):
            assert l["reward"] == 0.0  # промежуточная награда нулевая
    obs_text = "\n".join(json.dumps(l["observation"], ensure_ascii=False) for l in lines)
    for f in FORBIDDEN:
        assert f not in obs_text, f
    # ответ oracle не попадает в траекторию
    for l in lines:
        if l["policy"] == "oracle" and l["action"]:
            assert isinstance(l["action"]["args"]["findings"], str)


def test_committed_trajectories_have_no_truth():
    p = ROOT / "trajectories.jsonl"
    if not p.exists():
        pytest.skip("trajectories.jsonl ещё не сгенерирован")
    truth_ids = set()
    for split in ("train", "dev"):
        for d in (ROOT / "grader_data" / split).iterdir():
            if d.is_dir():
                truth_ids |= {i["issue_id"] for i in json.loads((d / "truth.json").read_text())["issues"]}
    with open(p, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            o = json.dumps(rec["observation"], ensure_ascii=False)
            for x in FORBIDDEN:
                assert x not in o
            assert not any(i in line for i in truth_ids)
            if rec["policy"] == "oracle" and rec["action"]:
                assert isinstance(rec["action"]["args"]["findings"], str)


def test_eval_mode_trajectories_have_zero_reward(grader):
    buf = io.StringIO()
    r = run_episode(AuditEnv(grader, "eval", policy_name="simple_reconciler"), "simple_reconciler",
                    {"episode_ref": refs("dev")[0]}, buf)
    assert r["return"] == 0.0 and r["metrics"] is None
    for l in buf.getvalue().splitlines():
        assert json.loads(l)["reward"] == 0.0
