"""P0.3: стабильный интерфейс политики и подключение чужих политик одной командой."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from auditenv.env.core import AuditEnv
from auditenv.policies.api import BUILTIN, Policy, load_policy
from auditenv.runner import run_episode

ROOT = Path(__file__).resolve().parents[1]
ENV = dict(os.environ, PYTHONPATH=str(ROOT))


def dev_refs():
    return [e["ref"] for e in json.loads((ROOT / "episodes" / "dev" / "index.json").read_text())]


def test_builtins_follow_protocol():
    for name in BUILTIN:
        assert isinstance(load_policy(name), Policy)


def test_unknown_or_bad_specs_are_clear():
    with pytest.raises(ValueError, match="неизвестная политика"):
        load_policy("nope")
    with pytest.raises(PermissionError):
        load_policy("oracle_from_truth")
    with pytest.raises(TypeError, match="reset"):
        load_policy("collections:OrderedDict")  # factory без reset/act -> понятная ошибка

def test_external_class_via_cli_run(tmp_path):
    r = subprocess.run([sys.executable, "-m", "auditenv.cli", "run", "--policies", "examples.my_policy:SilentAuditor",
                        "--splits", "dev", "--trajectories", str(tmp_path / "t.jsonl"),
                        "--grader-log", str(tmp_path / "g.jsonl"), "--summary", str(tmp_path / "s.json")],
                       cwd=ROOT, env=ENV, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    agg = json.loads((tmp_path / "s.json").read_text())["aggregate"]
    assert agg["examples.my_policy:SilentAuditor"]["episodes"] == 10


def test_external_class_via_agent_host(tmp_path):
    r = subprocess.run([sys.executable, "-m", "auditenv.agent_host", "--policy", "examples.my_policy:SilentAuditor",
                        "--split", "dev", "--mode", "eval", "--agent-stderr", str(tmp_path / "e.log"),
                        "--grader-log", str(tmp_path / "g.jsonl")],
                       cwd=ROOT, env=ENV, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr + (tmp_path / "e.log").read_text()
    out = json.loads(r.stdout)
    assert len(out["grader"]) == 10 and all(g["reason"] == "submit" for g in out["grader"])


def test_oracle_from_truth_trusted_only(grader, tmp_path):
    for ref in dev_refs()[:5]:
        r = run_episode(AuditEnv(grader, "train"), "oracle_from_truth", {"episode_ref": ref}, trusted=True,
                        truth_dir=str(ROOT / "grader_data" / "dev"))
        assert r["return"] == pytest.approx(1.0) and r["steps"] == 1  # обычный submit, k=0 шагов осмотра
    r = subprocess.run([sys.executable, "-m", "auditenv.agent_host", "--policy", "oracle_from_truth", "--split", "dev"],
                       cwd=ROOT, env=ENV, capture_output=True, text=True)
    assert r.returncode == 4 and "доверенном режиме" in r.stderr


def test_competent_reconciler_solves_open_episodes(grader):
    for split in ("train", "dev"):
        for e in json.loads((ROOT / "episodes" / split / "index.json").read_text()):
            r = run_episode(AuditEnv(grader, "train"), "competent_reconciler", {"episode_ref": e["ref"]})
            m = r["metrics"]
            assert (m["fp"], m["fn"]) == (0, 0), (e["ref"], m)
            assert r["steps"] <= 60
