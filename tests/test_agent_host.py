"""P0.1: внешний агент через CLI так, как его запустит жюри (относительные пути, падения, мусор, молчание)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def closed(tmp_path, cfg):
    f = tmp_path / "closed.json"
    f.write_text(json.dumps([{"seed": 41, "config": cfg}, {"seed": 42, "config": dict(cfg, clean=True)},
                             {"seed": 43, "config": cfg}]))
    s = tmp_path / "s.txt"
    s.write_text("jury-secret")
    return f, s


def host(tmp_path, closed, agent_cmd=None, policy=None, extra=()):
    f, s = closed
    args = [sys.executable, "-m", "auditenv.agent_host", "--closed-episodes-file", str(f), "--secret-file", str(s),
            "--mode", "eval", "--agent-stderr", str(tmp_path / "agent_stderr.log"),
            "--grader-log", str(tmp_path / "glog.jsonl"), *extra]
    args += ["--agent-cmd", agent_cmd] if agent_cmd else ["--policy", policy]
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    r = subprocess.run(args, cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    out = json.loads(r.stdout) if r.stdout.strip().startswith("{") else None
    return r, out


def test_relative_agent_path_works(tmp_path, closed):
    r, out = host(tmp_path, closed, agent_cmd="python examples/stdin_agent.py")
    assert r.returncode == 0, r.stderr
    assert len(out["grader"]) == 3 and len(out["agent_view"]) == 3
    assert all(g["reason"] == "submit" for g in out["grader"])


def test_crashing_agent_fails_loudly(tmp_path, closed):
    r, out = host(tmp_path, closed, agent_cmd='python -c "import sys; print(\'boom\', file=sys.stderr); sys.exit(3)"')
    assert r.returncode == 2
    assert "кодом 3" in r.stderr and "эпизоде 1" in r.stderr and "agent_stderr.log" in r.stderr
    assert len(out["grader"]) == 3  # ни один эпизод не пропал
    assert all(g["reason"] == "truncated_agent_crashed" for g in out["grader"])
    assert "boom" in (tmp_path / "agent_stderr.log").read_text()


def test_agent_crashing_midway(tmp_path, closed):
    agent = tmp_path / "mid.py"
    agent.write_text(
        "import json,sys\n"
        "n=0\n"
        "for line in sys.stdin:\n"
        "    m=json.loads(line)\n"
        "    if m.get('type')=='close': break\n"
        "    o=m['observation']\n"
        "    if m['terminated'] or m['truncated']:\n"
        "        n+=1\n"
        "        if n==1: raise SystemExit(5)\n"
        "        continue\n"
        "    eid=o['result']['manifest']['episode_id'] if o['result'] and 'manifest' in o['result'] else eid\n"
        "    print(json.dumps({'action':'submit','args':{'findings':{'episode_id':eid,'findings':[]}}}), flush=True)\n")
    r, out = host(tmp_path, closed, agent_cmd=f"python {agent}")
    assert r.returncode == 2 and "эпизоде 2" in r.stderr
    reasons = [g["reason"] for g in out["grader"]]
    assert reasons == ["submit", "truncated_agent_crashed", "truncated_agent_crashed"]


def test_garbage_agent_is_graded_not_crashed(tmp_path, closed):
    agent = tmp_path / "junk.py"
    agent.write_text("import sys\nfor line in sys.stdin:\n    print('not json at all', flush=True)\n")
    r, out = host(tmp_path, closed, agent_cmd=f"python {agent}")
    assert r.returncode == 0, r.stderr
    assert [g["reason"] for g in out["grader"]] == ["truncated_steps"] * 3
    assert all(g["inspection_steps"] == 100 for g in out["grader"])


def test_silent_agent_times_out(tmp_path, closed):
    agent = tmp_path / "silent.py"
    agent.write_text("import time\nwhile True: time.sleep(1)\n")
    r, out = host(tmp_path, closed, agent_cmd=f"python {agent}", extra=("--time-limit-s", "1"))
    assert r.returncode == 0, r.stderr
    assert [g["reason"] for g in out["grader"]] == ["truncated_time"] * 3


def test_missing_secret_warns(tmp_path, closed, cfg):
    f, _ = closed
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    r = subprocess.run([sys.executable, "-m", "auditenv.agent_host", "--closed-episodes-file", str(f), "--mode", "eval",
                        "--agent-cmd", "python examples/stdin_agent.py", "--agent-stderr", str(tmp_path / "e.log"),
                        "--grader-log", str(tmp_path / "g.jsonl")], cwd=ROOT, env=env, capture_output=True, text=True)
    assert r.returncode == 0 and "ПРЕДУПРЕЖДЕНИЕ" in r.stderr
