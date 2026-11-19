"""P0.4: режим жюри одной командой на чужих seed/параметрах/политиках."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = dict(os.environ, PYTHONPATH=str(ROOT))


def _split(tmp_path):
    f = tmp_path / "closed_split.json"
    f.write_text(json.dumps({"split": "hidden", "config_version": "hidden-0.1", "episodes": [
        {"seed": 777001, "overrides": {"clean": True}},
        {"seed": 777002, "overrides": {"interactions": {"partial_fx_boundary": 1, "duplicate_cutoff": 1}}},
        {"seed": 777003, "overrides": {"currencies": {"base": ["JPY"]}}},
        {"seed": 777004}]}))
    return f


def run(tmp_path, *extra):
    return subprocess.run([sys.executable, "-m", "auditenv.jury_run", "--split", str(_split(tmp_path)),
                           "--out", str(tmp_path / "out"), "--pairs", "4", *extra],
                          cwd=ROOT, env=ENV, capture_output=True, text=True, timeout=300)


def test_jury_run_end_to_end(tmp_path):
    s = tmp_path / "secret.txt"
    s.write_text("their-secret")
    r = run(tmp_path, "--secret-file", str(s), "--policies", "examples.my_policy:SilentAuditor", "competent_reconciler")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    assert "корректных эпизодов: 4/4" in r.stdout and "пройдено: 4/4" in r.stdout
    assert "examples.my_policy:SilentAuditor" in r.stdout and "| oracle | all |" in r.stdout
    assert "ПРЕДУПРЕЖДЕНИЕ" not in r.stderr
    out = tmp_path / "out"
    idx = json.loads((out / "episodes" / "hidden" / "index.json").read_text())
    for e in idx:  # открытая часть и эталон — в разных корнях
        assert not (out / "episodes" / e["ref"] / "truth.json").exists()
        assert (out / "grader" / e["ref"] / "truth.json").exists()
    summ = json.loads((out / "jury_summary.json").read_text())
    assert summ["aggregate"]["oracle"]["mean_return"] == 1.0
    assert summ["aggregate"]["competent_reconciler"]["micro"]["fn"] == 0


def test_jury_run_without_secret_warns_and_differs(tmp_path):
    r = run(tmp_path, "--policies", "flag_all")
    assert r.returncode == 0 and "ПРЕДУПРЕЖДЕНИЕ" in r.stderr
    ids_open = sorted(e["ref"] for e in json.loads((tmp_path / "out/episodes/hidden/index.json").read_text()))
    s = tmp_path / "secret.txt"
    s.write_text("x")
    run(tmp_path, "--secret-file", str(s), "--policies", "flag_all")
    ids_secret = sorted(e["ref"] for e in json.loads((tmp_path / "out/episodes/hidden/index.json").read_text()))
    assert set(ids_open).isdisjoint(ids_secret)  # секрет меняет эпизоды: перебор seed без секрета бесполезен
