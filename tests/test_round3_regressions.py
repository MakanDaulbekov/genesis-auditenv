"""Регрессии аудита раунда 3 (AUDIT.md). Каждый тест падал до своего фикса."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auditenv.env.core import AuditEnv  # noqa: E402
from auditenv.grader.service import GraderClient  # noqa: E402

REF = "dev/ep_053daf06ab62f121"


def _nest(d):
    x = []
    for _ in range(d):
        x = [x]
    return x


# ----------------------------------------------------------------------------- F3: глубоко вложенный submit

def test_deeply_nested_submit_never_crashes_and_is_always_graded(tmp_path):
    """Окно глубины, где json.dumps в step проходит, а сериализация запроса к оценщику (или его разбор) падает,
    давало RecursionError из env.step либо завершённый, но не оценённый эпизод (0 вместо −1).
    Перебор широкий: положение окна зависит от глубины стека вызывающего."""
    terminal = 0
    with GraderClient(str(ROOT / "episodes"), str(ROOT / "grader_data"), log_path=str(tmp_path / "g.jsonl")) as g:
        env = AuditEnv(g, "train")
        for d in range(850, 1151):
            for as_str in (False, True):
                obs, _ = env.reset(config={"episode_ref": REF})
                eid = obs["result"]["manifest"]["episode_id"]
                if as_str:
                    a = '{"action":"submit","args":{"findings":{"episode_id":"%s","findings":%s%s}}}' % (
                        eid, "[" * d, "]" * d)
                else:
                    a = {"action": "submit", "args": {"findings": {"episode_id": eid, "findings": _nest(d)}}}
                o, r, te, tr, info = env.step(a)  # не должно бросать
                if te:
                    terminal += 1
                    assert r == -1.0 and info["metrics"]["valid"] is False, (d, as_str, r, info)
                else:  # строка, которую не разобрать: ограниченная ошибка, шаг израсходован
                    assert as_str and o["error"]["code"] == "invalid_json" and not tr, (d, o)
        g.call("finalize")
        graded = g.call("results")["results"]
    assert terminal >= 301 and len(graded) == terminal
    assert all(x["event"] == "invalid_findings" and x["reward"] == -1.0 for x in graded)


# ----------------------------------------------------------------------------- F1: ID в формате жюри

import csv  # noqa: E402
import io  # noqa: E402

import pytest  # noqa: E402

from auditenv.grader.reward import oracle_findings  # noqa: E402

PUBLIC = ("manifest.json", "policy.json", "ledger.csv", "invoices.jsonl", "payments.jsonl", "fx.csv", "documents.jsonl")
ID_PREFIXES = ("R-", "INV-", "PAY-", "D-", "E-")


def _remap(o, m):
    if isinstance(o, dict):
        return {k: _remap(v, m) for k, v in o.items()}
    if isinstance(o, list):
        return [_remap(v, m) for v in o]
    return m.get(o, o) if isinstance(o, str) else o


def _jury_rename(ref, fmt, new_eid=None):
    """Harness жюри: биективная смена ВСЕХ ID открытых записей и эталона в свой формат + перестановка строк.
    manifest не пересчитывается (среда хэши не проверяет). -> (files, truth)."""
    files = {n: (ROOT / "episodes" / ref / n).read_bytes() for n in PUBLIC}
    truth = json.loads((ROOT / "grader_data" / ref / "truth.json").read_text(encoding="utf-8"))
    ids = set()

    def collect(o):
        if isinstance(o, dict):
            for v in o.values():
                collect(v)
        elif isinstance(o, list):
            for v in o:
                collect(v)
        elif isinstance(o, str) and o.startswith(ID_PREFIXES):
            ids.add(o)

    rows = list(csv.DictReader(io.StringIO(files["ledger.csv"].decode("utf-8"))))
    collect(rows)
    for n in ("invoices.jsonl", "payments.jsonl", "documents.jsonl"):
        collect([json.loads(x) for x in files[n].decode("utf-8").splitlines()])
    m = {x: fmt(i, x) for i, x in enumerate(sorted(ids))}
    assert len(set(m.values())) == len(m)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(reversed([_remap(r, m) for r in rows]))
    out = dict(files, **{"ledger.csv": buf.getvalue().encode("utf-8")})
    for n in ("invoices.jsonl", "payments.jsonl", "documents.jsonl"):
        recs = [_remap(json.loads(x), m) for x in files[n].decode("utf-8").splitlines()][::-1]
        out[n] = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in recs).encode("utf-8")
    t2 = _remap(truth, m)
    if new_eid:
        man = json.loads(files["manifest.json"])
        man["episode_id"] = new_eid
        out["manifest.json"] = json.dumps(man, ensure_ascii=False, sort_keys=True).encode("utf-8")
        t2["episode_id"] = new_eid
    return out, t2


def _put(root, name, files, truth):
    (root / "episodes" / "j" / name).mkdir(parents=True)
    (root / "grader" / "j" / name).mkdir(parents=True)
    for n, b in files.items():
        (root / "episodes" / "j" / name / n).write_bytes(b)
    (root / "grader" / "j" / name / "truth.json").write_text(json.dumps(truth, ensure_ascii=False), encoding="utf-8")
    return f"j/{name}"


JURY_FORMATS = {
    "slash": lambda i, x: f"{x.split('-')[0]}/2026/{i:06d}",
    "space": lambda i, x: f"{x.split('-')[0]} {i:06d}",
    "braced_uuid": lambda i, x: "{%08x-0000-4000-8000-%012x}" % (i, i),
    "cyrillic": lambda i, x: f"СЧ-{i:06d}",
    "hash_comma": lambda i, x: f"{x.split('-')[0]}#{i},{i}",
}
DIRTY_REF = "dev/ep_3254277a9552d3b2"  # все 4 типа, 8 issue


@pytest.mark.parametrize("fmt", sorted(JURY_FORMATS))
def test_jury_renamed_ids_keep_oracle_reward(tmp_path, fmt):
    """Смена ID жюри в формат вне [A-Za-z0-9_.:-] сохраняет ответ oracle и награду (раньше −1: malformed id)."""
    files, t2 = _jury_rename(DIRTY_REF, JURY_FORMATS[fmt])
    ref = _put(tmp_path, "e1", files, t2)
    with GraderClient(str(tmp_path / "episodes"), str(tmp_path / "grader"), allow_oracle=True) as g:
        env = AuditEnv(g, "train")
        obs, _ = env.reset(config={"episode_ref": ref})
        some = t2["issues"][0]["affected_ids"][0]
        assert env.step({"action": "get_record", "args": {"table": "invoices", "id": some}})[0]["error"] is None
        _, r, te, _, info = env.step({"action": "submit", "args": {"findings": oracle_findings(t2)}})
        assert te and info["metrics"]["valid"] and (info["metrics"]["tp"], info["metrics"]["fp"]) == (8, 0)
        assert r == pytest.approx(1 - 0.002 * 1)
        env2 = AuditEnv(g, "train")
        env2.reset(config={"episode_ref": ref})
        assert env2.oracle_step()[1] == 1.0  # oracle процесса оценщика


def test_odd_format_ids_still_invalid_unless_they_exist(tmp_path):
    """Правило не ослаблено для несуществующих ID: странный формат и Unicode-двойники по-прежнему −1."""
    files, t2 = _jury_rename(DIRTY_REF, JURY_FORMATS["slash"])
    ref = _put(tmp_path, "e1", files, t2)
    homoglyph = str.maketrans("ABCEHKMOPTXY", "АВСЕНКМОРТХУ")
    with GraderClient(str(tmp_path / "episodes"), str(tmp_path / "grader")) as g:
        for bad in ("INV/2026/999999", "../../grader_data/truth.json", "INV 000001"):
            env = AuditEnv(g, "train")
            env.reset(config={"episode_ref": ref})
            o = oracle_findings(t2)
            o["findings"][0]["evidence_ids"] = o["findings"][0]["evidence_ids"][:9] + [bad]
            _, r, te, _, info = env.step({"action": "submit", "args": {"findings": o}})
            assert te and r == -1.0 and info["metrics"]["valid"] is False, bad
    with GraderClient(str(ROOT / "episodes"), str(ROOT / "grader_data")) as g:
        env = AuditEnv(g, "train")
        env.reset(config={"episode_ref": DIRTY_REF})
        truth = json.loads((ROOT / "grader_data" / DIRTY_REF / "truth.json").read_text(encoding="utf-8"))
        o = oracle_findings(truth)
        o["findings"][0]["affected_ids"] = [x.translate(homoglyph) for x in o["findings"][0]["affected_ids"]]
        assert o["findings"][0]["affected_ids"] != oracle_findings(truth)["findings"][0]["affected_ids"]
        _, r, te, _, info = env.step({"action": "submit", "args": {"findings": o}})
        assert te and r == -1.0 and info["metrics"]["valid"] is False
