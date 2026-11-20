"""A. Имитация harness жюри: своё биективное переименование ID в разных форматах + перестановка,
затем честный ответ oracle (их — через обычный submit, наш — op oracle) через среду.

usage: python3.11 -I a_jury_rename.py <repo_root>
"""
import csv
import hashlib
import io
import json
import random
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(ROOT))

from auditenv.env.core import AuditEnv  # noqa: E402
from auditenv.grader.reward import oracle_findings  # noqa: E402
from auditenv.grader.service import GraderClient  # noqa: E402

PUBLIC = ("manifest.json", "policy.json", "ledger.csv", "invoices.jsonl", "payments.jsonl", "fx.csv", "documents.jsonl")
ID_PREFIXES = ("R-", "INV-", "PAY-", "D-", "E-")


def load(ref):
    p = ROOT / "episodes" / ref
    g = ROOT / "grader_data" / ref
    return {n: (p / n).read_bytes() for n in PUBLIC}, json.loads((g / "truth.json").read_text())


def collect_ids(files):
    ids = set()
    for r in csv.DictReader(io.StringIO(files["ledger.csv"].decode())):
        for k in ("row_id", "entity_id", "invoice_id", "payment_id"):
            if r[k]:
                ids.add(r[k])
    for n in ("invoices.jsonl", "payments.jsonl", "documents.jsonl"):
        for line in files[n].decode().splitlines():
            walk(json.loads(line), lambda s: ids.add(s) if s.startswith(ID_PREFIXES) else None)
    return ids


def walk(o, fn):
    if isinstance(o, dict):
        for v in o.values():
            walk(v, fn)
    elif isinstance(o, list):
        for v in o:
            walk(v, fn)
    elif isinstance(o, str):
        fn(o)


def remap(o, m):
    if isinstance(o, dict):
        return {k: remap(v, m) for k, v in o.items()}
    if isinstance(o, list):
        return [remap(v, m) for v in o]
    if isinstance(o, str):
        return m.get(o, o)
    return o


def jury_transform(files, truth, fmt, eid_fmt, rng):
    ids = sorted(collect_ids(files))
    m = {x: fmt(i, x) for i, x in enumerate(ids)}
    assert len(set(m.values())) == len(m)
    new_eid = eid_fmt(truth["episode_id"])
    out = {}
    # ledger: csv-модуль сам экранирует запятые/кавычки
    rows = list(csv.DictReader(io.StringIO(files["ledger.csv"].decode())))
    cols = list(rows[0].keys())
    rows = [{k: m.get(v, v) for k, v in r.items()} for r in rows]
    rng.shuffle(rows)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    out["ledger.csv"] = buf.getvalue().encode()
    for n in ("invoices.jsonl", "payments.jsonl", "documents.jsonl"):
        recs = [remap(json.loads(l), m) for l in files[n].decode().splitlines() if l.strip()]
        rng.shuffle(recs)
        out[n] = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in recs).encode()
    fx = files["fx.csv"].decode().splitlines()
    body = fx[1:]
    rng.shuffle(body)
    out["fx.csv"] = ("\n".join([fx[0]] + body) + "\n").encode()
    out["policy.json"] = files["policy.json"]
    man = json.loads(files["manifest.json"])
    man["episode_id"] = new_eid
    man["files"] = {n: hashlib.sha256(b).hexdigest() for n, b in sorted(out.items())}
    out["manifest.json"] = json.dumps(man, ensure_ascii=False, sort_keys=True).encode()
    t2 = remap(truth, m)
    t2["episode_id"] = new_eid
    return out, t2


def write(root, split, name, files, truth):
    (root / "episodes" / split / name).mkdir(parents=True)
    (root / "grader" / split / name).mkdir(parents=True)
    for n, b in files.items():
        (root / "episodes" / split / name / n).write_bytes(b)
    (root / "grader" / split / name / "truth.json").write_text(json.dumps(truth, ensure_ascii=False))
    return f"{split}/{name}"


FORMATS = {
    "control: наш формат (INV-XXXX)": lambda i, x: f"{x.split('-')[0]}-Z{i:06d}",
    "UUID без скобок": lambda i, x: str(uuid.UUID(int=i + 1)),
    "UUID в фигурных скобках": lambda i, x: "{" + str(uuid.UUID(int=i + 1)) + "}",
    "пробел ('INV 000123')": lambda i, x: f"{x.split('-')[0]} {i:06d}",
    "слэш ('INV/2026/000123')": lambda i, x: f"{x.split('-')[0]}/2026/{i:06d}",
    "кириллица ('СЧ-000123')": lambda i, x: f"СЧ-{i:06d}",
    "решётка ('INV#000123')": lambda i, x: f"{x.split('-')[0]}#{i:06d}",
    "запятая ('INV,000123')": lambda i, x: f"{x.split('-')[0]},{i:06d}",
    "длина 64": lambda i, x: f"ID{i:06d}".ljust(64, "x"),
    "длина 65": lambda i, x: f"ID{i:06d}".ljust(65, "x"),
    "длина 129": lambda i, x: f"ID{i:06d}".ljust(129, "x"),
    "нижний регистр ('inv-000123')": lambda i, x: f"{x.split('-')[0].lower()}-{i:06d}",
}
EID = {
    "eid как есть": lambda e: e,
    "eid UUID": lambda e: str(uuid.UUID(int=7)),
    "eid 65 символов": lambda e: "ep_" + "a" * 62,
}


def run(ref):
    files, truth = load(ref)
    tmp = Path(tempfile.mkdtemp(prefix="jury_rename_"))
    cases = []
    for fname, fmt in FORMATS.items():
        cases.append((fname, fmt, "eid как есть", EID["eid как есть"]))
    for ename in ("eid UUID", "eid 65 символов"):
        cases.append(("control: наш формат (INV-XXXX)", FORMATS["control: наш формат (INV-XXXX)"], ename, EID[ename]))
    refs = []
    for k, (fname, fmt, ename, efmt) in enumerate(cases):
        f2, t2 = jury_transform(files, truth, fmt, efmt, random.Random(k))
        refs.append((fname, ename, write(tmp, "j", f"c{k}", f2, t2), t2))
    print(f"episode {ref}: issues={len(truth['issues'])}")
    print(f"{'формат ID':34s} {'episode_id':16s} {'их oracle (submit)':>20s} {'наш oracle':>11s}  get_record  журнал")
    with GraderClient(str(tmp / "episodes"), str(tmp / "grader"), allow_oracle=True,
                      log_path=str(tmp / "log.jsonl")) as g:
        for fname, ename, r, t2 in refs:
            env = AuditEnv(g, "train")
            obs, _ = env.reset(config={"episode_ref": r})
            some = t2["issues"][0]["affected_ids"][0] if t2["issues"] else None
            gr = "-"
            if some:
                tbl = "invoices" if any(x in some for x in ("INV", "inv")) else None
                o = env.step({"action": "get_record", "args": {"table": "invoices", "id": some}})[0]
                o2 = env.step({"action": "get_record", "args": {"table": "payments", "id": some}})[0]
                gr = "ok" if (o["error"] is None or o2["error"] is None) else (o["error"]["code"])
            fnd = oracle_findings(t2)
            fnd["episode_id"] = obs["result"]["manifest"]["episode_id"]
            _, rew, te, tr, info = env.step({"action": "submit", "args": {"findings": fnd}})
            env2 = AuditEnv(g, "train")
            env2.reset(config={"episode_ref": r})
            _, rew2, *_ = env2.oracle_step()
            print(f"{fname:34s} {ename:16s} {rew:>20.3f} {rew2:>11.3f}  {gr:10s}")
    log = [json.loads(l) for l in (tmp / "log.jsonl").read_text().splitlines()]
    errs = [(x.get("event"), x.get("error")) for x in log if x.get("event") == "invalid_findings"]
    print("invalid_findings в журнале оценщика:", len(errs), sorted(set(e[1] for e in errs)))
    shutil.rmtree(tmp)


if __name__ == "__main__":
    for ref in sys.argv[2:] or ["dev/ep_053daf06ab62f121"]:
        run(ref)
