"""Режим жюри (раздел 06 PDF) одной командой — на ИХ seed и параметрах, с ИХ или нашими политиками.

make jury-run SPLIT=closed_split.json [SECRET=secret.txt] [POLICY=their.module:Factory]
python -m auditenv.jury_run --split closed_split.json --secret-file secret.txt --policies their.mod:Factory

Шаги:
  1. генерация: открытые записи -> <out>/episodes/<split>/, эталон -> <out>/grader/<split>/ (РАЗНЫЕ корни;
     эталон сохраняется на диск, чтобы жюри могло независимо проверить его и доказательства);
  2. независимый валидатор на каждом эпизоде;
  3. политики в режиме eval (по умолчанию наши 4 базовые + competent_reconciler; oracle — в процессе оценщика);
  4. 20 парных проверок: повтор seed/действий (побайтово) + перестановка и смена ID (transforms);
  5. таблица P/R/F1/возврат/шаги/время по политикам: все эпизоды, чистые, со взаимодействиями.
Ненулевой выход, если провалились валидация, парные проверки или какой-либо эпизод не оценён.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

from .cli import DEFAULT_SECRET_NOTE, cmd_generate, export_before, read_secret, split_configs
from .env.core import AuditEnv
from .env.store import parse_jsonl
from .generator.episode import generate_episode
from .generator.validate import audit, parse_public, validate_episode, validate_episode_dir
from .grader.matcher import alias_map_from_tables
from .grader.reward import grade, oracle_findings
from .grader.service import GraderClient
from .policies.api import load_policy
from .runner import run_policies
from .transforms import rename_findings, transform_episode

DEFAULT_POLICIES = ["random_explorer", "simple_reconciler", "competent_reconciler", "flag_all", "oracle"]
PUBLIC = ("manifest.json", "policy.json", "ledger.csv", "invoices.jsonl", "payments.jsonl", "fx.csv", "documents.jsonl")


def _load(ep_root: Path, gr_root: Path, ref: str):
    files = {n: (ep_root / ref / n).read_bytes() for n in PUBLIC}
    truth = json.loads((gr_root / ref / "truth.json").read_text(encoding="utf-8"))
    prov = json.loads((gr_root / ref / "provenance.json").read_text(encoding="utf-8"))
    return files, truth, prov


def _trace(g: GraderClient, ref: str, policy: str) -> str:
    env = AuditEnv(g, "train")
    obs, info = env.reset(config={"episode_ref": ref})
    h = hashlib.sha256(json.dumps([obs, info], sort_keys=True).encode())
    pol = load_policy(policy)
    pol.reset(obs)
    a = pol.act(obs)
    while True:
        out = env.step(a)
        h.update(json.dumps([a, list(out)], sort_keys=True).encode())
        if out[2] or out[3]:
            return h.hexdigest()
        a = pol.act(out[0])


def paired_checks(items, refs, ep_root: Path, gr_root: Path, secret: str, n: int = 20) -> list[dict]:
    out = []
    with GraderClient(str(ep_root), str(gr_root)) as g1, GraderClient(str(ep_root), str(gr_root)) as g2:
        for i in range(n):
            k = i % len(refs)
            seed, cfg = items[k]
            ref = refs[k]
            files, truth, prov = _load(ep_root, gr_root, ref)
            # (а) повтор seed+config+версии -> те же байты; повтор действий -> те же переходы и награды
            ep2 = generate_episode(seed, cfg, secret)
            regen_ok = ep2.public == files and json.loads(ep2.private["truth.json"]) == truth
            pol = "competent_reconciler" if i % 2 == 0 else "random_explorer"
            replay_ok = _trace(g1, ref, pol) == _trace(g2, ref, pol)
            # (б) перестановка + биективная смена ID
            f2, t2, p2, mp = transform_episode(files, truth, prov, seed=10_000 + i)
            al1 = alias_map_from_tables(parse_jsonl(files["invoices.jsonl"]), parse_jsonl(files["payments.jsonl"]))
            al2 = alias_map_from_tables(parse_jsonl(f2["invoices.jsonl"]), parse_jsonl(f2["payments.jsonl"]))
            steps = 7 * i
            r1 = grade(oracle_findings(truth), truth, al1, 0, steps)
            r2 = grade(rename_findings(oracle_findings(truth), mp), t2, al2, 0, steps)
            found, bad = audit(parse_public(f2))
            want = sorted((t["type"], tuple(sorted(mp.get(x, x) for x in t["affected_ids"])), t["expected_amount_minor"],
                           t["currency"]) for t in truth["issues"])
            got = sorted((t["type"], tuple(t["affected_ids"]), t["amount"], t["currency"]) for t in found)
            transform_ok = (r1["reward_exact"] == r2["reward_exact"] and got == want and not bad
                            and validate_episode(f2, t2, p2)["ok"])
            out.append({"pair": i + 1, "ref": ref, "replay_policy": pol, "regenerate_bytes": regen_ok,
                        "replay_actions": replay_ok, "transform": transform_ok,
                        "ok": regen_ok and replay_ok and transform_ok})
    return out


def _stats(rows: list[dict]) -> dict:
    if not rows:
        return None
    n = len(rows)
    tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))
    walls = sorted(r["wall_ms"] for r in rows)
    return {"n": n, "P": sum(r["precision"] for r in rows) / n, "R": sum(r["recall"] for r in rows) / n,
            "F1": sum(r["f1"] for r in rows) / n,
            "microF1": (2 * tp / (2 * tp + fp + fn)) if tp + fp + fn else 1.0,
            "return": sum(r["reward"] for r in rows) / n, "steps": sum(r["steps"] for r in rows) / n,
            "p50_ms": walls[(n - 1) // 2], "max_ms": walls[-1], "tp": tp, "fp": fp, "fn": fn}


def main(argv=None) -> int:
    from .common.console import utf8_stdio
    utf8_stdio()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", required=True, help="файл сплита: {split, config_version, episodes:[{seed, overrides}]}")
    ap.add_argument("--secret-file", default=None)
    ap.add_argument("--policies", "--policy", nargs="+", default=None,
                    help="политики: встроенные имена и/или module.path:factory (по умолчанию — наши базовые)")
    ap.add_argument("--out", default="jury_out")
    ap.add_argument("--pairs", type=int, default=20)
    a = ap.parse_args(argv)
    out = Path(a.out)
    ep_root, gr_root = out / "episodes", out / "grader"
    if not a.secret_file:
        print(DEFAULT_SECRET_NOTE, file=sys.stderr)
    secret = read_secret(a.secret_file)
    spec, items = split_configs(a.split)
    split = spec["split"]
    policies = a.policies or DEFAULT_POLICIES
    if "oracle" not in policies:
        policies = policies + ["oracle"]  # максимум награды проверяется всегда
    rc = 0

    print(f"== 1. генерация {split}: {len(items)} эпизодов -> {ep_root}/ (открытое), {gr_root}/ (эталон)")
    gen_args = argparse.Namespace(split=a.split, episodes_root=str(ep_root), grader_root=str(gr_root),
                                  secret_file=a.secret_file)
    import contextlib
    import io
    with contextlib.redirect_stderr(io.StringIO()):
        cmd_generate(gen_args)
    refs = [e["ref"] for e in json.loads((ep_root / split / "index.json").read_text(encoding="utf-8"))]

    print("== 2. независимый валидатор")
    n_ok = 0
    for ref in refs:
        rep = validate_episode_dir(ep_root / ref, gr_root / ref)
        n_ok += rep["ok"]
        if not rep["ok"]:
            print(f"   FAIL {ref}: {rep['errors'][:3]}")
    print(f"   корректных эпизодов: {n_ok}/{len(refs)}")
    rc |= n_ok != len(refs)
    # снимок «до внесения ошибок» рядом с эталоном: жюри проверяет равенства «до» своими средствами
    for ref in refs:
        export_before(ep_root / ref, gr_root / ref, gr_root / ref / "before")
    print(f"   состояние «до ошибок» для независимой проверки: {gr_root}/<split>/<episode>/before/")

    print(f"== 3. политики (eval): {policies}")
    summary = run_policies(policies, [split], str(ep_root), str(gr_root), trajectories=str(out / "trajectories.jsonl"),
                           grader_log=str(out / "grader_log.jsonl"), mode="eval")
    graded = summary["grader"]
    if len(graded) != len(policies) * len(refs):
        print(f"   ОШИБКА: оценено {len(graded)} из {len(policies) * len(refs)}")
        rc = 1

    print(f"== 4. парные проверки ({a.pairs})")
    pairs = paired_checks(items, refs, ep_root, gr_root, secret, a.pairs)
    n_pairs = sum(p["ok"] for p in pairs)
    for p in pairs:
        if not p["ok"]:
            print(f"   FAIL pair {p['pair']}: {p}")
    print(f"   пройдено: {n_pairs}/{len(pairs)}")
    rc |= n_pairs != len(pairs)

    print("== 5. таблица")
    kind = {}
    for ref in refs:
        t = json.loads((gr_root / ref / "truth.json").read_text(encoding="utf-8"))
        kind[t["episode_id"]] = "clean" if not t["issues"] else ("interacting" if t["interactions"] else "regular")
    wall = {(e["policy"], e["episode_id"]): e for e in summary["episodes"]}
    rows = defaultdict(list)
    for gr in graded:
        e = wall[(gr["policy"], gr["episode_id"])]
        r = dict(gr, wall_ms=e["wall_ms"], steps=e["steps"])
        rows[(gr["policy"], "all")].append(r)
        rows[(gr["policy"], kind[gr["episode_id"]])].append(r)
    lines = ["| политика | эпизоды | n | P | R | F1 | micro F1 | возврат | шаги | время p50/max, мс |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for pol in policies:
        for sub in ("all", "clean", "interacting"):
            s = _stats(rows.get((pol, sub), []))
            if s:
                lines.append(f"| {pol} | {sub} | {s['n']} | {s['P']:.3f} | {s['R']:.3f} | {s['F1']:.3f} | "
                             f"{s['microF1']:.3f} | {s['return']:.3f} | {s['steps']:.1f} | {s['p50_ms']}/{s['max_ms']} |")
    table = "\n".join(lines)
    print(table)
    report = [f"# Режим жюри: {split}", "", f"* эпизодов: {len(refs)} (чистых: "
              f"{sum(v == 'clean' for v in kind.values())}, со взаимодействиями: "
              f"{sum(v == 'interacting' for v in kind.values())})",
              f"* валидатор: {n_ok}/{len(refs)}", f"* парные проверки: {n_pairs}/{len(pairs)}",
              f"* секрет: {'задан' if a.secret_file else 'НЕ задан (открытый по умолчанию)'}", "", table, ""]
    (out / "jury_report.md").write_text("\n".join(report), encoding="utf-8", newline="\n")
    (out / "jury_summary.json").write_text(json.dumps({"validator_ok": n_ok, "episodes": len(refs), "pairs": pairs,
                                                       "aggregate": summary["aggregate"]}, ensure_ascii=False,
                                                      indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"== итог: {'OK' if rc == 0 else 'FAILED'}; отчёт {out / 'jury_report.md'}")
    return 1 if rc else 0


if __name__ == "__main__":
    sys.exit(main())
