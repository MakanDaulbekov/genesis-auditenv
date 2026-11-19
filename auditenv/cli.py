"""CLI: generate / validate / run / report / check / demo.

python -m auditenv.cli generate --split configs/splits/train.json
python -m auditenv.cli run --split train --policy simple_reconciler
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


from .generator.config import deep_merge  # noqa: E402  (реэкспорт для тестов/жюри)


DEFAULT_SECRET_NOTE = ("ПРЕДУПРЕЖДЕНИЕ: --secret-file не задан — используется открытый детерминированный секрет "
                       "по умолчанию (пустая строка). Повтор seed точен, но маленькие закрытые seed можно перебрать "
                       "открытым генератором. Для закрытого прогона передайте --secret-file.")


def read_secret(path: str | None) -> str:
    return Path(path).read_text(encoding="utf-8").strip() if path else ""


def split_configs(split_file: str) -> tuple[dict, list[tuple[int, dict]]]:
    """Сплит: {"split", "config_version", "generator_config"?, "episodes": [{"seed", "overrides"? | "config"?}]}.
    Без generator_config берётся configs/default.json; overrides/config мержатся поверх него."""
    spec = json.loads(Path(split_file).read_text(encoding="utf-8"))
    gc = spec.get("generator_config", "configs/default.json")
    gp = Path(gc) if Path(gc).is_absolute() or Path(gc).exists() else ROOT / gc
    base = json.loads(gp.read_text(encoding="utf-8"))
    out = []
    for e in spec["episodes"]:
        cfg = deep_merge(base, e.get("config", {}))
        cfg = deep_merge(cfg, e.get("overrides", {}))
        cfg["config_version"] = spec.get("config_version", cfg["config_version"])
        out.append((e["seed"], cfg))
    return spec, out


def cmd_generate(args) -> int:
    from .generator.episode import generate_episode
    spec, items = split_configs(args.split)
    name = spec["split"]
    pub_root = Path(args.episodes_root) / name
    prv_root = Path(args.grader_root) / name
    for d in (pub_root, prv_root):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    index = []
    from .generator.config import validate_config
    if not args.secret_file and name not in ("train", "dev"):
        print(DEFAULT_SECRET_NOTE, file=sys.stderr)
    secret = read_secret(args.secret_file)
    for seed, cfg in items:
        validate_config(cfg)
        ep = generate_episode(seed, cfg, secret)
        (pub_root / ep.episode_id).mkdir()
        (prv_root / ep.episode_id).mkdir()
        for n, b in sorted(ep.public.items()):
            (pub_root / ep.episode_id / n).write_bytes(b)
        for n, b in sorted(ep.private.items()):
            (prv_root / ep.episode_id / n).write_bytes(b)
        index.append({"ref": f"{name}/{ep.episode_id}", "config_version": cfg["config_version"]})
        print(f"[generate] {name}/{ep.episode_id}: issues={len(ep.truth['issues'])} "
              f"rows={json.loads(ep.public['manifest.json'])['counts']['ledger_rows']}")
    (pub_root / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return 0


def episode_refs(split: str, episodes_root: str) -> list[str]:
    idx = Path(episodes_root) / split / "index.json"
    return [e["ref"] for e in json.loads(idx.read_text(encoding="utf-8"))]


def cmd_validate(args) -> int:
    from .generator.validate import validate_episode_dir
    bad = 0
    for split in args.splits:
        for ref in episode_refs(split, args.episodes_root):
            rep = validate_episode_dir(Path(args.episodes_root) / ref, Path(args.grader_root) / ref)
            status = "OK" if rep["ok"] else "FAIL"
            print(f"[validate] {ref}: {status} issues={rep['n_issues']} checks={rep['n_checks']}")
            if not rep["ok"]:
                bad += 1
                for e in rep["errors"][:10]:
                    print("    -", e)
    return 1 if bad else 0


def cmd_validate_seeds(args) -> int:
    """Сотни случайных seed и сочетаний параметров через независимый валидатор."""
    from .common.drbg import DRBG
    from .generator.episode import generate_episode
    from .generator.validate import validate_episode
    base = json.loads((ROOT / "configs" / "default.json").read_text(encoding="utf-8"))
    rng = DRBG(b"validate-seeds")
    variants = _variants()
    bad = 0
    stats = {"episodes": 0, "issues": 0, "clean": 0}
    for i in range(args.n):
        seed = rng.randint(0, 2**63 - 1)
        cfg = deep_merge(base, variants[i % len(variants)])
        ep = generate_episode(seed, cfg)
        rep = validate_episode(ep.public, ep.truth, ep.provenance)
        stats["episodes"] += 1
        stats["issues"] += rep["n_issues"]
        stats["clean"] += rep["n_issues"] == 0
        if not rep["ok"]:
            bad += 1
            print(f"[validate-seeds] FAIL seed={seed} variant={i % len(variants)}: {rep['errors'][:3]}")
    print(f"[validate-seeds] {stats['episodes'] - bad}/{stats['episodes']} OK, issues={stats['issues']}, "
          f"clean={stats['clean']}")
    return 1 if bad else 0


def _variants():
    return [{}, {"clean": True}, {"interactions": {"partial_fx_boundary": 1, "duplicate_cutoff": 1}},
            {"errors": {"duplicate_invoice": 3, "missing_or_partial_allocation": 3,
                        "fx_wrong_date_or_rate": 3, "period_cutoff_error": 3},
             "interactions": {"partial_fx_boundary": 0, "duplicate_cutoff": 0}},
            {"size": {"n_invoices": [65, 75], "n_expenses": [500, 700]}},
            {"currencies": {"base": ["JPY", "GBP"], "foreign_customer_share_pct": 80}}]


def cmd_solvability(args) -> int:
    """Доказательство решаемости: competent_reconciler на N случайных seed × 6 сочетаний + худший размер."""
    from .common.drbg import DRBG
    from .env.core import AuditEnv
    from .grader.service import GraderClient
    from .policies.api import load_policy
    base = json.loads((ROOT / "configs" / "default.json").read_text(encoding="utf-8"))
    worst = deep_merge(base, {"size": {"n_invoices": [75, 75], "n_expenses": [700, 700], "n_customers": [14, 14],
                                       "n_prior_invoices": [6, 6]},
                              "distractors": {"prepayments": [3, 3], "recurring_groups": [2, 2]},
                              "currencies": {"foreign_customer_share_pct": 90}})
    rng = DRBG(b"solvability-seeds")
    names = ["default", "clean", "interactions", "12_errors", "large", "base_JPY_GBP"]
    jobs = [(f"{names[i % 6]}", rng.randint(0, 2**63 - 1), deep_merge(base, _variants()[i % 6])) for i in range(args.n)]
    jobs += [("worst_size", 5_000 + i, worst) for i in range(args.worst)]
    recs, fails = [], []
    with GraderClient(allow_oracle=False) as g:
        for name, seed, cfg in jobs:
            env = AuditEnv(g, "train")
            obs, _ = env.reset(seed=seed, config=cfg)
            pol = load_policy("competent_reconciler")
            pol.reset(obs)
            a, n = pol.act(obs), 0
            while True:
                o, r, te, tr, info = env.step(a)
                n += 1
                if te or tr:
                    break
                a = pol.act(o)
            m = info["metrics"]
            c = obs["result"]["manifest"]["counts"]
            rec = {"variant": name, "seed": seed, "steps": n, "truncated": tr, "reward": r, "f1": m["f1"],
                   "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "ledger_rows": c["ledger_rows"],
                   "documents": c["documents"]}
            recs.append(rec)
            if m["fp"] or m["fn"] or tr:
                fails.append(rec)
    steps = [r["steps"] for r in recs]
    hist = {}
    for s_ in steps:
        b = f"{s_ // 10 * 10:02d}-{s_ // 10 * 10 + 9:02d}"
        hist[b] = hist.get(b, 0) + 1
    out = {"episodes": len(recs), "perfect": sum(1 for r in recs if r["fp"] == 0 and r["fn"] == 0 and not r["truncated"]),
           "max_steps": max(steps), "mean_steps": round(sum(steps) / len(steps), 2), "steps_histogram": dict(sorted(hist.items())),
           "max_ledger_rows": max(r["ledger_rows"] for r in recs), "max_documents": max(r["documents"] for r in recs),
           "tp": sum(r["tp"] for r in recs), "fp": sum(r["fp"] for r in recs), "fn": sum(r["fn"] for r in recs),
           "by_variant": {v: {"n": sum(1 for r in recs if r["variant"] == v),
                              "perfect": sum(1 for r in recs if r["variant"] == v and not (r["fp"] or r["fn"])),
                              "max_steps": max(r["steps"] for r in recs if r["variant"] == v)}
                          for v in dict.fromkeys(r["variant"] for r in recs)},
           "failures": fails}
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[solvability] competent_reconciler: {out['perfect']}/{out['episodes']} эпизодов с F1=1 и без FP; "
          f"TP={out['tp']} FP={out['fp']} FN={out['fn']}; шаги max={out['max_steps']} mean={out['mean_steps']}")
    return 0 if out["max_steps"] < 100 else 1


def _peak_rss_mb() -> dict:
    import resource
    # Linux: ru_maxrss в КБ. RUSAGE_CHILDREN — пик среди завершённых дочерних (процесс оценщика).
    return {"peak_rss_mb_env_policy": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            "peak_rss_mb_grader": round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024, 1)}


def cmd_run(args) -> int:
    """Прогон политик. По умолчанию каждая политика — в своём процессе (--isolate): так пик RSS и
    время честно меряются по каждой политике отдельно; траектории склеиваются в порядке политик."""
    import subprocess
    import tempfile
    from .runner import run_policies
    if args.isolate and len(args.policies) > 1:
        merged = {"aggregate": {}, "episodes": [], "grader": []}
        with tempfile.TemporaryDirectory() as tmp, open(args.trajectories, "w", encoding="utf-8") as traj, \
                open(args.grader_log, "w", encoding="utf-8") if args.grader_log else open(os.devnull, "w") as glog:
            for i, pol in enumerate(args.policies):
                t, sm, gl = (Path(tmp) / f"{i}.{x}" for x in ("traj.jsonl", "summary.json", "glog.jsonl"))
                cmd = [sys.executable, "-m", "auditenv.cli", "run", "--no-isolate", "--policies", pol,
                       "--splits", *args.splits, "--mode", args.mode, "--episodes-root", args.episodes_root,
                       "--grader-root", args.grader_root, "--trajectories", str(t), "--grader-log", str(gl),
                       "--summary", str(sm)] + (["--trusted"] if args.trusted else []) + \
                      (["--truth-dir", args.truth_dir] if args.truth_dir else [])
                env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT), os.getcwd()]))
                subprocess.run(cmd, check=True, env=env, stdout=subprocess.DEVNULL)
                traj.write(t.read_text(encoding="utf-8"))
                glog.write(gl.read_text(encoding="utf-8"))
                one = json.loads(sm.read_text(encoding="utf-8"))
                merged["aggregate"].update(one["aggregate"])
                merged["episodes"] += one["episodes"]
                merged["grader"] += one["grader"]
        summary = merged
    else:
        summary = run_policies(policies=args.policies, splits=args.splits, episodes_root=args.episodes_root,
                               grader_root=args.grader_root, trajectories=args.trajectories,
                               grader_log=args.grader_log, mode=args.mode, trusted=args.trusted,
                               truth_dir=args.truth_dir)
        if len(args.policies) == 1:
            summary["aggregate"][args.policies[0]].update(_peak_rss_mb())
    print(json.dumps(summary["aggregate"], indent=2, ensure_ascii=False))
    if args.summary:
        Path(args.summary).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


CHECK_STEPS = [
    ("generate train", ["-m", "auditenv.cli", "generate", "--split", "configs/splits/train.json"]),
    ("generate dev", ["-m", "auditenv.cli", "generate", "--split", "configs/splits/dev.json"]),
    ("validate episodes", ["-m", "auditenv.cli", "validate", "--splits", "train", "dev"]),
    ("validate 300 random seeds", ["-m", "auditenv.cli", "validate-seeds", "--n", "300"]),
    ("pytest", ["-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"]),
    ("run baseline policies (+competent)", ["-m", "auditenv.cli", "run"]),
    ("solvability: competent_reconciler, 300 seeds + worst size", ["-m", "auditenv.cli", "solvability"]),
    ("trajectory leak scan", ["-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_policies.py", "-k",
                              "committed_trajectories"]),
    ("determinism (2 processes, PYTHONHASHSEED)", ["-m", "auditenv.checks", "determinism"]),
    ("self-check 4x20 (не оценка жюри)", ["tests/jury/run_jury.py"]),
    ("report", ["-m", "auditenv.report"]),
]


def cmd_check(args) -> int:
    """Единая проверка (то же, что make check): ненулевой выход при любом провале."""
    import subprocess
    import time
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1")
    rows = []
    ok_all = True
    for name, cmd in CHECK_STEPS:
        t0 = time.monotonic()
        r = subprocess.run([sys.executable] + cmd, cwd=ROOT, env=env, capture_output=True, text=True)
        dt = time.monotonic() - t0
        ok = r.returncode == 0
        ok_all &= ok
        tail = (r.stdout.strip().splitlines() or [""])[-1][:110]
        rows.append((name, ok, dt, tail))
        print(f"[{'OK' if ok else 'FAIL'}] {name:44s} {dt:6.1f}s  {tail}", flush=True)
        if "self-check" in name:
            print("\n".join("        " + l for l in r.stdout.strip().splitlines()), flush=True)
        if not ok:
            print(r.stdout[-4000:])
            print(r.stderr[-4000:])
            if not args.keep_going:
                break
    print("== check: " + ("OK ==" if ok_all else "FAILED =="))
    return 0 if ok_all else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="auditenv")
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--split", required=True)
    g.add_argument("--episodes-root", default="episodes")
    g.add_argument("--grader-root", default="grader_data")
    g.add_argument("--secret-file", default=None, help="секрет оценщика (закрытый прогон)")
    g.set_defaults(fn=cmd_generate)
    v = sub.add_parser("validate")
    v.add_argument("--splits", nargs="+", default=["train", "dev"])
    v.add_argument("--episodes-root", default="episodes")
    v.add_argument("--grader-root", default="grader_data")
    v.set_defaults(fn=cmd_validate)
    vs = sub.add_parser("validate-seeds")
    vs.add_argument("--n", type=int, default=300)
    vs.set_defaults(fn=cmd_validate_seeds)
    so = sub.add_parser("solvability")
    so.add_argument("--n", type=int, default=300)
    so.add_argument("--worst", type=int, default=10)
    so.add_argument("--out", default="grader_data/solvability.json")
    so.set_defaults(fn=cmd_solvability)
    r = sub.add_parser("run")
    r.add_argument("--policies", "--policy", nargs="+",
                   default=["random_explorer", "simple_reconciler", "competent_reconciler", "flag_all", "oracle"],
                   help="встроенные имена и/или module.path:factory")
    r.add_argument("--trusted", action="store_true", help="доверенный режим: разрешает oracle_from_truth")
    r.add_argument("--no-isolate", dest="isolate", action="store_false",
                   help="все политики в одном процессе (без замера пика RSS по политикам)")
    r.add_argument("--truth-dir", default=None, help="каталог эталона для oracle_from_truth (по умолчанию grader-root)")
    r.add_argument("--splits", nargs="+", default=["train", "dev"])
    r.add_argument("--mode", default="train", choices=["train", "eval"])
    r.add_argument("--episodes-root", default="episodes")
    r.add_argument("--grader-root", default="grader_data")
    r.add_argument("--trajectories", default="trajectories.jsonl")
    r.add_argument("--grader-log", default="grader_data/grader_log.jsonl")
    r.add_argument("--summary", default="grader_data/run_summary.json")
    r.set_defaults(fn=cmd_run)
    c = sub.add_parser("check")
    c.add_argument("--keep-going", action="store_true")
    c.set_defaults(fn=cmd_check)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
