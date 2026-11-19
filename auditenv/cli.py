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


def split_configs(split_file: str) -> tuple[dict, list[tuple[int, dict]]]:
    spec = json.loads(Path(split_file).read_text(encoding="utf-8"))
    base = json.loads((ROOT / spec["generator_config"]).read_text(encoding="utf-8"))
    out = []
    for e in spec["episodes"]:
        cfg = deep_merge(base, e.get("overrides", {}))
        cfg["config_version"] = spec["config_version"]
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
    for seed, cfg in items:
        validate_config(cfg)
        ep = generate_episode(seed, cfg)
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
    variants = [{}, {"clean": True}, {"interactions": {"partial_fx_boundary": 1, "duplicate_cutoff": 1}},
                {"errors": {"duplicate_invoice": 3, "missing_or_partial_allocation": 3,
                            "fx_wrong_date_or_rate": 3, "period_cutoff_error": 3},
                 "interactions": {"partial_fx_boundary": 0, "duplicate_cutoff": 0}},
                {"size": {"n_invoices": [80, 120], "n_expenses": [400, 700]}},
                {"currencies": {"base": ["JPY", "GBP"], "foreign_customer_share_pct": 80}}]
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


def cmd_run(args) -> int:
    from .runner import run_policies
    summary = run_policies(policies=args.policies, splits=args.splits, episodes_root=args.episodes_root,
                           grader_root=args.grader_root, trajectories=args.trajectories,
                           grader_log=args.grader_log, mode=args.mode)
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
    ("run 4 baseline policies", ["-m", "auditenv.cli", "run"]),
    ("trajectory leak scan", ["-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_policies.py", "-k",
                              "committed_trajectories"]),
    ("determinism (2 processes, PYTHONHASHSEED)", ["-m", "auditenv.checks", "determinism"]),
    ("shadow jury 4x20", ["tests/jury/run_jury.py"]),
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
        if "jury" in name:
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
    g.set_defaults(fn=cmd_generate)
    v = sub.add_parser("validate")
    v.add_argument("--splits", nargs="+", default=["train", "dev"])
    v.add_argument("--episodes-root", default="episodes")
    v.add_argument("--grader-root", default="grader_data")
    v.set_defaults(fn=cmd_validate)
    vs = sub.add_parser("validate-seeds")
    vs.add_argument("--n", type=int, default=300)
    vs.set_defaults(fn=cmd_validate_seeds)
    r = sub.add_parser("run")
    r.add_argument("--policies", nargs="+", default=["random_explorer", "simple_reconciler", "flag_all", "oracle"])
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
