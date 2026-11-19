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


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


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
    for seed, cfg in items:
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


def cmd_run(args) -> int:
    from .runner import run_policies
    summary = run_policies(policies=args.policies, splits=args.splits, episodes_root=args.episodes_root,
                           grader_root=args.grader_root, trajectories=args.trajectories,
                           grader_log=args.grader_log, mode=args.mode)
    print(json.dumps(summary["aggregate"], indent=2, ensure_ascii=False))
    if args.summary:
        Path(args.summary).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


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
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
