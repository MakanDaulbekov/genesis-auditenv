"""Прогон базовых политик на сплитах: траектории, метрики, диагностика по типам."""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

from .env.core import AuditEnv
from .grader.service import GraderClient
from .policies.api import load_policy

TYPES = ["duplicate_invoice", "missing_or_partial_allocation", "fx_wrong_date_or_rate", "period_cutoff_error"]
ORACLE_ACTION = {"action": "submit", "args": {"findings": "<built inside the grader process; not exposed>"}}


def run_episode(env: AuditEnv, policy: str, config: dict, traj_out=None, meta: dict | None = None,
                policy_obj=None, trusted: bool = False, truth_dir: str = "grader_data") -> dict:
    """policy: встроенное имя, "module.path:factory" или "oracle" (ответ строит оценщик)."""
    t0 = time.perf_counter()
    obs, info = env.reset(config=config)
    episode_id = obs["result"]["manifest"]["episode_id"]
    rec = dict(meta or {}, policy=policy, episode_id=episode_id)

    def log(step, action, o, r, term, trunc):
        if traj_out is not None:
            traj_out.write(json.dumps(dict(rec, step=step, action=action, observation=o, reward=r,
                                           terminated=term, truncated=trunc),
                                      sort_keys=True, ensure_ascii=False) + "\n")

    log(0, None, obs, 0.0, False, False)
    total, steps, term, trunc, last_info = 0.0, 0, False, False, {}
    if policy == "oracle":
        o, r, term, trunc, last_info = env.oracle_step()
        steps = 1
        log(1, ORACLE_ACTION, o, r, term, trunc)
        total = r
    else:
        pol = policy_obj if policy_obj is not None else load_policy(policy, trusted=trusted, truth_dir=truth_dir)
        pol.reset(obs)
        action = pol.act(obs)
        while True:
            o, r, term, trunc, last_info = env.step(action)
            steps += 1
            total += r
            log(steps, action, o, r, term, trunc)
            if term or trunc:
                break
            action = pol.act(o)
    return {"episode_id": episode_id, "policy": policy, "return": total, "steps": steps,
            "terminated": term, "truncated": trunc, "wall_ms": round((time.perf_counter() - t0) * 1000, 2),
            "metrics": last_info.get("metrics")}


def aggregate(results: list[dict], grader_results: list[dict], clean_ids: set) -> dict:
    by_pol = defaultdict(list)
    for r in results:
        by_pol[r["policy"]].append(r)
    g_by = {(g["policy"] or "", g["episode_id"]): g for g in grader_results}
    out = {}
    for pol, rs in sorted(by_pol.items()):
        gs = [g_by.get((pol, r["episode_id"])) for r in rs]
        gs = [g for g in gs if g]
        tp = sum(g["tp"] for g in gs)
        fp = sum(g["fp"] for g in gs)
        fn = sum(g["fn"] for g in gs)
        bt = {t: {k: sum(g["by_type"][t][k] for g in gs) for k in ("tp", "fp", "fn")} for t in TYPES}
        clean = [r for r in rs if r["episode_id"] in clean_ids]
        n = len(rs)
        out[pol] = {
            "episodes": n,
            "mean_return": round(sum(r["return"] for r in rs) / n, 4),
            "mean_precision": round(sum(g["precision"] for g in gs) / max(1, len(gs)), 4),
            "mean_recall": round(sum(g["recall"] for g in gs) / max(1, len(gs)), 4),
            "mean_f1": round(sum(g["f1"] for g in gs) / max(1, len(gs)), 4),
            "micro": {"tp": tp, "fp": fp, "fn": fn,
                      "precision": round(tp / (tp + fp), 4) if tp + fp else 0.0,
                      "recall": round(tp / (tp + fn), 4) if tp + fn else 0.0,
                      "f1": round(2 * tp / (2 * tp + fp + fn), 4) if tp + fp + fn else 1.0},
            "mean_steps": round(sum(r["steps"] for r in rs) / n, 2),
            "mean_wall_ms": round(sum(r["wall_ms"] for r in rs) / n, 2),
            "max_wall_ms": max(r["wall_ms"] for r in rs),
            "invalid_submissions": sum(1 for g in gs if not g["valid"]),
            "truncated": sum(1 for r in rs if r["truncated"]),
            "clean_episodes": len(clean),
            "clean_mean_return": round(sum(r["return"] for r in clean) / len(clean), 4) if clean else None,
            "by_type": bt,
        }
    return out


def run_policies(policies, splits, episodes_root="episodes", grader_root="grader_data",
                 trajectories="trajectories.jsonl", grader_log=None, mode="train", trusted: bool = False,
                 truth_dir: str | None = None) -> dict:
    refs = []
    for s in splits:
        idx = json.loads((Path(episodes_root) / s / "index.json").read_text(encoding="utf-8"))
        refs += [(s, e["ref"]) for e in idx]
    if grader_log:
        Path(grader_log).parent.mkdir(parents=True, exist_ok=True)
        Path(grader_log).write_text("", encoding="utf-8")
    results = []
    grader_results = []
    with open(trajectories, "w", encoding="utf-8") as traj:
        for pol in policies:
            # у каждой политики свой процесс оценщика; op "oracle" разрешён только процессу oracle,
            # поэтому никакая политика не может вызвать oracle через клиент среды
            with GraderClient(episodes_root, grader_root, log_path=grader_log, allow_oracle=(pol == "oracle")) as g:
                for split, ref in refs:
                    env = AuditEnv(g, mode=mode, policy_name=pol)
                    results.append(dict(run_episode(env, pol, {"episode_ref": ref}, traj, {"split": split},
                                                    trusted=trusted, truth_dir=truth_dir or grader_root),
                                        split=split))
                g.call("finalize")
                grader_results += g.call("results")["results"]
    clean_ids = {r["episode_id"] for r in grader_results if r["n_truth"] == 0}
    return {"aggregate": aggregate(results, grader_results, clean_ids), "episodes": results,
            "grader": grader_results}
