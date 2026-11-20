import json, sys, time
from pathlib import Path
ROOT = Path(sys.argv[1]).resolve(); sys.path.insert(0, str(ROOT))
from auditenv.generator.config import SIZE_KEYS, DISTRACTOR_RANGES, deep_merge, resolve_config
from auditenv.generator.episode import generate_episode
from auditenv.generator.world import GenerationError
from auditenv.env.core import AuditEnv
from auditenv.grader.service import GraderClient
from auditenv.policies.api import load_policy
size = {k: [hi, hi] for k, (lo, hi) in SIZE_KEYS.items()}
dis = {k: [hi, hi] for k, (lo, hi) in DISTRACTOR_RANGES.items()}
over = {"size": size, "distractors": dis, "currencies": {"foreign_customer_share_pct": 100, "base": ["JPY"]},
        "errors": {"duplicate_invoice": [3, 3], "missing_or_partial_allocation": [1, 1], "fx_wrong_date_or_rate": [1, 1], "period_cutoff_error": [1, 1]},
        "interactions": {"partial_fx_boundary": [3, 3], "duplicate_cutoff": [3, 3]}}
print("max config:", json.dumps(over)[:400])
cfg = resolve_config(over)
mx = (0, 0); worst_steps = 0; fails = 0; hist = {}
with GraderClient() as g:
    for seed in range(int(sys.argv[2]) if len(sys.argv) > 2 else 40):
        try:
            ep = generate_episode(seed, cfg)
        except GenerationError as e:
            fails += 1; continue
        c = json.loads(ep.public["manifest.json"])["counts"]
        mx = (max(mx[0], c["ledger_rows"]), max(mx[1], c["documents"]))
        env = AuditEnv(g, "train"); obs, _ = env.reset(seed=seed, config=over)
        pol = load_policy("competent_reconciler"); pol.reset(obs); a = pol.act(obs); n = 0
        while True:
            o, r, te, tr, info = env.step(a); n += 1
            if te or tr: break
            a = pol.act(o)
        worst_steps = max(worst_steps, n); hist[n // 5 * 5] = hist.get(n // 5 * 5, 0) + 1
        if info["metrics"]["fp"] or info["metrics"]["fn"] or tr:
            print("  seed", seed, "competent:", info["metrics"], "steps", n, "truncated", tr)
print(f"{seed+1} seed максимального config: max ledger_rows={mx[0]}, max documents={mx[1]}, competent max steps={worst_steps}, generation_failed={fails}")
print("гистограмма шагов (по 5):", dict(sorted(hist.items())))
