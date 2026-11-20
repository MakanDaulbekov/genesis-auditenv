"""B. Глубоко вложенный submit: RecursionError из env.step и неоценённые эпизоды.
usage: python3.11 -I b_deep_sweep.py <repo_root>
"""
import sys, json
from pathlib import Path
ROOT = Path(sys.argv[1]).resolve(); sys.path.insert(0, str(ROOT))
from auditenv.env.core import AuditEnv
from auditenv.grader.service import GraderClient
def nest(d):
    x = []
    for _ in range(d): x = [x]
    return x
exc, ungraded = [], []
terminal = 0
with GraderClient(str(ROOT/"episodes"), str(ROOT/"grader_data"), log_path=str(Path(__import__("tempfile").gettempdir()) / "sweep.jsonl")) as g:
    for kind in ("dict", "str"):
        for d in range(1, 1201):
            env = AuditEnv(g, "train"); obs,_ = env.reset(config={"episode_ref": "dev/ep_053daf06ab62f121"})
            eid = obs["result"]["manifest"]["episode_id"]
            a = {"action": "submit", "args": {"findings": {"episode_id": eid, "findings": nest(d)}}}
            try:
                s = None if kind == "dict" else ('{"action":"submit","args":{"findings":{"episode_id":"' + eid + '","findings":' + "[" * (d + 1) + "]" * (d + 1) + "}}}")
                o, r, te, tr, info = env.step(a if kind == "dict" else s)
                terminal += te
                if te and r != -1.0: ungraded.append((kind, d, r))
            except RecursionError:
                exc.append((kind, d))
    g.call("finalize"); n = len(g.call("results")["results"])
print("RecursionError из env.step:", exc)
print("награда != -1 (эпизод не оценён):", ungraded[:20], "... всего", len(ungraded))
print("терминальных submit:", terminal, "| записей graded/invalid_findings у оценщика:", n)
