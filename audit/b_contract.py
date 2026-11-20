"""B. Контракт reset/step: крайние seed, повторный reset, step до reset, JSON всех obs/info,
размеры наблюдений, terminated/truncated, тип и диапазон награды.
usage: python3.11 -I b_contract.py <repo_root>
"""
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(ROOT))
from auditenv.env.core import AuditEnv, EpisodeOpenError  # noqa: E402
from auditenv.grader.service import GraderClient  # noqa: E402

g = GraderClient(str(ROOT / "episodes"), str(ROOT / "grader_data"), allow_oracle=True)

print("== B1 seed")
for s in [None, 0, 1, 2**63 - 1, 2**63, 2**64, 2**64 + 1, -1, -(2**70), True, False, 1.5, 1.0, "42", [1], 10**400]:
    env = AuditEnv(g, "train")
    try:
        obs, info = env.reset(seed=s)
        eid = obs["result"]["manifest"]["episode_id"]
        env2 = AuditEnv(g, "train")
        same = env2.reset(seed=s)[0]["result"]["manifest"]["episode_id"] == eid if s is not None else "n/a"
        print(f"  seed={str(s)[:24]:26s} -> OK {eid}  повтор тот же: {same}")
    except EpisodeOpenError as e:
        print(f"  seed={str(s)[:24]:26s} -> EpisodeOpenError: {str(e)[:90]}")
    except Exception as e:
        print(f"  seed={str(s)[:24]:26s} -> НЕОЖИДАННО {type(e).__name__}: {str(e)[:90]}")

print("== B2 step до reset / после завершения / повторный reset посреди эпизода")
env = AuditEnv(g, "train")
print("  step до reset:", env.step({"action": "list_records", "args": {"table": "ledger"}}))
obs, _ = env.reset(seed=5)
e1 = obs["result"]["manifest"]["episode_id"]
for _ in range(3):
    env.step({"action": "list_records", "args": {"table": "ledger"}})
obs, _ = env.reset(seed=6)
o = env.step({"action": "list_records", "args": {"table": "fx", "limit": 1}})[0]
print("  повторный reset: steps_remaining после 1 шага =", o["steps_remaining"],
      "; новый эпизод:", obs["result"]["manifest"]["episode_id"] != e1)
r = env.step({"action": "submit", "args": {"findings": {"episode_id": obs["result"]["manifest"]["episode_id"], "findings": []}}})
r2 = env.step({"action": "submit", "args": {"findings": {"episode_id": obs["result"]["manifest"]["episode_id"], "findings": []}}})
print("  submit:", r[1:4], " повторный submit:", r2[0]["error"]["code"], r2[1:4])
# reset ПОСЛЕ submit того же seed: прежний grade не переигрывается
obs, _ = env.reset(seed=6)
print("  reset после submit, тот же seed -> steps_remaining", obs["steps_remaining"])

print("== B3 JSON всех obs/info в trajectories.jsonl; terminated^truncated; награда")
n = bad_json = both = bad_rew = nonfloat = 0
max_obs = (0, None)
rew_vals = set()
with open(ROOT / "trajectories.jsonl", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        n += 1
        s = json.dumps(rec["observation"], allow_nan=False, ensure_ascii=False)
        if len(s.encode()) > max_obs[0]:
            max_obs = (len(s.encode()), (rec["policy"], rec["episode_id"], rec["step"], (rec["action"] or {}).get("action")))
        if rec["terminated"] and rec["truncated"]:
            both += 1
        rw = rec["reward"]
        if not isinstance(rw, (int, float)) or isinstance(rw, bool) or not -1 <= rw <= 1 or math.isnan(rw):
            bad_rew += 1
        rew_vals.add(type(rw).__name__)
print(f"  строк={n} obs не-JSON={bad_json} terminated&truncated={both} награда вне [-1,1]={bad_rew} типы={rew_vals}")
print(f"  макс. наблюдение в траекториях: {max_obs[0]} байт @ {max_obs[1]}")

print("== B4 живые obs/info: round-trip JSON, типы, границы размеров (худший размер компании)")
from auditenv.cli import _variants  # noqa: E402
base = json.loads((ROOT / "configs/default.json").read_text())
from auditenv.generator.config import deep_merge  # noqa: E402
worst = deep_merge(base, {"size": {"n_invoices": [75, 75], "n_expenses": [700, 700], "n_customers": [14, 14],
                                   "n_prior_invoices": [6, 6]},
                          "distractors": {"prepayments": [3, 3], "recurring_groups": [2, 2]},
                          "currencies": {"foreign_customer_share_pct": 90}})
sizes = {}
rt_bad = 0
for seed in range(5000, 5006):
    env = AuditEnv(g, "train")
    obs, info = env.reset(seed=seed, config=worst)
    for x in (obs, info):
        if json.loads(json.dumps(x, allow_nan=False)) != x:
            rt_bad += 1
    sizes.setdefault("reset", []).append(len(json.dumps(obs, ensure_ascii=False).encode()))
    for t in ("ledger", "invoices", "payments", "fx"):
        cur = None
        while True:
            o, rew, te, tr, inf = env.step({"action": "list_records", "args": {"table": t, "cursor": cur}})
            if json.loads(json.dumps([o, rew, te, tr, inf], allow_nan=False)) != [o, rew, te, tr, inf]:
                rt_bad += 1
            assert type(rew) is float
            sizes.setdefault(f"page {t}", []).append(len(json.dumps(o, ensure_ascii=False).encode()))
            cur = o["result"]["next_cursor"] if o["result"] else None
            if cur is None or tr:
                break
sizes_s = {k: (min(v), max(v)) for k, v in sizes.items()}
print("  round-trip расхождений:", rt_bad)
for k, (a, b) in sizes_s.items():
    print(f"  {k:16s} min={a:7d} max={b:7d} байт")

# документы: верхняя граница по всем 300 seed validate-seeds невозможна дёшево -> берём худший размер
from auditenv.generator.episode import generate_episode  # noqa: E402
mx_doc = mx_inv = 0
for seed in range(5000, 5020):
    ep = generate_episode(seed, worst)
    for line in ep.public["documents.jsonl"].decode().splitlines():
        mx_doc = max(mx_doc, len(line.encode()) + 60)
    for line in ep.public["invoices.jsonl"].decode().splitlines():
        mx_inv = max(mx_inv, len(line.encode()))
print(f"  get_document: макс. ≈{mx_doc} байт; одна запись invoices: макс. {mx_inv} байт (20 seed худшего размера)")
g.close()
