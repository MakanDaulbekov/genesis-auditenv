"""C. Лимиты: худший размер компании + 100 самых тяжёлых действий; память; тайм-аут in-process.
usage: python3.11 -I c_limits.py <repo_root>
"""
import json
import resource
import sys
import time
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(ROOT))
from auditenv.env.core import AuditEnv, EpisodeOpenError  # noqa: E402
from auditenv.generator.config import deep_merge  # noqa: E402
from auditenv.grader.service import GraderClient  # noqa: E402
from auditenv.runner import run_episode  # noqa: E402

base = json.loads((ROOT / "configs/default.json").read_text())
print("== C0 границы размера в config")
for sz in ({"n_invoices": [76, 76]}, {"n_expenses": [701, 701]}, {"n_customers": [100, 100]},
           {"n_prior_invoices": [50, 50]}):
    with GraderClient() as g:
        try:
            AuditEnv(g).reset(seed=1, config={"size": sz})
            print("  принят:", sz)
        except EpisodeOpenError as e:
            print("  отклонён:", sz, "->", str(e)[:110])

worst = deep_merge(base, {"size": {"n_invoices": [75, 75], "n_expenses": [700, 700], "n_customers": [14, 14],
                                   "n_prior_invoices": [6, 6]},
                          "distractors": {"prepayments": [3, 3], "recurring_groups": [2, 2]},
                          "errors": {"duplicate_invoice": 3, "missing_or_partial_allocation": 3,
                                     "fx_wrong_date_or_rate": 3, "period_cutoff_error": 3},
                          "interactions": {"partial_fx_boundary": 0, "duplicate_cutoff": 0},
                          "currencies": {"foreign_customer_share_pct": 90}})


class Heavy:
    """99 самых тяжёлых инструментов + submit максимального размера (30×10 ID, 1000 символов)."""

    def reset(self, obs):
        self.eid = obs["result"]["manifest"]["episode_id"]
        self.n = 0
        self.cur = None
        self.docs = []
        self.ids = []

    def act(self, obs):
        self.n += 1
        res = obs.get("result") or {}
        for r in res.get("records", []) or []:
            if "doc_id" in r:
                self.docs.append(r["doc_id"])
            self.ids.append(r.get("invoice_id") or r.get("row_id") or r.get("payment_id") or "R-X")
        if res.get("table") == "invoices":
            self.cur = res.get("next_cursor")
        if self.n >= 100:
            f = [{"type": t, "affected_ids": self.ids[i:i + 2] or ["INV-X"], "claimed_amount_minor": 10**30,
                  "currency": "USD", "evidence_ids": (self.ids[i:i + 10] + ["R-X"] * 10)[:10], "explanation": "x" * 1000}
                 for i, t in zip(range(30), ["duplicate_invoice", "missing_or_partial_allocation",
                                             "fx_wrong_date_or_rate", "period_cutoff_error"] * 8)]
            return {"action": "submit", "args": {"findings": {"episode_id": self.eid, "findings": f}}}
        k = self.n % 3
        if k == 0:
            return {"action": "list_records", "args": {"table": "invoices", "cursor": self.cur, "limit": 50}}
        if k == 1 and self.docs:
            return {"action": "get_document", "args": {"doc_id": self.docs[self.n % len(self.docs)]}}
        return {"action": "calculator", "args": {"expression": "+".join(["(123456789.123456789*987654321.5)"] * 9)[:256]}}


print("== C1 худший размер + Heavy (100 шагов), 5 seed")
with GraderClient(allow_oracle=False) as g:
    walls, resets = [], []
    for seed in range(7000, 7005):
        env = AuditEnv(g, "train")
        t0 = time.perf_counter()
        obs, _ = env.reset(seed=seed, config=worst)
        t1 = time.perf_counter()
        pol = Heavy()
        pol.reset(obs)
        a = pol.act(obs)
        n = 0
        while True:
            o, r, te, tr, info = env.step(a)
            n += 1
            if te or tr:
                break
            a = pol.act(o)
        t2 = time.perf_counter()
        c = obs["result"]["manifest"]["counts"]
        walls.append(t2 - t1)
        resets.append(t1 - t0)
        print(f"  seed={seed} rows={c['ledger_rows']} docs={c['documents']} reset={t1 - t0:.2f}s "
              f"episode={t2 - t1:.2f}s steps={n} terminated={te} truncated={tr} reward={r} metrics={info.get('metrics', {}).get('fp')}")
    stats = g.call("stats")
print(f"  max reset {max(resets):.2f}s, max эпизод {max(walls):.2f}s (лимит 60 s)")
print(f"  пик RSS: этот процесс {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.1f} MiB, "
      f"оценщик {stats.get('peak_rss_mb')} MiB (лимит 4 ГБ)")

print("== C2 тайм-аут для in-process политики (лимит уменьшен до 1 с)")


class Slow:
    def reset(self, obs):
        self.eid = obs["result"]["manifest"]["episode_id"]
        self.k = 0

    def act(self, obs):
        self.k += 1
        if self.k == 2:
            time.sleep(1.3)  # «думает» дольше лимита
        if self.k == 3:
            return {"action": "submit", "args": {"findings": {"episode_id": self.eid, "findings": []}}}
        return {"action": "list_records", "args": {"table": "fx", "limit": 1}}


with GraderClient(log_path=str(Path(__import__("tempfile").gettempdir()) / "c2_log.jsonl")) as g:
    env = AuditEnv(g, "train", time_limit_s=1.0)
    res = run_episode(env, "slow", {"clean": True}, policy_obj=Slow())
    print("  ", {k: res[k] for k in ("steps", "terminated", "truncated", "return")})
    log = [json.loads(l) for l in open(str(Path(__import__("tempfile").gettempdir()) / "c2_log.jsonl"))]
    print("   журнал оценщика:", [(x["reason"], x["n_findings"], x["reward"]) for x in log])
(Path(__import__("tempfile").gettempdir()) / "c2_log.jsonl").unlink()
