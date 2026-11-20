"""E-drill: проверка «трюков по времени» как политик: награда против молчания, P/R на уровне документов.
usage: python3.11 -I e_drill.py <repo_root> [n]
"""
import csv
import io
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
N = int(sys.argv[2]) if len(sys.argv) > 2 else 300
sys.path.insert(0, str(ROOT))
from auditenv.cli import _variants  # noqa: E402
from auditenv.common.drbg import DRBG  # noqa: E402
from auditenv.env.store import parse_jsonl  # noqa: E402
from auditenv.generator.config import deep_merge  # noqa: E402
from auditenv.generator.episode import generate_episode  # noqa: E402
from auditenv.grader.matcher import alias_map_from_tables  # noqa: E402
from auditenv.grader.reward import grade  # noqa: E402

base = json.loads((ROOT / "configs/default.json").read_text())
rng = DRBG(b"validate-seeds")
V = _variants()


def tzof(off):
    sg = 1 if off[0] == "+" else -1
    return timezone(sg * timedelta(hours=int(off[1:3]), minutes=int(off[4:6])))


win_types = Counter()
var_cnt = Counter()
tricks = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "rew": 0.0, "n": 0})
silence = 0.0
for i in range(N):
    seed = rng.randint(0, 2**63 - 1)
    ep = generate_episode(seed, deep_merge(base, V[i % len(V)]))
    pub = ep.public
    pol = json.loads(pub["policy.json"])
    ctz = tzof(pol["timezone"])
    pend = (datetime.fromisoformat(pol["period"]["end"]) + timedelta(days=1)).replace(tzinfo=ctz)
    pstart = datetime.fromisoformat(pol["period"]["start"]).replace(tzinfo=ctz)
    led = list(csv.DictReader(io.StringIO(pub["ledger.csv"].decode())))
    invs = {r["invoice_id"]: r for r in parse_jsonl(pub["invoices.jsonl"])}
    pays = {r["payment_id"]: r for r in parse_jsonl(pub["payments.jsonl"])}
    alias = alias_map_from_tables(list(invs.values()), list(pays.values()))
    ev_type = {}
    for t in ep.truth["issues"]:
        for es in t["evidence_sets"]:
            for x in es:
                if x.startswith("R-"):
                    ev_type[x] = t["type"]
    for m in ep.provenance["mutations"]:
        if m["type"] == "period_cutoff_error":
            var_cnt[m["variant"]] += 1
    for r in led:
        if (r["invoice_id"] or r["payment_id"]):
            h = (pend - datetime.fromisoformat(r["timestamp"])).total_seconds() / 3600
            if 0 <= h < 6:
                win_types[ev_type.get(r["row_id"], "—не доказательство")] += 1

    def mk(kind_rows):
        f = []
        for r in kind_rows:
            ident = r["invoice_id"] if r["invoice_id"] and not r["payment_id"] else r["payment_id"]
            f.append({"type": "period_cutoff_error", "affected_ids": [ident],
                      "claimed_amount_minor": int(r["debit_minor"]) or int(r["credit_minor"]),
                      "currency": pol["base_currency"], "evidence_ids": [ident, r["row_id"]], "explanation": ""})
        return f[:30]

    def last6(r):
        h = (pend - datetime.fromisoformat(r["timestamp"])).total_seconds() / 3600
        return 0 <= h < 6

    def first6(r):
        h = (datetime.fromisoformat(r["timestamp"]) - pstart).total_seconds() / 3600
        return 0 <= h < 6

    post_inv = [r for r in led if r["invoice_id"] and not r["payment_id"] and r["account"] == "accounts_receivable"
                and int(r["debit_minor"]) > 0]
    post_pay = [r for r in led if r["payment_id"] and r["account"] == "bank"]
    cand = {
        "T1 AR-проводка в последние 6 ч (без документа)": [r for r in post_inv if last6(r)],
        "T2 + bank-проводка в последние 6 ч": [r for r in post_inv + post_pay if last6(r)],
        "T3 AR в первые/последние 6 ч": [r for r in post_inv if last6(r) or first6(r)],
        "T4 честно: проводка != момент документа (нужен документ)": [
            r for r in post_inv if datetime.fromisoformat(r["timestamp"]) != datetime.fromisoformat(invs[r["invoice_id"]]["issued_at"])] +
            [r for r in post_pay if datetime.fromisoformat(r["timestamp"]) != datetime.fromisoformat(pays[r["payment_id"]]["timestamp"])],
    }
    k = 4  # шагов на трюк: 2-4 страницы ledger, условно
    silence += grade({"episode_id": ep.truth["episode_id"], "findings": []}, ep.truth, alias, 0, 0)["reward"]
    for name, rows in cand.items():
        g = grade({"episode_id": ep.truth["episode_id"], "findings": mk(rows)}, ep.truth, alias, 0, k)
        tr = tricks[name]
        tr["tp"] += g["tp"]; tr["fp"] += g["fp"]; tr["fn"] += g["fn"]; tr["rew"] += g["reward"]; tr["n"] += 1
        tr["cut_tp"] = tr.get("cut_tp", 0) + g["by_type"]["period_cutoff_error"]["tp"]
        tr["cut_fn"] = tr.get("cut_fn", 0) + g["by_type"]["period_cutoff_error"]["fn"]

print("Строки проводок документов в последние 6 ч периода — чьи это доказательства:")
tot = sum(win_types.values())
for k_, v in win_types.most_common():
    print(f"  {k_:32s} {v:5d}  {100 * v / tot:5.1f}%")
print("\nВарианты cutoff:", dict(var_cnt))
print(f"\nСредняя награда молчания: {silence / N:+.4f}")
print(f"{'трюк (ответ: period_cutoff_error по строке)':58s} {'P':>6s} {'R(cutoff)':>9s} {'награда':>8s}")
for name, t in tricks.items():
    p = t["tp"] / max(1, t["tp"] + t["fp"])
    rc = t["cut_tp"] / max(1, t["cut_tp"] + t["cut_fn"])
    print(f"{name:58s} {p:6.3f} {rc:9.3f} {t['rew'] / t['n']:+8.4f}")
