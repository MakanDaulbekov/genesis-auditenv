"""E. Взаимная информация «признак открытых записей ↔ запись затронута ошибкой» на 300 seed
(те же seed и 6 вариантов config, что validate-seeds). Шум — MI при перемешивании меток внутри эпизода.
usage: python3.11 -I e_mi.py <repo_root> [n]
"""
import csv
import io
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
N = int(sys.argv[2]) if len(sys.argv) > 2 else 300
sys.path.insert(0, str(ROOT))
from auditenv.cli import _variants  # noqa: E402
from auditenv.common.drbg import DRBG  # noqa: E402
from auditenv.generator.config import deep_merge  # noqa: E402
from auditenv.generator.episode import generate_episode  # noqa: E402

base = json.loads((ROOT / "configs/default.json").read_text())
rng = DRBG(b"validate-seeds")
variants = _variants()


def ts(s):
    return datetime.fromisoformat(s)


def tzof(off):
    sg = 1 if off[0] == "+" else -1
    return timezone(sg * timedelta(hours=int(off[1:3]), minutes=int(off[4:6])))


def trailing0(n):
    n = abs(n)
    if n == 0:
        return 9
    k = 0
    while n % 10 == 0 and k < 6:
        n //= 10
        k += 1
    return k


def shape(s):
    return re.sub(r"[A-Z]", "A", re.sub(r"[a-z]", "a", re.sub(r"\d", "9", s or "")))


def bucket(x, edges):
    for i, e in enumerate(edges):
        if x < e:
            return i
    return len(edges)


# rows: (episode_index, table, feature_dict, labels_dict)
rows = []
for i in range(N):
    seed = rng.randint(0, 2**63 - 1)
    cfg = deep_merge(base, variants[i % len(variants)])
    ep = generate_episode(seed, cfg)
    pub = ep.public
    pol = json.loads(pub["policy.json"])
    ctz = tzof(pol["timezone"])
    pstart = datetime.fromisoformat(pol["period"]["start"]).replace(tzinfo=ctz)
    pend = (datetime.fromisoformat(pol["period"]["end"]) + timedelta(days=1)).replace(tzinfo=ctz)
    by_type = defaultdict(set)
    ev_rows = set()
    for t in ep.truth["issues"]:
        for x in t["affected_ids"]:
            by_type[t["type"]].add(x)
        for es in t["evidence_sets"]:
            ev_rows |= {x for x in es if x.startswith("R-")}
    aff = set().union(*by_type.values()) if by_type else set()
    ledger = list(csv.DictReader(io.StringIO(pub["ledger.csv"].decode())))
    invs = [json.loads(l) for l in pub["invoices.jsonl"].decode().splitlines()]
    pays = [json.loads(l) for l in pub["payments.jsonl"].decode().splitlines()]
    docs = [json.loads(l) for l in pub["documents.jsonl"].decode().splitlines()]
    docpos = {d["doc_id"]: k / max(1, len(docs) - 1) for k, d in enumerate(docs)}
    docmap = {d["doc_id"]: d for d in docs}
    ent_n = Counter([r["entity_id"] for r in invs] + [docmap[p["doc_id"]].get("payer_entity_id") for p in pays])
    post = defaultdict(list)
    for r in ledger:
        if r["invoice_id"] and not r["payment_id"] and r["account"] in ("accounts_receivable", "revenue", "opening_balance_equity"):
            post[("inv", r["invoice_id"])].append(r["timestamp"])
        if r["payment_id"] and r["account"] == "bank":
            post[("pay", r["payment_id"])].append(r["timestamp"])

    def lag(kind, ident, doc_ts):
        p = post.get((kind, ident))
        if not p:
            return "no_post"
        d = (ts(p[0]) - ts(doc_ts)).total_seconds()
        return "0" if d == 0 else ("neg" if d < 0 else "pos")

    def common(t):
        x = ts(t)
        lx = x.astimezone(ctz)
        return {"hour": x.hour, "minute": x.minute, "second": x.second, "sec_is_0": x.second == 0,
                "hour_company": lx.hour, "offset_eq_company": t[-6:] == pol["timezone"],
                "to_end_h": bucket((pend - x).total_seconds() / 3600, [-72, -6, 0, 6, 72]),
                "from_start_h": bucket((x - pstart).total_seconds() / 3600, [-72, -6, 0, 6, 72])}

    for k, r in enumerate(invs):
        d = docmap[r["doc_id"]]
        f = common(r["issued_at"])
        f.update({"pos_decile": int(10 * k / max(1, len(invs))), "docpos_decile": min(9, int(10 * docpos[r["doc_id"]])),
                  "n_lines": len(r["line_items"]), "round_tz": trailing0(r["total_minor"]),
                  "terms_days": round((ts(r["due_at"]) - ts(r["issued_at"])).total_seconds() / 86400),
                  "post_lag": lag("inv", r["invoice_id"], r["issued_at"]),
                  "entity_ndocs": min(12, ent_n[r["entity_id"]]), "po_shape": shape(d.get("po_number")),
                  "svc_end_minus_issue_d": bucket((datetime.fromisoformat(r["line_items"][0]["service_period_end"]).date()
                                                    - ts(r["issued_at"]).date()).days, [-40, -20, -5, 0, 1, 5, 20]),
                  "currency_is_base": r["currency"] == pol["base_currency"]})
        lab = {"any": r["invoice_id"] in aff, **{t: r["invoice_id"] in s for t, s in by_type.items()}}
        rows.append((i, "invoice", f, lab))
    for k, r in enumerate(pays):
        d = docmap[r["doc_id"]]
        f = common(r["timestamp"])
        br = d.get("bank_reference") or ""
        f.update({"pos_decile": int(10 * k / max(1, len(pays))), "docpos_decile": min(9, int(10 * docpos[r["doc_id"]])),
                  "bankref_len": len(br), "bankref_shape": shape(br), "bankref_prefix": br[:3],
                  "bankref_last_digit": br[-1:], "round_tz": trailing0(r["amount_minor"]),
                  "n_refs": len(r["invoice_refs"] or []), "refs_null": r["invoice_refs"] is None,
                  "post_lag": lag("pay", r["payment_id"], r["timestamp"]),
                  "entity_ndocs": min(12, ent_n[d.get("payer_entity_id")]), "sign": r["amount_minor"] < 0,
                  "currency_is_base": r["currency"] == pol["base_currency"]})
        lab = {"any": r["payment_id"] in aff, **{t: r["payment_id"] in s for t, s in by_type.items()}}
        rows.append((i, "payment", f, lab))
    for k, r in enumerate(ledger):
        if not (r["invoice_id"] or r["payment_id"]):
            continue
        f = common(r["timestamp"])
        f.update({"pos_decile": int(10 * k / len(ledger)), "round_tz": trailing0(int(r["debit_minor"]) or int(r["credit_minor"])),
                  "account": r["account"]})
        rows.append((i, "ledger_doc_row", f, {"any": r["row_id"] in ev_rows}))


def mi(pairs):
    n = len(pairs)
    c = Counter(pairs)
    px, py = Counter(), Counter()
    for (x, y), v in c.items():
        px[x] += v
        py[y] += v
    return sum(v / n * math.log2(v * n / (px[x] * py[y])) for (x, y), v in c.items())


def h(ys):
    n = len(ys)
    return -sum(v / n * math.log2(v / n) for v in Counter(ys).values())


out = []
for table in ("invoice", "payment", "ledger_doc_row"):
    R = [r for r in rows if r[1] == table]
    labels = ["any"] if table == "ledger_doc_row" else ["any", "duplicate_invoice", "missing_or_partial_allocation",
                                                       "fx_wrong_date_or_rate", "period_cutoff_error"]
    for lab in labels:
        ys = [r[3].get(lab, False) for r in R]
        if sum(ys) < 30:
            continue
        hy = h(ys)
        for feat in R[0][2]:
            xs = [r[2][feat] for r in R]
            m = mi(list(zip(xs, ys)))
            sh = []
            pr = random.Random(7)
            for _ in range(5):
                # перемешиваем метки ВНУТРИ эпизода
                yy = []
                grp = defaultdict(list)
                for (ep_i, *_), y in zip(R, ys):
                    grp[ep_i].append(y)
                for ep_i in grp:
                    pr.shuffle(grp[ep_i])
                it = {e: iter(v) for e, v in grp.items()}
                yy = [next(it[r[0]]) for r in R]
                sh.append(mi(list(zip(xs, yy))))
            mu = sum(sh) / 5
            sd = (sum((s - mu) ** 2 for s in sh) / 4) ** 0.5 or 1e-9
            # самое «обогащённое» значение (>=50 появлений)
            cx = Counter(xs)
            cxy = Counter(x for x, y in zip(xs, ys) if y)
            p0 = sum(ys) / len(ys)
            best = max(((cxy[v] / cx[v] / p0, v, cx[v], cxy[v]) for v in cx if cx[v] >= 50), default=(0, None, 0, 0))
            out.append({"table": table, "label": lab, "feature": feat, "n": len(R), "pos": sum(ys), "H_Y": hy,
                        "mi": m, "mi_shuffle": mu, "sd": sd, "excess": m - mu, "z": (m - mu) / sd,
                        "frac_HY": (m - mu) / hy, "best_lift": best})
(Path(__import__("tempfile").gettempdir()) / "e_mi.json").write_text(json.dumps(out, ensure_ascii=False, default=str, indent=1))
out.sort(key=lambda r: -r["excess"])
print(f"{'table':15s} {'label':30s} {'feature':22s} {'MI':>8s} {'шум':>8s} {'избыток':>8s} {'z':>7s} {'%H(Y)':>6s}  лучший lift (значение, n, n+)")
for r in out[:45]:
    b = r["best_lift"]
    print(f"{r['table']:15s} {r['label']:30s} {r['feature']:22s} {r['mi']:8.4f} {r['mi_shuffle']:8.4f} {r['excess']:8.4f} "
          f"{r['z']:7.1f} {100 * r['frac_HY']:6.2f}  {b[0]:.2f} ({b[1]!s:.12}, {b[2]}, {b[3]})")
