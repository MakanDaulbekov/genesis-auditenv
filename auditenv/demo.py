"""make demo: один dev-эпизод, лог шагов скриптовой политики и итоговые метрики."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from .env.core import AuditEnv
from .grader.service import GraderClient
from .policies.baselines import POLICIES

ROOT = Path(__file__).resolve().parent.parent


def short(action) -> str:
    a = action["args"]
    if action["action"] == "list_records":
        return f"list_records({a['table']}, cursor={'…' if a.get('cursor') else None}, limit={a.get('limit')})"
    if action["action"] == "get_document":
        return f"get_document({a['doc_id']})"
    if action["action"] == "submit":
        fs = a["findings"]["findings"]
        kinds = {}
        for f in fs:
            kinds[f["type"]] = kinds.get(f["type"], 0) + 1
        return f"submit({len(fs)} findings: {kinds})"
    return f"{action['action']}({a})"


def main(argv=None) -> int:
    idx = json.loads((ROOT / "episodes" / "dev" / "index.json").read_text(encoding="utf-8"))
    ref = idx[3]["ref"]
    print(f"== Демо: эпизод {ref}, политика simple_reconciler, режим train ==")
    with GraderClient(str(ROOT / "episodes"), str(ROOT / "grader_data"), allow_oracle=True) as g:
        env = AuditEnv(g, "train")
        obs, info = env.reset(config={"episode_ref": ref})
        m = obs["result"]["manifest"]
        print(f"компания: {m['company_name']} | base={m['base_currency']} | tz={m['timezone']} | "
              f"период {m['period_start']}..{m['period_end']} | counts={m['counts']}")
        gen = POLICIES["simple_reconciler"](obs)
        action = next(gen)
        while True:
            o, r, te, tr, info = env.step(action)
            err = f" ERROR={o['error']['code']}" if o["error"] else ""
            print(f"  шаг {100 - o['steps_remaining']:3d}: {short(action)}{err}  (осталось {o['steps_remaining']})")
            if te or tr:
                break
            action = gen.send(o)
        print(f"итог: reward={r:.4f} terminated={te} truncated={tr}")
        print("метрики:", json.dumps(info.get("metrics"), ensure_ascii=False))
        env2 = AuditEnv(g, "train")
        env2.reset(config={"episode_ref": ref})
        _, r2, *_ = env2.oracle_step()
        print(f"для сравнения oracle (ответ строится внутри процесса оценщика): reward={r2:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
