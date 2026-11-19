"""Адаптер «Python-политика -> внешний процесс» для agent_host (протокол stdin/stdout).

python -m auditenv.policies.process_adapter --policy module.path:factory
Эталон этому процессу недоступен: oracle_from_truth здесь запрещён.
"""
from __future__ import annotations

import argparse
import json
import sys

from .api import load_policy


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    a = ap.parse_args(argv)
    pol = load_policy(a.policy, trusted=False)
    out = sys.stdout
    sys.stdout = sys.stderr  # print() политики не ломает протокол
    for line in sys.stdin:
        msg = json.loads(line)
        if msg.get("type") == "close":
            break
        if msg["terminated"] or msg["truncated"]:
            continue
        obs = msg["observation"]
        res = obs.get("result")
        if isinstance(res, dict) and "manifest" in res and "task" in res:
            pol.reset(obs)
        out.write(json.dumps(pol.act(obs), ensure_ascii=False) + "\n")
        out.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
