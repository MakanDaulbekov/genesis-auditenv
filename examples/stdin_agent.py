"""Пример внешнего агента (отдельный процесс): читает наблюдения из stdin, пишет действия в stdout.
Стратегия: прочитать первую страницу счетов и отправить пустой ответ."""
import json
import sys

episode_id = None
step = 0
for line in sys.stdin:
    msg = json.loads(line)
    if msg["type"] == "close":
        break
    obs = msg["observation"]
    if msg["terminated"] or msg["truncated"]:
        step = 0
        continue
    if obs["result"] and "manifest" in obs["result"]:
        episode_id = obs["result"]["manifest"]["episode_id"]
        step = 0
    step += 1
    if step == 1:
        action = {"action": "list_records", "args": {"table": "invoices", "limit": 50}}
    else:
        action = {"action": "submit", "args": {"findings": {"episode_id": episode_id, "findings": []}}}
    sys.stdout.write(json.dumps(action) + "\n")
    sys.stdout.flush()
