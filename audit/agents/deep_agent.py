"""Внешний агент для agent_host: на каждом эпизоде сразу отправляет submit с вложенностью d (список d через запятую)."""
import json, sys
depths = [int(x) for x in sys.argv[1].split(",")]
i = 0
for line in sys.stdin:
    m = json.loads(line)
    if m["type"] == "close":
        break
    if m["terminated"] or m["truncated"]:
        continue
    res = m["observation"]["result"] or {}
    if "manifest" not in res:  # не первое наблюдение (например, invalid_json): пустой ответ
        sys.stdout.write(json.dumps({"action": "submit", "args": {"findings": {"episode_id": eid, "findings": []}}}) + "\n")
        sys.stdout.flush()
        continue
    eid = res["manifest"]["episode_id"]
    d = depths[i % len(depths)]; i += 1
    s = '{"action":"submit","args":{"findings":{"episode_id":"%s","findings":%s%s}}}' % (eid, "[" * d, "]" * d)
    sys.stdout.write(s + "\n"); sys.stdout.flush()
