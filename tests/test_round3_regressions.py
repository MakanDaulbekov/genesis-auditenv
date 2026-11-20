"""Регрессии аудита раунда 3 (AUDIT.md). Каждый тест падал до своего фикса."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auditenv.env.core import AuditEnv  # noqa: E402
from auditenv.grader.service import GraderClient  # noqa: E402

REF = "dev/ep_053daf06ab62f121"


def _nest(d):
    x = []
    for _ in range(d):
        x = [x]
    return x


# ----------------------------------------------------------------------------- F3: глубоко вложенный submit

def test_deeply_nested_submit_never_crashes_and_is_always_graded(tmp_path):
    """Окно глубины, где json.dumps в step проходит, а сериализация запроса к оценщику (или его разбор) падает,
    давало RecursionError из env.step либо завершённый, но не оценённый эпизод (0 вместо −1).
    Перебор широкий: положение окна зависит от глубины стека вызывающего."""
    terminal = 0
    with GraderClient(str(ROOT / "episodes"), str(ROOT / "grader_data"), log_path=str(tmp_path / "g.jsonl")) as g:
        env = AuditEnv(g, "train")
        for d in range(850, 1151):
            for as_str in (False, True):
                obs, _ = env.reset(config={"episode_ref": REF})
                eid = obs["result"]["manifest"]["episode_id"]
                if as_str:
                    a = '{"action":"submit","args":{"findings":{"episode_id":"%s","findings":%s%s}}}' % (
                        eid, "[" * d, "]" * d)
                else:
                    a = {"action": "submit", "args": {"findings": {"episode_id": eid, "findings": _nest(d)}}}
                o, r, te, tr, info = env.step(a)  # не должно бросать
                if te:
                    terminal += 1
                    assert r == -1.0 and info["metrics"]["valid"] is False, (d, as_str, r, info)
                else:  # строка, которую не разобрать: ограниченная ошибка, шаг израсходован
                    assert as_str and o["error"]["code"] == "invalid_json" and not tr, (d, o)
        g.call("finalize")
        graded = g.call("results")["results"]
    assert terminal >= 301 and len(graded) == terminal
    assert all(x["event"] == "invalid_findings" and x["reward"] == -1.0 for x in graded)
