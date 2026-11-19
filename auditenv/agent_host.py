"""Жёсткая граница «агент ↔ среда»: агент — отдельный процесс, видит только JSON-наблюдения.

python -m auditenv.agent_host --agent-cmd "python examples/stdin_agent.py" --split dev --mode eval

Протокол (по строке JSON):
  host -> agent: {"type": "observation", "observation", "reward", "terminated", "truncated", "info"}
  agent -> host: действие {"action": ..., "args": {...}} (любой мусор = ошибка среды и расход шага)
  host -> agent: {"type": "close"} в конце прогона.
У процесса агента нет дескриптора оценщика, секрета и закрытых файлов: закрытые эпизоды
генерируются в памяти процесса оценщика (closed_episodes_file + secret_file).
Таймаут ответа агента = остаток лимита времени эпизода; после него эпизод truncated.
"""
from __future__ import annotations

import argparse
import json
import os
import select
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .env.core import AuditEnv
from .grader.service import GraderClient


def _send(p, msg):
    p.stdin.write((json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8"))
    p.stdin.flush()


def _recv(p, timeout: float):
    r, _, _ = select.select([p.stdout], [], [], max(0.0, timeout))
    if not r:
        return None
    line = p.stdout.readline(1024 * 1024)
    return line.decode("utf-8", "replace") if line else None


def run_agent(agent_cmd: str, configs: list[dict], grader: GraderClient, mode: str, traj_path: str | None = None,
              time_limit_s: float = 60.0):
    workdir = tempfile.mkdtemp(prefix="agent_")
    env_vars = {"PATH": os.environ.get("PATH", ""), "HOME": workdir, "PYTHONHASHSEED": "0"}
    p = subprocess.Popen(shlex.split(agent_cmd), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, cwd=workdir, env=env_vars)
    out = []
    traj = open(traj_path, "w", encoding="utf-8") if traj_path else None
    try:
        for cfg in configs:
            env = AuditEnv(grader, mode=mode, policy_name="external_agent", time_limit_s=time_limit_s)
            obs, info = env.reset(config=cfg)
            t0 = time.monotonic()
            msg = {"type": "observation", "observation": obs, "reward": 0.0, "terminated": False,
                   "truncated": False, "info": info}
            total, steps = 0.0, 0
            while True:
                _send(p, msg)
                line = _recv(p, env.time_limit_s - (time.monotonic() - t0) + 0.05)
                if line is None:
                    o, r, te, tr, info = env.step(None)  # лимит времени истёк -> truncated
                    if not (te or tr):  # агент завершился раньше лимита: дожимаем пустым ответом
                        o, r, te, tr, info = env.step({"action": "submit", "args": {"findings": {
                            "episode_id": obs["result"]["manifest"]["episode_id"], "findings": []}}})
                else:
                    o, r, te, tr, info = env.step(line.strip())
                steps += 1
                total += r
                if traj:
                    traj.write(json.dumps({"episode_id": obs["result"]["manifest"]["episode_id"], "step": steps,
                                           "action": line.strip() if line else None, "observation": o,
                                           "reward": r, "terminated": te, "truncated": tr}, ensure_ascii=False) + "\n")
                msg = {"type": "observation", "observation": o, "reward": r, "terminated": te, "truncated": tr,
                       "info": info}
                if te or tr:
                    _send(p, msg)
                    break
            out.append({"return": total, "steps": steps, "wall_s": round(time.monotonic() - t0, 3)})
        _send(p, {"type": "close"})
    except (BrokenPipeError, OSError):
        pass
    finally:
        if traj:
            traj.close()
        try:
            p.kill()
        except Exception:
            pass
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent-cmd", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--mode", default="eval", choices=["train", "eval"])
    ap.add_argument("--closed-episodes-file", default=None)
    ap.add_argument("--secret-file", default=None)
    ap.add_argument("--trajectories", default=None)
    a = ap.parse_args(argv)
    with GraderClient(closed_episodes_file=a.closed_episodes_file, secret_file=a.secret_file,
                      log_path="grader_data/agent_host_log.jsonl") as g:
        if a.closed_episodes_file:
            n = len(json.loads(Path(a.closed_episodes_file).read_text(encoding="utf-8")))
            configs = [{"closed_index": i} for i in range(n)]
        else:
            idx = json.loads((Path("episodes") / a.split / "index.json").read_text(encoding="utf-8"))
            configs = [{"episode_ref": e["ref"]} for e in idx]
        res = run_agent(a.agent_cmd, configs, g, a.mode, a.trajectories)
        g.call("finalize")
        graded = g.call("results")["results"]
    print(json.dumps({"agent_view": res, "grader": [{k: r[k] for k in ("episode_id", "reward", "f1", "tp", "fp", "fn",
                                                                        "inspection_steps", "valid")} for r in graded]},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
