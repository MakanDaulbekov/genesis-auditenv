"""Жёсткая граница «агент ↔ среда»: агент — отдельный процесс, видит только JSON-наблюдения.

python -m auditenv.agent_host --agent-cmd "python examples/stdin_agent.py" --split dev --mode eval
python -m auditenv.agent_host --policy my_pkg.my_module:MyPolicy --closed-episodes-file closed.json --secret-file s.txt

Протокол (по строке JSON):
  host -> agent: {"type": "observation", "observation", "reward", "terminated", "truncated", "info"}
  agent -> host: действие {"action": ..., "args": {...}} (любой мусор = ошибка среды и расход шага)
  host -> agent: {"type": "close"} в конце прогона.
У процесса агента нет дескриптора оценщика, секрета и закрытых файлов: закрытые эпизоды
генерируются в памяти процесса оценщика (closed_episodes_file + secret_file).
Таймаут ответа агента = остаток лимита времени эпизода; после него эпизод truncated.
Падение агента: текущий и оставшиеся эпизоды -> пустой ответ (truncated_agent_crashed), exit 2,
stderr агента — в grader_data/agent_stderr.log. Число оценённых эпизодов != числу конфигов -> exit 3.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import select
import shlex
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from .env.core import AuditEnv
from .grader.service import GraderClient


class AgentCrashed(Exception):
    pass


def _writer(p):
    """Windows: буфер канала ~4 КБ, и запись агенту, который не читает stdin, блокировала бы хост навсегда
    (на Linux буфер 64 КБ). Поэтому там пишет фоновый поток; таймаут эпизода по-прежнему соблюдается."""
    w = getattr(p, "_auditenv_writer", None)
    if w is None:
        q, state = queue.Queue(), {"broken": False}

        def pump():
            while True:
                data = q.get()
                if data is None:
                    return
                try:
                    p.stdin.write(data)
                    p.stdin.flush()
                except (BrokenPipeError, OSError, ValueError):
                    state["broken"] = True
                    return

        threading.Thread(target=pump, daemon=True).start()
        w = p._auditenv_writer = (q, state)
    return w


def _send(p, msg):
    data = (json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8")
    if os.name == "nt":
        q, state = _writer(p)
        if state["broken"] or p.poll() is not None:
            raise AgentCrashed()
        q.put(data)
        return
    try:
        p.stdin.write(data)
        p.stdin.flush()
    except (BrokenPipeError, OSError, ValueError):
        raise AgentCrashed() from None


def _kill(p) -> None:
    """Завершить агента. На Windows — всем деревом: лаунчеры venv запускают интерпретатор дочерним
    процессом, и TerminateProcess одного лаунчера оставил бы агента живым."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    try:
        p.kill()
    except OSError:
        pass


def _line_queue(p) -> "queue.Queue":
    """Windows: select() не работает с каналами, поэтому строки агента читает фоновый поток в очередь."""
    q = getattr(p, "_auditenv_lines", None)
    if q is None:
        q = queue.Queue()

        def pump():
            try:
                while True:
                    line = p.stdout.readline(1024 * 1024)
                    q.put(line)
                    if not line:
                        return
            except (OSError, ValueError):
                q.put(b"")

        threading.Thread(target=pump, daemon=True).start()
        p._auditenv_lines = q
    return q


def _recv(p, timeout: float):
    """-> str (строка агента) | None (таймаут). Конец потока = агент завершился -> AgentCrashed."""
    if os.name == "nt":
        try:
            line = _line_queue(p).get(timeout=max(0.0, timeout))
        except queue.Empty:
            return None
        if not line:
            raise AgentCrashed()
        return line.decode("utf-8", "replace")
    r, _, _ = select.select([p.stdout], [], [], max(0.0, timeout))
    if not r:
        return None
    line = p.stdout.readline(1024 * 1024)
    if not line:
        raise AgentCrashed()
    return line.decode("utf-8", "replace")


def build_agent_argv(agent_cmd: str) -> list[str]:
    """`python`/`python3` в начале команды заменяются текущим интерпретатором: агент гарантированно
    получает тот же Python 3.11, что и среда (venv жюри), даже если `python` в PATH другой."""
    if os.name == "nt":  # пути Windows с обратной косой чертой: без POSIX-экранирования, кавычки снимаем
        argv = [a[1:-1] if len(a) >= 2 and a[0] == a[-1] and a[0] in "\"'" else a
                for a in shlex.split(agent_cmd, posix=False)]
    else:
        argv = shlex.split(agent_cmd)
    if not argv:
        raise ValueError("пустая --agent-cmd")
    if argv[0] in ("python", "python3", "python3.11"):
        argv[0] = sys.executable
    return argv


def run_agent(agent_cmd: str, configs: list[dict], grader: GraderClient, mode: str, traj_path: str | None = None,
              time_limit_s: float = 60.0, stderr_path: str | None = "grader_data/agent_stderr.log",
              cwd: str | None = None, extra_env: dict | None = None) -> dict:
    """Прогон внешнего агента. Агент запускается в cwd ВЫЗЫВАЮЩЕГО (относительные пути в
    --agent-cmd работают), с урезанным окружением и HOME=временный каталог. У агента нет
    дескриптора оценщика и секрета; это граница изоляции, а не песочница ФС.

    -> {"episodes": [...], "crashed": bool, "exit_code": int|None, "crash_episode": int|None,
        "stderr_path": str|None}
    Если агент завершился до конца прогона, текущий и все оставшиеся эпизоды оцениваются как
    пустой ответ (truncated, причина agent_crashed) — ни один эпизод не пропадает молча.
    """
    workdir = tempfile.mkdtemp(prefix="agent_home_")
    env_vars = {"PATH": os.environ.get("PATH", ""), "HOME": workdir, "PYTHONHASHSEED": "0",
                "LANG": os.environ.get("LANG", "C.UTF-8"), "PYTHONIOENCODING": "utf-8"}
    if os.name == "nt":  # без SYSTEMROOT интерпретатор Windows не стартует; это не каналы к оценщику
        env_vars.update({k: os.environ[k] for k in ("SYSTEMROOT", "WINDIR", "TEMP", "TMP", "PATHEXT", "COMSPEC")
                         if k in os.environ})
    env_vars.update(extra_env or {})
    err_f = None
    if stderr_path:
        Path(stderr_path).parent.mkdir(parents=True, exist_ok=True)
        err_f = open(stderr_path, "ab")
    p = subprocess.Popen(build_agent_argv(agent_cmd), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=err_f if err_f else subprocess.DEVNULL, cwd=cwd or os.getcwd(), env=env_vars)
    out = []
    crashed, crash_ep = False, None
    traj = open(traj_path, "w", encoding="utf-8", newline="\n") if traj_path else None

    def log_step(ep_id, step, action, o, r, te, tr):
        if traj:
            traj.write(json.dumps({"policy": "external_agent", "episode_id": ep_id, "step": step, "action": action,
                                   "observation": o, "reward": r, "terminated": te, "truncated": tr},
                                  ensure_ascii=False) + "\n")

    try:
        for i, cfg in enumerate(configs):
            env = AuditEnv(grader, mode=mode, policy_name="external_agent", time_limit_s=time_limit_s)
            obs, info = env.reset(config=cfg)
            ep_id = obs["result"]["manifest"]["episode_id"]
            t0 = time.monotonic()
            total, steps, reason = 0.0, 0, "ok"
            log_step(ep_id, 0, None, obs, 0.0, False, False)
            if crashed:
                o, r, te, tr, info = env.force_truncate("agent_crashed")
                log_step(ep_id, 1, None, o, r, te, tr)
                out.append({"episode_id": ep_id, "return": r, "steps": 0, "wall_s": 0.0, "status": "agent_crashed"})
                continue
            msg = {"type": "observation", "observation": obs, "reward": 0.0, "terminated": False,
                   "truncated": False, "info": info}
            while True:
                try:
                    _send(p, msg)
                    line = _recv(p, env.time_limit_s - (time.monotonic() - t0) + 0.05)
                except AgentCrashed:
                    crashed, crash_ep, reason = True, i, "agent_crashed"
                    o, r, te, tr, info = env.force_truncate("agent_crashed")
                    log_step(ep_id, steps + 1, None, o, r, te, tr)
                    total += r
                    break
                if line is None:
                    o, r, te, tr, info = env.step(None)  # лимит времени истёк -> truncated
                    reason = "timeout"
                else:
                    o, r, te, tr, info = env.step(line.strip())
                steps += 1
                total += r
                log_step(ep_id, steps, line.strip() if line else None, o, r, te, tr)
                msg = {"type": "observation", "observation": o, "reward": r, "terminated": te, "truncated": tr,
                       "info": info}
                if te or tr:
                    try:
                        _send(p, msg)
                    except AgentCrashed:
                        pass
                    break
            out.append({"episode_id": ep_id, "return": total, "steps": steps,
                        "wall_s": round(time.monotonic() - t0, 3), "status": reason})
        if not crashed:
            try:
                _send(p, {"type": "close"})
            except AgentCrashed:
                pass
    finally:
        if traj:
            traj.close()
        try:
            p.wait(timeout=2)
        except subprocess.TimeoutExpired:
            _kill(p)
            p.wait()
        if err_f:
            err_f.close()
    code = p.returncode
    if crashed and code is None:
        code = -1
    return {"episodes": out, "crashed": crashed, "exit_code": code if crashed else None,
            "crash_episode": crash_ep, "stderr_path": stderr_path}


def main(argv=None) -> int:
    from .common.console import utf8_stdio
    utf8_stdio()
    ap = argparse.ArgumentParser(description="Запуск внешнего агента (отдельный процесс, JSON через stdin/stdout)")
    g_ = ap.add_mutually_exclusive_group(required=True)
    g_.add_argument("--agent-cmd", help="команда агента; относительные пути — от текущего каталога")
    g_.add_argument("--policy", help="Python-политика module.path:factory (запускается отдельным процессом)")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--mode", default="eval", choices=["train", "eval"])
    ap.add_argument("--episodes-root", default="episodes")
    ap.add_argument("--grader-root", default="grader_data")
    ap.add_argument("--closed-episodes-file", default=None)
    ap.add_argument("--secret-file", default=None)
    ap.add_argument("--trajectories", default=None)
    ap.add_argument("--time-limit-s", type=float, default=60.0)
    ap.add_argument("--agent-stderr", default="grader_data/agent_stderr.log")
    ap.add_argument("--grader-log", default="grader_data/agent_host_log.jsonl")
    a = ap.parse_args(argv)
    if a.policy in ("oracle", "oracle_from_truth"):
        print("ОШИБКА: oracle работает только в процессе оценщика / доверенном режиме "
              "(python -m auditenv.cli run --trusted --policies oracle_from_truth), не во внешнем агенте.",
              file=sys.stderr)
        return 4
    if a.policy:
        agent_cmd = shlex.join([sys.executable, "-m", "auditenv.policies.process_adapter", "--policy", a.policy])
        extra_env = {"PYTHONPATH": os.pathsep.join([os.getcwd(), str(Path(__file__).resolve().parents[1])])}
    else:
        agent_cmd, extra_env = a.agent_cmd, None
    if a.closed_episodes_file and not a.secret_file:
        print("ПРЕДУПРЕЖДЕНИЕ: закрытый прогон без --secret-file — используется детерминированный секрет по "
              "умолчанию; маленькие seed можно перебрать открытым генератором.", file=sys.stderr)
    with GraderClient(a.episodes_root, a.grader_root, closed_episodes_file=a.closed_episodes_file,
                      secret_file=a.secret_file, log_path=a.grader_log) as g:
        if a.closed_episodes_file:
            n = len(json.loads(Path(a.closed_episodes_file).read_text(encoding="utf-8")))
            configs = [{"closed_index": i} for i in range(n)]
        else:
            idx = json.loads((Path(a.episodes_root) / a.split / "index.json").read_text(encoding="utf-8"))
            configs = [{"episode_ref": e["ref"]} for e in idx]
        res = run_agent(agent_cmd, configs, g, a.mode, a.trajectories, a.time_limit_s, a.agent_stderr,
                        extra_env=extra_env)
        g.call("finalize")
        graded = g.call("results")["results"]
    print(json.dumps({"agent_view": res["episodes"],
                      "grader": [{k: r[k] for k in ("episode_id", "reason", "reward", "f1", "tp", "fp", "fn",
                                                    "inspection_steps", "valid")} for r in graded]},
                     indent=2, ensure_ascii=False))
    rc = 0
    if res["crashed"]:
        print(f"ОШИБКА: агент завершился с кодом {res['exit_code']} на эпизоде {res['crash_episode'] + 1} из "
              f"{len(configs)}; этот и оставшиеся эпизоды оценены как пустой ответ. См. {res['stderr_path']}",
              file=sys.stderr)
        rc = 2
    if len(graded) != len(configs):
        print(f"ОШИБКА: оценено {len(graded)} эпизодов из {len(configs)}", file=sys.stderr)
        rc = 3
    return rc


if __name__ == "__main__":
    sys.exit(main())
