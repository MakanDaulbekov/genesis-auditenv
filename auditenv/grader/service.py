"""Оценщик как ОТДЕЛЬНЫЙ процесс. truth/provenance/seed живут только здесь.

Протокол: только строки JSON через stdin/stdout отдельного ОС-процесса — никакого
pickle от стороны агента, т.е. процесс агента не может «записать» в оценщик ничего,
кроме разрешённых операций: open, grade, oracle, finalize, results (+ stats: только пик RSS).
  * grade принимается ровно один раз на эпизод (token); повтор логируется и отклоняется.
  * В режиме eval grade возвращает 0 и ничего больше; результаты доступны через results
    только после finalize, после которого новые эпизоды не открываются.
"""
from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# <сплит>/<каталог эпизода>: оба компонента начинаются с буквы/цифры и не содержат '/', поэтому '..',
# абсолютные пути и выход за корни невозможны. Имя каталога не обязано быть ep_<16 hex> — жюри может
# положить свои преобразованные эпизоды под любыми именами.
REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}/[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
PUBLIC_FILES = ("manifest.json", "policy.json", "ledger.csv", "invoices.jsonl", "payments.jsonl",
                "fx.csv", "documents.jsonl")


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


class GraderCore:
    """Логика оценщика; работает только внутри процесса оценщика."""

    def __init__(self, settings: dict):
        self.episodes_root = Path(settings.get("episodes_root", "episodes"))
        self.grader_root = Path(settings.get("grader_root", "grader_data"))
        self.log_path = settings.get("log_path")
        secret_file = settings.get("secret_file")
        self.secret = ""
        if secret_file:
            self.secret = Path(secret_file).read_text(encoding="utf-8").strip()
        self.closed = []
        if settings.get("closed_episodes_file"):
            self.closed = json.loads(Path(settings["closed_episodes_file"]).read_text(encoding="utf-8"))
        self.allow_oracle = bool(settings.get("allow_oracle", False))
        self.sessions: dict[str, dict] = {}
        self.closed_opened: set[int] = set()
        self.counter = 0
        self.results: list[dict] = []
        self.finalized = False

    def log(self, rec: dict) -> None:
        if self.log_path:
            try:
                with open(self.log_path, "a", encoding="utf-8", newline="\n") as f:
                    f.write(json.dumps(rec, sort_keys=True, ensure_ascii=False) + "\n")
            except OSError as e:   # журнал недоступен (read-only ФС и т.п.): оценка не должна падать
                if not getattr(self, "_log_warned", False):
                    self._log_warned = True
                    sys.stderr.write(f"[grader] журнал недоступен ({type(e).__name__}); записи пропускаются\n")

    # ------------------------------------------------------------------ ops
    def op_open(self, mode: str, seed=None, config=None, ref=None, closed_index=None) -> dict:
        from ..generator.config import ConfigError, resolve_config
        from ..generator.episode import generate_episode
        from ..generator.world import GenerationError
        from ..env.store import parse_jsonl

        if self.finalized:
            return {"error": "run_finalized"}
        if mode not in ("train", "eval"):
            return {"error": "bad_mode"}
        if closed_index is not None:
            if isinstance(closed_index, bool) or not isinstance(closed_index, int) \
                    or not 0 <= closed_index < len(self.closed):
                return {"error": "bad_index"}
            # закрытый эпизод: только eval и только один раз за прогон (нельзя переоткрыть и
            # «прощупать» эталон повторными оценками или train-наградой)
            if mode != "eval":
                self.log({"event": "rejected_closed_train", "closed_index": closed_index})
                return {"error": "closed_requires_eval"}
            if closed_index in self.closed_opened:
                self.log({"event": "rejected_closed_reopen", "closed_index": closed_index})
                return {"error": "closed_already_opened"}
            self.closed_opened.add(closed_index)
            spec = self.closed[closed_index]
            ep = generate_episode(spec["seed"], resolve_config(spec.get("config")), self.secret)
            public, truth = ep.public, ep.truth
        elif ref is not None:
            if not isinstance(ref, str) or not REF_RE.match(ref):
                return {"error": "bad_ref"}
            pdir, gdir = self.episodes_root / ref, self.grader_root / ref
            if not pdir.is_dir() or not gdir.is_dir():
                return {"error": "not_found"}
            missing = [n for n in PUBLIC_FILES if not (pdir / n).is_file()]
            if missing or not (gdir / "truth.json").is_file():
                return {"error": "missing_files",
                        "detail": f"в каталоге эпизода нет файлов: {missing or ['truth.json (эталон)']}"}
            public = {n: (pdir / n).read_bytes() for n in PUBLIC_FILES}
            truth = json.loads((gdir / "truth.json").read_text(encoding="utf-8"))
        else:
            if isinstance(seed, bool) or not isinstance(seed, int):
                return {"error": "bad_seed", "detail": "seed: ожидается целое число"}
            try:
                cfg = resolve_config(config)
            except ConfigError as e:
                return {"error": "bad_config", "detail": str(e)[:500]}
            try:
                ep = generate_episode(seed, cfg, self.secret)
            except GenerationError:
                return {"error": "generation_failed",
                        "detail": "не удалось построить эпизод с этим config (слишком мало кандидатов для ошибок)"}
            public, truth = ep.public, ep.truth
        from ..grader.matcher import alias_map_from_tables, public_ids_from_files, tolerance_from_files
        alias = alias_map_from_tables(parse_jsonl(public["invoices.jsonl"]), parse_jsonl(public["payments.jsonl"]))
        # все ID открытых записей: несуществующий evidence_id -> FP (определения проверок жюри, matcher п. 3)
        known = public_ids_from_files(public)
        tolerance = tolerance_from_files(public)
        self.counter += 1
        token = f"t{self.counter}"
        self.sessions[token] = {"mode": mode, "truth": truth, "alias": alias, "tolerance": tolerance, "known": known,
                                "graded": False, "episode_id": truth["episode_id"], "ref": ref,
                                "closed_index": closed_index}
        return {"token": token, "files": {n: _b64(b) for n, b in public.items()}}

    def _grade(self, token: str, findings, inspection_steps: int, reason: str, policy: str | None) -> dict:
        from .reward import grade
        s = self.sessions.get(token)
        if s is None:
            return {"error": "unknown_token"}
        if s["graded"]:
            self.log({"event": "rejected_regrade", "token": token, "episode_id": s["episode_id"]})
            return {"error": "already_graded"}
        s["graded"] = True
        t0 = time.perf_counter()
        res = grade(findings, s["truth"], s["alias"], s["tolerance"], inspection_steps, s["known"])
        rec = {"event": "graded", "token": token, "episode_id": s["episode_id"], "mode": s["mode"],
               "reason": reason, "policy": policy, "ref": s["ref"], "closed_index": s["closed_index"],
               "n_truth": len(s["truth"]["issues"]), **res,
               "grade_ms": round((time.perf_counter() - t0) * 1000, 3)}
        if not res["valid"]:
            rec["event"] = "invalid_findings"
        self.results.append(rec)
        self.log(rec)
        return {"ok": True, "res": res, "mode": s["mode"]}

    def op_grade(self, token: str, findings, inspection_steps: int, reason: str, policy: str | None = None) -> dict:
        out = self._grade(token, findings, inspection_steps, reason, policy)
        if "error" in out:
            return out
        res = out["res"]
        if out["mode"] == "eval":
            return {"reward": 0.0, "metrics": None}
        return {"reward": res["reward"], "metrics": {k: res[k] for k in (
            "valid", "precision", "recall", "f1", "tp", "fp", "fn", "inspection_steps")}}

    def op_oracle(self, token: str) -> dict:
        """Oracle работает только здесь: эталон не покидает процесс оценщика."""
        from .reward import oracle_findings
        if not self.allow_oracle:
            self.log({"event": "rejected_oracle", "token": token})
            return {"error": "oracle_disabled"}
        s = self.sessions.get(token)
        if s is None:
            return {"error": "unknown_token"}
        out = self._grade(token, oracle_findings(s["truth"]), 0, "submit", "oracle")
        if "error" in out:
            return out
        return {"reward": out["res"]["reward"] if out["mode"] == "train" else 0.0,
                "n_findings": out["res"]["n_findings"]}

    def op_finalize(self) -> dict:
        self.finalized = True
        return {"ok": True, "n_results": len(self.results)}

    def op_results(self) -> dict:
        if not self.finalized:
            return {"error": "not_finalized"}
        return {"results": [{k: v for k, v in r.items() if k != "matches"} for r in self.results]}

    def op_stats(self) -> dict:
        """Только диагностика ресурсов (пик RSS процесса оценщика); ничего об эталоне/наградах."""
        from ..common.rss import self_peak_rss_mb
        return {"peak_rss_mb": self_peak_rss_mb()}


def serve(inp, out, settings: dict) -> None:
    """Цикл оценщика: одна JSON-строка запроса -> одна JSON-строка ответа."""
    core = GraderCore(settings)
    ops = {"open": core.op_open, "grade": core.op_grade, "oracle": core.op_oracle,
           "finalize": core.op_finalize, "results": core.op_results, "stats": core.op_stats}
    while True:
        line = inp.readline(MAX_MSG + 1)
        if not line:
            break
        if len(line) > MAX_MSG:
            res = {"error": "message_too_large"}
            # дочитываем остаток строки
            while line and not line.endswith(b"\n"):
                line = inp.readline(MAX_MSG)
        else:
            try:
                msg = json.loads(line.decode("utf-8"))
                op = msg.get("op") if isinstance(msg, dict) else None
                kw = msg.get("kw", {}) if isinstance(msg, dict) else None
                if op == "shutdown":
                    out.write(b'{"ok": true}\n')
                    out.flush()
                    break
                fn = ops.get(op)
                if fn is None or not isinstance(kw, dict):
                    res = {"error": "bad_op"}
                else:
                    res = fn(**kw)
            except TypeError:
                res = {"error": "bad_arguments"}
            except Exception as e:  # оценщик не падает и не отдаёт стектрейс наружу
                core.log({"event": "internal_error", "type": type(e).__name__})
                res = {"error": "internal"}
        out.write(json.dumps(res, ensure_ascii=True).encode("ascii") + b"\n")
        out.flush()


MAX_MSG = 16 * 1024 * 1024


class GraderClient:
    """Клиент в процессе среды. Оценщик — отдельный ОС-процесс (subprocess), общение —
    только строками JSON через stdin/stdout. Клиент не хранит ничего закрытого."""

    def __init__(self, episodes_root: str = "episodes", grader_root: str = "grader_data",
                 log_path: str | None = None, secret_file: str | None = None,
                 closed_episodes_file: str | None = None, allow_oracle: bool = False):
        settings = {"allow_oracle": bool(allow_oracle),"episodes_root": os.path.abspath(episodes_root), "grader_root": os.path.abspath(grader_root),
                    "log_path": os.path.abspath(log_path) if log_path else None,
                    "secret_file": os.path.abspath(secret_file) if secret_file else None,
                    "closed_episodes_file": os.path.abspath(closed_episodes_file) if closed_episodes_file else None}
        pkg_parent = str(Path(__file__).resolve().parents[2])
        env = {k: v for k, v in os.environ.items() if k not in ("AUDITENV_GRADER_SECRET",)}
        env["PYTHONPATH"] = pkg_parent
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "auditenv.grader.service", json.dumps(settings)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env, cwd=pkg_parent)

    def call(self, op: str, **kw) -> dict:
        self._proc.stdin.write(json.dumps({"op": op, "kw": kw}, ensure_ascii=True).encode("ascii") + b"\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            raise RuntimeError("grader process is not available")
        return json.loads(line.decode("ascii"))

    @property
    def pid(self) -> int:
        return self._proc.pid

    def close(self) -> None:
        if self._proc.poll() is None:
            try:
                self.call("shutdown")
            except Exception:
                pass
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        for f in (self._proc.stdin, self._proc.stdout):
            try:
                f.close()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        return "<GraderClient>"


def decode_files(files: dict[str, str]) -> dict[str, bytes]:
    return {n: base64.b64decode(v) for n, v in files.items()}


def main() -> None:
    settings = json.loads(sys.argv[1])
    proto_in, proto_out = sys.stdin.buffer, sys.stdout.buffer
    sys.stdout = sys.stderr  # любой случайный print не испортит протокол
    serve(proto_in, proto_out, settings)


if __name__ == "__main__":
    main()
