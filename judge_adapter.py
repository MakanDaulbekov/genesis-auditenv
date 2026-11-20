"""judge_adapter.py — единый адаптер жюри к среде auditenv.

Реализует интерфейс из «Genesis FinTech: определения проверок» (20.11.2026), раздел «Единый способ
подключения». Тонкая обёртка над кодом команды: генератор, среда (reset/step, 5 действий, лимиты) и
оценщик (matcher и общая награда раздела 05) — те же, что в `make check`; адаптер только переводит
оболочку действий и форму наблюдений.

    generate_episode(seed, config, output_dir)        -> None
    make_env()                                        -> объект с reset(seed, config) и step(action)
    score_episode(episode_dir, findings, inspection_steps) -> {"tp","fp","fn","precision","recall","f1","return"}
    validate_episode(episode_dir)                     -> отчёт независимого валидатора (дополнительно)

Запуск из корня репозитория (рабочая директория образа — /app):  `python -c "import judge_adapter"`.
Только stdlib, Python 3.11.

Граница доступа. Объект среды не читает и не разбирает truth/provenance: при reset с episode_dir открытые
файлы и закрытые файлы копируются в РАЗНЫЕ временные корни; закрытые читает только отдельный ОС-процесс
оценщика (auditenv.grader.service, общение строками JSON). В закрытом режиме reward = 0 на каждом шаге,
info = {}; истинная награда пишется в журнал оценщика (AUDITENV_JUDGE_LOG, по умолчанию
./judge_grader_log.jsonl). Секрет генератора (если нужен) — переменная AUDITENV_GRADER_SECRET.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auditenv.env.core import MAX_STEPS, TASK_TEXT, TIME_LIMIT_S, AuditEnv, _too_deep  # noqa: E402
from auditenv.env.store import parse_jsonl  # noqa: E402
from auditenv.env.tools import ACTION_ARGS  # noqa: E402
from auditenv.generator.config import resolve_config  # noqa: E402
from auditenv.generator.episode import generate_episode as _generate  # noqa: E402
from auditenv.grader.matcher import alias_map_from_tables, public_ids_from_files, structure_error  # noqa: E402
from auditenv.grader.reward import grade  # noqa: E402
from auditenv.grader.service import PUBLIC_FILES, GraderClient  # noqa: E402

PRIVATE_FILES = ("truth.json", "provenance.json")
TABLES = ("ledger", "invoices", "payments", "fx")
ERROR_TYPES = ("duplicate_invoice", "missing_or_partial_allocation", "fx_wrong_date_or_rate", "period_cutoff_error")

TASK = TASK_TEXT + (
    " Действия (оболочка {\"name\": str, \"args\": object}): list_records(table, cursor=null, limit<=50) -> "
    "{records, next_cursor}; get_record(table, id) -> запись; get_document(doc_id) -> документ; "
    "calculator(expression) -> десятичная строка; submit(findings) -> {accepted: true}. "
    "Таблицы: ledger, invoices, payments, fx (ID строки курса: fx_id = fx:<date>:<source>:<target>).")


def _log_path() -> str:
    return os.path.abspath(os.environ.get("AUDITENV_JUDGE_LOG", "judge_grader_log.jsonl"))


def _log(rec: dict) -> None:
    with open(_log_path(), "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, sort_keys=True, ensure_ascii=False) + "\n")


def _secret() -> str:
    return os.environ.get("AUDITENV_GRADER_SECRET", "")


def _team_config(config):
    """Принимает config генератора команды или обёртку {"team_config": {...}, ...}."""
    if isinstance(config, dict) and "team_config" in config:
        return config.get("team_config")
    return config


# ----------------------------------------------------------------------------- generate_episode

def generate_episode(seed: int, config: dict, output_dir: str) -> None:
    """Эпизод генератора команды в каталог жюри: 7 открытых файлов раздела 03 + truth.json и
    provenance.json. config — config генератора (частичный dict мержится с configs/default.json,
    см. README «Адаптер жюри» — рецепты D01–D20); None/{} — дефолт."""
    cfg = resolve_config(_team_config(config) or None)
    ep = _generate(seed, cfg, _secret())
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, data in {**ep.public, **ep.private}.items():
        (out / name).write_bytes(data)


# ----------------------------------------------------------------------------- make_env

def _obs(result, error, remaining: int) -> dict:
    return {"result": result, "error": error, "remaining_steps": int(remaining)}


class JudgeEnv:
    """reset(seed, config) -> (observation, info); step(action) -> (observation, reward, terminated, truncated, info).

    config: {"mode": "closed"|"train", "episode_dir": "<каталог>" (необязательно), "team_config": {...}}.
    С episode_dir загружается именно этот эпизод (ID не меняются, повторный reset начинает его заново);
    без него — генератор команды с seed и team_config."""

    def __init__(self, time_limit_s: float = TIME_LIMIT_S, max_steps: int = MAX_STEPS):
        self._time_limit_s = time_limit_s
        self._max_steps = max_steps
        self._tmp = Path(tempfile.mkdtemp(prefix="auditenv_judge_"))
        (self._tmp / "pub" / "j").mkdir(parents=True)
        (self._tmp / "grd" / "j").mkdir(parents=True)
        secret_file = None
        if _secret():
            secret_file = self._tmp / "secret.txt"
            secret_file.write_text(_secret(), encoding="utf-8")
        self._grader = GraderClient(str(self._tmp / "pub"), str(self._tmp / "grd"), log_path=_log_path(),
                                    secret_file=str(secret_file) if secret_file else None)
        self._env: AuditEnv | None = None
        self._episode_id: str | None = None
        self._n = 0

    # -- reset
    def _stage(self, episode_dir: str) -> str:
        src = Path(episode_dir)
        missing = [n for n in PUBLIC_FILES + ("truth.json",) if not (src / n).is_file()]
        if missing:
            raise ValueError(f"episode_dir: нет файлов {missing}")
        for old in list((self._tmp / "pub" / "j").iterdir()) + list((self._tmp / "grd" / "j").iterdir()):
            shutil.rmtree(old, ignore_errors=True)
        self._n += 1
        name = f"e{self._n}"
        pub, grd = self._tmp / "pub" / "j" / name, self._tmp / "grd" / "j" / name
        pub.mkdir()
        grd.mkdir()
        for n in PUBLIC_FILES:
            shutil.copyfile(src / n, pub / n)
        for n in PRIVATE_FILES:          # копируются байты; разбирает их только процесс оценщика
            if (src / n).is_file():
                shutil.copyfile(src / n, grd / n)
        return f"j/{name}"

    def reset(self, seed: int | None = None, config: dict | None = None, options: dict | None = None):
        cfg = config if config is not None else (options or {}).get("config") or {}
        if not isinstance(cfg, dict):
            raise ValueError("config: ожидается объект {mode, episode_dir?, team_config?}")
        mode = cfg.get("mode", "train")
        if mode not in ("closed", "train"):
            raise ValueError("config.mode: ожидается 'closed' или 'train'")
        self._env = AuditEnv(self._grader, mode="eval" if mode == "closed" else "train",
                             max_steps=self._max_steps, time_limit_s=self._time_limit_s)
        if cfg.get("episode_dir"):
            obs, _ = self._env.reset(config={"episode_ref": self._stage(cfg["episode_dir"])})
        else:
            obs, _ = self._env.reset(seed=seed, config=cfg.get("team_config"))
        r = obs["result"]
        self._episode_id = r["manifest"]["episode_id"]
        return {"task": TASK, "manifest": r["manifest"], "policy": r["policy"],
                "remaining_steps": obs["steps_remaining"]}, {}

    # -- step
    def step(self, action):
        if self._env is None:
            return _obs(None, {"code": "not_reset", "message": "Сначала вызовите reset."}, 0), 0.0, False, False, {}
        if isinstance(action, (str, bytes)):
            try:
                action = json.loads(action)
            except (ValueError, UnicodeDecodeError, RecursionError):
                action = {"__invalid_json__": True}
        name, core, submit_err = None, {"__invalid__": True}, None
        if isinstance(action, dict) and "name" in action and set(action) <= {"name", "args"}:
            name = action["name"]
            core = {"action": name, "args": action.get("args", {})}
            if name == "submit":
                args = action.get("args")
                if not isinstance(args, dict) or set(args) != {"findings"}:
                    submit_err = "submit args: ожидается {findings}"
                elif _too_deep(args["findings"]):
                    submit_err = "findings: слишком глубокая вложенность"
                else:
                    submit_err = structure_error(args["findings"], self._episode_id)
        elif isinstance(action, dict) and "__invalid_json__" in action:
            core = "\x00"                               # -> invalid_json в среде, шаг расходуется
        obs, reward, terminated, truncated, _ = self._env.step(core)
        result, error = obs["result"], obs["error"]
        if error is None and result is not None and isinstance(name, str) and name in ACTION_ARGS:
            if name == "list_records":
                result = {"records": result["records"], "next_cursor": result["next_cursor"]}
            elif name == "get_record":
                result = result["record"]
            elif name == "get_document":
                result = result["document"]
            elif name == "calculator":
                result = result["value"]
            elif name == "submit":
                if submit_err is None:
                    result = {"accepted": True}
                else:
                    result, error = None, {"code": "invalid_findings",
                                           "message": "Неверная структура findings: " + submit_err[:200]}
        return _obs(result, error, obs["steps_remaining"]), float(reward), bool(terminated), bool(truncated), {}

    def close(self) -> None:
        try:
            self._grader.close()
        finally:
            shutil.rmtree(self._tmp, ignore_errors=True)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        return "<JudgeEnv>"


def make_env() -> JudgeEnv:
    return JudgeEnv()


# ----------------------------------------------------------------------------- score_episode

def score_episode(episode_dir: str, findings: dict, inspection_steps: int) -> dict:
    """Закрытый вызов оценщика: тот же matcher и та же общая награда, что в среде.
    Валидный submit -> {"tp","fp","fn","precision","recall","f1","return"}; неверная структура ->
    return = -1, остальные поля null, ошибка — в журнале оценщика."""
    d = Path(episode_dir)
    truth = json.loads((d / "truth.json").read_text(encoding="utf-8"))
    public = {n: (d / n).read_bytes() for n in PUBLIC_FILES}
    alias = alias_map_from_tables(parse_jsonl(public["invoices.jsonl"]), parse_jsonl(public["payments.jsonl"]))
    tolerance = json.loads(public["manifest.json"])["rounding"]["tolerance_minor"]
    res = grade(findings, truth, alias, tolerance, inspection_steps, public_ids_from_files(public))
    _log({"event": "score_episode", "episode_id": truth.get("episode_id"), "valid": res["valid"],
          "error": res["error"], "tp": res["tp"], "fp": res["fp"], "fn": res["fn"], "reward": res["reward"],
          "reward_exact": res["reward_exact"], "inspection_steps": inspection_steps, "matches": res["matches"]})
    if not res["valid"]:
        return {"tp": None, "fp": None, "fn": None, "precision": None, "recall": None, "f1": None, "return": -1.0}
    return {k: res[k] for k in ("tp", "fp", "fn", "precision", "recall", "f1")} | {"return": res["reward"]}


def validate_episode(episode_dir: str) -> dict:
    """Независимый валидатор команды (не импортирует генератор): равенства до ошибки, объявленное
    расхождение после, достаточность каждого набора доказательств. Нужен provenance.json."""
    from auditenv.generator.validate import validate_episode_dir
    d = Path(episode_dir)
    return validate_episode_dir(d, d)
