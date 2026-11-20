"""AuditEnv: Gymnasium-подобный JSON-контракт reset/step (раздел 04).

reset(seed=None, config=None) -> (observation, info)
step(action) -> (observation, reward, terminated, truncated, info)

Все наблюдения имеют одну форму: {"result": ..., "error": null|{code,message}, "steps_remaining": int}.
Среда не генерирует эпизоды и не видит truth: эпизод открывает процесс оценщика и
передаёт сюда только открытые файлы. seed не хранится в объекте среды.

Режимы:
  train — терминальная награда и агрегированные метрики (P/R/F1/TP/FP/FN) возвращаются политике;
  eval  — политике всегда reward=0 и info={}; итоги у оценщика до finalize.
Промежуточная награда всегда 0. Подсказок о правильности по ходу нет.
"""
from __future__ import annotations

import copy
import json
import time

from ..grader.service import GraderClient, decode_files
from .store import ReadOnlyStore
from .tools import ACTION_ARGS, TOOLS_DESCRIPTION, ToolError, err, run_tool

MAX_STEPS = 100
TIME_LIMIT_S = 60.0
MAX_ACTION_BYTES = 512 * 1024
# Корректный findings имеет глубину 4 (объект → список → вывод → список ID). Предел с большим запасом
# отсекает вложенность до сериализации: иначе в узком окне глубины json.dumps здесь проходил, а запрос
# к оценщику падал RecursionError (AUDIT.md, F3).
MAX_FINDINGS_DEPTH = 32


def _too_deep(obj, limit: int = MAX_FINDINGS_DEPTH) -> bool:
    """Вложенность контейнеров глубже limit. Итеративно: сама проверка не рекурсивна."""
    stack = [(obj, 1)]
    while stack:
        x, d = stack.pop()
        if isinstance(x, (dict, list, tuple)):
            if d > limit:
                return True
            stack.extend((v, d + 1) for v in (x.values() if isinstance(x, dict) else x))
    return False

TASK_TEXT = (
    "Вы — аудитор. Изучите открытые записи компании за отчётный период с помощью инструментов и "
    "отправьте findings.json с найденными ошибками и доказательствами. Типы ошибок: duplicate_invoice, "
    "missing_or_partial_allocation, fx_wrong_date_or_rate, period_cutoff_error. Правила (курс, период, "
    "округление, распределение, канонические affected_ids и суммы) — в policy. Если ошибок нет, "
    "отправьте пустой список findings: ложные обвинения штрафуются. Бюджет — 100 шагов, включая submit; "
    "каждый шаг до submit немного уменьшает награду."
)


class EpisodeOpenError(ValueError):
    """reset не смог открыть эпизод; сообщение называет причину (поле config, seed, ref)."""


class AuditEnv:
    metadata = {"render_modes": []}

    def __init__(self, grader: GraderClient, mode: str = "train", max_steps: int = MAX_STEPS,
                 time_limit_s: float = TIME_LIMIT_S, clock=time.monotonic, policy_name: str | None = None):
        if mode not in ("train", "eval"):
            raise ValueError("mode must be 'train' or 'eval'")
        self.__grader = grader
        self.mode = mode
        self.max_steps = max_steps
        self.time_limit_s = time_limit_s
        self._clock = clock
        self._policy_name = policy_name
        self._auto_seed = 0
        self._reset_state()

    def _reset_state(self) -> None:
        self._store = None
        self._token = None
        self._episode_id = None
        self._steps = 0
        self._done = True
        self._t0 = None

    # ------------------------------------------------------------------ helpers
    def _obs(self, result=None, error=None) -> dict:
        return {"result": result, "error": error, "steps_remaining": max(0, self.max_steps - self._steps)}

    def _finish(self, findings, reason: str) -> tuple[float, dict]:
        self._done = True
        inspection = self._steps - 1 if reason == "submit" else self._steps
        out = self.__grader.call("grade", token=self._token, findings=findings,
                                 inspection_steps=inspection, reason=reason, policy=self._policy_name)
        if self.mode == "eval" or "error" in out:
            return 0.0, {}
        return float(out["reward"]), {"metrics": out["metrics"]}

    # ------------------------------------------------------------------ API
    def reset(self, seed: int | None = None, config: dict | None = None, options: dict | None = None):
        """Открыть эпизод.

        config:
          * None или {} — configs/default.json (config_version default-1.0.0);
          * частичный dict генератора — глубоко мержится поверх дефолта ({"clean": true} и т.п.);
          * {"episode_ref": "dev/ep_..."} — готовый эпизод с диска оценщика;
          * {"closed_index": i} — i-й закрытый эпизод (только eval).
        seed: int. Если None — берётся счётчик этого объекта среды (0, 1, 2, …): поведение
        детерминировано, но для воспроизводимости лучше передавать seed явно.
        Ошибки: EpisodeOpenError (подкласс ValueError) с причиной — какое поле и что ожидалось.
        """
        self._reset_state()
        cfg = config if config is not None else (options or {}).get("config")
        if seed is None and not (isinstance(cfg, dict) and ({"episode_ref", "closed_index"} & set(cfg))):
            seed = self._auto_seed
            self._auto_seed += 1
        if isinstance(cfg, dict) and "episode_ref" in cfg:
            out = self.__grader.call("open", mode=self.mode, ref=cfg["episode_ref"])
        elif isinstance(cfg, dict) and "closed_index" in cfg:
            out = self.__grader.call("open", mode=self.mode, closed_index=cfg["closed_index"])
        else:
            out = self.__grader.call("open", mode=self.mode, seed=seed, config=cfg)
        if "error" in out:
            raise EpisodeOpenError(f"эпизод не открыт: {out.get('detail') or out['error']}")
        self._store = ReadOnlyStore(decode_files(out["files"]))
        self._token = out["token"]
        manifest = json.loads(self._store.manifest)
        self._episode_id = manifest["episode_id"]
        self._done = False
        self._t0 = self._clock()
        result = {"task": TASK_TEXT, "manifest": manifest, "policy": json.loads(self._store.policy),
                  "tools": copy.deepcopy(TOOLS_DESCRIPTION),
                  "budget": {"max_steps": self.max_steps, "time_limit_s": self.time_limit_s}}
        return self._obs(result=result), {}

    def step(self, action):
        if self._store is None:
            return self._obs(error=err("not_reset")), 0.0, False, False, {}
        if self._done:
            # повторный submit / шаг после завершения: шаг не тратится, оценка не меняется
            return self._obs(error=err("episode_finished")), 0.0, True, False, {}

        # лимит времени проверяется до выполнения действия
        if self._clock() - self._t0 > self.time_limit_s:
            reward, info = self._finish({"episode_id": self._episode_id, "findings": []}, "truncated_time")
            return self._obs(), reward, False, True, info

        self._steps += 1
        result, error, submitted = None, None, False
        parsed = self._parse(action)
        if isinstance(parsed, dict) and "__error__" in parsed:
            error = err(parsed["__error__"])
        else:
            name, args = parsed
            if name == "submit":
                if not isinstance(args, dict) or set(args) != {"findings"}:
                    findings = {"__invalid__": "bad_submit_args"}
                else:
                    findings = args["findings"]
                    try:
                        if _too_deep(findings):
                            findings = {"__invalid__": "too_deep"}
                        elif len(json.dumps(findings)) > MAX_ACTION_BYTES:
                            findings = {"__invalid__": "too_large"}
                    except (TypeError, ValueError, RecursionError):
                        findings = {"__invalid__": "not_serializable"}
                reward, info = self._finish(findings, "submit")
                return self._obs(result={"status": "submitted"}), reward, True, False, info
            try:
                result = run_tool(self._store, name, args)
            except ToolError as e:
                error = err(e.code)

        if self._steps >= self.max_steps:
            reward, info = self._finish({"episode_id": self._episode_id, "findings": []}, "truncated_steps")
            return self._obs(result=result, error=error), reward, False, True, info
        return self._obs(result=result, error=error), 0.0, False, False, {}

    def oracle_step(self):
        """ТОЛЬКО для harness: ответ oracle строит процесс оценщика по эталону; эталон сюда
        не передаётся, возвращается лишь награда (train) или 0 (eval)."""
        if self._store is None or self._done:
            return self._obs(error=err("episode_finished")), 0.0, True, False, {}
        self._steps += 1
        self._done = True
        out = self.__grader.call("oracle", token=self._token)
        reward = 0.0 if self.mode == "eval" or "error" in out else float(out["reward"])
        return self._obs(result={"status": "submitted"}), reward, True, False, {}

    def force_truncate(self, reason: str = "harness_abort"):
        """ТОЛЬКО для harness: досрочно закрыть эпизод (напр. упал процесс агента).
        Оценивается пустой ответ, как при truncation; причина пишется в журнал оценщика."""
        if self._store is None or self._done:
            return self._obs(error=err("episode_finished")), 0.0, True, False, {}
        reward, info = self._finish({"episode_id": self._episode_id, "findings": []}, f"truncated_{reason}")
        return self._obs(), reward, False, True, info

    def _parse(self, action):
        if isinstance(action, (str, bytes)):
            if len(action) > MAX_ACTION_BYTES:
                return {"__error__": "invalid_json"}
            try:
                action = json.loads(action)
            except (ValueError, UnicodeDecodeError, RecursionError):
                return {"__error__": "invalid_json"}
        if not isinstance(action, dict) or set(action) - {"action", "args"} or "action" not in action:
            return {"__error__": "invalid_action_format"}
        name = action["action"]
        args = action.get("args", {})
        if not isinstance(name, str) or name not in ACTION_ARGS:
            return {"__error__": "unknown_action"}
        if not isinstance(args, dict):
            return {"__error__": "invalid_arguments"}
        return name, args

    def close(self) -> None:
        self._reset_state()

    def __repr__(self) -> str:
        return f"<AuditEnv mode={self.mode}>"
