"""Стабильный интерфейс политики для harness жюри.

    class Policy(Protocol):
        def reset(self, observation: dict) -> None   # первое наблюдение эпизода (результат env.reset)
        def act(self, observation: dict) -> dict     # текущее наблюдение -> действие {"action", "args"}

Цикл harness: policy.reset(obs0); a = policy.act(obs0); obs = env.step(a)[0]; a = policy.act(obs); …
до terminated/truncated. Политика видит только наблюдения.

Подключение без правки кода:
  * встроенные имена: random_explorer, simple_reconciler, competent_reconciler, flag_all,
    oracle (ответ строит процесс оценщика), oracle_from_truth (доверенный режим, см. ниже);
  * своя политика: "module.path:factory" — factory() возвращает объект с reset/act
    (класс или функция без аргументов); модуль ищется также в текущем каталоге;
  * политика на любом языке — отдельный процесс через auditenv.agent_host --agent-cmd.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from typing import Callable, Protocol, runtime_checkable

from . import baselines


@runtime_checkable
class Policy(Protocol):
    def reset(self, observation: dict) -> None: ...

    def act(self, observation: dict) -> dict: ...


class GeneratorPolicy:
    """Тонкий адаптер: политика-генератор (yield действие, send наблюдение) -> Policy."""

    def __init__(self, fn: Callable, **kwargs):
        self._fn, self._kw = fn, kwargs
        self._gen = None
        self._first = None
        self._episode_id = None

    def reset(self, observation: dict) -> None:
        self._episode_id = observation["result"]["manifest"]["episode_id"]
        self._gen = self._fn(observation, **self._kw)
        self._first = next(self._gen)

    def act(self, observation: dict) -> dict:
        if self._first is not None:
            a, self._first = self._first, None
            return a
        try:
            return self._gen.send(observation)
        except StopIteration:
            return {"action": "submit", "args": {"findings": {"episode_id": self._episode_id, "findings": []}}}


class OracleFromTruth:
    """Oracle жюри по НАШЕМУ эталону на диске: читает <truth_dir>/**/<episode_id>/truth.json и сразу
    отправляет верные выводы с первым допустимым набором доказательств через обычный submit.
    Только доверенный режим (harness оценщика): процессу агента эталон недоступен."""

    def __init__(self, truth_dir: str = "grader_data"):
        self.truth_dir = Path(truth_dir)
        self._findings = None

    def reset(self, observation: dict) -> None:
        eid = observation["result"]["manifest"]["episode_id"]
        hits = sorted(self.truth_dir.rglob(f"{eid}/truth.json"))
        if not hits:
            # каталог эпизода жюри может называться иначе, чем episode_id: ищем по содержимому (AUDIT.md, F6)
            hits = [p for p in sorted(self.truth_dir.rglob("truth.json"))
                    if json.loads(p.read_text(encoding="utf-8")).get("episode_id") == eid]
        if not hits:
            raise FileNotFoundError(f"oracle_from_truth: нет truth.json для {eid} в {self.truth_dir}")
        truth = json.loads(hits[0].read_text(encoding="utf-8"))
        self._findings = {"episode_id": eid, "findings": [
            {"type": t["type"], "affected_ids": t["affected_ids"], "claimed_amount_minor": t["expected_amount_minor"],
             "currency": t["currency"], "evidence_ids": t["evidence_sets"][0], "explanation": "oracle_from_truth"}
            for t in truth["issues"]]}

    def act(self, observation: dict) -> dict:
        return {"action": "submit", "args": {"findings": self._findings}}


BUILTIN: dict[str, Callable[[], Policy]] = {
    "random_explorer": lambda: GeneratorPolicy(baselines.random_explorer),
    "simple_reconciler": lambda: GeneratorPolicy(baselines.simple_reconciler),
    "competent_reconciler": lambda: GeneratorPolicy(baselines.competent_reconciler),
    "flag_all": lambda: GeneratorPolicy(baselines.flag_all),
}
TRUSTED_ONLY = {"oracle_from_truth"}


def load_policy(spec: str, trusted: bool = False, truth_dir: str = "grader_data") -> Policy:
    """Имя встроенной политики или "module.path:factory" -> объект Policy."""
    if spec in BUILTIN:
        return BUILTIN[spec]()
    if spec == "oracle_from_truth":
        if not trusted:
            raise PermissionError("oracle_from_truth читает эталон с диска: только в доверенном режиме (--trusted)")
        return OracleFromTruth(truth_dir)
    if spec == "oracle":
        raise ValueError("'oracle' исполняется процессом оценщика (runner), а не как объект Policy")
    if ":" not in spec:
        raise ValueError(f"неизвестная политика '{spec}': встроенные {sorted(BUILTIN)} или 'module.path:factory'")
    mod_name, attr = spec.split(":", 1)
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    factory = getattr(importlib.import_module(mod_name), attr)
    pol = factory()
    if not (callable(getattr(pol, "reset", None)) and callable(getattr(pol, "act", None))):
        raise TypeError(f"{spec}: factory() должна вернуть объект с методами reset(observation) и act(observation)")
    return pol
