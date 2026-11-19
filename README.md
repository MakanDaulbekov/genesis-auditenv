# auditenv — воспроизводимая RL-среда для ИИ-аудитора

GENESIS Hackathon // FinTech. Генератор синтетических компаний с внедрёнными финансовыми ошибками, контракт `reset/step` в стиле Gymnasium, узкие инструменты расследования, открытый matcher и общая награда, оценщик в отдельном процессе, 4 базовые политики, траектории, тесты.

* Python 3.11, рантайм — только стандартная библиотека; dev-зависимость — `pytest==8.3.4` (`requirements-dev.lock`). Лицензия MIT.
* Без сети, без платных API, без GPU. Полная проверка занимает ≈ 1 минуту на 2 CPU; эпизод — десятки миллисекунд.
* Устройство и допущения описаны в [DESIGN.md](DESIGN.md), измеренный разбор сбоев — в [REPORT.md](REPORT.md), соответствие требованиям PDF — в [CHECKLIST.md](CHECKLIST.md).

## Запуск одной командой

```bash
make check                 # или без make: python3.11 -m auditenv.cli check
```
Шаги проверки:
1. генерирует 10 train + 10 dev эпизодов;
2. валидирует их независимым валидатором;
3. прогоняет 300 случайных seed × 6 сочетаний параметров;
4. запускает все тесты;
5. гоняет 4 базовые политики и пишет `trajectories.jsonl`;
6. проверяет траектории на утечки;
7. сравнивает SHA-256 генерации в двух процессах с разным `PYTHONHASHSEED`;
8. запускает теневое жюри 4×20;
9. пишет `REPORT.md`.

При любом провале выход ненулевой.

Контейнер:
```bash
docker build -t auditenv .
docker run --rm --network none auditenv          # = python -m auditenv.cli check
```
Другие цели: `make demo` (один эпизод с логом шагов), `make jury`, `make report`, `make test`.

## Контракт среды

```python
from auditenv.grader.service import GraderClient   # оценщик = отдельный ОС-процесс
from auditenv.env.core import AuditEnv

with GraderClient() as grader:
    env = AuditEnv(grader, mode="train")            # "eval": награда и диагностика скрыты до finalize
    obs, info = env.reset(seed=42, config=cfg)      # или config={"episode_ref": "dev/ep_..."}
    obs, reward, terminated, truncated, info = env.step(
        {"action": "list_records", "args": {"table": "ledger", "cursor": None, "limit": 50}})
```
* Наблюдение: `{"result": ..., "error": null | {"code", "message"}, "steps_remaining": int}`. Все значения JSON-совместимы.
* Действия:
  * `list_records(table, cursor, limit≤50)`;
  * `get_record(table, id)`;
  * `get_document(doc_id)`;
  * `calculator(expression)`;
  * `submit(findings)`.
  Таблицы: `ledger`, `invoices`, `payments`, `fx`.
* Лимиты: 100 шагов включая submit и 60 с на эпизод. Промежуточная награда всегда 0. При truncation оценивается пустой ответ.
* `findings.json`: `{episode_id, findings: [{type, affected_ids, claimed_amount_minor, currency, evidence_ids, explanation}]}`. Лимиты: ≤30 выводов, ≤10 ID доказательств, explanation ≤1000 символов. Неверная структура → −1.
* Награда: `clip(2*F1 − 1 − 0.002*inspection_steps, −1, 1)`. Правила сопоставления и допуски записаны в `policy.matching` каждого эпизода и в `grader/matcher.py`.

## Данные эпизода

`episodes/<split>/<episode_id>/` — открытая часть: `manifest.json`, `policy.json`, `ledger.csv`, `invoices.jsonl`, `payments.jsonl`, `fx.csv`, `documents.jsonl`. Деньги хранятся только как целые minor units, курсы — десятичными строками, время — ISO-8601 со смещением, отсутствующие ссылки — `null`. Схемы лежат в `auditenv/schemas/`.

`grader_data/<split>/<episode_id>/` — закрытая часть: `truth.json`, `provenance.json`. Среда её не видит.

## Как организаторам подставить закрытые seed и новые сочетания параметров (без правки кода)

1. Создайте `closed.json` — список `{"seed": <int>, "config": {...}}`. `config` — любой вариант `configs/default.json`: включение и счётчики типов ошибок, сценарии взаимодействий, `clean`, размеры, валюты, доли отвлекающих случаев. Также укажите `config_version`.
2. Создайте файл секрета, доступный только оценщику (без него маленькие seed перебираются открытым генератором).
3. Запуск внешнего агента отдельным процессом (только JSON через stdin/stdout):
   ```bash
   python -m auditenv.agent_host --closed-episodes-file closed.json --secret-file secret.txt \
       --mode eval --agent-cmd "python my_agent.py"
   ```
   или из своего кода: `GraderClient(closed_episodes_file=..., secret_file=...)` и `env.reset(config={"closed_index": i})`.
4. Безвредные преобразования делаются через `auditenv.transforms.transform_episode(public, truth, provenance, seed)`: перестановка строк и биективная смена всех ID, согласованно в эталоне.

Пример внешнего агента — `examples/stdin_agent.py`.

## Базовые политики

`random_explorer` (фиксированный seed), `simple_reconciler` (честная скриптовая сверка), `flag_all` (атака) и `oracle`, ответ которого строится только в процессе оценщика. Результаты — в REPORT.md.

## Структура

```
auditenv/common     деньги (Decimal), время, ГПСЧ SHA-256, ID, мини-валидатор JSON Schema
auditenv/schemas    JSON Schema: manifest, records, document, action, observation, findings, truth, policy
auditenv/generator  world → render → mutations → truth; validate.py — независимый валидатор
auditenv/env        AuditEnv, read-only store и курсоры, инструменты, калькулятор
auditenv/grader     matcher, reward, процесс оценщика
auditenv/policies   базовые политики
auditenv/transforms.py, runner.py, agent_host.py, report.py, demo.py, cli.py, checks.py
configs/            default.json + splits/{train,dev}.json (config_version)
tests/              контракт, награда/атаки, данные, парные, изоляция, политики; tests/jury — теневое жюри 4×20
```
