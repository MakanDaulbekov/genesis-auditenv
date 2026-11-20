# auditenv — воспроизводимая RL-среда для ИИ-аудитора

Среда генерирует синтетические компании с внедрёнными финансовыми ошибками. Агент расследует их через `reset/step` (5 узких инструментов) и отправляет findings с доказательствами. Оценщик работает в отдельном процессе и выдаёт общую награду раздела 05. Python 3.11, только stdlib, без сети и GPU.

```bash
make check                        # без make: python3.11 -m auditenv.cli check   (≈2 мин, 2 CPU)
```
Команда печатает по строке `[OK]/[FAIL]` на шаг и в конце `== check: OK ==`; при любом провале выход ненулевой. Шаги:
1. генерация 10 train + 10 dev эпизодов;
2. независимый валидатор;
3. 300 случайных seed;
4. ≈200 тестов;
5. 5 политик и `trajectories.jsonl`;
6. решаемость на 310 эпизодах;
7. детерминизм в 2 процессах;
8. самопроверка 80/80;
9. `REPORT.md`.

| Что | Где |
|---|---|
| эпизоды: открытая часть / эталон оценщика | `episodes/{train,dev}/<id>/` / `grader_data/{train,dev}/<id>/` |
| метрики, решаемость, разбор сбоев | [REPORT.md](REPORT.md) |
| устройство, допущения, честные ограничения | [DESIGN.md](DESIGN.md) |
| требование PDF → файл → тест | [CHECKLIST.md](CHECKLIST.md) |
| режим жюри на ваших seed и политиках | `make jury-run` (ниже) |

---

## Для жюри: как подключить свои политики

**Интерфейс** (`auditenv/policies/api.py`):
```python
class Policy(Protocol):
    def reset(self, observation: dict) -> None   # первое наблюдение эпизода (результат env.reset)
    def act(self, observation: dict) -> dict     # текущее наблюдение -> {"action": ..., "args": {...}}
```
Цикл: `reset(obs0)`, затем `a = act(obs0)`, `obs = env.step(a)[0]`, `a = act(obs)` и так далее до `terminated` или `truncated`.

**Пример** (`examples/my_policy.py`):
```python
class SilentAuditor:
    def reset(self, observation):
        self.episode_id = observation["result"]["manifest"]["episode_id"]
        self.pages = 0

    def act(self, observation):
        res = observation["result"] or {}
        if self.pages and res.get("next_cursor") is None or self.pages >= 3:
            return {"action": "submit", "args": {"findings": {"episode_id": self.episode_id, "findings": []}}}
        cursor = res.get("next_cursor") if self.pages else None
        self.pages += 1
        return {"action": "list_records", "args": {"table": "invoices", "cursor": cursor, "limit": 50}}
```

**Запуск одной командой:**
```bash
# 1) в процессе среды (factory() -> объект с reset/act; модуль ищется и в текущем каталоге)
python -m auditenv.cli run --policies your.module:YourPolicy --splits dev
# 2) отдельным процессом (жёсткая граница: у политики нет дескриптора оценщика)
python -m auditenv.agent_host --policy your.module:YourPolicy --split dev
python -m auditenv.agent_host --agent-cmd "python your_agent.py" --split dev   # любой язык, JSON stdin/stdout
# 3) полный режим жюри на 20 ваших скрытых эпизодах (см. следующий раздел)
make jury-run SPLIT=closed_split.json SECRET=secret.txt POLICY="your.module:A your.module:B"
```
* Встроенные политики: `random_explorer`, `simple_reconciler`, `competent_reconciler`, `flag_all`, `oracle`. Ответ `oracle` строит процесс оценщика.
* **Свой oracle по нашему эталону:** `python -m auditenv.cli run --trusted --policies oracle_from_truth --truth-dir grader_data/dev --splits dev`. Он читает `truth.json` с диска и отправляет ответ обычным `submit`. Разрешён только с `--trusted`, а в `agent_host` запрещён.
* **Протокол `agent_host`** (по строке JSON):
  * в агента приходит `{"type":"observation","observation","reward","terminated","truncated","info"}`, в конце прогона — `{"type":"close"}`;
  * агент отвечает действием.
* **Как ведёт себя `agent_host`:**
  * агент запускается в текущем каталоге, поэтому относительные пути работают; `python` в начале команды заменяется текущим интерпретатором;
  * stderr агента пишется в `grader_data/agent_stderr.log`;
  * если агент упал, текущий и оставшиеся эпизоды оцениваются пустым ответом, выход — код 2 с сообщением;
  * если оценённых эпизодов меньше, чем конфигов, выход — код 3.

## Режим жюри: ваши seed, параметры, секрет и политики

```bash
make jury-run SPLIT=closed_split.json [SECRET=secret.txt] [POLICY="your.module:Factory ..."]
# = python -m auditenv.jury_run --split closed_split.json --secret-file secret.txt --policies ...
```
`closed_split.json` (пример — `examples/closed_split_example.json`, 20 эпизодов: 5 чистых, 5 со взаимодействиями):
```json
{"split": "hidden", "config_version": "hidden-1.0.0",
 "episodes": [{"seed": 123456789, "overrides": {"clean": true}},
              {"seed": 42, "overrides": {"interactions": {"partial_fx_boundary": 1}, "currencies": {"base": ["JPY"]}}}]}
```
`overrides` — любой частичный config (поля `configs/default.json`), который мержится с дефолтом. Неверное поле даёт понятную ошибку с его именем.

Шаги внутри:
1. генерация: открытые записи пишутся в `jury_out/episodes/`, **эталон — в `jury_out/grader/`**, отдельный корень, чтобы вы могли независимо проверить эталон и доказательства по записям; рядом с эталоном каждого эпизода лежит `before/` — открытые таблицы в состоянии **до** внесения ошибок, в том же формате раздела 03;
2. независимый валидатор;
3. политики в режиме eval, `oracle` добавляется всегда;
4. 20 парных проверок: повтор seed/действий побайтово, перестановка и смена ID через `auditenv.transforms`;
5. таблица P/R/F1/возврат/шаги/время по политикам — для всех, чистых и взаимодействующих эпизодов. Она пишется в `jury_out/jury_report.md`.

**Независимая проверка эталона без нашего кода.**
* `jury_out/grader/<split>/<episode>/truth.json`: `issue_id`, `type`, канонические `affected_ids`, `expected_amount_minor`, `currency`, `evidence_sets` (любой полный набор достаточен) и `interaction_group`;
* `provenance.json`: причинные мутации «было → стало» и дифф открытых записей;
* `before/`: состояние до ошибок. Для открытых train/dev то же самое делает `python -m auditenv.cli export-before --refs dev/ep_… --out before_state`;
* правила, по которым считается расхождение (дата курса, граница периода, округление, распределение, требуемые доказательства), агент получает в `policy.json` первого наблюдения;
* наш независимый валидатор: `python -m auditenv.cli validate --splits <split> --episodes-root jury_out/episodes --grader-root jury_out/grader`.

**Секрет.** Генератор открыт, поэтому без секрета маленькие закрытые seed можно перебрать и получить эталон. Секрет входит в ключ эпизода, и это закрывает перебор. Без `--secret-file` выводится предупреждение и используется открытый детерминированный секрет по умолчанию, так что повтор seed остаётся точным.

## Контракт среды

```python
from auditenv.grader.service import GraderClient   # оценщик = отдельный ОС-процесс
from auditenv.env.core import AuditEnv

with GraderClient() as grader:
    env = AuditEnv(grader, mode="train")            # "eval": награда/диагностика скрыты до finalize
    obs, info = env.reset(seed=42)                  # config=None -> configs/default.json; частичный config мержится
    obs, reward, terminated, truncated, info = env.step({"action": "list_records", "args": {"table": "ledger"}})
```
**Наблюдение и `reset`:**
* наблюдение всегда имеет вид `{"result", "error": null | {"code", "message"}, "steps_remaining"}`;
* `config` для `reset` может быть: `None` / `{}` / частичный dict / `{"episode_ref": "dev/ep_…"}` / `{"closed_index": i}`. Ошибка открытия — `EpisodeOpenError` с причиной.

**Действия:**
* `list_records(table, cursor=None, limit=50)` — `cursor` и `limit` необязательны, лишние аргументы дают `invalid_arguments`;
* `get_record(table, id)`;
* `get_document(doc_id)`;
* `calculator(expression)`;
* `submit(findings)`.

Таблицы: `ledger`, `invoices`, `payments`, `fx`. В каждой строке `fx` есть вычисляемое поле `fx_id` = `fx:<date>:<src>:<tgt>`: это ID доказательства. Сам `fx.csv` не меняется.

**Лимиты и награда:**
* 100 шагов включая submit и 60 с на эпизод; промежуточная награда 0; при truncation оценивается пустой ответ;
* `findings`: ≤30 выводов, ≤10 ID доказательств, explanation ≤1000 символов; неверная структура даёт −1;
* награда — `clip(2*F1 − 1 − 0.002*inspection_steps, −1, 1)`;
* сопоставление и допуски описаны в `grader/matcher.py` и в `policy.matching` каждого эпизода;
* какие ID обязаны быть в `evidence_ids` для каждого типа, сказано в `policy.matching.required_evidence`. Например, для дубля нужны оба счёта и строка выпуска **каждого** из них, потому что одна строка доказывает только одну проводку.

## Docker

```bash
make docker-pin      # docker pull python:3.11-slim; закрепляет digest в docker/base-image.txt (закоммитьте)
make docker-check    # docker build --build-arg PYTHON_IMAGE=<digest> + docker run --network none --cpus 2 --memory 4g
# без make:
docker build --build-arg PYTHON_IMAGE="$(cat docker/base-image.txt)" -t auditenv .
docker run --rm --network none --cpus 2 --memory 4g auditenv      # CMD = python -m auditenv.cli check
```
* pytest и его зависимости лежат в `vendor/wheels` и ставятся по SHA-256 из `requirements-dev.lock` с `--no-index`, так что сборке нужна сеть только для базового образа.
* В образе нет `make`: `CMD` вызывает проверку напрямую.
* Чтобы обновить digest, выполните `make docker-pin` (по умолчанию тег `python:3.11-slim`; другой тег — `sh docker/pin_base_image.sh python:3.11.X-slim`).

### Проверено после доработок (20.11.2026, без Docker)
Windows 11 x64, Python 3.12.11 (msys2), 12 логических CPU; `python -m auditenv.cli check`:
```
[OK] generate train                                  1.5s  [generate] train/ep_7c609c9a4771e31a: issues=5 rows=700
[OK] generate dev                                    1.5s  [generate] dev/ep_053daf06ab62f121: issues=3 rows=446
[OK] validate episodes                               5.3s  [validate] dev/ep_053daf06ab62f121: OK issues=3 checks=67
[OK] validate 300 random seeds                     300.5s  [validate-seeds] 300/300 OK, issues=1880, clean=50
[OK] pytest                                        546.9s  206 passed in 545.87s (0:09:05)
[OK] run baseline policies (+competent)              9.7s  }
[OK] solvability: competent_reconciler, 300 seeds + worst size  159.6s  [solvability] competent_reconciler: 310/310 эпизодов с F1=1 и без FP; TP=1963 FP=0 FN=0; шаги max=73 mean=41.6
[OK] trajectory leak scan                            2.5s  1 passed, 7 deselected in 1.54s
[OK] determinism (2 processes, PYTHONHASHSEED)       8.3s  [determinism] files=182 identical_across_processes=True
[OK] self-check 4x20 (не оценка жюри)               76.9s  [self-check] Не покрыто самопроверкой: harness жюри, их политики, их преобразования и закрытые seed.
[OK] report                                          0.5s  [report] REPORT.md: 108 строк; simple_reconciler FP=346 FN=27
== check: OK ==
```
Машина медленнее Linux-песочницы (≈6× на генерации), отсюда время. Эпизоды и все числа (issues=1880, TP=1963, 80/80) совпадают с прогонами на Linux/3.11. Два блока ниже сняты 19.11 **до** доработок; Dockerfile и digest с тех пор не менялись, но после изменений кода `make docker-check` стоит прогнать заново.

### Проверено командой
```
дата: 19-11-2026   машина: Windows x64, Docker Desktop   docker --version: 29.9.3
digest: python@sha256:07d4824036af339fa1b6ef769cc99d9c10a141769538b52b3b597c84f7a7b535

$ docker build --build-arg PYTHON_IMAGE=python@sha256:07d4824036af339fa1b6ef769cc99d9c10a141769538b52b3b597c84f7a7b535 -t auditenv .
$ docker run --rm --network none --cpus 2 --memory 4g auditenv
[OK] generate train                                  0.5s  [generate] train/ep_7c609c9a4771e31a: issues=5 rows=700
[OK] generate dev                                    0.5s  [generate] dev/ep_053daf06ab62f121: issues=3 rows=446
[OK] validate episodes                               1.6s  [validate] dev/ep_053daf06ab62f121: OK issues=3 checks=67
[OK] validate 300 random seeds                      28.8s  [validate-seeds] 300/300 OK, issues=1880, clean=50
[OK] pytest                                         69.6s  192 passed in 69.15s (0:01:09)
[OK] run baseline policies (+competent)              3.5s
[OK] solvability: competent_reconciler, 300 seeds + worst size   16.7s  310/310 эпизодов с F1=1 и без FP; TP=1963 FP=0 FN=0; шаги max=73 mean=41.6
[OK] trajectory leak scan                            0.9s
[OK] determinism (2 processes, PYTHONHASHSEED)       2.0s  files=182 identical_across_processes=True
[OK] self-check 4x20 (не оценка жюри)               14.5s  80/80
[OK] report                                          0.3s
== check: OK ==
```
Тот же результат (те же episode_id, issues=1880, TP=1963) получен на Linux вне Docker на Python 3.11: генерация детерминирована и между машинами.

### Проверено в песочнице разработки
Чистый `git clone` + свежий venv Python 3.11.18 (`pip install --no-index --find-links vendor/wheels --require-hashes -r requirements-dev.lock`), 2 vCPU, 19.11.2026; `make check`:
```
[OK] generate train                                  0.3s  [generate] train/ep_7c609c9a4771e31a: issues=5 rows=700
[OK] generate dev                                    0.3s  [generate] dev/ep_053daf06ab62f121: issues=3 rows=446
[OK] validate episodes                               1.0s  [validate] dev/ep_053daf06ab62f121: OK issues=3 checks=67
[OK] validate 300 random seeds                      25.9s  [validate-seeds] 300/300 OK, issues=1880, clean=50
[OK] pytest                                         58.4s  192 passed in 58.08s
[OK] run baseline policies (+competent)              2.1s  }
[OK] solvability: competent_reconciler, 300 seeds + worst size   16.5s  [solvability] competent_reconciler: 310/310 эпизодов с F1=1 и без FP; TP=1963 FP=0 FN=0; шаги max=73 mean=41.6
[OK] trajectory leak scan                            0.7s  1 passed, 7 deselected in 0.46s
[OK] determinism (2 processes, PYTHONHASHSEED)       1.2s  [determinism] files=182 identical_across_processes=True
[OK] self-check 4x20 (не оценка жюри)                8.9s  [self-check] Не покрыто самопроверкой: harness жюри, их политики, их преобразования и закрытые seed.
[OK] report                                          0.1s  [report] REPORT.md: 108 строк; simple_reconciler FP=346 FN=27
== check: OK ==
real	1m55.399s
```
После прогона в рабочем дереве меняется только `REPORT.md` (замеры времени и RSS); эпизоды и `trajectories.jsonl` совпадают с закоммиченными побайтово.

Docker в песочнице: Docker Hub закрыт, поэтому образ собран на локальной базе Python 3.11 (rootfs песочницы) — `docker run --rm --network none --cpus 2 --memory 4g auditenv` → `== check: OK ==` за 107 с под непривилегированным пользователем.

## Структура

```
auditenv/common     деньги (Decimal), время, ГПСЧ SHA-256, ID, мини-валидатор JSON Schema
auditenv/schemas    JSON Schema: manifest, records, fx_record, document, action, observation, findings, truth, policy
auditenv/generator  config (дефолт/слияние/валидация) → world → render → mutations → truth; validate.py — независимый валидатор
auditenv/env        AuditEnv, read-only store и курсоры, инструменты, калькулятор
auditenv/grader     matcher, reward, процесс оценщика
auditenv/policies   api.py (Policy, load_policy, oracle_from_truth), baselines.py, process_adapter.py
auditenv/{transforms,runner,agent_host,jury_run,report,demo,cli,checks}.py
configs/            default.json + splits/{train,dev}.json (config_version)
examples/           my_policy.py, stdin_agent.py, closed_split_example.json
tests/              контракт, награда/атаки, данные/MI, парные, изоляция, agent_host, reset/config, API политик, jury-run
tests/jury          самопроверка 4×20 (зеркало раздела 10; не оценка жюри)
vendor/wheels       pytest + зависимости (офлайн-установка)
```
Лицензия MIT.
