# auditenv — воспроизводимая RL-среда для ИИ-аудитора

Среда генерирует синтетические компании с внедрёнными финансовыми ошибками. Агент расследует их через `reset/step` (5 узких инструментов) и отправляет findings с доказательствами. Оценщик работает в отдельном процессе и выдаёт общую награду раздела 05. Python 3.11, только stdlib, без сети и GPU.

```bash
python3.11 -m pip install --no-index --find-links vendor/wheels --require-hashes -r requirements-dev.lock  # pytest офлайн (в venv), 1 раз
make check                        # без make: python3.11 -m auditenv.cli check   (≈2.5 мин, 2 CPU)
```
Команда печатает по строке `[OK]/[FAIL]` на шаг и в конце `== check: OK ==`; при любом провале выход ненулевой. Шаги:
1. генерация 10 train + 10 dev эпизодов;
2. независимый валидатор;
3. 300 случайных seed;
4. ≈260 тестов (из них 40 — адаптер жюри по определениям проверок);
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
| **адаптер жюри** (определения проверок 20.11) | [`judge_adapter.py`](judge_adapter.py), раздел ниже, тесты `tests/test_judge_adapter.py` |
| режим жюри на ваших seed и политиках | `make jury-run` (ниже) |

---

## Адаптер жюри `judge_adapter.py` (определения проверок от 20.11.2026)

Файл `judge_adapter.py` лежит в корне репозитория. **Рабочая директория** — корень репозитория (`/app` в образе). Импорт: `import judge_adapter as ja`, путь к пакету адаптер добавляет сам. В образе:
```bash
docker run --rm --network none auditenv python -c "import judge_adapter as ja; print(ja.make_env())"
docker run --rm --network none auditenv python -m pytest -q tests/test_judge_adapter.py   # C01–C20, T01–T20, M01–M12, P, D
```
```python
import judge_adapter as ja
ja.generate_episode(seed, config, output_dir)   # 7 открытых файлов §03 + truth.json + provenance.json в output_dir
env = ja.make_env()
obs, info = env.reset(seed, {"mode": "closed", "episode_dir": output_dir})        # или "mode": "train"
obs, reward, terminated, truncated, info = env.step({"name": "list_records", "args": {"table": "ledger", "cursor": None, "limit": 50}})
ja.score_episode(output_dir, findings, inspection_steps)  # {"tp","fp","fn","precision","recall","f1","return"}
ja.validate_episode(output_dir)                  # дополнительно: наш независимый валидатор (нужен provenance.json)
```
* **`generate_episode(seed, config, output_dir)`** — обёртка над генератором команды. `config` — config генератора: частичный dict мержится с `configs/default.json`; принимается и обёртка `{"team_config": {...}}`. Повтор с теми же seed и config даёт побайтово те же файлы.
* **`reset(seed, config)`**, где `config = {"mode": "closed"|"train", "episode_dir"?: str, "team_config"?: dict}`.
  * С `episode_dir` загружается именно переданный эпизод: ID не меняются, повторный reset начинает его заново.
  * Без `episode_dir` эпизод строит генератор команды по `seed` и `team_config`.
  * Открытые и закрытые файлы копируются в **разные** временные корни. `truth.json` и `provenance.json` читает только отдельный ОС-процесс оценщика, а объект среды их не разбирает.
* **Первое наблюдение:** `{"task": str, "manifest": object, "policy": object, "remaining_steps": 100}`. Следующие: `{"result": value|null, "error": null|{"code","message"}, "remaining_steps": int}`. `info` всегда `{}`.
* **Действия** (`{"name", "args"}`) и их результаты:

| name | args | result |
|---|---|---|
| `list_records` | `table` ∈ `ledger, invoices, payments, fx`; `cursor` (первый — `null`); `limit` 1..50, по умолчанию 50 | `{"records": [...], "next_cursor": str\|null}` |
| `get_record` | `table`, `id` (row_id / invoice_id или doc_id / payment_id или doc_id / fx_id) | объект записи (поля §03; у строк fx есть вычисляемое `fx_id`) |
| `get_document` | `doc_id` | объект документа (remittance advice, PO, memo, строки) |
| `calculator` | `expression` — только десятичные `+ - * / ( )` | десятичная строка, например `"0.3"` |
| `submit` | `findings` — объект findings.json §05 | `{"accepted": true}` или ошибка `invalid_findings` (только структура) |

* **Коды ошибок:** `invalid_json`, `invalid_action_format`, `unknown_action`, `invalid_arguments`, `unknown_table`, `not_found`, `invalid_cursor`, `limit_out_of_range`, `calc_error`, `invalid_findings`, `episode_finished`, `not_reset`. Каждый ошибочный вызов расходует шаг. Код не раскрывает совпадение с эталоном.
* **Лимиты.** 100 действий, submit входит в лимит; submit на 100-м действии принимается. `inspection_steps` — все действия до submit, включая ошибочные, без самого submit. 100 исследовательских действий без submit дают `truncated=true` и пустые findings. Лимит 60 с считается от reset и включает время политики.
* **Режимы.** В `closed` reward политике всегда `0.0`, включая завершение. Истинная награда пишется в журнал оценщика: `AUDITENV_JUDGE_LOG`, по умолчанию `./judge_grader_log.jsonl`. В `train` терминальная награда возвращается политике.
* **`score_episode`.** Тот же matcher и та же общая награда, что в среде. При неверной структуре `return = -1`, остальные поля `null`, а ошибка пишется в тот же журнал.

**Соответствие четырёх типов ошибок** (`type` в findings и truth; подробно — `policy.matching` каждого эпизода):

| type | канонические `affected_ids` | `claimed_amount_minor`, валюта | минимальный набор доказательств |
|---|---|---|---|
| `duplicate_invoice` | оба `invoice_id` | `total_minor` дубля, валюта счёта | оба счёта + строка проводки (AR или revenue) **каждого** |
| `missing_or_partial_allocation` | `[invoice_id, payment_id]` | нераспределённая по remittance часть, валюта платежа | счёт + платёж + строка `customer_advances` платежа |
| `fx_wrong_date_or_rate` | `[invoice_id или payment_id]` | \|проведено − верно\| в base | документ + его проводка + `fx_id` верного курса |
| `period_cutoff_error` | `[invoice_id или payment_id]` | сумма ошибочной проводки в base | документ + его проводка |

**Matcher** соответствует разделу «Точные соглашения matcher»:
* ID, `type` и `currency` сравниваются точно, без смены регистра и обрезки пробелов. Единственное документированное правило — `doc_id` счёта или платежа равнозначен его `invoice_id`/`payment_id`.
* `affected_ids` и `evidence_ids` — множества. Лимит 10 для `evidence_ids` проверяется до удаления повторов.
* Несуществующий `evidence_id` даёт FP. Неизвестный `type`, неверная валюта, сумма или доказательства тоже дают FP.
* Неверная структура (−1): нет обязательного поля, неверный JSON-тип (сумма — строка, bool или дробь), больше 30 выводов, больше 10 доказательств, explanation длиннее 1000 code points, чужой `episode_id`. Лишние ключи игнорируются.
* Допуск τ — опубликованное целое: `manifest.rounding.tolerance_minor` и `policy.matching.tolerance_minor`. Если они различаются (жюри изменило одно поле), действует большее. У всех наших эпизодов оба равны 0.
* Эталон нормализуется так же, как ответ: `affected_ids` — отсортированное множество, `doc_id` в эталоне равнозначен ID записи. Порядок `affected_ids` в `truth.json` после смены ID значения не имеет.
* Ничьи разрешаются так: максимальное паросочетание, а среди равных — лексикографически наименьшая последовательность `(issue_id, индекс вывода)`.

**Минимальный цикл политики через адаптер** (политика жюри может работать в своём процессе — адаптеру нужен только транспорт `step`):
```python
import judge_adapter as ja
ja.generate_episode(42, {"clean": True}, "jury/ep42")
env = ja.make_env()
obs, _ = env.reset(42, {"mode": "closed", "episode_dir": "jury/ep42"})
eid, cur, rows, steps = obs["manifest"]["episode_id"], None, [], 0
while obs["remaining_steps"] > 1:
    obs, r, term, trunc, _ = env.step({"name": "list_records", "args": {"table": "invoices", "cursor": cur}})
    steps += 1
    rows += obs["result"]["records"]
    cur = obs["result"]["next_cursor"]
    if cur is None:
        break
findings = {"episode_id": eid, "findings": []}          # ... выводы политики
obs, r, term, trunc, _ = env.step({"name": "submit", "args": {"findings": findings}})
print(obs["result"], r)                                  # {'accepted': True} 0.0  (closed)
print(ja.score_episode("jury/ep42", findings, steps))    # чистый эпизод: tp=fp=fn=0, return = 1 - 0.002*steps
env.close()
```

**Поля со ссылками на ID** (для согласованной замены ID в парных проверках): `ledger.csv` — `row_id`, `entity_id`, `invoice_id`, `payment_id`; `invoices.jsonl` — `invoice_id`, `doc_id`, `entity_id`; `payments.jsonl` — `payment_id`, `doc_id`, `invoice_refs[].invoice_id`; `documents.jsonl` — `doc_id`, `record.*` (копия записи счёта/платежа), `payer_entity_id`, `refund_of`, `remittance.lines[].invoice_id`; `truth.json` — `affected_ids`, `evidence_sets`. `fx_id` = `fx:<date>:<source>:<target>` вычисляется из содержимого `fx.csv` и не переименовывается.

**Рецепты config для матрицы D01–D20.** Все ключи документированы в `auditenv/generator/config.py`, остальное берётся из дефолта; seed задаёт жюри. Частичный config **мержится** с дефолтом, поэтому `Z` и `I` обязательно обнуляют остальные типы и сценарии: иначе они возьмутся из дефолта `[0, 2]` и `[0, 1]`. В тесте `tests/test_judge_adapter.py::test_D_recipes_generate_valid_episodes` каждый рецепт проверяется валидатором.
```text
Z = {"duplicate_invoice": 0, "missing_or_partial_allocation": 0, "fx_wrong_date_or_rate": 0, "period_cutoff_error": 0}
I = {"partial_fx_boundary": 0, "duplicate_cutoff": 0}
D01 обычная сверка        {"clean": true}
D02 разбитый платёж       {"clean": true, "distractors": {"payment_mix_pct": {"none": 10, "full": 30, "split": 60, "partial": 0, "overpay": 0}}}
D03 корректный возврат    {"clean": true, "distractors": {"prepayments": [2, 3], "refund_pct": 100}}
D04 повторяющиеся суммы   {"clean": true, "distractors": {"recurring_groups": [3, 4], "same_total_diff_lines": [1, 2]}}
D05 граница валюты/периода {"clean": true, "currencies": {"foreign_customer_share_pct": 70}, "distractors": {"tz_boundary_invoices": [3, 5], "tz_boundary_next_invoices": [2, 3], "tz_boundary_receipts": [2, 3]}}
D06–D08 дубль             {"errors": Z + {"duplicate_invoice": 1}, "interactions": I}
D09 пропуск распределения {"errors": Z + {"missing_or_partial_allocation": 1}, "interactions": I, "error_variants": {"missing_or_partial_allocation": ["missing"]}}
D10 частичное             … то же с ["partial"];  D11 — без error_variants (вариант по seed)
D12–D13 FX                {"errors": Z + {"fx_wrong_date_or_rate": 1}, "interactions": I}   (вариант: error_variants.fx_wrong_date_or_rate ⊆ [naive_date, shifted_date, due_date, wrong_pair])
D14–D15 период            {"errors": Z + {"period_cutoff_error": 1}, "interactions": I}     (⊆ [next_period_booked, prior_period_booked, next_period_payment_booked])
D16–D18 сценарий 1        {"errors": Z, "interactions": I + {"partial_fx_boundary": 1}}  — частичный платёж в иностранной валюте у границы периода: allocation + fx
D19–D20 сценарий 2        {"errors": Z, "interactions": I + {"duplicate_cutoff": 1}}     — дубль счёта, копия за границей периода: duplicate + cutoff
```
Внутренние варианты ошибок в сценариях фиксированы: `partial_fx_boundary` — частичное распределение + FX по «наивной» дате; `duplicate_cutoff` — копия счёта, начисленная на закрытии периода. `error_variants` действует только на одиночные ошибки. Параметры сценариев меняются ключами `size`, `currencies` и `distractors`, например:
```text
D16  {"errors": Z, "interactions": I + {"partial_fx_boundary": 1}, "currencies": {"base": ["KZT"], "foreign": ["USD", "EUR"]}}
D17  … то же с "currencies": {"base": ["EUR"], "foreign": ["JPY", "GBP"], "foreign_customer_share_pct": 80}
D18  … то же с "size": {"n_invoices": [60, 75]}, "distractors": {"tz_boundary_receipts": [4, 6]}
D19  {"errors": Z, "interactions": I + {"duplicate_cutoff": 1}, "currencies": {"base": ["USD"]}}
D20  … то же с "size": {"n_invoices": [20, 30]}, "distractors": {"tz_boundary_next_invoices": [4, 6]}
```

## Для жюри: как подключить свои политики

> Этот раздел — **внутренний** API команды (`{"action", "args"}`, `observation["result"]`). Harness жюри по определениям проверок использует **адаптер** `judge_adapter.py` (раздел выше) с оболочкой `{"name", "args"}`.

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
python -m auditenv.cli run --policies your.module:YourPolicy --splits dev --trajectories your_traj.jsonl
#    (без --trajectories прогон перезапишет сдаваемый trajectories.jsonl; вернуть: git checkout trajectories.jsonl)
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

### Свои преобразованные эпизоды (перестановка строк, смена ID)
`make jury-run` применяет наши `auditenv.transforms`. Свои преобразования подложите так (имена сплита и каталога — `[A-Za-z0-9_.-]`, любые):
```
my/episodes/<split>/<name>/   manifest.json policy.json ledger.csv invoices.jsonl payments.jsonl fx.csv documents.jsonl
my/grader/<split>/<name>/     truth.json   (с теми же новыми ID; provenance.json нужен только валидатору)
my/episodes/<split>/index.json   [{"ref": "<split>/<name>"}, ...]
```
```bash
python -m auditenv.cli run --policies oracle competent_reconciler your.module:YourPolicy --splits <split> \
  --episodes-root my/episodes --grader-root my/grader \
  --trajectories my/traj.jsonl --grader-log my/grader_log.jsonl --summary my/summary.json
```
* `oracle` строит ответ в процессе оценщика по вашему `truth.json`. Свой oracle по диску — `--trusted --policies oracle_from_truth --truth-dir my/grader` (каталог может называться не так, как `episode_id`).
* **ID** — любые строки до 128 символов (`INV/2026/0001`, `{uuid}`, пробелы, кириллица) сравниваются точно. Несуществующий `evidence_id` даёт FP, а не неверную структуру.
* **`episode_id`** — до 128 символов, одинаковый в `manifest.json` и `truth.json`; с ним агент отправляет findings.
* **`fx_id`** = `fx:<date>:<src>:<tgt>` вычисляется из содержимого `fx.csv` (в нём нет столбца ID). В эталоне его не переименовывайте, перестановка строк `fx.csv` на него не влияет.
* **Хэши `manifest.files`** среда не проверяет, поэтому после перестановки эпизод открывается и без пересчёта. Наш валидатор их проверяет; пересчитать:
```bash
python -c "import hashlib,json,pathlib,sys; d=pathlib.Path(sys.argv[1]); p=d/'manifest.json'; m=json.loads(p.read_text('utf-8')); m['files']={n: hashlib.sha256((d/n).read_bytes()).hexdigest() for n in m['files']}; p.write_bytes((json.dumps(m, sort_keys=True, indent=2, ensure_ascii=False) + '\n').encode('utf-8'))" my/episodes/<split>/<name>
```
* Валидатор: `python -m auditenv.cli validate --splits <split> --episodes-root my/episodes --grader-root my/grader`. Ему нужен `provenance.json` с той же биекцией ID, а схемы записей валидатора рассчитаны на наш формат ID (`[A-Za-z0-9_.:-]`, ≤64): ID в другом формате он отметит как `schema: … invalid … records`. Это ограничение валидатора, а не среды.

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
* `reset` посреди эпизода открывает новый эпизод с полным бюджетом; брошенный эпизод не оценивается и в итоги оценщика не попадает.

**Действия:**
* `list_records(table, cursor=None, limit=50)` — `cursor` и `limit` необязательны, лишние аргументы дают `invalid_arguments`;
* `get_record(table, id)`;
* `get_document(doc_id)`;
* `calculator(expression)`;
* `submit(findings)`.

Таблицы: `ledger`, `invoices`, `payments`, `fx`. В каждой строке `fx` есть вычисляемое поле `fx_id` = `fx:<date>:<src>:<tgt>`: это ID доказательства. Сам `fx.csv` не меняется.

**Лимиты и награда:**
* 100 шагов включая submit и 60 с на эпизод; промежуточная награда 0; при truncation оценивается пустой ответ;
* `findings`: ≤30 выводов, ≤10 ID доказательств, explanation ≤1000 символов, ID и `episode_id` ≤128 символов, вложенность ≤32; неверная структура даёт −1;
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
Машина медленнее Linux-песочницы (≈6× на генерации), отсюда время. Эпизоды и все числа (issues=1880, TP=1963, 80/80) совпадают с прогонами на Linux/3.11. Блок «Проверено командой» ниже снят 20.11 после добавления адаптера жюри `judge_adapter.py` (263 теста, из них 40 — адаптер по определениям проверок и регрессии независимого ревью) на официальном `python:3.11-slim` по digest; блок «Проверено в песочнице» — 19.11, до доработок (192 теста). Dockerfile и digest с 19.11 не менялись.

### Проверено командой
```
дата: 20-11-2026   машина: Windows x64, Docker Desktop   docker --version: 29.9.3
digest: python@sha256:07d4824036af339fa1b6ef769cc99d9c10a141769538b52b3b597c84f7a7b535
версия: адаптер жюри + matcher по определениям проверок 20.11.2026 + исправления независимого ревью

$ docker build --build-arg PYTHON_IMAGE=python@sha256:07d4824036af339fa1b6ef769cc99d9c10a141769538b52b3b597c84f7a7b535 -t auditenv .
$ docker run --rm --network none --cpus 2 --memory 4g auditenv
[OK] generate train                                  0.5s  [generate] train/ep_7c609c9a4771e31a: issues=5 rows=700
[OK] generate dev                                    0.5s  [generate] dev/ep_053daf06ab62f121: issues=3 rows=446
[OK] validate episodes                               1.1s  [validate] dev/ep_053daf06ab62f121: OK issues=3 checks=67
[OK] validate 300 random seeds                      29.4s  [validate-seeds] 300/300 OK, issues=1880, clean=50
[OK] pytest                                        101.7s  263 passed in 101.20s (0:01:41)
[OK] run baseline policies (+competent)              4.3s
[OK] solvability: competent_reconciler, 300 seeds + worst size   19.5s  310/310 эпизодов с F1=1 и без FP; TP=1963 FP=0 FN=0; шаги max=73 mean=41.6
[OK] trajectory leak scan                            0.9s
[OK] determinism (2 processes, PYTHONHASHSEED)       2.3s  files=182 identical_across_processes=True
[OK] self-check 4x20 (не оценка жюри)               17.4s  80/80
[OK] report                                          0.3s
== check: OK ==

$ docker run --rm --network none auditenv python -m pytest -q tests/test_judge_adapter.py
40 passed in 7.32s
```
Тот же результат (те же episode_id, issues=1880, TP=1963, 263 теста) получен на Linux вне Docker на Python 3.11, а эпизоды, эталон и траектории побайтово совпали с файлами, сгенерированными на Windows: генерация детерминирована и между машинами.

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
