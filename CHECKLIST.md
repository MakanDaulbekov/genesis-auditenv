# CHECKLIST: требование PDF → файл → тест → статус

## Раздел 09 «Сдача» — материалы

| Материал (PDF 09) | Что требуется | Файл(ы) | Тест / проверка | Статус |
|---|---|---|---|---|
| Исходный код | исходники с лицензией и зафиксированными зависимостями: генератор и среда | `auditenv/`, `LICENSE` (MIT), `pyproject.toml`, `requirements-dev.lock` (версии + SHA-256), `vendor/wheels` | `make check`; офлайн-установка `pip --no-index --require-hashes` | ✅ |
| README | инструкции контейнера и единая команда проверки | `README.md` (первые 20 строк — одна команда), `Dockerfile`, `Makefile: check, docker-check, docker-pin, jury-run` | `make check`; `docker run --network none --cpus 2 --memory 4g` на официальном `python:3.11-slim` по digest (машина команды, 20.11, после аудита раунда 3, 220 тестов — README) | ✅ |
| Matcher и награда | общий код сопоставления и награды, схемы и политики | `auditenv/grader/matcher.py`, `reward.py`, `auditenv/schemas/*.schema.json`, `policy.json → matching` | `tests/test_reward.py`, самопроверка reward 20/20 | ✅ |
| Эпизоды | 10 открытых train и 10 dev | `episodes/{train,dev}` (10+10; в dev есть base=JPY), эталон — `grader_data/{train,dev}`, конфиги — `configs/splits/*.json` | `test_committed_episodes_validate`, `make validate` | ✅ |
| Тесты | тесты среды, сопоставления и награды | `tests/` (≈200 тестов): контракт, награда/атаки, данные/MI, парные, изоляция, agent_host, reset/config, API политик, jury-run, крайние config и доказательства (`test_robustness.py`) | `make test` | ✅ |
| Базовые политики | четыре адаптера базовых политик | `auditenv/policies/api.py` (`Policy`, `load_policy`), `baselines.py`: `random_explorer`, `simple_reconciler`, `flag_all`, `oracle` (оценщик) + `competent_reconciler`, `oracle_from_truth` (`--trusted`), `process_adapter.py`; `--policy module:factory` | `tests/test_policy_api.py`, `tests/test_policies.py` | ✅ |
| Траектории | `trajectories.jsonl` с episode_id, step, action, observation, reward, terminated, truncated; без эталона в наблюдениях | `trajectories.jsonl` (5 политик × 20 эпизодов) | `test_committed_trajectories_have_no_truth`, `test_trajectory_fields_and_no_truth` | ✅ |
| Записка о дизайне | краткое описание устройства среды | `DESIGN.md` (Assumptions, Честные ограничения) | — | ✅ |
| Демо | показ работы среды скриптовой политикой | `make demo` → `auditenv/demo.py` | — | ✅ |

## Остальные разделы

| Раздел PDF | Требование | Файл(ы) | Тест | Статус |
|---|---|---|---|---|
| 02 | 4 независимо настраиваемых типа ошибок | `generator/mutations.py`, `config.errors` | `test_each_error_type_independently_configurable`, `test_variants_present_across_seeds` | ✅ |
| 02 | 2 сценария взаимодействий | `mutations.SCENARIOS` | `test_interaction_scenarios` | ✅ |
| 02 | Чистые компании + отвлекающие (10 видов) | `generator/world.py` | `test_clean_companies_have_distractors`, валидатор п.5 | ✅ |
| 02 | Сначала история, потом документы, потом ошибки; эталон из причинных изменений | `world → render → mutations → truth`, `provenance.diff` | валидатор: состояние «до» чисто | ✅ |
| 02 | Варьирование имён, ID, порядка, сумм, валют, часовых поясов (вкл. +05:30/+05:45), сочетаний | `world.py`, `render.py`, `config.py` | `test_minute_offsets_and_jpy_base`, `validate-seeds` (6 сочетаний) | ✅ |
| 02 | Сложность против простых трюков | порядок по ключевому хешу, ГПСЧ SHA-256, кандидаты мутаций включают отвлекающие | `test_position_*`, `test_id_order_*`, `test_round_amounts_*`, `test_categorical_fields_carry_no_label_signal` | ✅ |
| 02/10 | Решаемо по открытым данным (не «неразрешимые данные») | `competent_reconciler`, `cli solvability` | 310/310 эпизодов F1=1, ≤73 шагов; `test_competent_reconciler_solves_*` | ✅ |
| 03 | manifest, форматы, таблицы ровно по схеме, явные null | `render.py`, `schemas/` | `test_contract.py`, валидатор п.1 | ✅ |
| 03 | эталон и provenance закрыты, оценщик вне процесса агента | `grader/service.py` (subprocess, JSON), `agent_host.py` | `tests/test_isolation.py` | ✅ |
| 04 | reset(seed, config) → (obs, info); config=None → дефолт; понятные ошибки | `env/core.py`, `generator/config.py` | `tests/test_reset_config.py` | ✅ |
| 04 | 5 действий, страница ≤50, стабильные курсоры, дефолты list_records, fx_id | `env/tools.py`, `env/store.py` | `test_pagination_*`, `test_list_records_defaults_*`, `test_fx_records_carry_fx_id` | ✅ |
| 04 | калькулятор, ограниченные ошибки, лимиты 100 шагов/60 с, truncated → пустой ответ, eval скрывает награду | `env/` | `test_contract.py`, самопроверка contract 20/20 | ✅ |
| 05 | findings, лимиты, −1 за структуру, взаимно-однозначное сопоставление, P/R/F1, общая награда, атаки | `grader/` | `test_reward.py`, `test_hedging_attack_*` | ✅ |
| 06 | 4 политики организаторов на 20 скрытых эпизодах (5 чистых, 5 со взаимодействиями) + 20 парных проверок | `make jury-run` → `auditenv/jury_run.py` | `tests/test_jury_run.py`; пример `examples/closed_split_example.json` | ✅ |
| 06 | закрытые seed и новые сочетания без правки кода; секрет | `--split`, `--secret-file`, `--closed-episodes-file` | `test_jury_run_without_secret_warns_and_differs`, `test_closed_episodes_eval_only_and_single_open` | ✅ |
| 07 | официальная проверка без сети, платных API, GPU | stdlib, `vendor/wheels`, `--network none` | — | ✅ |
| 10 | воспроизводимость: seed/действия, перестановка и смена ID | `common/drbg.py`, `transforms.py`, `checks.py` | `tests/test_paired.py`, `make determinism`, самопроверка paired 20/20 | ✅ |
| 10 | P, R, F1, возврат, шаги, время (p50/p95/max), пик RSS по политикам | `REPORT.md` ← `report.py`, `cli run` (политики в отдельных процессах) | `make check` | ✅ |
| 10 | измеренный разбор сбоев, честные ограничения | `REPORT.md` (разделы 2–3), `DESIGN.md` §7 | — | ✅ |
| 12 | всё вымышлено, без реальных данных | генератор из словарей | — | ✅ |

| 06/10 | эталон и доказательства проверяются жюри независимо по записям; равенства «до» ошибки | `cli export-before` → таблицы «до» в формате раздела 03; `jury-run` кладёт их в `jury_out/grader/<split>/<ep>/before/`; `policy.matching.required_evidence` | `test_export_before_is_clean_and_balanced`, `test_duplicate_evidence_needs_posting_of_both_invoices` | ✅ |
| 06/10 | смена ID жюри в свой формат (`/`, `{uuid}`, кириллица, `episode_id` ≤128) сохраняет ответ oracle; глубокий `submit` → −1 без падения; путь «свои преобразованные эпизоды» | `grader/matcher.py`, `env/core.py`, README «Свои преобразованные эпизоды», `AUDIT.md` | `tests/test_round3_regressions.py` | ✅ |
| 10 | крайние сочетания параметров не ломают генерацию | добор кандидатов в `world.py`, равномерное урезание сверх 12 в `episode._plan` | `test_extreme_configs_generate_and_validate`, `test_over_max_issues_is_balanced_across_types` | ✅ |

## Что команде сделать руками

1. ~~Выполнить `make docker-pin && make docker-check`~~ — сделано: digest закреплён 19.11; прогон на официальном образе после аудита раунда 3 — 20.11, 220 тестов, вывод в README.
2. Отправить ссылку на репозиторий/архив в форму сдачи (ссылку организаторы сообщат отдельно).
