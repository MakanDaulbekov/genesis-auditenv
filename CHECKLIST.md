# CHECKLIST: требование PDF → файл/тест → статус

| Раздел PDF | Требование | Где реализовано | Тест | Статус |
|---|---|---|---|---|
| 02 | 4 независимо настраиваемых типа ошибок | `generator/mutations.py`, `configs/default.json:errors` | `test_each_error_type_independently_configurable`, `test_variants_present_across_seeds` | ✅ |
| 02 | 2 сценария взаимодействующих ошибок | `mutations.py: SCENARIOS` | `test_interaction_scenarios` (+валидатор) | ✅ |
| 02 | Чистые компании + отвлекающие случаи (9 видов) | `world.py` | `test_clean_companies_have_distractors`, валидатор п.5 | ✅ |
| 02 | Сначала история, потом документы, потом ошибки | `world.build_world → render → mutations → post_all → render` | — | ✅ |
| 03 | manifest с schema_version, episode_id, base, период, tz, счётчики, округление | `render.public_files`, `schemas/manifest.schema.json` | `test_reset_signature_and_first_observation` | ✅ |
| 03 | ledger/invoices/payments/fx/policy ровно по схеме | `render.py`, `schemas/*.schema.json` | `test_step_signature_and_json` | ✅ |
| 03 | деньги int minor, FX — десятичные строки, ISO-8601 со смещением, явные null | `common/money.py`, `common/timeutil.py` | схемы | ✅ |
| 03 | truth/provenance закрыты, вне каталога среды; эталон не утекает через наблюдения, пути, ошибки, debug, объяснение награды | `grader_data/`, `grader/service.py`, `agent_host.py` | `tests/test_isolation.py` (16 тестов), jury c19–c20 | ✅ |
| 04 | reset(seed, config) -> (obs, info); step -> 5-tuple | `env/core.py` | `test_contract.py` | ✅ |
| 04 | 5 действий, страница ≤50, стабильные курсоры | `env/tools.py`, `env/store.py` | `test_pagination_*`, `test_cursor_*`, `test_list_limit_bounds` | ✅ |
| 04 | калькулятор только арифметика | `env/calculator.py` | `test_calculator_*` | ✅ |
| 04 | неизвестный ID/действие → ограниченная ошибка, шаг расходуется | `env/tools.py: ERRORS` | `test_unknown_ids_tables_actions` | ✅ |
| 04 | 100 шагов включая submit; truncated → пустой ответ | `env/core.py` | `test_truncation_*`, `test_submit_as_100th_step_terminates`, `test_time_limit_truncates` | ✅ |
| 04 | промежуточная награда 0; eval скрывает награду | `env/core.py` | `test_eval_mode_hides_reward` | ✅ |
| 05 | findings.json, лимиты 30/10/1000, неверная структура → −1 + журнал | `grader/matcher.py`, `schemas/findings.schema.json` | `test_invalid_*`, `test_invalid_findings_logged_by_grader` | ✅ |
| 05 | взаимно-однозначное сопоставление, TP-условия, дубли → FP | `grader/matcher.py` | `test_duplicate_findings_give_no_gain`, `test_findings_order_invariance` | ✅ |
| 05 | P/R/F1, краевые случаи, общая награда | `grader/reward.py` | `test_metric_edge_cases`, `test_step_cost_small` | ✅ |
| 05 | 5 свойств награды + атаки | — | `test_reward.py` (16 групп, fuzz 800 ответов) | ✅ |
| 06 | 4 базовые политики | `policies/baselines.py`, `runner.py`, oracle — `grader/service.py:op_oracle` | `tests/test_policies.py` | ✅ |
| 07 | траектории без эталона | `trajectories.jsonl`, `runner.py` | `test_trajectory_fields_and_no_truth`, `test_committed_trajectories_have_no_truth` | ✅ |
| 07 | Контейнер + единая команда | `Makefile: check` → `auditenv/cli.py check`, `Dockerfile` | `make check` (в чистом clone + venv) | ✅ (образ не собран в песочнице: Docker Hub закрыт) |
| 10 | детерминизм seed/config/version | `common/drbg.py`, `checks.py` | `make determinism` (2 процесса, разный PYTHONHASHSEED) | ✅ |
| 10 | перестановки и смена ID сохраняют ответ oracle и награду | `transforms.py` | `tests/test_paired.py` (24 пары + env-уровень), jury paired 20/20 | ✅ |
| 06/10 | Независимая проверка эталона и доказательств | `generator/validate.py` (не импортирует генератор) | `make validate`, `make validate-seeds` (300 seed × 6 сочетаний), `test_validator_detects_corruption` | ✅ |
| 02 | Анти-трюки: позиция, порядок ID, круглые суммы | `render.py` (порядок по ключевому хешу), `common/drbg.py` | `test_position_heuristic_*`, `test_id_order_*`, `test_round_amounts_*` | ✅ |
| 06 | Oracle только для оценщика | `grader/service.py:op_oracle` (`allow_oracle`), `runner.py` | `test_oracle_disabled_for_agent_runs` | ✅ |
| 09 | README с инструкциями контейнера и единой командой | `README.md` | — | ✅ |
| 09 | Записка о дизайне | `DESIGN.md` (Assumptions, Честные ограничения) | jury engineering | ✅ |
| 09 | Демо (скриптовая политика) | `make demo` → `auditenv/demo.py` | — | ✅ |
| 09 | 10 train + 10 dev эпизодов | `episodes/{train,dev}`, `configs/splits/*.json` | `test_committed_episodes_validate` | ✅ |
| 10 | P, R, F1, возврат, шаги, время агентов как диагностика | `REPORT.md` ← `auditenv/report.py` | `make check` | ✅ |
| 10 | Теневое жюри 4×20 + прогноз | `tests/jury/run_jury.py` | `make jury` | ✅ |
| 12 | Всё вымышлено, без сети | генератор из словарей, `--network none` | — | ✅ |
