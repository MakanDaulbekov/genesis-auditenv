# CHECKLIST: требование PDF → файл/тест → статус

| Раздел PDF | Требование | Где реализовано | Тест | Статус |
|---|---|---|---|---|
| 02 | 4 независимо настраиваемых типа ошибок | `generator/mutations.py`, `configs/default.json:errors` | `tests/test_reward.py` (все 4 типа в эпизоде) | ✅ Ф1 (валидатор — Ф2) |
| 02 | 2 сценария взаимодействующих ошибок | `mutations.py: SCENARIOS` | — | 🟡 код есть, тесты Ф3 |
| 02 | Чистые компании + отвлекающие случаи | `world.py` (recurring, split, partial, overpay, batch, prepay/refund, tz-boundary) | `test_clean_episode_empty_findings_is_max` | 🟡 независимая проверка Ф2 |
| 02 | Сначала история, потом документы, потом ошибки | `world.build_world → render → mutations → post_all → render` | — | ✅ |
| 03 | manifest с schema_version, episode_id, base, период, tz, счётчики, округление | `render.public_files`, `schemas/manifest.schema.json` | `test_reset_signature_and_first_observation` | ✅ |
| 03 | ledger/invoices/payments/fx/policy ровно по схеме | `render.py`, `schemas/*.schema.json` | `test_step_signature_and_json` | ✅ |
| 03 | деньги int minor, FX — десятичные строки, ISO-8601 со смещением, явные null | `common/money.py`, `common/timeutil.py` | схемы | ✅ |
| 03 | truth/provenance закрыты, вне каталога среды | `grader_data/`, `grader/service.py` | — | 🟡 тесты утечек Ф3 |
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
| 06 | 4 базовые политики | — | — | ⏳ Ф2 |
| 07 | траектории без эталона | — | — | ⏳ Ф2 |
| 07 | Контейнер + единая команда | `Makefile: check` | `make check` | 🟡 Dockerfile Ф3 |
| 10 | детерминизм seed/config/version | `common/drbg.py`, `checks.py` | `make determinism` (2 процесса, разный PYTHONHASHSEED) | ✅ |
| 10 | перестановки и смена ID | — | — | ⏳ Ф3 |
