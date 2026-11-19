PY ?= python3.11
export PYTHONPATH := $(CURDIR)
export PYTHONDONTWRITEBYTECODE := 1

.PHONY: check generate validate test run report determinism leaks jury demo clean

## Единая команда проверки: ненулевой выход при любом провале
check: generate validate test run determinism
	@echo "== make check: OK =="

generate:
	$(PY) -m auditenv.cli generate --split configs/splits/train.json
	$(PY) -m auditenv.cli generate --split configs/splits/dev.json

validate:
	@if [ -f auditenv/generator/validate.py ]; then $(PY) -m auditenv.cli validate --splits train dev; \
	else echo "[validate] (фаза 2)"; fi

test:
	$(PY) -m pytest -q tests

run:
	@if [ -f auditenv/runner.py ]; then $(PY) -m auditenv.cli run; else echo "[run] (фаза 2)"; fi

determinism:
	$(PY) -m auditenv.checks determinism

clean:
	rm -rf episodes/train episodes/dev grader_data/train grader_data/dev .pytest_cache
