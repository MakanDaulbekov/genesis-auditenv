PY ?= python3.11
export PYTHONPATH := $(CURDIR)
export PYTHONDONTWRITEBYTECODE := 1

.PHONY: check generate validate validate-seeds test run report determinism leaks jury demo clean

## Единая команда проверки: ненулевой выход при любом провале
check: generate validate validate-seeds test run determinism
	@echo "== make check: OK =="

generate:
	$(PY) -m auditenv.cli generate --split configs/splits/train.json
	$(PY) -m auditenv.cli generate --split configs/splits/dev.json

validate:
	$(PY) -m auditenv.cli validate --splits train dev

validate-seeds:
	$(PY) -m auditenv.cli validate-seeds --n 300

test:
	$(PY) -m pytest -q tests

run:
	$(PY) -m auditenv.cli run > /dev/null
	$(PY) -m pytest -q tests/test_policies.py -k committed_trajectories

determinism:
	$(PY) -m auditenv.checks determinism

clean:
	rm -rf episodes/train episodes/dev grader_data/train grader_data/dev .pytest_cache
