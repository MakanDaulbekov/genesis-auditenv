# python3.11, если есть; иначе python3 (код — только stdlib; генерация побайтово совпадает на 3.11 и 3.12)
PY ?= $(shell command -v python3.11 >/dev/null 2>&1 && echo python3.11 || echo python3)
export PYTHONPATH := $(CURDIR)
export PYTHONDONTWRITEBYTECODE := 1

.PHONY: check generate validate validate-seeds test run report determinism leaks jury demo clean

## Единая команда проверки: ненулевой выход при любом провале
check:
	$(PY) -m auditenv.cli check

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

demo:
	$(PY) -m auditenv.demo

jury:
	$(PY) tests/jury/run_jury.py

report:
	$(PY) -m auditenv.report

## Режим жюри: make jury-run SPLIT=closed_split.json [SECRET=secret.txt] [POLICY="their.module:Factory ..."]
SPLIT ?= examples/closed_split_example.json
jury-run:
	$(PY) -m auditenv.jury_run --split $(SPLIT) $(if $(SECRET),--secret-file $(SECRET),) $(if $(POLICY),--policies $(POLICY),)

## Docker: закрепить базовый образ по digest и прогнать проверку в контейнере без сети
docker-pin:
	sh docker/pin_base_image.sh

BASE_IMAGE := $(shell cat docker/base-image.txt 2>/dev/null || echo python:3.11-slim)
docker-check:
	docker build --build-arg PYTHON_IMAGE=$(BASE_IMAGE) -t auditenv .
	docker run --rm --network none --cpus 2 --memory 4g auditenv
