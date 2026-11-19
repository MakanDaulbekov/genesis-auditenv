# Воспроизводимая проверка: Python 3.11, без сети во время сборки зависимостей и проверки, без GPU.
#
# Базовый образ закрепляется по digest командой `make docker-pin` (пишет docker/base-image.txt):
#   docker pull python:3.11-slim && docker inspect --format '{{index .RepoDigests 0}}' python:3.11-slim
# make docker-check подставляет его через --build-arg PYTHON_IMAGE=python:3.11-slim@sha256:<digest>.
# Без --build-arg используется тег python:3.11-slim (не закреплён).
ARG PYTHON_IMAGE=python:3.11-slim
FROM ${PYTHON_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
# dev-зависимость (pytest) ставится из вложенных колёс по SHA-256 — сеть не нужна
COPY requirements-dev.lock /app/
COPY vendor/wheels /app/vendor/wheels
RUN pip install --no-index --find-links /app/vendor/wheels --require-hashes -r requirements-dev.lock

COPY . /app
RUN useradd --create-home runner && chown -R runner /app
USER runner

# make в slim-образе нет — та же проверка, что `make check`, напрямую:
CMD ["python", "-m", "auditenv.cli", "check"]
