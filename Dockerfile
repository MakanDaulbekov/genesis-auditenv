# Воспроизводимая проверка: python 3.11, без сети во время проверки, без GPU.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
# только dev-зависимость (pytest) с точными версиями; рантайм — стандартная библиотека
COPY requirements-dev.lock /app/
RUN pip install --no-deps -r requirements-dev.lock

COPY . /app
RUN useradd --create-home runner && chown -R runner /app
USER runner

# make не обязателен: та же проверка, что и `make check`
CMD ["python", "-m", "auditenv.cli", "check"]
