FROM python:3.11

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUTF8=1 \
    PYTHONUNBUFFERED=1 \
    HOMR_PORT=8010 \
    HOMR_PRELOAD_MODELS=true \
    POETRY_DYNAMIC_VERSIONING_BYPASS=0.1.0

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libxcb1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

COPY pyproject.toml README.md LICENSE MODIFICATIONS.md ./
COPY homr ./homr
COPY homr_service ./homr_service

RUN pip install --no-cache-dir .

RUN python -m homr.main --init

RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /models \
    && chown -R appuser:appuser /app /models

USER appuser

EXPOSE 8010

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=5 \
    CMD curl -fsS "http://127.0.0.1:${HOMR_PORT}/health" || exit 1

CMD ["sh", "-c", "uvicorn homr_service.main:app --host 0.0.0.0 --port ${HOMR_PORT}"]
