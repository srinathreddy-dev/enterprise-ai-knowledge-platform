FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/appuser/.cache/huggingface

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY app ./app
COPY data/sample ./data/sample

RUN pip install --upgrade pip \
    && pip install .

RUN useradd --create-home appuser \
    && mkdir -p /home/appuser/.cache/huggingface \
    && chown -R appuser:appuser /home/appuser

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
