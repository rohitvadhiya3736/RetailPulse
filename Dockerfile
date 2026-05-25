# RetailPulse – Multi-stage production Dockerfile
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    RETAILPULSE_HOME=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

RUN mkdir -p data/raw data/processed artifacts/models artifacts/plots mlflow/mlruns

# Generate data and train on build (optional – comment out for faster builds)
RUN python scripts/generate_sample_data.py && python scripts/train_all.py || true

EXPOSE 8000 8501

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["dashboard"]
