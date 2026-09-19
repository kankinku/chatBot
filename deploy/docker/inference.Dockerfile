FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config ./config
COPY data ./data
COPY src ./src
COPY services/__init__.py ./services/__init__.py
COPY services/inference ./services/inference

RUN mkdir -p logs out/benchmarks out/tests vector_store

EXPOSE 8000

ENV PYTHONPATH=/app/src:/app
ENV PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

CMD ["uvicorn", "services.inference.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
