FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/backend

WORKDIR /app

ARG REQUIREMENTS_FILE=backend/requirements.txt

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/backend/requirements.txt
COPY backend/requirements.deploy.txt /app/backend/requirements.deploy.txt
RUN pip install --upgrade pip && pip install -r /app/${REQUIREMENTS_FILE}

COPY backend /app/backend
COPY bench_data /app/bench_data
COPY scripts /app/scripts

WORKDIR /app/backend

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
