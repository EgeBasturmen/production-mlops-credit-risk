FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /app/pyproject.toml
COPY .env /app/.env
COPY src /app/src

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e . \
    && pip install --no-cache-dir streamlit requests

RUN mkdir -p /app/artifacts /app/logs /app/data/prod

EXPOSE 8000

CMD ["uvicorn", "credit_risk.app:app", "--host", "0.0.0.0", "--port", "8000"]
