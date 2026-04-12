FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY services/ ./services/

WORKDIR /app/services/monitoring

# Monitoring API + background analytics (drift, SLO polling, etc.)
CMD ["uvicorn", "control_plane.app:app", "--host", "0.0.0.0", "--port", "9104"]
