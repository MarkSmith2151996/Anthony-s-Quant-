FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && \
    rm -rf /var/lib/apt/lists/*

# Python deps (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . .
RUN pip install --no-cache-dir -e .

# Create data dirs
RUN mkdir -p data/raw data/processed data/features results reports logs

# Default: run the backtest
CMD ["python", "run_backtest.py"]
