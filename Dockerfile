# Multi-stage production Dockerfile
FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Final minimal runtime image
FROM python:3.12-slim AS runtime

WORKDIR /app

# Create non-root system user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy installed python dependencies from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source code, configuration and pre-trained champion model
COPY pyproject.toml config.yaml ./
COPY src/ ./src/
COPY api/ ./api/
COPY models/ ./models/
COPY data/ ./data/

# Set environment paths and ownership
ENV PATH=/usr/local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src:/app

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Health check probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
