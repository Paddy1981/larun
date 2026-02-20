# LARUN TinyML - Docker Image
# Multi-stage build for optimized image size

# ============================================================================
# Stage 1: Builder
# ============================================================================
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements-railway.txt requirements.txt

# Install Python dependencies with timeout settings
RUN pip install --no-cache-dir --user --timeout=300 -r requirements.txt

# ============================================================================
# Stage 2: Runtime
# ============================================================================
FROM python:3.11-slim as runtime

LABEL maintainer="Padmanaban Veeraragavalu <larun@example.com>"
LABEL description="LARUN TinyML - Astronomical Data Analysis for Exoplanet Discovery"
LABEL version="2.0.0"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    LARUN_HOME=/app

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY src/ ./src/
COPY skills/ ./skills/
COPY config/ ./config/
COPY models/ ./models/
COPY nodes/ ./nodes/
COPY larun.py .
COPY larun_chat.py .
COPY api.py .
COPY api-minimal.py .
COPY cloud_endpoints.py .

# Create data directories
RUN mkdir -p data/cache data/raw output logs output/submissions

# Expose ports
# 8000 - FastAPI
# 8080 - Flask Dashboard
EXPOSE 8000 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Default command: Run FastAPI server (use minimal API for faster startup)
CMD ["uvicorn", "api-minimal:app", "--host", "0.0.0.0", "--port", "8000"]

# ============================================================================
# Alternative entrypoints (use with docker run --entrypoint)
# ============================================================================
# CLI Mode:        docker run -it --entrypoint python larun larun_pipeline.py
# FastAPI:         docker run --entrypoint uvicorn larun api:app --host 0.0.0.0 --port 8000
# Chat Mode:       docker run -it --entrypoint python larun larun_chat.py
# Training:        docker run --entrypoint python larun train_production.py
