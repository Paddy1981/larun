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
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

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
COPY larun.py .
COPY larun_chat.py .
COPY larun_pipeline.py .
COPY api.py .

# Create data directories
RUN mkdir -p data/cache data/raw output logs output/submissions

# Expose ports
# 8000 - FastAPI
# 8080 - Flask Dashboard
EXPOSE 8000 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || curl -f http://localhost:${PORT:-8000}/ || exit 1

# Default command: Run Discovery Pipeline Dashboard
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]

# ============================================================================
# Alternative entrypoints (use with docker run --entrypoint)
# ============================================================================
# CLI Mode:        docker run -it --entrypoint python larun larun_pipeline.py
# FastAPI:         docker run --entrypoint uvicorn larun api:app --host 0.0.0.0 --port 8000
# Chat Mode:       docker run -it --entrypoint python larun larun_chat.py
# Training:        docker run --entrypoint python larun train_production.py
