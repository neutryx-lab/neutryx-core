# Multi-stage build for Neutryx
FROM python:3.12-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

# Labels
LABEL org.opencontainers.image.created=$BUILD_DATE \
      org.opencontainers.image.authors="Neutryx Platform Team <platform@neutryx.io>" \
      org.opencontainers.image.url="https://neutryx.io" \
      org.opencontainers.image.source="https://github.com/neutryx-lab/neutryx-core" \
      org.opencontainers.image.version=$VERSION \
      org.opencontainers.image.revision=$VCS_REF \
      org.opencontainers.image.vendor="Neutryx" \
      org.opencontainers.image.title="Neutryx Core" \
      org.opencontainers.image.description="Quantitative Finance Platform" \
      org.opencontainers.image.licenses="Apache-2.0"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy requirements
COPY requirements.txt .
COPY requirements-dev.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Final stage
FROM python:3.12-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    liblapack3 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r neutryx && useradd -r -g neutryx neutryx

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=neutryx:neutryx . .

# Set Python path
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Create directories for runtime
RUN mkdir -p /app/logs /app/cache /app/tmp && \
    chown -R neutryx:neutryx /app

# Switch to non-root user
USER neutryx

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose ports
EXPOSE 8080 9090

# Default command (override in k8s deployment)
CMD ["python", "-m", "uvicorn", "neutryx.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
