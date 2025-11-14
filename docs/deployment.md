# Deployment Guide

This guide provides comprehensive instructions for deploying Neutryx API across various infrastructure targets, from local development to production-grade multi-region deployments.

## Table of Contents

1. [Overview](#overview)
2. [Deployment Options](#deployment-options)
3. [Local Development](#local-development)
4. [Docker Deployment](#docker-deployment)
5. [Kubernetes Deployment](#kubernetes-deployment)
6. [Cloud Providers](#cloud-providers)
7. [Environment Configuration](#environment-configuration)
8. [Database Setup](#database-setup)
9. [Market Data Infrastructure](#market-data-infrastructure)
10. [Security & Compliance](#security-compliance)
11. [CI/CD Integration](#cicd-integration)
12. [Monitoring & Observability](#monitoring-observability)
13. [Troubleshooting](#troubleshooting)

---

## Overview

### Deployment Architecture

Neutryx API supports multiple deployment architectures tailored to different use cases:

```
┌─────────────────────────────────────────────────────────────────┐
│                   Deployment Options                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Local      │  │    Docker    │  │  Kubernetes  │         │
│  │  Development │  │   Compose    │  │   Cluster    │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│        │                   │                   │                │
│        └───────────────────┴───────────────────┘                │
│                            │                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Cloud Infrastructure                        │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐             │   │
│  │  │   GCP    │  │   AWS    │  │  Azure   │             │   │
│  │  └──────────┘  └──────────┘  └──────────┘             │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Features

- **Multi-Environment**: Development, staging, and production configurations
- **Auto-Scaling**: Horizontal and vertical scaling based on load
- **High Availability**: Multi-region deployment with automatic failover
- **Security**: SSO, RBAC, encryption at rest and in transit
- **Observability**: Prometheus metrics, Grafana dashboards, Jaeger tracing
- **Disaster Recovery**: Automated backups and recovery procedures

---

## Deployment Options

### Comparison Matrix

| Feature | Local Dev | Docker Compose | Kubernetes | Cloud Managed |
|---------|-----------|----------------|------------|---------------|
| **Complexity** | Low | Medium | High | Medium |
| **Setup Time** | 5 min | 15 min | 1-2 hours | 30-60 min |
| **Cost** | Free | Free | Infrastructure | Pay-as-you-go |
| **Scalability** | Single machine | Single machine | Horizontal | Horizontal + Vertical |
| **HA Support** | No | No | Yes | Yes |
| **Production Ready** | No | No | Yes | Yes |
| **GPU Support** | Limited | Yes | Yes | Yes |
| **Multi-Region** | No | No | Yes | Yes |

### Use Case Recommendations

- **Local Development**: Prototyping, testing, debugging
- **Docker Compose**: Small deployments, demos, integration testing
- **Kubernetes**: Production deployments, auto-scaling, high availability
- **Cloud Managed**: Enterprise deployments, compliance requirements, global distribution

---

## Local Development

### Prerequisites

```bash
# Required
Python 3.10+
pip 23.0+
virtualenv or conda

# Optional
Docker Desktop (for containerized dependencies)
PostgreSQL 14+ (or use Docker)
Redis 7+ (or use Docker)
```

### Quick Start

```bash
# 1. Clone repository
git clone https://github.com/neutryx-lab/neutryx-api.git
cd neutryx-api

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
pip install -e ".[dev]"

# 4. Run tests
pytest -v

# 5. Start development server (if REST API available)
# uvicorn neutryx.api.rest:app --reload --host 0.0.0.0 --port 8000
```

### Using Docker for Dependencies

```bash
# Start PostgreSQL
docker run -d --name neutryx-postgres \
  -e POSTGRES_DB=neutryx \
  -e POSTGRES_USER=neutryx \
  -e POSTGRES_PASSWORD=neutryx \
  -p 5432:5432 \
  postgres:15

# Start Redis
docker run -d --name neutryx-redis \
  -p 6379:6379 \
  redis:7-alpine

# Start TimescaleDB (for market data)
docker run -d --name neutryx-timescaledb \
  -e POSTGRES_DB=market_data \
  -e POSTGRES_USER=neutryx \
  -e POSTGRES_PASSWORD=neutryx \
  -p 5433:5432 \
  timescale/timescaledb:latest-pg15
```

### Environment Variables

Create `.env` file:

```bash
# Database
DATABASE_URL=postgresql://neutryx:neutryx@localhost:5432/neutryx
TIMESCALEDB_URL=postgresql://neutryx:neutryx@localhost:5433/market_data

# Redis
REDIS_URL=redis://localhost:6379

# API
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Security
SECRET_KEY=your-secret-key-here
JWT_SECRET=your-jwt-secret-here

# Market Data
BLOOMBERG_HOST=localhost
BLOOMBERG_PORT=8194
REFINITIV_API_KEY=your-api-key

# Observability
PROMETHEUS_ENABLED=true
JAEGER_ENABLED=false

# Development
DEBUG=true
LOG_LEVEL=DEBUG
```

---

## Docker Deployment

### Docker Compose

#### Production Stack

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  # API Server
  neutryx-api:
    image: neutryx/neutryx-core:latest
    command: uvicorn neutryx.api.rest:app --host 0.0.0.0 --port 8000
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://neutryx:neutryx@postgres:5432/neutryx
      - REDIS_URL=redis://redis:6379
      - JAX_PLATFORM_NAME=cpu
    depends_on:
      - postgres
      - redis
    restart: unless-stopped
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2'
          memory: 4G

  # Worker (GPU-enabled for pricing)
  neutryx-worker:
    image: neutryx/neutryx-core:gpu
    command: python -m neutryx.worker
    environment:
      - DATABASE_URL=postgresql://neutryx:neutryx@postgres:5432/neutryx
      - REDIS_URL=redis://redis:6379
      - JAX_PLATFORM_NAME=gpu
    depends_on:
      - postgres
      - redis
    restart: unless-stopped
    deploy:
      replicas: 2
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # PostgreSQL Database
  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=neutryx
      - POSTGRES_USER=neutryx
      - POSTGRES_PASSWORD=neutryx
    volumes:
      - postgres-data:/var/lib/postgresql/data
    restart: unless-stopped

  # TimescaleDB for Market Data
  timescaledb:
    image: timescale/timescaledb:latest-pg15
    environment:
      - POSTGRES_DB=market_data
      - POSTGRES_USER=neutryx
      - POSTGRES_PASSWORD=neutryx
    volumes:
      - timescale-data:/var/lib/postgresql/data
    restart: unless-stopped

  # Redis Cache
  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis-data:/data
    restart: unless-stopped

  # Prometheus
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    restart: unless-stopped

  # Grafana
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_INSTALL_PLUGINS=grafana-piechart-panel
    volumes:
      - grafana-data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
    depends_on:
      - prometheus
    restart: unless-stopped

  # Jaeger (Distributed Tracing)
  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"  # Jaeger UI
      - "14268:14268"  # Collector HTTP
    environment:
      - COLLECTOR_ZIPKIN_HOST_PORT=:9411
    restart: unless-stopped

volumes:
  postgres-data:
  timescale-data:
  redis-data:
  prometheus-data:
  grafana-data:
```

#### Usage

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Scale workers
docker-compose up -d --scale neutryx-worker=5

# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

### Multi-Stage Dockerfile

```dockerfile
# syntax=docker/dockerfile:1

# Stage 1: Base
FROM python:3.10-slim as base
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Dependencies
FROM base as dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 3: Application
FROM dependencies as application
COPY . .
RUN pip install --no-cache-dir -e .

# Stage 4: Production
FROM application as production
RUN useradd -m -u 1000 neutryx && \
    chown -R neutryx:neutryx /app
USER neutryx
EXPOSE 8000
CMD ["uvicorn", "neutryx.api.rest:app", "--host", "0.0.0.0", "--port", "8000"]

# Stage 5: GPU-enabled
FROM nvidia/cuda:12.0.0-base-ubuntu22.04 as gpu-base
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

FROM gpu-base as gpu
WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
COPY . .
RUN pip3 install --no-cache-dir -e .
RUN useradd -m -u 1000 neutryx && chown -R neutryx:neutryx /app
USER neutryx
ENV JAX_PLATFORM_NAME=gpu
CMD ["python3", "-m", "neutryx.worker"]
```

### Building Images

```bash
# Build CPU image
docker build -t neutryx/neutryx-core:latest --target production .

# Build GPU image
docker build -t neutryx/neutryx-core:gpu --target gpu .

# Push to registry
docker push neutryx/neutryx-core:latest
docker push neutryx/neutryx-core:gpu
```

---

## Kubernetes Deployment

### Quick Start

```bash
# Deploy Neutryx API to Kubernetes
kubectl create namespace neutryx-api

# Deploy API services
kubectl apply -f deployment.yaml -n neutryx-api

# Check deployment status
kubectl get pods -n neutryx-api
kubectl get svc -n neutryx-api

# Access API service
kubectl port-forward svc/neutryx-api 8000:8000 -n neutryx-api
```

### Key Features

- **Auto-Scaling**: HorizontalPodAutoscaler for API pods based on CPU/memory
- **Multi-Region**: Active-active deployment across regions
- **High Availability**: Multiple replicas with load balancing
- **Security**: Network policies, pod security, secrets management
- **Monitoring**: Prometheus metrics, health checks, and logging

---

## Cloud Providers

### Google Cloud Platform (GCP)

#### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         GCP Region                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────────┐      │
│  │   GKE Cluster (neutryx-api-prod)                 │      │
│  │   ┌────────────┐  ┌────────────┐                │      │
│  │   │ API Pool   │  │ Spot Pool  │                │      │
│  │   │ (n2-std-4) │  │ (preempt)  │                │      │
│  │   └────────────┘  └────────────┘                │      │
│  └──────────────────────────────────────────────────┘      │
│                                                              │
│  ┌──────────────────────────────────────────────────┐      │
│  │   Cloud SQL (PostgreSQL)                          │      │
│  │   - HA: Multi-zone replication                    │      │
│  │   - Backups: Automated daily                      │      │
│  │   - Size: db-n1-highmem-8                         │      │
│  └──────────────────────────────────────────────────┘      │
│                                                              │
│  ┌──────────────────────────────────────────────────┐      │
│  │   Memorystore (Redis)                             │      │
│  │   - HA: Standard tier                             │      │
│  │   - Size: 10GB                                    │      │
│  └──────────────────────────────────────────────────┘      │
│                                                              │
│  ┌──────────────────────────────────────────────────┐      │
│  │   Cloud Storage (Backups)                         │      │
│  │   - Lifecycle: 30 days → nearline → coldline     │      │
│  └──────────────────────────────────────────────────┘      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

#### Setup

```bash
# 1. Create GKE cluster for API
gcloud container clusters create neutryx-api-prod \
  --region=us-central1 \
  --num-nodes=3 \
  --machine-type=n2-standard-4 \
  --enable-autoscaling \
  --min-nodes=2 \
  --max-nodes=20 \
  --enable-autorepair \
  --enable-autoupgrade \
  --addons=HorizontalPodAutoscaling,HttpLoadBalancing,GcePersistentDiskCsiDriver

# 2. Create Cloud SQL instance
gcloud sql instances create neutryx-api-db \
  --database-version=POSTGRES_15 \
  --tier=db-n1-standard-4 \
  --region=us-central1 \
  --availability-type=REGIONAL \
  --backup-start-time=03:00

# 3. Create Memorystore instance for caching
gcloud redis instances create neutryx-api-cache \
  --size=5 \
  --region=us-central1 \
  --tier=standard

# 5. Deploy application
kubectl apply -k k8s/overlays/prod
```

#### Cost Optimization

```bash
# Use preemptible VMs for worker nodes (save 60-80%)
gcloud container node-pools create spot-pool \
  --cluster=neutryx-prod \
  --region=us-central1 \
  --machine-type=n2-standard-16 \
  --spot \
  --num-nodes=5 \
  --enable-autoscaling \
  --min-nodes=0 \
  --max-nodes=50

# Set storage lifecycle policy
gsutil lifecycle set lifecycle-policy.json gs://neutryx-backups
```

### Amazon Web Services (AWS)

#### Architecture

```bash
# 1. Create EKS cluster
eksctl create cluster \
  --name neutryx-prod \
  --region us-east-1 \
  --nodegroup-name standard-workers \
  --node-type m5.2xlarge \
  --nodes 3 \
  --nodes-min 3 \
  --nodes-max 50 \
  --managed

# 2. Add GPU node group
eksctl create nodegroup \
  --cluster neutryx-prod \
  --region us-east-1 \
  --name gpu-workers \
  --node-type p3.2xlarge \
  --nodes 2 \
  --nodes-min 0 \
  --nodes-max 10

# 3. Create RDS instance
aws rds create-db-instance \
  --db-instance-identifier neutryx-db \
  --db-instance-class db.r5.2xlarge \
  --engine postgres \
  --engine-version 15.3 \
  --master-username neutryx \
  --master-user-password <password> \
  --allocated-storage 100 \
  --storage-type gp3 \
  --multi-az \
  --backup-retention-period 30

# 4. Create ElastiCache cluster
aws elasticache create-replication-group \
  --replication-group-id neutryx-cache \
  --replication-group-description "Neutryx Redis Cache" \
  --cache-node-type cache.r5.large \
  --engine redis \
  --num-cache-clusters 2 \
  --automatic-failover-enabled

# 5. Deploy application
kubectl apply -k k8s/overlays/prod
```

### Microsoft Azure

#### Setup

```bash
# 1. Create AKS cluster
az aks create \
  --resource-group neutryx-rg \
  --name neutryx-prod \
  --location eastus \
  --node-count 3 \
  --node-vm-size Standard_D8s_v3 \
  --enable-cluster-autoscaler \
  --min-count 3 \
  --max-count 50 \
  --generate-ssh-keys

# 2. Add GPU node pool
az aks nodepool add \
  --resource-group neutryx-rg \
  --cluster-name neutryx-prod \
  --name gpupool \
  --node-count 2 \
  --node-vm-size Standard_NC6s_v3 \
  --enable-cluster-autoscaler \
  --min-count 0 \
  --max-count 10

# 3. Create PostgreSQL server
az postgres flexible-server create \
  --resource-group neutryx-rg \
  --name neutryx-db \
  --location eastus \
  --admin-user neutryx \
  --admin-password <password> \
  --sku-name Standard_D8s_v3 \
  --tier GeneralPurpose \
  --version 15 \
  --high-availability Enabled

# 4. Create Redis cache
az redis create \
  --resource-group neutryx-rg \
  --name neutryx-cache \
  --location eastus \
  --sku Premium \
  --vm-size P1

# 5. Deploy application
kubectl apply -k k8s/overlays/prod
```

---

## Environment Configuration

### Development Environment

```yaml
# config/dev.yaml
environment: development
debug: true
log_level: DEBUG

api:
  host: 0.0.0.0
  port: 8000
  workers: 2
  reload: true

database:
  url: postgresql://neutryx:neutryx@localhost:5432/neutryx
  pool_size: 5
  max_overflow: 10

redis:
  url: redis://localhost:6379
  db: 0

security:
  enable_auth: false
  cors_origins: ["*"]

market_data:
  bloomberg:
    enabled: false
  refinitiv:
    enabled: false
  mock:
    enabled: true

observability:
  prometheus:
    enabled: true
    port: 9090
  jaeger:
    enabled: false
  profiling:
    enabled: true
```

### Staging Environment

```yaml
# config/staging.yaml
environment: staging
debug: false
log_level: INFO

api:
  host: 0.0.0.0
  port: 8000
  workers: 4
  reload: false

database:
  url: ${DATABASE_URL}
  pool_size: 20
  max_overflow: 40
  ssl_mode: require

redis:
  url: ${REDIS_URL}
  db: 0
  ssl: true

security:
  enable_auth: true
  jwt_secret: ${JWT_SECRET}
  cors_origins:
    - https://staging.neutryx.io

market_data:
  bloomberg:
    enabled: true
    host: ${BLOOMBERG_HOST}
    port: 8194
  refinitiv:
    enabled: true
    api_key: ${REFINITIV_API_KEY}

observability:
  prometheus:
    enabled: true
    port: 9090
  jaeger:
    enabled: true
    endpoint: ${JAEGER_ENDPOINT}
```

### Production Environment

```yaml
# config/prod.yaml
environment: production
debug: false
log_level: WARNING

api:
  host: 0.0.0.0
  port: 8000
  workers: 16
  reload: false
  keepalive: 5

database:
  url: ${DATABASE_URL}
  pool_size: 100
  max_overflow: 200
  ssl_mode: require
  statement_timeout: 30000

redis:
  url: ${REDIS_URL}
  db: 0
  ssl: true
  max_connections: 1000

security:
  enable_auth: true
  jwt_secret: ${JWT_SECRET}
  mfa_required: true
  cors_origins:
    - https://neutryx.io
    - https://app.neutryx.io

market_data:
  bloomberg:
    enabled: true
    host: ${BLOOMBERG_HOST}
    port: 8194
    timeout: 5000
    retry_attempts: 3
  refinitiv:
    enabled: true
    api_key: ${REFINITIV_API_KEY}
    endpoint: ${REFINITIV_ENDPOINT}

observability:
  prometheus:
    enabled: true
    port: 9090
  jaeger:
    enabled: true
    endpoint: ${JAEGER_ENDPOINT}
  profiling:
    enabled: false

rate_limiting:
  enabled: true
  requests_per_minute: 1000
  burst: 200

caching:
  enabled: true
  ttl: 300
  max_size: 10000
```

---

## Database Setup

### PostgreSQL

#### Schema Initialization

```sql
-- Create database
CREATE DATABASE neutryx;

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS trading;
CREATE SCHEMA IF NOT EXISTS risk;
CREATE SCHEMA IF NOT EXISTS market_data;
CREATE SCHEMA IF NOT EXISTS audit;

-- Create tables (example)
CREATE TABLE trading.trades (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    trade_date DATE NOT NULL,
    counterparty VARCHAR(100) NOT NULL,
    product_type VARCHAR(50) NOT NULL,
    notional DECIMAL(20, 2) NOT NULL,
    maturity_date DATE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE risk.positions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    portfolio_id UUID NOT NULL,
    instrument VARCHAR(50) NOT NULL,
    quantity DECIMAL(20, 4) NOT NULL,
    mark_to_market DECIMAL(20, 2),
    delta DECIMAL(10, 6),
    gamma DECIMAL(10, 6),
    vega DECIMAL(10, 6),
    as_of_date DATE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes
CREATE INDEX idx_trades_date ON trading.trades(trade_date);
CREATE INDEX idx_trades_counterparty ON trading.trades(counterparty);
CREATE INDEX idx_positions_portfolio ON risk.positions(portfolio_id);
CREATE INDEX idx_positions_date ON risk.positions(as_of_date);

-- Create audit trigger
CREATE OR REPLACE FUNCTION audit.log_changes()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' OR TG_OP = 'UPDATE' THEN
        INSERT INTO audit.log (table_name, action, row_data, changed_at, changed_by)
        VALUES (TG_TABLE_NAME, TG_OP, to_jsonb(NEW), NOW(), current_user);
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        INSERT INTO audit.log (table_name, action, row_data, changed_at, changed_by)
        VALUES (TG_TABLE_NAME, TG_OP, to_jsonb(OLD), NOW(), current_user);
        RETURN OLD;
    END IF;
END;
$$ LANGUAGE plpgsql;
```

### TimescaleDB (Market Data)

```sql
-- Create database
CREATE DATABASE market_data;
\c market_data

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Create market data table
CREATE TABLE ticks (
    timestamp TIMESTAMPTZ NOT NULL,
    instrument VARCHAR(50) NOT NULL,
    price DOUBLE PRECISION,
    volume BIGINT,
    bid DOUBLE PRECISION,
    ask DOUBLE PRECISION,
    source VARCHAR(20)
);

-- Convert to hypertable
SELECT create_hypertable('ticks', 'timestamp');

-- Enable compression
ALTER TABLE ticks SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'instrument'
);

-- Add compression policy (compress data older than 7 days)
SELECT add_compression_policy('ticks', INTERVAL '7 days');

-- Add retention policy (drop data older than 2 years)
SELECT add_retention_policy('ticks', INTERVAL '2 years');

-- Create continuous aggregates for OHLC bars
CREATE MATERIALIZED VIEW ohlc_1min
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 minute', timestamp) AS bucket,
    instrument,
    FIRST(price, timestamp) AS open,
    MAX(price) AS high,
    MIN(price) AS low,
    LAST(price, timestamp) AS close,
    SUM(volume) AS volume
FROM ticks
GROUP BY bucket, instrument;

-- Add refresh policy
SELECT add_continuous_aggregate_policy('ohlc_1min',
    start_offset => INTERVAL '1 hour',
    end_offset => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 minute');

-- Create indexes
CREATE INDEX idx_ticks_instrument ON ticks (instrument, timestamp DESC);
CREATE INDEX idx_ticks_source ON ticks (source, timestamp DESC);
```

### Migration Strategy

```bash
# Using Alembic for schema migrations
alembic init migrations

# Create migration
alembic revision --autogenerate -m "Initial schema"

# Apply migration
alembic upgrade head

# Rollback
alembic downgrade -1
```

---

## Market Data Infrastructure

### Bloomberg Integration

```python
# config/bloomberg.yaml
bloomberg:
  host: localhost
  port: 8194
  timeout: 30
  retry_attempts: 3
  retry_delay: 5

  subscriptions:
    - type: equity
      tickers:
        - AAPL US Equity
        - MSFT US Equity
        - GOOGL US Equity
      fields:
        - LAST_PRICE
        - BID
        - ASK
        - VOLUME

    - type: fx
      tickers:
        - EURUSD Curncy
        - GBPUSD Curncy
      fields:
        - LAST_PRICE
        - BID
        - ASK

  storage:
    backend: timescaledb
    batch_size: 1000
    flush_interval: 1
```

### Refinitiv Integration

```python
# config/refinitiv.yaml
refinitiv:
  platform: rdp  # or eikon
  api_key: ${REFINITIV_API_KEY}
  endpoint: https://api.refinitiv.com

  subscriptions:
    - service: pricing
      instruments:
        - EUR=
        - GBP=
      fields:
        - BID
        - ASK
        - LAST

  storage:
    backend: timescaledb
    compression: true
```

### Data Validation Pipeline

```python
# Validation rules
validation:
  price_range:
    enabled: true
    max_jump_pct: 20  # Alert if price jumps > 20%

  spread_check:
    enabled: true
    max_spread_pct: 5  # Alert if spread > 5% of mid

  volume_spike:
    enabled: true
    max_multiplier: 10  # Alert if volume > 10x average

  staleness:
    enabled: true
    max_age_seconds: 300  # Alert if data > 5 minutes old

  actions:
    on_validation_failure:
      - log_warning
      - send_alert
      - use_last_good_price
      - mark_stale
```

---

## Security & Compliance

### Authentication & Authorization

```yaml
# config/security.yaml
authentication:
  providers:
    - type: oauth2
      issuer: https://auth.neutryx.io
      client_id: ${OAUTH_CLIENT_ID}
      client_secret: ${OAUTH_CLIENT_SECRET}

    - type: ldap
      server: ldap://ldap.company.com
      base_dn: dc=company,dc=com
      bind_dn: cn=admin,dc=company,dc=com

    - type: saml
      entity_id: neutryx
      sso_url: https://sso.company.com
      certificate: ${SAML_CERT}

authorization:
  rbac:
    enabled: true
    roles:
      - name: trader
        permissions:
          - pricing:execute
          - pricing:view
          - portfolio:view
          - portfolio:trade

      - name: risk_manager
        permissions:
          - risk:view
          - risk:modify
          - limits:view
          - limits:modify

      - name: admin
        permissions: ["*"]

audit:
  enabled: true
  retention_days: 2555  # 7 years
  log_events:
    - authentication
    - authorization_failures
    - pricing_requests
    - trade_bookings
    - limit_modifications
    - configuration_changes
```

### Network Security

```yaml
# Network policies
network_policies:
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              name: neutryx
      ports:
        - protocol: TCP
          port: 8000

  egress:
    - to:
        - namespaceSelector:
            matchLabels:
              name: kube-system
      ports:
        - protocol: TCP
          port: 53  # DNS
    - to:
        - podSelector:
            matchLabels:
              app: postgres
      ports:
        - protocol: TCP
          port: 5432
```

### Encryption

```yaml
# Encryption configuration
encryption:
  at_rest:
    enabled: true
    provider: kms  # or vault
    key_id: ${KMS_KEY_ID}

  in_transit:
    tls:
      enabled: true
      min_version: "1.3"
      cert_path: /etc/certs/tls.crt
      key_path: /etc/certs/tls.key

  database:
    ssl_mode: require
    ssl_cert: /etc/certs/db-client.crt
    ssl_key: /etc/certs/db-client.key
```

---

## CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches:
      - main
    tags:
      - 'v*'

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -e ".[dev]"

      - name: Run tests
        run: pytest -v --cov=neutryx

      - name: Security scan
        run: |
          pip install bandit
          bandit -r src/neutryx

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2

      - name: Login to Docker Hub
        uses: docker/login-action@v2
        with:
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}

      - name: Build and push
        uses: docker/build-push-action@v4
        with:
          context: .
          push: true
          tags: |
            neutryx/neutryx-core:latest
            neutryx/neutryx-core:${{ github.sha }}

  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Configure kubectl
        uses: azure/k8s-set-context@v3
        with:
          method: kubeconfig
          kubeconfig: ${{ secrets.KUBE_CONFIG }}

      - name: Deploy to Kubernetes
        run: |
          kubectl set image deployment/neutryx-api \
            neutryx-api=neutryx/neutryx-core:${{ github.sha }} \
            -n neutryx
          kubectl rollout status deployment/neutryx-api -n neutryx
```

### GitLab CI/CD

```yaml
# .gitlab-ci.yml
stages:
  - test
  - build
  - deploy

test:
  stage: test
  image: python:3.10
  script:
    - pip install -r requirements.txt
    - pip install -e ".[dev]"
    - pytest -v --cov=neutryx

build:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker build -t neutryx/neutryx-core:$CI_COMMIT_SHA .
    - docker push neutryx/neutryx-core:$CI_COMMIT_SHA

deploy_production:
  stage: deploy
  image: bitnami/kubectl:latest
  script:
    - kubectl config use-context production
    - kubectl set image deployment/neutryx-api
        neutryx-api=neutryx/neutryx-core:$CI_COMMIT_SHA
        -n neutryx
  only:
    - main
```

---

## Monitoring & Observability

### Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'neutryx-api'
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
            - neutryx
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        action: keep
        regex: neutryx-api
      - source_labels: [__meta_kubernetes_pod_name]
        target_label: pod

  - job_name: 'neutryx-worker'
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
            - neutryx
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        action: keep
        regex: neutryx-worker

alerting:
  alertmanagers:
    - static_configs:
        - targets:
            - alertmanager:9093

rule_files:
  - /etc/prometheus/rules/*.yml
```

### Alert Rules

```yaml
# monitoring/rules/alerts.yml
groups:
  - name: neutryx_alerts
    interval: 30s
    rules:
      - alert: HighErrorRate
        expr: |
          rate(http_requests_total{status=~"5.."}[5m]) /
          rate(http_requests_total[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }}"

      - alert: HighLatency
        expr: |
          histogram_quantile(0.95,
            rate(http_request_duration_seconds_bucket[5m])
          ) > 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High latency detected"
          description: "P95 latency is {{ $value }}s"

      - alert: QueueBacklog
        expr: queue_depth > 10000
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Large queue backlog"
          description: "Queue depth is {{ $value }}"
```

### Grafana Dashboards

See [monitoring.md](monitoring.md) for detailed dashboard configurations.

---

## Troubleshooting

### Common Issues

#### 1. Pods Not Starting

```bash
# Check pod status
kubectl get pods -n neutryx

# Describe pod for events
kubectl describe pod <pod-name> -n neutryx

# Check logs
kubectl logs <pod-name> -n neutryx
kubectl logs <pod-name> -n neutryx --previous

# Common causes:
# - Image pull errors: Check registry credentials
# - Resource limits: Check CPU/memory requests
# - Config errors: Validate ConfigMaps and Secrets
```

#### 2. Database Connection Failures

```bash
# Test database connectivity
kubectl run -it --rm debug --image=postgres:15 --restart=Never -- \
  psql -h <db-host> -U <db-user> -d <db-name>

# Check connection pool
SELECT count(*) FROM pg_stat_activity WHERE datname = 'neutryx';

# Check for long-running queries
SELECT pid, now() - query_start AS duration, query
FROM pg_stat_activity
WHERE state = 'active' AND now() - query_start > interval '1 minute';
```

#### 3. High Memory Usage

```bash
# Check resource usage
kubectl top pods -n neutryx
kubectl top nodes

# Adjust memory limits in deployment
kubectl set resources deployment neutryx-api \
  --limits=memory=8Gi \
  --requests=memory=4Gi \
  -n neutryx
```

#### 4. Market Data Feed Issues

```bash
# Check adapter logs
kubectl logs -f deployment/neutryx-market-data -n neutryx

# Verify connectivity to vendor
kubectl exec -it <pod-name> -n neutryx -- \
  nc -zv bloomberg-server 8194

# Check data freshness
SELECT max(timestamp), instrument
FROM ticks
GROUP BY instrument;
```

### Performance Tuning

See [performance_tuning.md](performance_tuning.md) for detailed optimization strategies.

### Support Channels

- **Documentation**: https://docs.neutryx.io
- **GitHub Issues**: https://github.com/neutryx-lab/neutryx-core/issues
- **Community Forum**: https://community.neutryx.io
- **Email**: support@neutryx.io

---

## References

- [Architecture Guide](architecture.md)
- [Security Audit](security_audit.md)
- [Monitoring Guide](monitoring.md)
- [Performance Tuning](performance_tuning.md)
- [Troubleshooting Guide](troubleshooting.md)

---

**Document Ownership:** Neutryx DevOps Team
**Last Updated:** November 2025
**Next Review:** Q1 2026
