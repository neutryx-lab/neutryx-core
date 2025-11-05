# Monitoring and Observability Guide

This guide covers the comprehensive monitoring and observability stack for Neutryx Core, including metrics collection, distributed tracing, performance profiling, and alerting.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Prometheus Metrics](#prometheus-metrics)
- [Grafana Dashboards](#grafana-dashboards)
- [Distributed Tracing](#distributed-tracing)
- [Performance Profiling](#performance-profiling)
- [Alerting and Notifications](#alerting-and-notifications)
- [Configuration](#configuration)
- [Production Deployment](#production-deployment)

## Overview

Neutryx Core includes a production-ready observability stack with:

- **Prometheus**: Time-series metrics collection and storage
- **Grafana**: Rich visualization and dashboards
- **Jaeger**: Distributed tracing with OpenTelemetry
- **AlertManager**: Alert routing and notification management
- **Custom Metrics**: Domain-specific metrics for pricing, risk, and calibration operations

## Quick Start

### 1. Start the Monitoring Stack

```bash
# Navigate to the monitoring directory
cd dev/monitoring

# Start all monitoring services
docker-compose up -d

# Verify services are running
docker-compose ps
```

### 2. Access the Services

- **Grafana**: http://localhost:3000 (admin/neutryx)
- **Prometheus**: http://localhost:9090
- **Jaeger UI**: http://localhost:16686
- **AlertManager**: http://localhost:9093

### 3. Enable Observability in Your Application

```python
from neutryx.infrastructure.observability import ObservabilityConfig, setup_observability
from fastapi import FastAPI

app = FastAPI()

# Configure observability with environment variables or code
config = ObservabilityConfig.from_env()
observability = setup_observability(app, config=config)

# Access the metrics recorder
metrics = observability.metrics
```

### 4. Start the Neutryx API

```bash
# Enable all observability features
export NEUTRYX_PROMETHEUS_ENABLED=true
export NEUTRYX_TRACING_ENABLED=true
export NEUTRYX_TRACING_EXPORTER=otlp
export NEUTRYX_TRACING_OTLP_ENDPOINT=http://localhost:4318/v1/traces
export NEUTRYX_PROFILING_ENABLED=true
export NEUTRYX_ALERTING_ENABLED=true

# Start the API
uvicorn neutryx.api.rest:app --host 0.0.0.0 --port 8000
```

## Prometheus Metrics

### Available Metrics

Neutryx Core exposes the following custom metrics:

#### HTTP Metrics

- `neutryx_api_requests_total` - Total HTTP requests by method, route, and status
- `neutryx_api_request_latency_seconds` - Request latency histogram

#### Operation Metrics

- `neutryx_api_operations_total` - Total domain operations by type, status, channel, and product
- `neutryx_api_operation_latency_seconds` - Operation latency histogram

#### Pricing Metrics

- `neutryx_api_pricing_calculations_total` - Pricing calculations by product type, model, and status
- `neutryx_api_monte_carlo_paths` - Distribution of Monte Carlo path counts
- `neutryx_api_xva_calculations_total` - XVA calculations (CVA, FVA, MVA) by type and status

#### Calibration Metrics

- `neutryx_api_calibration_iterations` - Distribution of calibration iterations by model

### Recording Custom Metrics

```python
from neutryx.infrastructure.observability import get_metrics_recorder

metrics = get_metrics_recorder()

# Record a pricing calculation
metrics.record_pricing_calculation(
    product_type="vanilla_option",
    model="black_scholes",
    success=True
)

# Record Monte Carlo simulation
metrics.record_monte_carlo_paths(
    product_type="asian_option",
    num_paths=100000
)

# Record XVA calculation
metrics.record_xva_calculation(
    xva_type="cva",
    success=True
)

# Record calibration
metrics.record_calibration_iterations(
    model="heston",
    iterations=150
)

# Time an operation
with metrics.time("custom_operation", labels={"channel": "batch", "product": "portfolio"}):
    # Your code here
    pass
```

### Querying Metrics

Access the Prometheus UI at http://localhost:9090 to run queries:

```promql
# Request rate by endpoint
rate(neutryx_api_requests_total[5m])

# 95th percentile latency
histogram_quantile(0.95, rate(neutryx_api_request_latency_seconds_bucket[5m]))

# Error rate
sum(rate(neutryx_api_requests_total{status=~"5.."}[5m])) / sum(rate(neutryx_api_requests_total[5m]))

# Pricing calculations per second
rate(neutryx_api_pricing_calculations_total[1m])

# Average Monte Carlo paths
avg(rate(neutryx_api_monte_carlo_paths_sum[5m]) / rate(neutryx_api_monte_carlo_paths_count[5m]))
```

## Grafana Dashboards

### Pre-built Dashboards

Two production-ready dashboards are included:

#### 1. Neutryx Core - Overview
**Location**: `dev/monitoring/grafana/dashboards/neutryx-overview.json`

Features:
- HTTP request rate and status codes
- Request latency (p95)
- Pricing calculations by product type
- XVA calculations rate
- Monte Carlo path distribution
- Calibration iterations

#### 2. Neutryx Core - Performance Analysis
**Location**: `dev/monitoring/grafana/dashboards/neutryx-performance.json`

Features:
- Operation latency percentiles (p50, p95, p99)
- Operations throughput table
- Error rate by operation
- Request latency heatmap

### Importing Dashboards

Dashboards are automatically provisioned when using docker-compose. To manually import:

1. Open Grafana at http://localhost:3000
2. Navigate to **Dashboards** → **Import**
3. Upload the JSON file or paste its content
4. Select the Prometheus datasource
5. Click **Import**

### Creating Custom Dashboards

Use PromQL queries in Grafana panels:

```promql
# Panel: Pricing Success Rate
sum(rate(neutryx_api_pricing_calculations_total{status="success"}[5m]))
/
sum(rate(neutryx_api_pricing_calculations_total[5m]))

# Panel: Top 5 Slowest Operations
topk(5, histogram_quantile(0.95, sum(rate(neutryx_api_operation_latency_seconds_bucket[5m])) by (le, operation)))
```

## Distributed Tracing

### Overview

Neutryx uses OpenTelemetry for distributed tracing, with Jaeger as the backend.

### Configuration

```bash
# Enable tracing
export NEUTRYX_TRACING_ENABLED=true

# Configure exporter (console or otlp)
export NEUTRYX_TRACING_EXPORTER=otlp
export NEUTRYX_TRACING_OTLP_ENDPOINT=http://localhost:4318/v1/traces

# Set service name
export NEUTRYX_TRACING_SERVICE_NAME=neutryx-core

# Configure sampling (0.0 to 1.0)
export NEUTRYX_TRACING_SAMPLE_RATIO=1.0

# Enable automatic instrumentation
export NEUTRYX_TRACING_FASTAPI=true
export NEUTRYX_TRACING_GRPC=true
```

### Viewing Traces

1. Access Jaeger UI at http://localhost:16686
2. Select **neutryx-core** from the Service dropdown
3. Click **Find Traces** to view recent traces
4. Click on a trace to see detailed span information

### Adding Custom Spans

```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

def complex_calculation():
    with tracer.start_as_current_span("complex_calculation") as span:
        span.set_attribute("calculation.type", "monte_carlo")
        span.set_attribute("paths", 100000)

        # Your calculation logic
        result = perform_calculation()

        span.set_attribute("result.value", result)
        return result
```

### Trace Propagation

Traces are automatically propagated across:
- FastAPI HTTP requests
- gRPC calls
- Internal service calls

Headers used for propagation:
- `traceparent`
- `tracestate`

## Performance Profiling

### Overview

The profiling middleware captures detailed performance profiles for slow requests, using Python's cProfile.

### Configuration

```bash
# Enable profiling
export NEUTRYX_PROFILING_ENABLED=true

# Set output directory
export NEUTRYX_PROFILING_OUTPUT_DIR=dev/profiles

# Minimum duration to trigger profiling (seconds)
export NEUTRYX_PROFILING_MIN_DURATION=0.25

# Number of profile files to retain
export NEUTRYX_PROFILING_RETAIN=20

# Generate text reports alongside binary profiles
export NEUTRYX_PROFILING_TEXT=true
```

### Analyzing Profiles

Profile files are saved to `dev/profiles/` with timestamps:

```bash
# List generated profiles
ls -lh dev/profiles/

# View text report
cat dev/profiles/20250105-143022_post_price_vanilla.txt

# Analyze binary profile with pstats
python -m pstats dev/profiles/20250105-143022_post_price_vanilla.prof
```

### Using SnakeViz for Visualization

```bash
# Install snakeviz
pip install snakeviz

# Visualize profile
snakeviz dev/profiles/20250105-143022_post_price_vanilla.prof
```

## Alerting and Notifications

### Built-in Alerts

Alert rules are defined in `dev/monitoring/prometheus/rules/alerts.yml`:

#### Critical Alerts
- **HighErrorRate**: Error rate > 5% for 5 minutes
- **ServiceDown**: Service unavailable for > 1 minute

#### Warning Alerts
- **HighLatency**: p95 latency > 2s for 5 minutes
- **PricingCalculationFailures**: Pricing failures > 0.1/sec for 3 minutes
- **XVACalculationFailures**: XVA failures > 0.05/sec for 3 minutes
- **SlowOperations**: Operation p95 latency > 5s for 5 minutes

#### Info Alerts
- **HighMonteCarloPathCount**: p95 paths > 1M for 10 minutes
- **HighCalibrationIterations**: p95 iterations > 500 for 10 minutes

### In-Process Alerting

Neutryx also includes lightweight in-process alerting:

```bash
# Enable alerting
export NEUTRYX_ALERTING_ENABLED=true

# Configuration
export NEUTRYX_ALERT_WINDOW=300  # 5 minutes
export NEUTRYX_ALERT_ERROR_THRESHOLD=0.05  # 5%
export NEUTRYX_ALERT_LATENCY_THRESHOLD=2.0  # 2 seconds
export NEUTRYX_ALERT_COOLDOWN=120  # 2 minutes
export NEUTRYX_ALERT_MIN_REQUESTS=25  # Minimum requests before alerting
```

### Custom Notifiers

Implement custom alert notifiers:

```python
from neutryx.infrastructure.observability.alerting import BaseNotifier, AlertMessage

class SlackNotifier(BaseNotifier):
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    def notify(self, message: AlertMessage) -> None:
        # Send to Slack
        import requests
        requests.post(self.webhook_url, json={
            "text": f"[{message.severity.upper()}] {message.name}",
            "attachments": [{
                "text": message.summary,
                "fields": [
                    {"title": k, "value": str(v), "short": True}
                    for k, v in message.details.items()
                ]
            }]
        })

# Use in setup
from neutryx.infrastructure.observability import setup_observability

observability = setup_observability(
    app,
    notifiers=[SlackNotifier(webhook_url="YOUR_WEBHOOK_URL")]
)
```

### AlertManager Configuration

Configure AlertManager routing in `dev/monitoring/alertmanager/config.yml`:

```yaml
receivers:
  - name: 'slack-critical'
    slack_configs:
      - api_url: 'YOUR_SLACK_WEBHOOK_URL'
        channel: '#alerts-critical'
        title: 'Critical Alert: {{ .GroupLabels.alertname }}'
```

## Configuration

### Environment Variables

Complete list of observability configuration options:

```bash
# Prometheus Metrics
NEUTRYX_PROMETHEUS_ENABLED=true
NEUTRYX_PROMETHEUS_ENDPOINT=/metrics
NEUTRYX_PROMETHEUS_NAMESPACE=neutryx
NEUTRYX_PROMETHEUS_SUBSYSTEM=api

# Distributed Tracing
NEUTRYX_TRACING_ENABLED=true
NEUTRYX_TRACING_SERVICE_NAME=neutryx-core
NEUTRYX_TRACING_EXPORTER=otlp  # console or otlp
NEUTRYX_TRACING_OTLP_ENDPOINT=http://localhost:4318/v1/traces
NEUTRYX_TRACING_OTLP_INSECURE=true
NEUTRYX_TRACING_SAMPLE_RATIO=1.0
NEUTRYX_TRACING_FASTAPI=true
NEUTRYX_TRACING_GRPC=true

# Performance Profiling
NEUTRYX_PROFILING_ENABLED=true
NEUTRYX_PROFILING_OUTPUT_DIR=dev/profiles
NEUTRYX_PROFILING_MIN_DURATION=0.25
NEUTRYX_PROFILING_RETAIN=20
NEUTRYX_PROFILING_TEXT=true

# Alerting
NEUTRYX_ALERTING_ENABLED=true
NEUTRYX_ALERT_WINDOW=300
NEUTRYX_ALERT_ERROR_THRESHOLD=0.05
NEUTRYX_ALERT_LATENCY_THRESHOLD=2.0
NEUTRYX_ALERT_COOLDOWN=120
NEUTRYX_ALERT_MIN_REQUESTS=25
```

### Programmatic Configuration

```python
from neutryx.infrastructure.observability import (
    ObservabilityConfig,
    PrometheusConfig,
    TracingConfig,
    ProfilingConfig,
    AlertingConfig,
    setup_observability
)

config = ObservabilityConfig(
    metrics=PrometheusConfig(
        enabled=True,
        namespace="neutryx",
        subsystem="pricing"
    ),
    tracing=TracingConfig(
        enabled=True,
        exporter="otlp",
        sample_ratio=0.1  # Sample 10% of traces
    ),
    profiling=ProfilingConfig(
        enabled=True,
        min_duration_seconds=0.5
    ),
    alerting=AlertingConfig(
        enabled=True,
        error_rate_threshold=0.02  # 2%
    )
)

observability = setup_observability(app, config=config)
```

## Production Deployment

### Best Practices

1. **Metrics Retention**: Configure appropriate retention periods
   ```bash
   --storage.tsdb.retention.time=90d  # 90 days
   ```

2. **Sampling**: Use sampling for high-traffic services
   ```bash
   NEUTRYX_TRACING_SAMPLE_RATIO=0.01  # 1% sampling
   ```

3. **Security**: Enable authentication and TLS
   ```yaml
   # prometheus.yml
   global:
     external_labels:
       cluster: 'production'
   ```

4. **High Availability**: Deploy Prometheus with Thanos or Cortex

5. **Alert Routing**: Configure proper escalation paths
   ```yaml
   # alertmanager config
   routes:
     - match:
         severity: critical
       receiver: pagerduty
   ```

### Kubernetes Deployment

For Kubernetes deployments, use the Prometheus Operator:

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: neutryx-api
spec:
  selector:
    matchLabels:
      app: neutryx-api
  endpoints:
    - port: metrics
      interval: 30s
```

### Cloud-Managed Services

Consider using managed services:
- **AWS**: CloudWatch + X-Ray
- **GCP**: Cloud Monitoring + Cloud Trace
- **Azure**: Application Insights

Configure exporters accordingly:
```python
# For AWS X-Ray
config = TracingConfig(
    exporter="xray",
    # Additional X-Ray configuration
)
```

## Troubleshooting

### Metrics Not Appearing

1. Check if Prometheus is scraping successfully:
   ```bash
   curl http://localhost:9090/api/v1/targets
   ```

2. Verify metrics endpoint is accessible:
   ```bash
   curl http://localhost:8000/metrics
   ```

3. Check Prometheus logs:
   ```bash
   docker-compose logs prometheus
   ```

### Traces Not Visible

1. Verify Jaeger is receiving traces:
   ```bash
   docker-compose logs jaeger
   ```

2. Check OTLP endpoint configuration:
   ```bash
   echo $NEUTRYX_TRACING_OTLP_ENDPOINT
   ```

3. Verify sampling ratio:
   ```bash
   echo $NEUTRYX_TRACING_SAMPLE_RATIO
   ```

### High Memory Usage

1. Reduce trace sampling:
   ```bash
   NEUTRYX_TRACING_SAMPLE_RATIO=0.1
   ```

2. Disable profiling for high-traffic endpoints:
   ```bash
   NEUTRYX_PROFILING_MIN_DURATION=2.0
   ```

3. Adjust Prometheus retention:
   ```bash
   --storage.tsdb.retention.time=15d
   ```

## Resources

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Jaeger Documentation](https://www.jaegertracing.io/docs/)
- [OpenTelemetry Documentation](https://opentelemetry.io/docs/)
- [AlertManager Documentation](https://prometheus.io/docs/alerting/latest/alertmanager/)
