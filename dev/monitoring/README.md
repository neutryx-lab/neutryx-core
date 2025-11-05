# Neutryx Monitoring Stack

This directory contains a complete monitoring and observability stack for Neutryx Core.

## Quick Start

### 1. Start All Services

```bash
docker-compose up -d
```

This will start:
- Prometheus (metrics) on port 9090
- Grafana (dashboards) on port 3000
- Jaeger (tracing) on port 16686
- AlertManager (alerts) on port 9093
- Node Exporter (system metrics) on port 9100
- cAdvisor (container metrics) on port 8080

### 2. Access the Services

- **Grafana**: http://localhost:3000
  - Username: `admin`
  - Password: `neutryx`

- **Prometheus**: http://localhost:9090
- **Jaeger**: http://localhost:16686
- **AlertManager**: http://localhost:9093

### 3. Configure Neutryx API

Create a `.env` file or export environment variables:

```bash
# Enable all observability features
export NEUTRYX_PROMETHEUS_ENABLED=true
export NEUTRYX_TRACING_ENABLED=true
export NEUTRYX_TRACING_EXPORTER=otlp
export NEUTRYX_TRACING_OTLP_ENDPOINT=http://localhost:4318/v1/traces
export NEUTRYX_PROFILING_ENABLED=true
export NEUTRYX_ALERTING_ENABLED=true

# Start the API
cd ../..
uvicorn neutryx.api.rest:app --host 0.0.0.0 --port 8000
```

### 4. View Dashboards

Open Grafana and navigate to **Dashboards**:
- **Neutryx Core - Overview**: High-level metrics and business KPIs
- **Neutryx Core - Performance Analysis**: Detailed performance analysis

## Directory Structure

```
dev/monitoring/
├── docker-compose.yml              # Main orchestration file
├── README.md                       # This file
├── prometheus/
│   ├── prometheus.yml              # Prometheus configuration
│   └── rules/
│       └── alerts.yml              # Alert rules
├── grafana/
│   ├── datasources.yml             # Datasource configuration
│   └── dashboards/
│       ├── neutryx-overview.json   # Overview dashboard
│       └── neutryx-performance.json # Performance dashboard
└── alertmanager/
    └── config.yml                  # AlertManager configuration
```

## Pre-configured Dashboards

### Neutryx Core - Overview
- HTTP request rate and status codes
- Request latency (p95)
- Pricing calculations by product type
- XVA calculations rate
- Monte Carlo path distribution
- Calibration iterations

### Neutryx Core - Performance Analysis
- Operation latency percentiles (p50, p95, p99)
- Operations throughput table
- Error rate by operation
- Request latency heatmap

## Alert Rules

Pre-configured alerts include:
- **HighErrorRate**: Triggers when error rate > 5%
- **HighLatency**: Triggers when p95 latency > 2s
- **PricingCalculationFailures**: Monitors pricing calculation failures
- **XVACalculationFailures**: Monitors XVA calculation failures
- **ServiceDown**: Triggers when service is unavailable

## Customization

### Adding Custom Metrics

1. Add metrics in your code:
```python
from neutryx.infrastructure.observability import get_metrics_recorder

metrics = get_metrics_recorder()
metrics.record_pricing_calculation(
    product_type="exotic_option",
    model="monte_carlo",
    success=True
)
```

2. Query in Prometheus:
```promql
rate(neutryx_api_pricing_calculations_total{product_type="exotic_option"}[5m])
```

### Adding Custom Dashboards

1. Create dashboard in Grafana UI
2. Export as JSON: **Dashboard Settings** → **JSON Model**
3. Save to `grafana/dashboards/`
4. Restart Grafana: `docker-compose restart grafana`

### Configuring Notifications

Edit `alertmanager/config.yml` to add notification channels:

```yaml
receivers:
  - name: 'slack-alerts'
    slack_configs:
      - api_url: 'YOUR_SLACK_WEBHOOK_URL'
        channel: '#alerts'
```

## Management Commands

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f [service_name]

# Restart a service
docker-compose restart [service_name]

# Remove all data (caution!)
docker-compose down -v
```

## Troubleshooting

### Metrics not showing up
1. Check if Prometheus is scraping the target:
   ```bash
   curl http://localhost:9090/api/v1/targets
   ```

2. Verify the Neutryx API is exposing metrics:
   ```bash
   curl http://localhost:8000/metrics
   ```

### Grafana dashboards not loading
1. Check datasource configuration in Grafana
2. Verify Prometheus is accessible: http://localhost:9090

### Jaeger not showing traces
1. Verify OTLP endpoint configuration:
   ```bash
   echo $NEUTRYX_TRACING_OTLP_ENDPOINT
   ```

2. Check Jaeger logs:
   ```bash
   docker-compose logs jaeger
   ```

## Additional Resources

- [Full Monitoring Documentation](../../docs/monitoring.md)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Jaeger Documentation](https://www.jaegertracing.io/docs/)
