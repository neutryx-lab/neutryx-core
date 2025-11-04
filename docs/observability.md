---
title: Observability Stack
summary: Prometheus metrics, Grafana dashboards, distributed tracing, profiling, and in-app alerting for Neutryx services.
---

# Observability Stack

Neutryx Core now ships with an integrated observability layer that covers runtime metrics, tracing, profiling, dashboards, and lightweight alerting. The instrumentation is enabled for both the FastAPI REST service (`neutryx.api.rest`) and the gRPC service (`neutryx.api.grpc`), and can be extended to custom workloads through a simple Python API.

## Prometheus Metrics

- **Endpoint**: When metrics are enabled, FastAPI automatically adds a `GET /metrics` endpoint that exposes Prometheus text exposition format.
- **Collectors**:
  - `neutryx_api_requests_total{method,route,status}` – HTTP ingress volume.
  - `neutryx_api_request_latency_seconds` – histogram with configurable buckets for request latency.
  - `neutryx_api_operations_total{operation,status,channel,product}` – domain-specific operations (pricing, XVA, portfolio).
  - `neutryx_api_operation_latency_seconds` – histogram tracking compute latency for domain operations.
- **Access**: Metrics are registered in the default Prometheus registry (`prometheus_client`). They can be scraped by pointing Prometheus to the service and polling the metrics endpoint.

### Usage in Code

```python
from neutryx.infrastructure.observability import get_metrics_recorder

metrics = get_metrics_recorder()

def run_pricing_job():
    with metrics.time("pricing.batch", labels={"channel": "offline", "product": "cva"}):
        # perform computation
        ...
```

## Grafana Dashboards

Provisioning-ready dashboards are provided under `dev/monitoring/grafana/`:

- `pricing_overview.json` – HTTP throughput, latency percentiles, error rates, and pricing/XVA operation latency.
- `xva_operations.json` – Domain operation throughput, failure rate, and transport-specific latency percentiles.

Import the JSON files into Grafana (`Dashboard -> Import -> Upload JSON`) and bind them to a Prometheus data source. Both dashboards expect a data source variable named `Prometheus`; Grafana will prompt for mapping the variable to an existing data source on import.

## Distributed Tracing (OpenTelemetry)

Tracing is powered by OpenTelemetry and can stream spans to either the console (default) or an OTLP collector.

Environment toggles:

| Variable | Default | Description |
| --- | --- | --- |
| `NEUTRYX_TRACING_ENABLED` | `false` | Master switch for tracing. |
| `NEUTRYX_TRACING_EXPORTER` | `console` | `console` or `otlp`. |
| `NEUTRYX_TRACING_OTLP_ENDPOINT` | `http://localhost:4318/v1/traces` | Collector endpoint for OTLP. |
| `NEUTRYX_TRACING_SAMPLE_RATIO` | `1.0` | Probability sampler between `0.0` and `1.0`. |
| `NEUTRYX_TRACING_FASTAPI` | `true` | Instrument FastAPI routes. |
| `NEUTRYX_TRACING_GRPC` | `true` | Instrument gRPC handlers. |

When enabled the OpenTelemetry `TracerProvider` is configured during app startup and emits spans tagged with `service.name = neutryx-core`.

## Performance Profiling

Slow request profiling uses the standard library `cProfile`:

- Enable via `NEUTRYX_PROFILING_ENABLED=true`.
- Profiles are stored under `dev/profiles/` (`.prof` dumps plus optional text summaries).
- Minimum capture duration defaults to 250 ms and can be changed with `NEUTRYX_PROFILING_MIN_DURATION`.
- The middleware excludes the `/metrics` and `/healthz` endpoints by default to avoid noisy captures.

The generated `.prof` files can be analysed with `snakeviz`, `pyinstrument`, or the `pstats` module:

```bash
python -m pstats dev/profiles/20240531-101500_post_price_vanilla.prof
```

## Alerting and Notifications

A lightweight in-process alert manager provides error-rate and latency guards without external dependencies:

- Enabled via `NEUTRYX_ALERTING_ENABLED=true`.
- Configurable evaluation window (`NEUTRYX_ALERT_WINDOW`), error-rate threshold, latency threshold, minimum request volume, and cooldown between notifications.
- Alerts are emitted through the standard logging subsystem (`neutryx.alerts` logger). Custom notifiers can be added by supplying `BaseNotifier` instances to `setup_observability`.

Example log message when the HTTP error rate breaches the configured threshold:

```
[neutryx.alerts] [http_error_rate] HTTP error rate 12.50% exceeds threshold 5.00% | details={'total': 80, 'errors': 10, 'error_rate': 0.125}
```

## Configuration Summary

All observability features accept environment-based overrides. The key variables are summarised below:

| Feature | Variable(s) | Purpose |
| --- | --- | --- |
| Metrics | `NEUTRYX_PROMETHEUS_ENABLED`, `NEUTRYX_PROMETHEUS_ENDPOINT`, `NEUTRYX_PROMETHEUS_NAMESPACE`, `NEUTRYX_PROMETHEUS_SUBSYSTEM` | Control Prometheus export. |
| Tracing | `NEUTRYX_TRACING_*` | Configure OpenTelemetry tracing. |
| Profiling | `NEUTRYX_PROFILING_*` | Manage profiling middleware. |
| Alerting | `NEUTRYX_ALERT_*` | Tune in-process alert rules. |

Defaults are defined in `neutryx.infrastructure.observability.config.ObservabilityConfig`. Custom configurations can also be supplied programmatically:

```python
from neutryx.infrastructure.observability import ObservabilityConfig, setup_observability

config = ObservabilityConfig.from_env()
config.profiling.enabled = True
config.profiling.min_duration_seconds = 0.5

app = create_app()
setup_observability(app, config=config)
```

## Quickstart

1. **Install dependencies** (now part of `pyproject.toml`): `prometheus-client`, `opentelemetry-sdk`, `opentelemetry-instrumentation-fastapi`, `opentelemetry-instrumentation-grpc`, and the OTLP HTTP exporter.
2. **Run the API server** with observability toggles, e.g.:

   ```bash
   NEUTRYX_PROMETHEUS_ENABLED=true \
   NEUTRYX_TRACING_ENABLED=true \
   NEUTRYX_ALERTING_ENABLED=true \
   uvicorn neutryx.api.rest:create_app --factory --reload
   ```

3. **Point Prometheus** at `http://localhost:8000/metrics` and import the Grafana dashboards.
4. **Inspect traces** in your chosen collector (Jaeger, Tempo, etc.) when OTLP export is enabled.
5. **Review alerts** in application logs or extend with custom notifiers for Slack, PagerDuty, or other tooling.

With these components enabled, Neutryx services gain immediate visibility into request health, latency, and pricing workloads, while remaining extensible for enterprise observability stacks.
