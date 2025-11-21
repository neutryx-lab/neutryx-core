"""Tests for the observability subsystem."""

from __future__ import annotations

import os
from typing import List

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from neutryx.infrastructure.observability import (
    ObservabilityConfig,
    ProfilingConfig,
    PrometheusConfig,
    TracingConfig,
    get_metrics_recorder,
    setup_observability,
)
from neutryx.infrastructure.observability.alerting import (
    AlertManager,
    AlertMessage,
    AlertingConfig,
    BaseNotifier,
)
from neutryx.infrastructure.observability.metrics import MetricsRecorder


class _CapturingNotifier(BaseNotifier):
    def __init__(self) -> None:
        self.messages: List[AlertMessage] = []

    def notify(self, message: AlertMessage) -> None:
        self.messages.append(message)


def test_observability_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEUTRYX_PROMETHEUS_ENABLED", "false")
    monkeypatch.setenv("NEUTRYX_TRACING_ENABLED", "true")
    monkeypatch.setenv("NEUTRYX_TRACING_EXPORTER", "otlp")
    monkeypatch.setenv("NEUTRYX_TRACING_SAMPLE_RATIO", "0.5")
    monkeypatch.setenv("NEUTRYX_PROFILING_MIN_DURATION", "1.5")
    monkeypatch.setenv("NEUTRYX_ALERT_ERROR_THRESHOLD", "0.25")
    config = ObservabilityConfig.from_env()

    assert config.metrics.enabled is False
    assert config.tracing.enabled is True
    assert config.tracing.exporter == "otlp"
    assert config.tracing.sample_ratio == pytest.approx(0.5)
    assert config.profiling.min_duration_seconds == pytest.approx(1.5)
    assert config.alerting.error_rate_threshold == pytest.approx(0.25)


def test_metrics_timer_records_success(monkeypatch: pytest.MonkeyPatch) -> None:
    # Use a unique namespace per test run to avoid collector clashes.
    namespace = f"test{os.getpid()}"
    config = PrometheusConfig(enabled=True, namespace=namespace, subsystem="unit")
    metrics = MetricsRecorder(config=config)

    with metrics.time(
        "unit.test_operation",
        labels={"channel": "test", "product": "demo"},
    ):
        pass

    metric_name = f"{namespace}_unit_operations_total"
    sample_labels = {
        "operation": "unit.test_operation",
        "status": "success",
        "channel": "test",
        "product": "demo",
    }
    from prometheus_client import REGISTRY

    value = REGISTRY.get_sample_value(metric_name, sample_labels)
    assert value == pytest.approx(1.0)


def test_alert_manager_emits_when_threshold_exceeded() -> None:
    notifier = _CapturingNotifier()
    config = AlertingConfig(
        enabled=True,
        evaluation_window_seconds=60.0,
        error_rate_threshold=0.25,
        latency_threshold_seconds=2.0,
        minimum_requests=4,
        cooldown_seconds=0.0,
    )
    manager = AlertManager(config=config, notifiers=[notifier])

    # 2 successes followed by 2 failures => 50% error rate over 4 requests.
    for success in (True, True, False, False):
        manager.record_event(
            kind="http",
            success=success,
            duration=0.1,
            attributes={"route": "/price/vanilla", "method": "POST"},
        )

    assert notifier.messages, "Expected alert to be emitted for elevated error rate"
    assert notifier.messages[0].name == "http_error_rate"


def test_prometheus_middleware_exposes_metrics(tmp_path) -> None:
    app = FastAPI()
    observability_config = ObservabilityConfig(
        metrics=PrometheusConfig(
            enabled=True,
            endpoint="/metrics-test",
            namespace="testapi",
            subsystem="http",
        ),
        tracing=TracingConfig(enabled=False),
        profiling=ProfilingConfig(enabled=False, output_dir=str(tmp_path)),
        alerting=AlertingConfig(enabled=False),
    )
    setup_observability(app, config=observability_config)

    @app.get("/ping")
    def ping() -> dict[str, str]:
        return {"status": "ok"}

    with TestClient(app) as client:
        response = client.get("/ping")
        assert response.status_code == 200

        metrics_response = client.get("/metrics-test")
        body = metrics_response.text
        assert "testapi_http_requests_total" in body
        assert 'route="/ping"' in body

    # Ensure global recorder matches the app state.
    global_metrics = get_metrics_recorder()
    assert isinstance(global_metrics, MetricsRecorder)
