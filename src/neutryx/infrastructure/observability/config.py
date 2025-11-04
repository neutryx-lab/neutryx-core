"""Configuration objects for the observability subsystem."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Iterable, Optional


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_choice(name: str, default: str, choices: Iterable[str]) -> str:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip().lower()
    valid = {choice.lower(): choice for choice in choices}
    if value in valid:
        return valid[value]
    return default


@dataclass
class PrometheusConfig:
    """Configuration for Prometheus metrics exposure."""

    enabled: bool = True
    endpoint: str = "/metrics"
    namespace: str = "neutryx"
    subsystem: str = "api"
    latency_buckets: tuple[float, ...] = (
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
    )


@dataclass
class TracingConfig:
    """Configuration for distributed tracing."""

    enabled: bool = False
    service_name: str = "neutryx-core"
    exporter: str = "console"
    otlp_endpoint: str = "http://localhost:4318/v1/traces"
    otlp_insecure: bool = True
    sample_ratio: float = 1.0
    instrument_fastapi: bool = True
    instrument_grpc: bool = True


@dataclass
class ProfilingConfig:
    """Configuration for request-level profiling."""

    enabled: bool = False
    output_dir: str = "dev/profiles"
    min_duration_seconds: float = 0.25
    retain: int = 20
    emit_text_reports: bool = True
    include_paths: Optional[tuple[str, ...]] = None
    exclude_paths: tuple[str, ...] = ("/metrics", "/healthz")


@dataclass
class AlertingConfig:
    """Configuration for lightweight, in-process alerting."""

    enabled: bool = False
    evaluation_window_seconds: float = 300.0
    error_rate_threshold: float = 0.05
    latency_threshold_seconds: float = 2.0
    cooldown_seconds: float = 120.0
    minimum_requests: int = 25


@dataclass
class ObservabilityConfig:
    """Top-level configuration container."""

    metrics: PrometheusConfig = field(default_factory=PrometheusConfig)
    tracing: TracingConfig = field(default_factory=TracingConfig)
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)
    alerting: AlertingConfig = field(default_factory=AlertingConfig)

    @classmethod
    def from_env(cls) -> "ObservabilityConfig":
        """Create configuration populated from environment variables."""

        metrics = PrometheusConfig(
            enabled=_env_flag("NEUTRYX_PROMETHEUS_ENABLED", True),
            endpoint=os.getenv("NEUTRYX_PROMETHEUS_ENDPOINT", "/metrics"),
            namespace=os.getenv("NEUTRYX_PROMETHEUS_NAMESPACE", "neutryx"),
            subsystem=os.getenv("NEUTRYX_PROMETHEUS_SUBSYSTEM", "api"),
        )

        tracing = TracingConfig(
            enabled=_env_flag("NEUTRYX_TRACING_ENABLED", False),
            service_name=os.getenv("NEUTRYX_TRACING_SERVICE_NAME", "neutryx-core"),
            exporter=_env_choice("NEUTRYX_TRACING_EXPORTER", "console", {"console", "otlp"}),
            otlp_endpoint=os.getenv("NEUTRYX_TRACING_OTLP_ENDPOINT", "http://localhost:4318/v1/traces"),
            otlp_insecure=_env_flag("NEUTRYX_TRACING_OTLP_INSECURE", True),
            sample_ratio=_env_float("NEUTRYX_TRACING_SAMPLE_RATIO", 1.0),
            instrument_fastapi=_env_flag("NEUTRYX_TRACING_FASTAPI", True),
            instrument_grpc=_env_flag("NEUTRYX_TRACING_GRPC", True),
        )

        profiling = ProfilingConfig(
            enabled=_env_flag("NEUTRYX_PROFILING_ENABLED", False),
            output_dir=os.getenv("NEUTRYX_PROFILING_OUTPUT_DIR", "dev/profiles"),
            min_duration_seconds=_env_float("NEUTRYX_PROFILING_MIN_DURATION", 0.25),
            retain=_env_int("NEUTRYX_PROFILING_RETAIN", 20),
            emit_text_reports=_env_flag("NEUTRYX_PROFILING_TEXT", True),
        )

        alerting = AlertingConfig(
            enabled=_env_flag("NEUTRYX_ALERTING_ENABLED", False),
            evaluation_window_seconds=_env_float("NEUTRYX_ALERT_WINDOW", 300.0),
            error_rate_threshold=_env_float("NEUTRYX_ALERT_ERROR_THRESHOLD", 0.05),
            latency_threshold_seconds=_env_float("NEUTRYX_ALERT_LATENCY_THRESHOLD", 2.0),
            cooldown_seconds=_env_float("NEUTRYX_ALERT_COOLDOWN", 120.0),
            minimum_requests=_env_int("NEUTRYX_ALERT_MIN_REQUESTS", 25),
        )

        return cls(
            metrics=metrics,
            tracing=tracing,
            profiling=profiling,
            alerting=alerting,
        )


__all__ = [
    "AlertingConfig",
    "ObservabilityConfig",
    "ProfilingConfig",
    "PrometheusConfig",
    "TracingConfig",
]
