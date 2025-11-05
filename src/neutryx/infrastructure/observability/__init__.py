"""Unified entrypoints for the observability stack."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

from fastapi import FastAPI

from .alerting import BaseAlertManager, BaseNotifier, create_alert_manager
from .config import (
    AlertingConfig,
    ObservabilityConfig,
    ProfilingConfig,
    PrometheusConfig,
    TracingConfig,
)
from .metrics import (
    MetricsRecorder,
    PrometheusMiddleware,
    configure_metrics,
    create_metrics_endpoint,
    get_metrics_recorder,
    reset_metrics_recorder,
)
from .profiling import ProfilingMiddleware
from .tracing import setup_tracing


@dataclass
class ObservabilityState:
    """Aggregates the configured observability components."""

    config: ObservabilityConfig
    metrics: MetricsRecorder
    alert_manager: BaseAlertManager
    tracer_provider: Optional[Any]


def _has_middleware(app: FastAPI, middleware_cls) -> bool:
    return any(middleware.cls is middleware_cls for middleware in app.user_middleware)


def _register_metrics_route(app: FastAPI, config: PrometheusConfig) -> None:
    if any(getattr(route, "path", None) == config.endpoint for route in app.router.routes):
        return
    handler = create_metrics_endpoint()
    app.add_api_route(
        config.endpoint,
        handler,
        methods=["GET"],
        include_in_schema=False,
        name="metrics",
    )


def setup_observability(
    app: Optional[FastAPI] = None,
    *,
    config: Optional[ObservabilityConfig] = None,
    notifiers: Optional[Sequence[BaseNotifier]] = None,
) -> ObservabilityState:
    """Initialise metrics, tracing, profiling, and alerting."""

    config = config or ObservabilityConfig.from_env()
    alert_manager = create_alert_manager(config.alerting, notifiers=notifiers)
    metrics = configure_metrics(config.metrics, alert_manager=alert_manager)

    if app is not None and config.metrics.enabled:
        if not _has_middleware(app, PrometheusMiddleware):
            app.add_middleware(PrometheusMiddleware, recorder=metrics)
        _register_metrics_route(app, config.metrics)

    if app is not None and config.profiling.enabled:
        if not _has_middleware(app, ProfilingMiddleware):
            app.add_middleware(ProfilingMiddleware, config=config.profiling)

    tracer_provider = setup_tracing(config.tracing, app=app)

    state = ObservabilityState(
        config=config,
        metrics=metrics,
        alert_manager=alert_manager,
        tracer_provider=tracer_provider,
    )
    if app is not None:
        app.state.observability = state
    return state


__all__ = [
    "AlertingConfig",
    "BaseAlertManager",
    "BaseNotifier",
    "ObservabilityConfig",
    "ObservabilityState",
    "ProfilingConfig",
    "PrometheusConfig",
    "TracingConfig",
    "get_metrics_recorder",
    "reset_metrics_recorder",
    "setup_observability",
]
