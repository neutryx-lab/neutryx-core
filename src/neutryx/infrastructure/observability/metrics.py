"""Prometheus metrics instrumentation helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional

from fastapi import HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, REGISTRY, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response as StarletteResponse

from .alerting import BaseAlertManager, NullAlertManager
from .config import PrometheusConfig


def _safe_route_name(route: Optional[str]) -> str:
    if not route:
        return "unknown"
    if route.startswith("/"):
        return route
    return f"/{route}"


class ObservationTimer:
    """Context manager used to time arbitrary operations."""

    def __init__(
        self,
        recorder: "MetricsRecorder",
        operation: str,
        labels: Optional[Mapping[str, str]] = None,
        kind: str = "operation",
    ):
        self._recorder = recorder
        self._operation = operation
        self._labels = dict(labels or {})
        self._kind = kind
        self._start = time.perf_counter()
        self._stopped = False

    def stop(self, success: bool = True) -> None:
        if self._stopped:
            return
        duration = time.perf_counter() - self._start
        self._recorder.observe_operation(
            operation=self._operation,
            duration=duration,
            success=success,
            labels=self._labels,
            kind=self._kind,
        )
        self._stopped = True

    def __enter__(self) -> "ObservationTimer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop(success=exc_type is None)


class NullObservationTimer(ObservationTimer):
    """No-op timer used when metrics are disabled."""

    def __init__(self) -> None:
        super().__init__(recorder=_NullMetricsRecorder(), operation="noop")

    def stop(self, success: bool = True) -> None:  # pragma: no cover - trivial
        self._stopped = True
        return

    def __enter__(self) -> "NullObservationTimer":  # pragma: no cover - trivial
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - trivial
        return


class MetricsRecorder:
    """Encapsulates Prometheus collectors and recording logic."""

    def __init__(
        self,
        config: PrometheusConfig,
        alert_manager: Optional[BaseAlertManager] = None,
    ):
        self.config = config
        self.enabled = config.enabled
        self._alert_manager = alert_manager or NullAlertManager()

        if not self.enabled:
            # Initialise placeholders to simplify attribute access when disabled.
            self._http_requests = None
            self._http_latency = None
            self._operation_counter = None
            self._operation_latency = None
            return

        namespace = self.config.namespace
        subsystem = self.config.subsystem
        buckets = self.config.latency_buckets

        self._http_requests = _reusable_counter(
            namespace,
            subsystem,
            name="requests_total",
            documentation="Total number of HTTP requests processed.",
            labelnames=("method", "route", "status"),
        )
        self._http_latency = _reusable_histogram(
            namespace,
            subsystem,
            name="request_latency_seconds",
            documentation="HTTP request latency in seconds.",
            labelnames=("method", "route"),
            buckets=buckets,
        )
        self._operation_counter = _reusable_counter(
            namespace,
            subsystem,
            name="operations_total",
            documentation="Total number of domain operations executed.",
            labelnames=("operation", "status", "channel", "product"),
        )
        self._operation_latency = _reusable_histogram(
            namespace,
            subsystem,
            name="operation_latency_seconds",
            documentation="Latency for domain operations in seconds.",
            labelnames=("operation", "channel", "product"),
            buckets=buckets,
        )

    def time(
        self,
        operation: str,
        *,
        labels: Optional[Mapping[str, str]] = None,
        kind: str = "operation",
    ) -> ObservationTimer:
        if not self.enabled:
            return NullObservationTimer()
        return ObservationTimer(self, operation=operation, labels=labels, kind=kind)

    def observe_operation(
        self,
        *,
        operation: str,
        duration: float,
        success: bool,
        labels: Optional[Mapping[str, str]] = None,
        kind: str = "operation",
    ) -> None:
        if not self.enabled:
            return

        normalised = self._normalise_operation_labels(labels)
        status = "success" if success else "failure"
        self._operation_counter.labels(
            operation=operation,
            status=status,
            channel=normalised["channel"],
            product=normalised["product"],
        ).inc()
        self._operation_latency.labels(
            operation=operation,
            channel=normalised["channel"],
            product=normalised["product"],
        ).observe(duration)

        attributes = {
            "operation": operation,
            "channel": normalised["channel"],
            "product": normalised["product"],
        }
        self._alert_manager.record_event(
            kind=kind,
            success=success,
            duration=duration,
            attributes=attributes,
        )

    def observe_http_request(
        self,
        *,
        method: str,
        route: str,
        status_code: int,
        duration: float,
        success: bool,
    ) -> None:
        if not self.enabled:
            return

        route = _safe_route_name(route)
        labels = {"method": method, "route": route, "status": str(status_code)}
        self._http_requests.labels(**labels).inc()
        self._http_latency.labels(method=method, route=route).observe(duration)
        self._alert_manager.record_event(
            kind="http",
            success=success,
            duration=duration,
            attributes={"method": method, "route": route},
        )

    def _normalise_operation_labels(self, labels: Optional[Mapping[str, str]]) -> Dict[str, str]:
        data = {"channel": "internal", "product": "unknown"}
        if labels:
            if "channel" in labels and labels["channel"]:
                data["channel"] = labels["channel"]
            if "product" in labels and labels["product"]:
                data["product"] = labels["product"]
        return data


class _NullMetricsRecorder(MetricsRecorder):
    """Internal helper used by the null observation timer."""

    def __init__(self) -> None:  # pragma: no cover - only for guard paths
        super().__init__(config=PrometheusConfig(enabled=False))


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Starlette middleware that records HTTP metrics."""

    def __init__(self, app, recorder: MetricsRecorder):
        super().__init__(app)
        self._recorder = recorder

    async def dispatch(self, request: Request, call_next):
        if not self._recorder.enabled or request.url.path == self._recorder.config.endpoint:
            return await call_next(request)

        method = request.method.upper()
        route = getattr(request.scope.get("route"), "path", request.url.path)
        start = time.perf_counter()
        status_code = 500
        success = False
        try:
            response: StarletteResponse = await call_next(request)
            status_code = response.status_code
            success = 200 <= status_code < 500
            return response
        except HTTPException as exc:
            status_code = exc.status_code
            success = 200 <= status_code < 500
            raise
        except Exception:
            status_code = 500
            success = False
            raise
        finally:
            duration = time.perf_counter() - start
            self._recorder.observe_http_request(
                method=method,
                route=route,
                status_code=status_code,
                duration=duration,
                success=success,
            )


def create_metrics_endpoint() -> Callable[[], Response]:
    """Return a FastAPI-compatible handler that exposes metrics."""

    def metrics_endpoint() -> Response:
        payload = generate_latest()
        return Response(payload, media_type=CONTENT_TYPE_LATEST)

    return metrics_endpoint


_RECORDER: MetricsRecorder | None = None


def configure_metrics(
    config: PrometheusConfig,
    alert_manager: Optional[BaseAlertManager] = None,
) -> MetricsRecorder:
    """Initialise the metrics recorder with the provided configuration."""

    global _RECORDER  # noqa: PLW0603 - module level singleton
    if _RECORDER is not None:
        return _RECORDER
    _RECORDER = MetricsRecorder(config=config, alert_manager=alert_manager)
    return _RECORDER


def get_metrics_recorder() -> MetricsRecorder:
    """Return the configured metrics recorder (or a disabled one)."""

    if _RECORDER is None:
        return MetricsRecorder(PrometheusConfig(enabled=False))
    return _RECORDER


__all__ = [
    "MetricsRecorder",
    "ObservationTimer",
    "PrometheusMiddleware",
    "configure_metrics",
    "create_metrics_endpoint",
    "get_metrics_recorder",
]
