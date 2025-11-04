"""Distributed tracing utilities built on OpenTelemetry."""

from __future__ import annotations

import logging
from typing import Optional

try:  # pragma: no cover - optional dependency handling
    from opentelemetry import trace as ot_trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.grpc import GrpcAioInstrumentor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

    _OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency handling
    TracerProvider = object  # type: ignore[misc, assignment]
    _OTEL_AVAILABLE = False

from .config import TracingConfig

LOGGER = logging.getLogger("neutryx.tracing")

_TRACING_INITIALISED = False


def _build_exporter(config: TracingConfig):
    exporter = config.exporter.lower()
    if exporter == "console":
        return ConsoleSpanExporter()
    if exporter == "otlp":
        return OTLPSpanExporter(
            endpoint=config.otlp_endpoint,
            insecure=config.otlp_insecure,
        )
    raise ValueError(f"Unsupported tracing exporter '{config.exporter}'")


def setup_tracing(
    config: TracingConfig,
    *,
    app=None,
) -> Optional[TracerProvider]:
    """Initialise OpenTelemetry tracing according to configuration."""

    global _TRACING_INITIALISED  # noqa: PLW0603 - module level singleton
    if not config.enabled:
        return None
    if not _OTEL_AVAILABLE:
        LOGGER.warning("OpenTelemetry is not installed; tracing setup skipped.")
        return None
    if _TRACING_INITIALISED:
        return ot_trace.get_tracer_provider()

    sample_ratio = max(0.0, min(config.sample_ratio, 1.0))
    resource = Resource.create({"service.name": config.service_name})
    tracer_provider = TracerProvider(
        sampler=TraceIdRatioBased(sample_ratio),
        resource=resource,
    )
    exporter = _build_exporter(config)
    tracer_provider.add_span_processor(BatchSpanProcessor(exporter))

    ot_trace.set_tracer_provider(tracer_provider)

    if config.instrument_fastapi and app is not None:
        try:
            FastAPIInstrumentor.instrument_app(app, tracer_provider=tracer_provider)
        except Exception:  # pragma: no cover - defensive
            LOGGER.warning("FastAPI already instrumented; skipping duplicate instrumentation.")
    elif config.instrument_fastapi and app is None:
        FastAPIInstrumentor().instrument()

    if config.instrument_grpc:
        try:
            GrpcAioInstrumentor().instrument(tracer_provider=tracer_provider)
        except Exception:  # pragma: no cover - defensive
            LOGGER.warning("gRPC already instrumented; skipping duplicate instrumentation.")

    _TRACING_INITIALISED = True
    return tracer_provider


__all__ = ["setup_tracing"]
