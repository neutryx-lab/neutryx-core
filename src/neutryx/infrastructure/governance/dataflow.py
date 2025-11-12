"""Lightweight data lineage and data-flow recording utilities.

The goal of this module is to provide a minimal yet expressive mechanism to
capture provenance metadata for computation jobs and API driven workflows.  It
introduces a lineage context that can be activated while executing a job or
servicing an API request.  Any storage layer that opts-in can automatically
enrich persisted metadata with the active lineage identifier, while also
emitting structured events that downstream governance tooling can consume.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock
from types import MappingProxyType
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional
from uuid import uuid4
import contextvars


_CURRENT_CONTEXT: contextvars.ContextVar["LineageContext | None"] = contextvars.ContextVar(
    "neutryx_current_lineage_context", default=None
)


def generate_lineage_id() -> str:
    """Return a random lineage identifier."""

    return uuid4().hex


@dataclass(frozen=True)
class LineageContext:
    """Represents the active lineage scope for a job or API request."""

    lineage_id: str
    source: str | None = None
    job_id: str | None = None
    api_request_id: str | None = None
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> Dict[str, Any]:
        """Return a metadata dictionary describing the context."""

        metadata: Dict[str, Any] = {"lineage_id": self.lineage_id}
        if self.source:
            metadata.setdefault("lineage_source", self.source)
        if self.job_id:
            metadata.setdefault("job_id", self.job_id)
        if self.api_request_id:
            metadata.setdefault("api_request_id", self.api_request_id)
        for key, value in self.attributes.items():
            metadata.setdefault(str(key), value)
        return metadata


@dataclass(frozen=True)
class DataFlowEvent:
    """Structured record emitted whenever an artefact is produced."""

    event_type: str
    lineage_id: str
    timestamp: datetime
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the event into a serialisable dictionary."""

        return {
            "event_type": self.event_type,
            "lineage_id": self.lineage_id,
            "timestamp": self.timestamp.isoformat(),
            "metadata": dict(self.metadata),
        }


class DataFlowRecorder:
    """In-memory recorder that stores data-flow events."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._events: List[DataFlowEvent] = []
        self._subscribers: List[Callable[[DataFlowEvent], None]] = []

    def publish(self, event_type: str, metadata: Optional[Mapping[str, Any]] = None) -> DataFlowEvent:
        """Create an event and notify all subscribers."""

        enriched = embed_lineage_metadata(metadata)
        lineage_id = enriched["lineage_id"]

        event = DataFlowEvent(
            event_type=event_type,
            lineage_id=lineage_id,
            timestamp=datetime.now(timezone.utc),
            metadata=MappingProxyType(dict(enriched)),
        )

        with self._lock:
            self._events.append(event)
            subscribers = list(self._subscribers)

        for callback in subscribers:
            callback(event)

        return event

    def subscribe(self, callback: Callable[[DataFlowEvent], None]) -> None:
        """Register a callback invoked whenever an event is published."""

        with self._lock:
            if callback not in self._subscribers:
                self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[DataFlowEvent], None]) -> None:
        """Remove a previously registered subscriber."""

        with self._lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

    def get_events(self) -> List[DataFlowEvent]:
        """Return a snapshot of all recorded events."""

        with self._lock:
            return list(self._events)

    def clear(self) -> None:
        """Remove all stored events."""

        with self._lock:
            self._events.clear()


_DEFAULT_RECORDER = DataFlowRecorder()


def get_default_recorder() -> DataFlowRecorder:
    """Return the process-wide default recorder."""

    return _DEFAULT_RECORDER


def set_default_recorder(recorder: DataFlowRecorder) -> DataFlowRecorder:
    """Replace the global recorder, returning the previous instance."""

    global _DEFAULT_RECORDER
    previous = _DEFAULT_RECORDER
    _DEFAULT_RECORDER = recorder
    return previous


@contextmanager
def use_recorder(recorder: DataFlowRecorder) -> Iterator[DataFlowRecorder]:
    """Context manager installing a temporary default recorder."""

    previous = set_default_recorder(recorder)
    try:
        yield recorder
    finally:
        set_default_recorder(previous)


def embed_lineage_metadata(metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Merge lineage information into ``metadata``."""

    base: Dict[str, Any] = dict(metadata or {})
    context = _CURRENT_CONTEXT.get()

    lineage_id = base.get("lineage_id")
    if context is not None:
        lineage_id = context.lineage_id

    if lineage_id is None:
        lineage_id = generate_lineage_id()

    base["lineage_id"] = lineage_id

    if context is not None:
        context_metadata = context.to_metadata()
        context_metadata["lineage_id"] = lineage_id
        for key, value in context_metadata.items():
            base.setdefault(key, value)

    return base


def publish_event(event_type: str, metadata: Optional[Mapping[str, Any]] = None) -> DataFlowEvent:
    """Publish an event to the default recorder."""

    recorder = get_default_recorder()
    return recorder.publish(event_type, metadata)


def record_artifact(
    artifact_id: str,
    *,
    kind: str,
    metadata: Optional[Mapping[str, Any]] = None,
    extra_event_metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Record the creation of an artefact and return enriched metadata."""

    enriched_metadata = embed_lineage_metadata(metadata)
    event_payload: Dict[str, Any] = dict(enriched_metadata)
    event_payload.setdefault("artifact_id", artifact_id)
    event_payload.setdefault("artifact_kind", kind)
    if extra_event_metadata:
        for key, value in extra_event_metadata.items():
            event_payload.setdefault(str(key), value)

    publish_event("data_artifact_saved", event_payload)
    return enriched_metadata


@contextmanager
def data_flow_context(
    *,
    lineage_id: Optional[str] = None,
    source: Optional[str] = None,
    job_id: Optional[str] = None,
    api_request_id: Optional[str] = None,
    attributes: Optional[Mapping[str, Any]] = None,
    emit_events: bool = True,
) -> Iterator[LineageContext]:
    """Activate a lineage context for the duration of the ``with`` block."""

    resolved_lineage = lineage_id or generate_lineage_id()
    context = LineageContext(
        lineage_id=resolved_lineage,
        source=source,
        job_id=job_id,
        api_request_id=api_request_id,
        attributes=dict(attributes or {}),
    )

    token = _CURRENT_CONTEXT.set(context)

    if emit_events:
        publish_event(
            "data_flow_context_started",
            {"lineage_id": resolved_lineage, "source": source, "job_id": job_id, "api_request_id": api_request_id},
        )

    try:
        yield context
    finally:
        _CURRENT_CONTEXT.reset(token)
        if emit_events:
            publish_event(
                "data_flow_context_finished",
                {"lineage_id": resolved_lineage, "source": source, "job_id": job_id, "api_request_id": api_request_id},
            )


def current_context() -> Optional[LineageContext]:
    """Return the currently active lineage context, if any."""

    return _CURRENT_CONTEXT.get()


def is_context_active() -> bool:
    """Return ``True`` when a lineage context is active."""

    return current_context() is not None


__all__ = [
    "DataFlowEvent",
    "DataFlowRecorder",
    "LineageContext",
    "current_context",
    "data_flow_context",
    "embed_lineage_metadata",
    "generate_lineage_id",
    "get_default_recorder",
    "is_context_active",
    "publish_event",
    "record_artifact",
    "set_default_recorder",
    "use_recorder",
]

