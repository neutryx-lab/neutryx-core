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
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Protocol
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


# Global default recorder for the event-based API
_DEFAULT_RECORDER: DataFlowRecorder | None = None
_RECORDER_LOCK = RLock()


def get_default_recorder() -> DataFlowRecorder:
    """Return the global default recorder, creating if necessary."""

    global _DEFAULT_RECORDER  # noqa: PLW0603
    with _RECORDER_LOCK:
        if _DEFAULT_RECORDER is None:
            _DEFAULT_RECORDER = DataFlowRecorder()
        return _DEFAULT_RECORDER


def set_default_recorder(recorder: DataFlowRecorder) -> None:
    """Replace the global default recorder."""

    global _DEFAULT_RECORDER  # noqa: PLW0603
    with _RECORDER_LOCK:
        _DEFAULT_RECORDER = recorder


@contextmanager
def use_recorder(recorder: DataFlowRecorder) -> Iterator[DataFlowRecorder]:
    """Context manager to temporarily use a specific recorder."""

    previous = get_default_recorder()
    set_default_recorder(recorder)
    try:
        yield recorder
    finally:
        set_default_recorder(previous)


def current_context() -> LineageContext | None:
    """Return the currently active lineage context, if any."""

    return _CURRENT_CONTEXT.get()


def is_context_active() -> bool:
    """Check if a lineage context is currently active."""

    return _CURRENT_CONTEXT.get() is not None


def embed_lineage_metadata(metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Enrich metadata with active lineage context information."""

    enriched: Dict[str, Any] = dict(metadata or {})
    ctx = current_context()
    if ctx:
        enriched.update(ctx.to_metadata())
    else:
        # Generate a new lineage ID if no context is active
        enriched.setdefault("lineage_id", generate_lineage_id())
    return enriched


@contextmanager
def data_flow_context(
    *,
    source: str,
    job_id: Optional[str] = None,
    api_request_id: Optional[str] = None,
    **attributes: Any,
) -> Iterator[LineageContext]:
    """Context manager that establishes a lineage context for the enclosed work."""

    lineage_id = generate_lineage_id()
    ctx = LineageContext(
        lineage_id=lineage_id,
        source=source,
        job_id=job_id,
        api_request_id=api_request_id,
        attributes=dict(attributes),
    )

    token = _CURRENT_CONTEXT.set(ctx)
    try:
        yield ctx
    finally:
        _CURRENT_CONTEXT.reset(token)


def publish_event(event_type: str, metadata: Optional[Mapping[str, Any]] = None) -> DataFlowEvent:
    """Publish a data flow event using the default recorder."""

    recorder = get_default_recorder()
    return recorder.publish(event_type, metadata)


def record_artifact(
    key: str,
    kind: str,
    metadata: Optional[Dict[str, Any]] = None,
    extra_event_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Record an artifact and publish an event.

    Args:
        key: Unique identifier for the artifact
        kind: Type of artifact (e.g., "array", "metadata", "portfolio")
        metadata: Metadata to be enriched and associated with the artifact
        extra_event_metadata: Additional metadata for the event only

    Returns:
        Enriched metadata dictionary including lineage information
    """

    enriched_metadata = embed_lineage_metadata(metadata)

    event_metadata = {
        "artifact_key": key,
        "artifact_kind": kind,
        **enriched_metadata,
    }
    if extra_event_metadata:
        event_metadata.update(extra_event_metadata)

    publish_event("data_artifact_saved", event_metadata)

    return enriched_metadata


class DataFlowSink(Protocol):
    """Protocol describing sinks that can consume lineage records."""

    def emit(self, record: "DataFlowRecord") -> None:
        """Persist or forward the lineage record."""


@dataclass(slots=True)
class DataFlowRecord:
    """Structured lineage information for a single data flow event."""

    lineage_id: str
    job_id: str
    source: str
    timestamp: datetime
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a serialisable dictionary representation."""

        return {
            "lineage_id": self.lineage_id,
            "job_id": self.job_id,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "inputs": dict(self.inputs),
            "outputs": dict(self.outputs),
            "context": dict(self.context),
        }


class InMemorySink(DataFlowSink):
    """Simple sink storing lineage records for inspection (e.g. in tests)."""

    def __init__(self) -> None:
        self._records: List[DataFlowRecord] = []
        self._lock = RLock()

    def emit(self, record: DataFlowRecord) -> None:
        with self._lock:
            self._records.append(record)

    def records(self) -> List[DataFlowRecord]:
        with self._lock:
            return list(self._records)


class DataFlowRecorder:
    """Registry coordinating lineage ID generation and sink fan-out."""

    def __init__(self, sinks: Optional[Iterable[DataFlowSink]] = None) -> None:
        self._sinks: List[DataFlowSink] = list(sinks or [])
        self._lock = RLock()

    def register_sink(self, sink: DataFlowSink) -> None:
        """Register an additional sink."""

        with self._lock:
            self._sinks.append(sink)

    def record_flow(
        self,
        *,
        job_id: str,
        source: str,
        inputs: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        lineage_id: Optional[str] = None,
    ) -> DataFlowRecord:
        """Create a lineage record and notify registered sinks."""

        record = DataFlowRecord(
            lineage_id=lineage_id or str(uuid4()),
            job_id=job_id,
            source=source,
            timestamp=datetime.now(timezone.utc),
            inputs=dict(inputs or {}),
            outputs=dict(outputs or {}),
            context=dict(context or {}),
        )
        self._notify(record)
        return record

    def _notify(self, record: DataFlowRecord) -> None:
        with self._lock:
            sinks = list(self._sinks)
        for sink in sinks:
            sink.emit(record)

    @staticmethod
    def inject_lineage(metadata: MutableMapping[str, Any], lineage_id: str) -> None:
        """Inject the lineage identifier into the provided metadata mapping."""

        metadata["lineage_id"] = lineage_id


_DEFAULT_RECORDER = DataFlowRecorder()


def get_dataflow_recorder() -> DataFlowRecorder:
    """Return the global data flow recorder instance."""

    return _DEFAULT_RECORDER


def set_dataflow_recorder(recorder: DataFlowRecorder) -> None:
    """Replace the global data flow recorder (primarily for testing)."""

    global _DEFAULT_RECORDER  # noqa: PLW0603 - module level singleton override
    _DEFAULT_RECORDER = recorder


@contextmanager
def dataflow_context(
    *,
    job_id: str,
    source: str,
    inputs: Optional[Dict[str, Any]] = None,
    recorder: Optional[DataFlowRecorder] = None,
) -> Iterator[DataFlowRecord]:
    """Context manager that yields a lineage record for the enclosed work."""

    active_recorder = recorder or get_dataflow_recorder()
    record = active_recorder.record_flow(job_id=job_id, source=source, inputs=inputs)
    yield record
