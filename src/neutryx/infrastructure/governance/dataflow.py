"""Data flow lineage recording utilities.

This module provides a lightweight registry for capturing lineage metadata
produced by compute jobs and API requests.  Downstream services can register
sinks which receive structured :class:`DataFlowRecord` instances and persist
or forward them to external governance systems.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Dict, Iterable, Iterator, List, MutableMapping, Optional, Protocol
from uuid import uuid4


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
