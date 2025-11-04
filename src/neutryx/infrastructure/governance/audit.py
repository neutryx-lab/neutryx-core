"""Audit logging utilities for governance features."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Callable, Deque, List, MutableSequence, Optional


@dataclass(slots=True)
class AuditRecord:
    """Structured representation of an audit event."""

    timestamp: datetime
    tenant_id: str | None
    user_id: str | None
    action: str
    severity: str = "INFO"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable representation of the record."""

        return {
            "timestamp": self.timestamp.isoformat(),
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "action": self.action,
            "severity": self.severity,
            "metadata": dict(self.metadata),
        }


class AuditLogger:
    """Thread-safe audit log with optional retention limits."""

    def __init__(self, *, retention: int | None = 1000):
        self._records: Deque[AuditRecord] = deque(maxlen=retention)
        self._lock = RLock()
        self._subscribers: MutableSequence[Callable[[AuditRecord], None]] = []

    def log(
        self,
        *,
        action: str,
        tenant_id: str | None = None,
        user_id: str | None = None,
        severity: str = "INFO",
        metadata: Optional[dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> AuditRecord:
        """Record an audit event and notify subscribers."""

        record = AuditRecord(
            timestamp=(timestamp or datetime.now(timezone.utc)),
            tenant_id=tenant_id,
            user_id=user_id,
            action=action,
            severity=severity.upper(),
            metadata=dict(metadata or {}),
        )
        with self._lock:
            self._records.append(record)
            subscribers = list(self._subscribers)
        for callback in subscribers:
            callback(record)
        return record

    def tail(self, limit: int | None = None) -> List[AuditRecord]:
        """Return the most recent records, respecting ``limit`` if provided."""

        with self._lock:
            items = list(self._records)
        if limit is None or limit >= len(items):
            return items
        return items[-limit:]

    def filter(self, *, tenant_id: str | None = None, action_prefix: str | None = None) -> List[AuditRecord]:
        """Return records matching the filter criteria."""

        with self._lock:
            items = list(self._records)
        result: List[AuditRecord] = []
        for record in items:
            if tenant_id is not None and record.tenant_id != tenant_id:
                continue
            if action_prefix is not None and not record.action.startswith(action_prefix):
                continue
            result.append(record)
        return result

    def subscribe(self, callback: Callable[[AuditRecord], None]) -> None:
        """Register a callback invoked for every new audit record."""

        with self._lock:
            self._subscribers.append(callback)

    def export(self) -> List[dict[str, Any]]:
        """Return audit records in serialisable format."""

        return [record.to_dict() for record in self.tail()]


__all__ = ["AuditLogger", "AuditRecord"]
