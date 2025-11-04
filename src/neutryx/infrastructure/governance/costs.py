"""Cost tracking and allocation utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Dict, List, Mapping, Optional

from .audit import AuditLogger


@dataclass(slots=True)
class CostEntry:
    """Individual cost entry for a tenant."""

    tenant_id: str
    resource: str
    quantity: float
    unit_cost: float
    timestamp: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def total_cost(self) -> float:
        return self.quantity * self.unit_cost


class CostTracker:
    """Capture tenant costs and support allocation strategies."""

    def __init__(self, audit_logger: AuditLogger | None = None):
        self._entries: List[CostEntry] = []
        self._lock = RLock()
        self._audit = audit_logger

    def record(
        self,
        tenant_id: str,
        resource: str,
        quantity: float,
        unit_cost: float,
        *,
        timestamp: datetime | None = None,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> CostEntry:
        """Store a cost entry and return it."""

        entry = CostEntry(
            tenant_id=tenant_id,
            resource=resource,
            quantity=quantity,
            unit_cost=unit_cost,
            timestamp=timestamp or datetime.now(timezone.utc),
            metadata=dict(metadata or {}),
        )
        with self._lock:
            self._entries.append(entry)
        self._log(
            action="cost.record",
            tenant_id=tenant_id,
            metadata={
                "resource": resource,
                "quantity": quantity,
                "unit_cost": unit_cost,
                "total_cost": entry.total_cost,
            },
        )
        return entry

    def total_cost(self, tenant_id: str | None = None) -> float:
        """Return aggregated cost for ``tenant_id`` or all tenants."""

        with self._lock:
            entries = list(self._entries)
        total = 0.0
        for entry in entries:
            if tenant_id is not None and entry.tenant_id != tenant_id:
                continue
            total += entry.total_cost
        return total

    def cost_by_resource(self, tenant_id: str | None = None) -> Dict[str, float]:
        """Return cost totals grouped by resource."""

        result: Dict[str, float] = {}
        with self._lock:
            entries = list(self._entries)
        for entry in entries:
            if tenant_id is not None and entry.tenant_id != tenant_id:
                continue
            result.setdefault(entry.resource, 0.0)
            result[entry.resource] += entry.total_cost
        return result

    def allocate(self, tenant_id: str, allocations: Mapping[str, float]) -> Dict[str, float]:
        """Allocate tenant cost according to the provided weights."""

        total = self.total_cost(tenant_id)
        if total == 0.0:
            return {name: 0.0 for name in allocations}
        total_weight = sum(weight for weight in allocations.values() if weight > 0)
        if total_weight <= 0.0:
            raise ValueError("Allocation weights must sum to a positive value")
        allocation_result = {
            name: total * (weight / total_weight) if weight > 0 else 0.0
            for name, weight in allocations.items()
        }
        self._log(
            action="cost.allocate",
            tenant_id=tenant_id,
            metadata={
                "total": total,
                "weights": dict(allocations),
                "allocated": dict(allocation_result),
            },
        )
        return allocation_result

    def list_entries(self, tenant_id: str | None = None) -> List[CostEntry]:
        """Return a copy of stored entries, optionally filtered by tenant."""

        with self._lock:
            entries = [
                entry
                for entry in self._entries
                if tenant_id is None or entry.tenant_id == tenant_id
            ]
        return entries

    def _log(
        self,
        *,
        action: str,
        tenant_id: str,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> None:
        if self._audit is not None:
            self._audit.log(action=action, tenant_id=tenant_id, metadata=dict(metadata or {}))


__all__ = ["CostEntry", "CostTracker"]
