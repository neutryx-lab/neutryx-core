"""Service level agreement monitoring utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Dict, List, Optional

from .audit import AuditLogger


@dataclass(slots=True)
class SLAPolicy:
    """Target thresholds for SLA evaluation."""

    availability_target: float = 99.0
    max_latency_ms: float = 250.0
    max_error_rate: float = 0.01
    max_incidents: int = 0


@dataclass(slots=True)
class SLAStats:
    """Aggregated SLA metrics."""

    availability_sum: float = 0.0
    availability_count: int = 0
    latency_sum: float = 0.0
    latency_count: int = 0
    error_requests: int = 0
    total_requests: int = 0
    incidents: int = 0
    last_update: datetime | None = None


@dataclass(slots=True)
class SLAEvaluation:
    """Evaluation result for a tenant."""

    tenant_id: str
    compliant: bool
    breaches: List[str]
    metrics: Dict[str, float | int | None]
    policy: SLAPolicy
    generated_at: datetime

    def to_dict(self) -> Dict[str, object]:
        return {
            "tenant_id": self.tenant_id,
            "compliant": self.compliant,
            "breaches": list(self.breaches),
            "metrics": dict(self.metrics),
            "policy": {
                "availability_target": self.policy.availability_target,
                "max_latency_ms": self.policy.max_latency_ms,
                "max_error_rate": self.policy.max_error_rate,
                "max_incidents": self.policy.max_incidents,
            },
            "generated_at": self.generated_at.isoformat(),
        }


class SLAMonitor:
    """Track SLA metrics and evaluate against configured policies."""

    def __init__(
        self,
        *,
        default_policy: Optional[SLAPolicy] = None,
        audit_logger: AuditLogger | None = None,
    ):
        self._default_policy = default_policy or SLAPolicy()
        self._policies: Dict[str, SLAPolicy] = {}
        self._stats: Dict[str, SLAStats] = {}
        self._lock = RLock()
        self._audit = audit_logger

    def set_policy(self, tenant_id: str, policy: SLAPolicy) -> None:
        """Configure SLA thresholds for ``tenant_id``."""

        with self._lock:
            self._policies[tenant_id] = policy
            self._stats.setdefault(tenant_id, SLAStats())
        self._log(
            action="sla.policy.update",
            tenant_id=tenant_id,
            metadata={
                "availability_target": policy.availability_target,
                "max_latency_ms": policy.max_latency_ms,
                "max_error_rate": policy.max_error_rate,
                "max_incidents": policy.max_incidents,
            },
        )

    def record_availability(
        self,
        tenant_id: str,
        availability: float,
        *,
        timestamp: datetime | None = None,
    ) -> None:
        stats = self._stats_for(tenant_id)
        with self._lock:
            stats.availability_sum += availability
            stats.availability_count += 1
            stats.last_update = timestamp or datetime.now(timezone.utc)
        self._log(
            action="sla.metric.availability",
            tenant_id=tenant_id,
            metadata={"value": availability},
        )

    def record_latency(
        self,
        tenant_id: str,
        latency_ms: float,
        *,
        timestamp: datetime | None = None,
    ) -> None:
        stats = self._stats_for(tenant_id)
        with self._lock:
            stats.latency_sum += latency_ms
            stats.latency_count += 1
            stats.last_update = timestamp or datetime.now(timezone.utc)
        self._log(
            action="sla.metric.latency",
            tenant_id=tenant_id,
            metadata={"value": latency_ms},
        )

    def record_requests(
        self,
        tenant_id: str,
        *,
        errors: int,
        total: int,
        timestamp: datetime | None = None,
    ) -> None:
        stats = self._stats_for(tenant_id)
        with self._lock:
            stats.error_requests += errors
            stats.total_requests += total
            stats.last_update = timestamp or datetime.now(timezone.utc)
        self._log(
            action="sla.metric.requests",
            tenant_id=tenant_id,
            metadata={"errors": errors, "total": total},
        )

    def record_incident(self, tenant_id: str, *, timestamp: datetime | None = None) -> None:
        stats = self._stats_for(tenant_id)
        with self._lock:
            stats.incidents += 1
            stats.last_update = timestamp or datetime.now(timezone.utc)
        self._log(
            action="sla.metric.incident",
            tenant_id=tenant_id,
            metadata={"incidents": stats.incidents},
        )

    def evaluate(self, tenant_id: str) -> SLAEvaluation:
        """Evaluate SLA compliance for ``tenant_id``."""

        policy = self._policies.get(tenant_id, self._default_policy)
        stats = self._stats_for(tenant_id)
        breaches: List[str] = []

        availability = (
            stats.availability_sum / stats.availability_count if stats.availability_count else None
        )
        latency = stats.latency_sum / stats.latency_count if stats.latency_count else None
        error_rate = (
            stats.error_requests / stats.total_requests if stats.total_requests else None
        )
        incidents = stats.incidents

        if availability is not None and availability < policy.availability_target:
            breaches.append(
                f"availability {availability:.2f}% below target {policy.availability_target:.2f}%"
            )
        if latency is not None and latency > policy.max_latency_ms:
            breaches.append(f"latency {latency:.2f}ms above target {policy.max_latency_ms:.2f}ms")
        if error_rate is not None and error_rate > policy.max_error_rate:
            breaches.append(f"error rate {error_rate:.4f} above {policy.max_error_rate:.4f}")
        if incidents > policy.max_incidents:
            breaches.append(f"incidents {incidents} exceed allowance {policy.max_incidents}")

        evaluation = SLAEvaluation(
            tenant_id=tenant_id,
            compliant=not breaches,
            breaches=breaches,
            metrics={
                "availability": availability,
                "latency": latency,
                "error_rate": error_rate,
                "incidents": incidents,
            },
            policy=policy,
            generated_at=datetime.now(timezone.utc),
        )
        return evaluation

    def evaluate_all(self) -> List[SLAEvaluation]:
        """Evaluate all tenants with recorded metrics."""

        with self._lock:
            tenant_ids = sorted(set(self._stats.keys()) | set(self._policies.keys()))
        return [self.evaluate(tenant_id) for tenant_id in tenant_ids]

    def _stats_for(self, tenant_id: str) -> SLAStats:
        with self._lock:
            return self._stats.setdefault(tenant_id, SLAStats())

    def _log(
        self,
        *,
        action: str,
        tenant_id: str,
        metadata: Optional[Dict[str, float | int]] = None,
    ) -> None:
        if self._audit is not None:
            payload = dict(metadata or {})
            self._audit.log(action=action, tenant_id=tenant_id, metadata=payload)


__all__ = ["SLAMonitor", "SLAPolicy", "SLAEvaluation"]
