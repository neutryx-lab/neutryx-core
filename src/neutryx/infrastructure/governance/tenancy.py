"""Multi-tenancy management utilities."""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Dict, Iterator, Optional

from .audit import AuditLogger

_TENANT_CONTEXT: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "neutryx_tenant_id", default=None
)


@dataclass(slots=True)
class Tenant:
    """Tenant configuration and metadata."""

    tenant_id: str
    name: str
    tier: str = "standard"
    status: str = "active"
    metadata: dict[str, Any] = field(default_factory=dict)

    def activate(self) -> None:
        self.status = "active"

    def suspend(self) -> None:
        self.status = "suspended"

    def deactivate(self) -> None:
        self.status = "deactivated"


@dataclass(slots=True)
class TenantLimits:
    """Quota allocations for a tenant."""

    max_users: int | None = None
    compute_seconds: float | None = None
    storage_gb: float | None = None


@dataclass(slots=True)
class TenantUsage:
    """Runtime usage statistics for a tenant."""

    active_users: set[str] = field(default_factory=set)
    jobs_run: int = 0
    compute_seconds: float = 0.0
    storage_gb: float = 0.0
    last_activity: datetime | None = None

    def touch(
        self,
        *,
        user_id: str | None = None,
        jobs: int = 0,
        compute_seconds: float = 0.0,
        storage_gb: float = 0.0,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Update usage aggregates."""

        if user_id:
            self.active_users.add(user_id)
        if jobs:
            self.jobs_run += jobs
        if compute_seconds:
            self.compute_seconds += compute_seconds
        if storage_gb:
            self.storage_gb += storage_gb
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        self.last_activity = timestamp


class TenantManager:
    """Register tenants and maintain isolation state."""

    def __init__(self, audit_logger: AuditLogger | None = None):
        self._tenants: Dict[str, Tenant] = {}
        self._limits: Dict[str, TenantLimits] = {}
        self._usage: Dict[str, TenantUsage] = {}
        self._lock = RLock()
        self._audit = audit_logger

    def register(self, tenant: Tenant, *, overwrite: bool = False) -> Tenant:
        """Register a new tenant."""

        with self._lock:
            exists = tenant.tenant_id in self._tenants
            if exists and not overwrite:
                raise ValueError(f"Tenant '{tenant.tenant_id}' already registered")
            self._tenants[tenant.tenant_id] = tenant
            self._usage.setdefault(tenant.tenant_id, TenantUsage())
        self._log(action="tenant.register", tenant_id=tenant.tenant_id, metadata={"overwrite": overwrite})
        return tenant

    def ensure(self, tenant_id: str) -> Tenant:
        """Return the tenant or raise ``KeyError``."""

        with self._lock:
            tenant = self._tenants.get(tenant_id)
        if tenant is None:
            raise KeyError(f"Tenant '{tenant_id}' is not registered")
        return tenant

    def set_limits(self, tenant_id: str, limits: TenantLimits) -> None:
        """Assign resource limits to a tenant."""

        self.ensure(tenant_id)
        with self._lock:
            self._limits[tenant_id] = limits
        self._log(
            action="tenant.limits.update",
            tenant_id=tenant_id,
            metadata=asdict(limits),
        )

    def get_limits(self, tenant_id: str) -> TenantLimits | None:
        """Return configured limits for the tenant."""

        with self._lock:
            return self._limits.get(tenant_id)

    def record_activity(
        self,
        tenant_id: str,
        *,
        user_id: str | None = None,
        jobs: int = 0,
        compute_seconds: float = 0.0,
        storage_gb: float = 0.0,
        timestamp: Optional[datetime] = None,
    ) -> TenantUsage:
        """Update usage metrics for the tenant."""

        tenant = self.ensure(tenant_id)
        with self._lock:
            usage = self._usage.setdefault(tenant.tenant_id, TenantUsage())
            usage.touch(
                user_id=user_id,
                jobs=jobs,
                compute_seconds=compute_seconds,
                storage_gb=storage_gb,
                timestamp=timestamp,
            )
        self._log(
            action="tenant.activity.record",
            tenant_id=tenant_id,
            user_id=user_id,
            metadata={
                "jobs": jobs,
                "compute_seconds": compute_seconds,
                "storage_gb": storage_gb,
            },
        )
        return usage

    def get_usage(self, tenant_id: str) -> TenantUsage:
        """Return usage metrics for the tenant."""

        self.ensure(tenant_id)
        with self._lock:
            usage = self._usage.get(tenant_id, TenantUsage())
        return usage

    def list_tenants(self) -> Iterator[Tenant]:
        """Iterate over registered tenants."""

        with self._lock:
            tenants = list(self._tenants.values())
        return iter(tenants)

    def update_status(self, tenant_id: str, *, status: str) -> Tenant:
        """Update the tenant status."""

        tenant = self.ensure(tenant_id)
        with self._lock:
            tenant.status = status
        self._log(action="tenant.status.update", tenant_id=tenant_id, metadata={"status": status})
        return tenant

    def unregister(self, tenant_id: str) -> None:
        """Remove a tenant."""

        with self._lock:
            removed = self._tenants.pop(tenant_id, None)
            self._limits.pop(tenant_id, None)
            self._usage.pop(tenant_id, None)
        if removed:
            self._log(action="tenant.unregister", tenant_id=tenant_id)

    def _log(
        self,
        *,
        action: str,
        tenant_id: str | None = None,
        user_id: str | None = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        if self._audit is not None:
            self._audit.log(action=action, tenant_id=tenant_id, user_id=user_id, metadata=metadata)


def current_tenant_id() -> str | None:
    """Return the tenant id within the active context."""

    return _TENANT_CONTEXT.get()


@contextmanager
def tenant_scope(tenant_manager: TenantManager, tenant_id: str, *, user_id: str | None = None):
    """Context manager that activates a tenant scope."""

    tenant = tenant_manager.ensure(tenant_id)
    token = _TENANT_CONTEXT.set(tenant_id)
    tenant_manager._log(action="tenant.scope.enter", tenant_id=tenant_id, user_id=user_id)
    try:
        yield tenant
    finally:
        _TENANT_CONTEXT.reset(token)
        tenant_manager._log(action="tenant.scope.exit", tenant_id=tenant_id, user_id=user_id)


__all__ = [
    "Tenant",
    "TenantLimits",
    "TenantManager",
    "TenantUsage",
    "current_tenant_id",
    "tenant_scope",
]
