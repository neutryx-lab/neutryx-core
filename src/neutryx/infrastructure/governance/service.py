"""High level governance service integrating tenancy, RBAC, SLA, and cost tracking."""

from __future__ import annotations

from typing import Iterable, Mapping, Optional

from .audit import AuditLogger
from .compliance import ComplianceReport, ComplianceReporter, ComplianceRule
from .costs import CostTracker
from .rbac import RBACManager, Role
from .sla import SLAEvaluation, SLAMonitor, SLAPolicy
from .tenancy import LimitEvaluation, Tenant, TenantLimits, TenantManager, tenant_scope


class GovernanceService:
    """Facade providing a cohesive governance interface."""

    def __init__(
        self,
        *,
        audit_logger: Optional[AuditLogger] = None,
        rules: Optional[Iterable[ComplianceRule]] = None,
    ):
        self.audit = audit_logger or AuditLogger()
        self.tenants = TenantManager(audit_logger=self.audit)
        self.rbac = RBACManager(tenant_manager=self.tenants, audit_logger=self.audit)
        self.sla = SLAMonitor(audit_logger=self.audit)
        self.costs = CostTracker(audit_logger=self.audit)
        self.compliance = ComplianceReporter(self.tenants, self.rbac, self.audit, rules=rules)

    def register_tenant(self, tenant: Tenant, *, limits: TenantLimits | None = None) -> Tenant:
        registered = self.tenants.register(tenant)
        if limits is not None:
            self.tenants.set_limits(tenant.tenant_id, limits)
        return registered

    def assign_role(
        self,
        user_id: str,
        role: Role,
        *,
        tenant_id: str | None = None,
        overwrite: bool = False,
    ) -> None:
        scope = tenant_id
        if not self.rbac.has_role(role.name, tenant_id=scope) or overwrite:
            self.rbac.define_role(role, tenant_id=scope, overwrite=overwrite)
        self.rbac.assign_role(user_id, role.name, tenant_id=scope)

    def record_activity(
        self,
        tenant_id: str,
        *,
        user_id: str | None = None,
        jobs: int = 0,
        compute_seconds: float = 0.0,
        storage_gb: float = 0.0,
    ) -> None:
        self.tenants.record_activity(
            tenant_id,
            user_id=user_id,
            jobs=jobs,
            compute_seconds=compute_seconds,
            storage_gb=storage_gb,
        )

    def record_cost(
        self,
        tenant_id: str,
        resource: str,
        quantity: float,
        unit_cost: float,
        *,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> None:
        self.costs.record(
            tenant_id,
            resource,
            quantity,
            unit_cost,
            metadata=metadata,
        )

    def set_sla_policy(self, tenant_id: str, policy: SLAPolicy) -> None:
        self.sla.set_policy(tenant_id, policy)

    def evaluate_sla(self, tenant_id: str) -> SLAEvaluation:
        return self.sla.evaluate(tenant_id)

    def generate_compliance_report(self, tenant_id: str | None = None) -> ComplianceReport:
        return self.compliance.generate_report(tenant_id=tenant_id)

    def check_limits(self, tenant_id: str) -> LimitEvaluation:
        return self.tenants.check_limits(tenant_id)

    def allocate_costs(self, tenant_id: str, allocations: Mapping[str, float]) -> Mapping[str, float]:
        return self.costs.allocate(tenant_id, allocations)

    def scoped(self, tenant_id: str, *, user_id: str | None = None):
        """Return a tenant scoped context manager."""

        return tenant_scope(self.tenants, tenant_id, user_id=user_id)


__all__ = ["GovernanceService"]
