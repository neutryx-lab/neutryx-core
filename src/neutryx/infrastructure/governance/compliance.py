"""Compliance reporting utilities leveraging audit trails."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterable, List, Optional, Sequence

from .audit import AuditLogger, AuditRecord
from .rbac import RBACManager
from .tenancy import TenantManager, TenantUsage


@dataclass(slots=True)
class ComplianceIssue:
    """Recorded compliance issue."""

    rule: str
    message: str
    severity: str
    tenant_id: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class ComplianceReport:
    """Structured compliance report."""

    generated_at: datetime
    tenant_id: str | None
    issues: List[ComplianceIssue]
    summary: dict[str, object]

    def is_compliant(self) -> bool:
        """Return ``True`` if no high severity issues were detected."""

        return not any(issue.severity.upper() in {"ERROR", "HIGH", "CRITICAL"} for issue in self.issues)


class ComplianceRule:
    """Interface for compliance rules."""

    name: str
    description: str

    def evaluate(
        self,
        *,
        tenant_manager: TenantManager,
        rbac_manager: RBACManager,
        audit_records: Sequence[AuditRecord],
        tenant_id: str | None = None,
    ) -> Iterable[ComplianceIssue]:
        raise NotImplementedError


class TenantLimitsRule(ComplianceRule):
    """Ensure tenants have configured limits."""

    name = "tenant_limits"
    description = "Tenants must have explicit resource limits configured."

    def evaluate(
        self,
        *,
        tenant_manager: TenantManager,
        rbac_manager: RBACManager,
        audit_records: Sequence[AuditRecord],
        tenant_id: str | None = None,
    ) -> Iterable[ComplianceIssue]:
        tenants = (
            [tenant_manager.ensure(tenant_id)] if tenant_id else list(tenant_manager.list_tenants())
        )
        for tenant in tenants:
            limits = tenant_manager.get_limits(tenant.tenant_id)
            if limits is None:
                yield ComplianceIssue(
                    rule=self.name,
                    severity="WARN",
                    tenant_id=tenant.tenant_id,
                    message="Missing tenant limits configuration",
                )


class SuspendedTenantAccessRule(ComplianceRule):
    """Ensure suspended tenants do not retain active role assignments."""

    name = "suspended_tenant_access"
    description = "Suspended or deactivated tenants must not have user assignments."

    def evaluate(
        self,
        *,
        tenant_manager: TenantManager,
        rbac_manager: RBACManager,
        audit_records: Sequence[AuditRecord],
        tenant_id: str | None = None,
    ) -> Iterable[ComplianceIssue]:
        tenant_ids = (
            [tenant_id] if tenant_id else [tenant.tenant_id for tenant in tenant_manager.list_tenants()]
        )
        assignments = rbac_manager.all_tenant_assignments()
        for tid in tenant_ids:
            tenant = tenant_manager.ensure(tid)
            if tenant.status == "active":
                continue
            user_roles = assignments.get(tid, {})
            if user_roles:
                yield ComplianceIssue(
                    rule=self.name,
                    severity="HIGH",
                    tenant_id=tid,
                    message=f"Tenant '{tid}' is {tenant.status} but still has role assignments",
                    metadata={"assignments": {user: list(roles) for user, roles in user_roles.items()}},
                )


class AuditCoverageRule(ComplianceRule):
    """Ensure recent audit activity exists for each tenant."""

    name = "audit_coverage"
    description = "Tenants must have audit activity within the configured window."

    def __init__(self, *, window_days: int = 30, severity: str = "WARN"):
        self._window = timedelta(days=window_days)
        self._severity = severity

    def evaluate(
        self,
        *,
        tenant_manager: TenantManager,
        rbac_manager: RBACManager,
        audit_records: Sequence[AuditRecord],
        tenant_id: str | None = None,
    ) -> Iterable[ComplianceIssue]:
        cutoff = datetime.now(timezone.utc) - self._window
        scoped_records = (
            [record for record in audit_records if record.tenant_id == tenant_id]
            if tenant_id
            else list(audit_records)
        )
        tenant_ids = (
            [tenant_id] if tenant_id else [tenant.tenant_id for tenant in tenant_manager.list_tenants()]
        )
        per_tenant = {tid: [] for tid in tenant_ids}
        for record in scoped_records:
            if record.tenant_id in per_tenant and record.timestamp >= cutoff:
                per_tenant[record.tenant_id].append(record)
        for tid, records in per_tenant.items():
            if not records:
                yield ComplianceIssue(
                    rule=self.name,
                    severity=self._severity,
                    tenant_id=tid,
                    message=f"No audit activity within {self._window.days} days",
                )


class InactivityRule(ComplianceRule):
    """Detect tenants without recent activity."""

    name = "tenant_inactivity"
    description = "Tenants must exhibit activity within the configured window."

    def __init__(self, *, max_inactive_days: int = 60, severity: str = "INFO"):
        self._max_inactive = timedelta(days=max_inactive_days)
        self._severity = severity

    def evaluate(
        self,
        *,
        tenant_manager: TenantManager,
        rbac_manager: RBACManager,
        audit_records: Sequence[AuditRecord],
        tenant_id: str | None = None,
    ) -> Iterable[ComplianceIssue]:
        now = datetime.now(timezone.utc)
        tenant_ids = (
            [tenant_id] if tenant_id else [tenant.tenant_id for tenant in tenant_manager.list_tenants()]
        )
        for tid in tenant_ids:
            usage: TenantUsage = tenant_manager.get_usage(tid)
            if usage.last_activity is None:
                yield ComplianceIssue(
                    rule=self.name,
                    severity=self._severity,
                    tenant_id=tid,
                    message="Tenant has no recorded activity",
                )
                continue
            if now - usage.last_activity > self._max_inactive:
                yield ComplianceIssue(
                    rule=self.name,
                    severity=self._severity,
                    tenant_id=tid,
                    message=f"Tenant inactive for {(now - usage.last_activity).days} days",
                )


class LimitBreachRule(ComplianceRule):
    """Detect tenants that exceed their configured limits."""

    name = "tenant_limit_breach"
    description = "Tenants must operate within assigned quotas."

    def __init__(self, *, severity: str = "HIGH"):
        self._severity = severity

    def evaluate(
        self,
        *,
        tenant_manager: TenantManager,
        rbac_manager: RBACManager,
        audit_records: Sequence[AuditRecord],
        tenant_id: str | None = None,
    ) -> Iterable[ComplianceIssue]:
        tenant_ids = (
            [tenant_id] if tenant_id else [tenant.tenant_id for tenant in tenant_manager.list_tenants()]
        )
        for tid in tenant_ids:
            evaluation = tenant_manager.check_limits(tid)
            if evaluation.limits is None or evaluation.within_limits:
                continue
            yield ComplianceIssue(
                rule=self.name,
                severity=self._severity,
                tenant_id=tid,
                message="Tenant usage exceeds configured limits",
                metadata={"breaches": dict(evaluation.breaches)},
            )


def default_rules() -> List[ComplianceRule]:
    """Return the default rule set used by the reporter."""

    return [
        TenantLimitsRule(),
        LimitBreachRule(),
        SuspendedTenantAccessRule(),
        AuditCoverageRule(),
        InactivityRule(),
    ]


class ComplianceReporter:
    """Generate compliance reports from governance state."""

    def __init__(
        self,
        tenant_manager: TenantManager,
        rbac_manager: RBACManager,
        audit_logger: AuditLogger,
        *,
        rules: Optional[Iterable[ComplianceRule]] = None,
    ):
        self._tenant_manager = tenant_manager
        self._rbac_manager = rbac_manager
        self._audit_logger = audit_logger
        self._rules: List[ComplianceRule] = list(rules) if rules else default_rules()

    def register_rule(self, rule: ComplianceRule) -> None:
        self._rules.append(rule)

    def generate_report(self, tenant_id: str | None = None) -> ComplianceReport:
        """Run rules and return a compliance report."""

        records = self._audit_logger.tail()
        issues: List[ComplianceIssue] = []
        for rule in self._rules:
            issues.extend(
                rule.evaluate(
                    tenant_manager=self._tenant_manager,
                    rbac_manager=self._rbac_manager,
                    audit_records=records,
                    tenant_id=tenant_id,
                )
            )
        summary: dict[str, object] = {"total_issues": len(issues), "by_severity": {}}
        severity_counts: dict[str, int] = {}
        for issue in issues:
            level = issue.severity.upper()
            severity_counts[level] = severity_counts.get(level, 0) + 1
        summary["by_severity"] = severity_counts
        return ComplianceReport(
            generated_at=datetime.now(timezone.utc),
            tenant_id=tenant_id,
            issues=issues,
            summary=summary,
        )


__all__ = [
    "ComplianceIssue",
    "ComplianceReport",
    "ComplianceReporter",
    "ComplianceRule",
    "LimitBreachRule",
    "AuditCoverageRule",
    "InactivityRule",
    "SuspendedTenantAccessRule",
    "TenantLimitsRule",
    "default_rules",
]
