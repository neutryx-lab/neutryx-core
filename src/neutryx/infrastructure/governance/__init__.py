"""Governance utilities for multi-tenancy, RBAC, and compliance features."""

from __future__ import annotations

from .audit import AuditLogger, AuditRecord
from .compliance import (
    AuditCoverageRule,
    ComplianceIssue,
    ComplianceReport,
    ComplianceReporter,
    ComplianceRule,
    LimitBreachRule,
    InactivityRule,
    SuspendedTenantAccessRule,
    TenantLimitsRule,
    default_rules,
)
from .costs import CostEntry, CostTracker
from .dataflow import (
    DataFlowEvent,
    DataFlowRecorder,
    LineageContext,
    current_context,
    data_flow_context,
    embed_lineage_metadata,
    generate_lineage_id,
    get_default_recorder,
    is_context_active,
    publish_event,
    record_artifact,
    set_default_recorder,
    use_recorder,
)
from .rbac import RBACManager, Role
from .service import GovernanceService
from .sla import SLAEvaluation, SLAMonitor, SLAPolicy
from .tenancy import (
    LimitEvaluation,
    Tenant,
    TenantLimits,
    TenantManager,
    TenantUsage,
    current_tenant_id,
    tenant_scope,
)

__all__ = [
    "AuditLogger",
    "AuditRecord",
    "RBACManager",
    "Role",
    "SLAEvaluation",
    "SLAMonitor",
    "SLAPolicy",
    "CostEntry",
    "CostTracker",
    "ComplianceIssue",
    "ComplianceReport",
    "ComplianceReporter",
    "ComplianceRule",
    "AuditCoverageRule",
    "LimitBreachRule",
    "InactivityRule",
    "SuspendedTenantAccessRule",
    "TenantLimitsRule",
    "default_rules",
    "GovernanceService",
    "LimitEvaluation",
    "Tenant",
    "TenantLimits",
    "TenantManager",
    "TenantUsage",
    "current_tenant_id",
    "tenant_scope",
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
