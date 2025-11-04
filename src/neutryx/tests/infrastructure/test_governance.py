"""Tests for governance infrastructure components."""

from __future__ import annotations

import math

import pytest

from neutryx.infrastructure.governance import (
    GovernanceService,
    Role,
    SLAPolicy,
    Tenant,
    TenantLimits,
    current_tenant_id,
)


def test_governance_end_to_end_flow():
    service = GovernanceService()

    service.register_tenant(Tenant("tenant-a", "Alpha"), limits=TenantLimits(max_users=25))
    service.register_tenant(Tenant("tenant-b", "Beta"))
    service.tenants.update_status("tenant-b", status="suspended")

    service.assign_role(
        "alice",
        Role("admin", {"pricing:read", "pricing:write", "risk:view"}),
        tenant_id="tenant-a",
    )
    service.assign_role("bob", Role("viewer", {"pricing:read"}), tenant_id="tenant-b")

    service.record_activity(
        "tenant-a",
        user_id="alice",
        jobs=3,
        compute_seconds=180.0,
        storage_gb=2.5,
    )

    service.sla.record_availability("tenant-a", 99.8)
    service.sla.record_latency("tenant-a", 150.0)
    service.sla.record_requests("tenant-a", errors=1, total=200)
    service.sla.record_incident("tenant-a")
    service.set_sla_policy(
        "tenant-a",
        SLAPolicy(
            availability_target=99.0,
            max_latency_ms=200.0,
            max_error_rate=0.05,
            max_incidents=1,
        ),
    )
    sla_evaluation = service.evaluate_sla("tenant-a")
    assert sla_evaluation.compliant
    assert sla_evaluation.metrics["error_rate"] == pytest.approx(0.005, abs=1e-6)

    service.record_cost("tenant-a", "compute", 100.0, 0.5)
    service.record_cost("tenant-a", "storage", 10.0, 0.2)
    cost_summary = service.costs.cost_by_resource("tenant-a")
    assert cost_summary["compute"] > cost_summary["storage"]

    allocation = service.allocate_costs("tenant-a", {"team-x": 2.0, "team-y": 1.0})
    assert math.isclose(
        sum(allocation.values()),
        service.costs.total_cost("tenant-a"),
        rel_tol=1e-6,
    )

    report = service.generate_compliance_report()
    issues = [issue for issue in report.issues if issue.tenant_id == "tenant-b"]
    codes = {issue.rule: issue for issue in issues}
    assert "tenant_limits" in codes
    assert codes["tenant_limits"].severity == "WARN"
    assert "suspended_tenant_access" in codes
    assert codes["suspended_tenant_access"].severity == "HIGH"

    with service.scoped("tenant-a", user_id="alice"):
        assert current_tenant_id() == "tenant-a"
    assert current_tenant_id() is None
