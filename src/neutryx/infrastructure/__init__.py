"""Infrastructure for distributed execution, governance, and experiment tracking.

This module also includes configuration management.
"""

from __future__ import annotations

from . import config  # noqa: F401
from .cluster import *  # noqa: F401, F403
from .governance import *  # noqa: F401, F403
try:  # pragma: no cover - optional dependency glue
    from .observability import (  # noqa: F401
        ObservabilityConfig,
        ObservabilityState,
        get_metrics_recorder,
        setup_observability,
    )
except ImportError as _obs_exc:  # pragma: no cover
    ObservabilityConfig = None  # type: ignore[assignment]
    ObservabilityState = None  # type: ignore[assignment]

    def get_metrics_recorder(*args, **kwargs):  # type: ignore[no-redef]
        raise RuntimeError("Observability dependencies are not installed") from _obs_exc

    def setup_observability(*args, **kwargs):  # type: ignore[no-redef]
        raise RuntimeError("Observability dependencies are not installed") from _obs_exc
from .tracking import *  # noqa: F401, F403
from .workflows import CheckpointManager, ModelWorkflow

__all__ = [
    # config
    "config",
    # tracking
    "BaseTracker",
    "TrackingConfig",
    "TrackingError",
    "calibration_metric_template",
    "calibration_param_template",
    "resolve_tracker",
    # cluster
    "ClusterConfig",
    "load_cluster_config",
    # workflows
    "CheckpointManager",
    "ModelWorkflow",
    # governance
    "AuditLogger",
    "AuditRecord",
    "RBACManager",
    "Role",
    "Tenant",
    "TenantLimits",
    "TenantManager",
    "TenantUsage",
    "current_tenant_id",
    "tenant_scope",
    # observability
    "ObservabilityConfig",
    "ObservabilityState",
    "get_metrics_recorder",
    "setup_observability",
]
