"""Distributed computing backends for scalable portfolio processing.

Provides execution backends that can scale from single CPU to multi-GPU clusters:

- **LocalCPU**: Single-process execution (default)
- **LocalGPU**: Multi-GPU execution via JAX pmap
- **RayCluster**: Distributed execution across cluster nodes via Ray

Design allows seamless switching via configuration without code changes.
"""

from neutryx.distributed.backend import BackendType, ExecutionBackend, ExecutionConfig
from neutryx.distributed.factory import (
    create_backend,
    create_default_backend,
    create_gpu_backend,
    create_ray_backend,
)

__all__ = [
    # Core types
    "ExecutionBackend",
    "ExecutionConfig",
    "BackendType",
    # Factory functions
    "create_backend",
    "create_default_backend",
    "create_gpu_backend",
    "create_ray_backend",
]
