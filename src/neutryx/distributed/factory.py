"""Factory functions to create execution backends."""

from __future__ import annotations

from neutryx.distributed.backend import BackendType, ExecutionBackend, ExecutionConfig
from neutryx.distributed.local_cpu import LocalCPUBackend
from neutryx.distributed.local_gpu import LocalGPUBackend
from neutryx.distributed.ray_cluster import RayClusterBackend


def create_backend(config: ExecutionConfig) -> ExecutionBackend:
    """Create an execution backend from configuration.

    Parameters
    ----------
    config : ExecutionConfig
        Backend configuration

    Returns
    -------
    ExecutionBackend
        Instantiated backend

    Raises
    ------
    ValueError
        If backend type is not supported

    Examples
    --------
    >>> # Local CPU (default)
    >>> from neutryx.distributed import create_backend, ExecutionConfig, BackendType
    >>> config = ExecutionConfig(backend=BackendType.LOCAL_CPU)
    >>> backend = create_backend(config)
    >>>
    >>> # Multi-GPU
    >>> config = ExecutionConfig(backend=BackendType.LOCAL_GPU)
    >>> backend = create_backend(config)
    >>>
    >>> # Ray cluster
    >>> config = ExecutionConfig(
    ...     backend=BackendType.RAY_CLUSTER,
    ...     num_devices=8,
    ...     metadata={"ray_address": "ray://cluster-head:10001"},
    ... )
    >>> backend = create_backend(config)
    """
    if config.backend == BackendType.LOCAL_CPU:
        return LocalCPUBackend(config)

    elif config.backend == BackendType.LOCAL_GPU:
        return LocalGPUBackend(config)

    elif config.backend == BackendType.RAY_CLUSTER:
        return RayClusterBackend(config)

    else:
        raise ValueError(f"Unsupported backend type: {config.backend}")


def create_default_backend() -> ExecutionBackend:
    """Create default execution backend (Local CPU).

    Returns
    -------
    ExecutionBackend
        Local CPU backend

    Examples
    --------
    >>> backend = create_default_backend()
    >>> backend.num_devices()
    1
    """
    config = ExecutionConfig(backend=BackendType.LOCAL_CPU)
    return create_backend(config)


def create_gpu_backend(num_gpus: int | None = None) -> ExecutionBackend:
    """Create GPU execution backend.

    Parameters
    ----------
    num_gpus : int, optional
        Number of GPUs to use (default: auto-detect all)

    Returns
    -------
    ExecutionBackend
        GPU backend

    Examples
    --------
    >>> # Use all available GPUs
    >>> backend = create_gpu_backend()
    >>>
    >>> # Use specific number of GPUs
    >>> backend = create_gpu_backend(num_gpus=2)
    """
    config = ExecutionConfig(
        backend=BackendType.LOCAL_GPU,
        num_devices=num_gpus,
        auto_detect_devices=(num_gpus is None),
    )
    return create_backend(config)


def create_ray_backend(
    num_workers: int = 4,
    ray_address: str | None = None,
) -> ExecutionBackend:
    """Create Ray cluster execution backend.

    Parameters
    ----------
    num_workers : int, optional
        Number of Ray workers (default: 4)
    ray_address : str, optional
        Ray cluster address (default: None, starts local cluster)

    Returns
    -------
    ExecutionBackend
        Ray cluster backend

    Examples
    --------
    >>> # Local Ray cluster
    >>> backend = create_ray_backend(num_workers=8)
    >>>
    >>> # Connect to existing cluster
    >>> backend = create_ray_backend(
    ...     num_workers=16,
    ...     ray_address="auto",  # Auto-discover cluster
    ... )
    """
    config = ExecutionConfig(
        backend=BackendType.RAY_CLUSTER,
        num_devices=num_workers,
        metadata={"ray_address": ray_address} if ray_address else {},
    )
    return create_backend(config)
