"""Ray cluster execution backend for distributed computing."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import jax.numpy as jnp
from jax import Array

from neutryx.distributed.backend import ExecutionBackend, ExecutionConfig

try:
    import ray

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    ray = None


class RayClusterBackend(ExecutionBackend):
    """Distributed execution backend using Ray.

    Distributes computation across multiple nodes in a Ray cluster.

    Examples
    --------
    >>> config = ExecutionConfig(
    ...     backend=BackendType.RAY_CLUSTER,
    ...     num_devices=4,
    ...     metadata={"ray_address": "auto"},  # Connect to existing cluster
    ... )
    >>> backend = create_backend(config)
    >>> # Distribute portfolio chunks across cluster
    >>> results = backend.execute_batch(price_counterparty, cp_batches)

    Notes
    -----
    Requires Ray to be installed: pip install ray
    """

    def __init__(self, config: ExecutionConfig):
        """Initialize Ray cluster backend."""
        if not RAY_AVAILABLE:
            raise ImportError(
                "Ray is not installed. Install with: pip install ray\n"
                "For cluster support: pip install ray[default]"
            )

        super().__init__(config)
        self._init_ray()

    def _init_ray(self) -> None:
        """Initialize Ray cluster connection."""
        ray_address = self.config.metadata.get("ray_address", None)

        if not ray.is_initialized():
            if ray_address:
                # Connect to existing cluster
                ray.init(address=ray_address)
            else:
                # Start local Ray cluster
                ray.init(
                    num_cpus=self.config.num_devices,
                    ignore_reinit_error=True,
                )

        self._num_workers = self.config.num_devices or ray.cluster_resources().get("CPU", 1)

    def _validate_config(self) -> None:
        """Validate Ray-specific configuration."""
        if self.config.num_devices is not None and self.config.num_devices <= 0:
            raise ValueError("num_devices must be positive")

    def execute(
        self,
        fn: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute function as Ray remote task.

        Parameters
        ----------
        fn : callable
            Function to execute
        *args
            Positional arguments
        **kwargs
            Keyword arguments

        Returns
        -------
        Any
            Function result
        """
        # Create remote function
        remote_fn = ray.remote(fn)

        # Execute and get result
        future = remote_fn.remote(*args, **kwargs)
        return ray.get(future)

    def execute_batch(
        self,
        fn: Callable,
        inputs: Array,
        **kwargs: Any,
    ) -> Array:
        """Execute function in parallel across Ray cluster.

        Parameters
        ----------
        fn : callable
            Function to parallelize
        inputs : Array
            Batched inputs [batch_size, ...]
        **kwargs
            Additional arguments to fn

        Returns
        -------
        Array
            Batched results

        Notes
        -----
        Splits inputs into chunks and distributes across workers.
        """
        batch_size = inputs.shape[0]
        n_workers = int(self._num_workers)

        # Split inputs into chunks
        chunk_size = (batch_size + n_workers - 1) // n_workers
        chunks = [inputs[i : i + chunk_size] for i in range(0, batch_size, chunk_size)]

        # Create remote function
        @ray.remote
        def process_chunk(chunk):
            import jax

            # Process chunk with vmap
            vmapped_fn = jax.vmap(lambda x: fn(x, **kwargs))
            return vmapped_fn(chunk)

        # Distribute chunks to workers
        futures = [process_chunk.remote(chunk) for chunk in chunks]

        # Gather results
        results = ray.get(futures)

        # Concatenate results
        return jnp.concatenate(results, axis=0)

    def num_devices(self) -> int:
        """Get number of Ray workers."""
        return int(self._num_workers)

    def device_info(self) -> Dict[str, Any]:
        """Get Ray cluster information."""
        cluster_resources = ray.cluster_resources()
        return {
            "platform": "ray_cluster",
            "num_workers": int(self._num_workers),
            "cluster_resources": dict(cluster_resources),
            "available_resources": dict(ray.available_resources()),
        }

    def shutdown(self) -> None:
        """Shutdown Ray cluster."""
        if ray.is_initialized():
            ray.shutdown()
