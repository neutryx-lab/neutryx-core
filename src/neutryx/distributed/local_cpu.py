"""Local CPU execution backend."""

from __future__ import annotations

from typing import Any, Callable, Dict

import jax
from jax import Array

from neutryx.distributed.backend import ExecutionBackend, ExecutionConfig


class LocalCPUBackend(ExecutionBackend):
    """Single-process CPU execution backend.

    Uses JAX's default CPU execution. No parallelization across devices.

    Examples
    --------
    >>> from neutryx.distributed import create_backend, ExecutionConfig, BackendType
    >>> config = ExecutionConfig(backend=BackendType.LOCAL_CPU)
    >>> backend = create_backend(config)
    >>> result = backend.execute(my_function, *args)
    """

    def _validate_config(self) -> None:
        """Validate CPU-specific configuration."""
        # No special validation needed for CPU backend
        pass

    def execute(
        self,
        fn: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute function on CPU.

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
        return fn(*args, **kwargs)

    def execute_batch(
        self,
        fn: Callable,
        inputs: Array,
        **kwargs: Any,
    ) -> Array:
        """Execute function on batched inputs using vmap.

        Parameters
        ----------
        fn : callable
            Function to map
        inputs : Array
            Batched inputs [batch_size, ...]
        **kwargs
            Additional arguments to fn

        Returns
        -------
        Array
            Batched results
        """
        # Use vmap for CPU vectorization
        vmapped_fn = jax.vmap(lambda x: fn(x, **kwargs))
        return vmapped_fn(inputs)

    def num_devices(self) -> int:
        """Get number of CPU devices (always 1)."""
        return 1

    def device_info(self) -> Dict[str, Any]:
        """Get CPU device information."""
        cpu_devices = jax.devices("cpu")
        return {
            "platform": "cpu",
            "num_devices": 1,
            "devices": [str(d) for d in cpu_devices],
        }

    def shutdown(self) -> None:
        """Shutdown CPU backend (no-op)."""
        pass
