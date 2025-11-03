"""Local multi-GPU execution backend."""

from __future__ import annotations

from typing import Any, Callable, Dict

import jax
from jax import Array

from neutryx.distributed.backend import ExecutionBackend, ExecutionConfig


class LocalGPUBackend(ExecutionBackend):
    """Multi-GPU execution backend using JAX pmap.

    Distributes computation across multiple GPUs on a single node.

    Examples
    --------
    >>> config = ExecutionConfig(backend=BackendType.LOCAL_GPU, auto_detect_devices=True)
    >>> backend = create_backend(config)
    >>> # Function will be parallelized across available GPUs
    >>> result = backend.execute_batch(pricing_fn, portfolio_batch)
    """

    def __init__(self, config: ExecutionConfig):
        """Initialize GPU backend."""
        super().__init__(config)
        self._devices = self._detect_gpus()

    def _detect_gpus(self) -> list:
        """Detect available GPU devices."""
        try:
            gpus = jax.devices("gpu")
            if not gpus:
                raise RuntimeError("No GPU devices found")
            return gpus
        except Exception as exc:
            raise RuntimeError("Failed to detect GPUs. Is JAX GPU version installed?") from exc

    def _validate_config(self) -> None:
        """Validate GPU-specific configuration."""
        if self.config.auto_detect_devices:
            # Auto-detect will be done in __init__
            pass
        elif self.config.num_devices is not None:
            if self.config.num_devices <= 0:
                raise ValueError("num_devices must be positive")

    def execute(
        self,
        fn: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute function on single GPU (first device).

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
        # Execute on first GPU
        with jax.default_device(self._devices[0]):
            return fn(*args, **kwargs)

    def execute_batch(
        self,
        fn: Callable,
        inputs: Array,
        **kwargs: Any,
    ) -> Array:
        """Execute function across multiple GPUs using pmap.

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
        Input batch_size should be divisible by num_devices for optimal performance.
        """
        n_devices = len(self._devices)

        # Reshape inputs for pmap: [n_devices, batch_per_device, ...]
        batch_size = inputs.shape[0]
        if batch_size % n_devices != 0:
            # Pad to make divisible
            pad_size = n_devices - (batch_size % n_devices)
            inputs = jax.numpy.pad(inputs, [(0, pad_size)] + [(0, 0)] * (inputs.ndim - 1))
            padded = True
        else:
            padded = False

        # Reshape for pmap
        inputs_reshaped = inputs.reshape(n_devices, -1, *inputs.shape[1:])

        # pmap function
        pmapped_fn = jax.pmap(lambda x: jax.vmap(lambda y: fn(y, **kwargs))(x))

        # Execute across GPUs
        results_reshaped = pmapped_fn(inputs_reshaped)

        # Flatten back
        results = results_reshaped.reshape(-1, *results_reshaped.shape[2:])

        # Remove padding if added
        if padded:
            results = results[:batch_size]

        return results

    def num_devices(self) -> int:
        """Get number of GPU devices."""
        return len(self._devices)

    def device_info(self) -> Dict[str, Any]:
        """Get GPU device information."""
        return {
            "platform": "gpu",
            "num_devices": len(self._devices),
            "devices": [str(d) for d in self._devices],
        }

    def shutdown(self) -> None:
        """Shutdown GPU backend (no-op)."""
        pass
