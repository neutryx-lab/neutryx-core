"""Abstract backend interface for distributed execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional

from jax import Array


class BackendType(str, Enum):
    """Supported execution backend types."""

    LOCAL_CPU = "local_cpu"
    LOCAL_GPU = "local_gpu"
    RAY_CLUSTER = "ray_cluster"


@dataclass
class ExecutionConfig:
    """Configuration for execution backend.

    Attributes
    ----------
    backend : BackendType
        Type of execution backend
    num_devices : int, optional
        Number of devices to use (for GPU/cluster backends)
    auto_detect_devices : bool
        Automatically detect available devices
    batch_size : int, optional
        Batch size for distributed execution
    metadata : dict, optional
        Backend-specific configuration

    Examples
    --------
    >>> # Local CPU (default)
    >>> config = ExecutionConfig(backend=BackendType.LOCAL_CPU)
    >>>
    >>> # Multi-GPU with auto-detection
    >>> config = ExecutionConfig(
    ...     backend=BackendType.LOCAL_GPU,
    ...     auto_detect_devices=True,
    ... )
    >>>
    >>> # Ray cluster with 4 workers
    >>> config = ExecutionConfig(
    ...     backend=BackendType.RAY_CLUSTER,
    ...     num_devices=4,
    ...     metadata={"ray_address": "auto"},
    ... )
    """

    backend: BackendType
    num_devices: Optional[int] = None
    auto_detect_devices: bool = True
    batch_size: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Initialize metadata."""
        if self.metadata is None:
            self.metadata = {}


class ExecutionBackend(ABC):
    """Abstract base class for execution backends."""

    def __init__(self, config: ExecutionConfig):
        """Initialize backend.

        Parameters
        ----------
        config : ExecutionConfig
            Backend configuration
        """
        self.config = config
        self._validate_config()

    @abstractmethod
    def _validate_config(self) -> None:
        """Validate backend-specific configuration."""
        pass

    @abstractmethod
    def execute(
        self,
        fn: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute a function with this backend.

        Parameters
        ----------
        fn : callable
            Function to execute
        *args
            Positional arguments to fn
        **kwargs
            Keyword arguments to fn

        Returns
        -------
        Any
            Result of fn execution
        """
        pass

    @abstractmethod
    def execute_batch(
        self,
        fn: Callable,
        inputs: Array,
        **kwargs: Any,
    ) -> Array:
        """Execute function on batched inputs.

        Parameters
        ----------
        fn : callable
            Function to map over inputs
        inputs : Array
            Batched inputs [batch_size, ...]
        **kwargs
            Additional arguments to fn

        Returns
        -------
        Array
            Batched results
        """
        pass

    @abstractmethod
    def num_devices(self) -> int:
        """Get number of available devices."""
        pass

    @abstractmethod
    def device_info(self) -> Dict[str, Any]:
        """Get information about available devices."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown backend and release resources."""
        pass

    def __enter__(self) -> ExecutionBackend:
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.shutdown()

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(backend={self.config.backend}, devices={self.num_devices()})"
