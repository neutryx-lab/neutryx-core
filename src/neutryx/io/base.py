"""Base interfaces for storage backends.

Defines the common API for all storage implementations, allowing
seamless switching between Zarr, memory-mapped arrays, and other formats.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

from jax import Array


class StorageBackend(str, Enum):
    """Supported storage backend types."""

    ZARR = "zarr"
    MMAP = "mmap"
    PICKLE = "pickle"  # Legacy, not recommended


@dataclass
class StorageConfig:
    """Configuration for storage backend.

    Attributes
    ----------
    backend : StorageBackend
        Type of storage backend to use
    path : str
        Storage path (local directory or cloud URI)
    compression : str, optional
        Compression algorithm ("blosc", "gzip", "lz4", None)
    compression_level : int, optional
        Compression level (1-9, backend-specific)
    read_only : bool
        Whether to open in read-only mode
    create_if_missing : bool
        Create storage if it doesn't exist
    metadata : dict, optional
        Additional backend-specific configuration

    Examples
    --------
    >>> # Zarr with Blosc compression for persistence
    >>> config = StorageConfig(
    ...     backend=StorageBackend.ZARR,
    ...     path="s3://my-bucket/simulations/",
    ...     compression="blosc",
    ...     compression_level=5,
    ... )
    >>>
    >>> # Memory-mapped arrays for temporary cache
    >>> config = StorageConfig(
    ...     backend=StorageBackend.MMAP,
    ...     path="/tmp/neutryx_cache/",
    ...     compression=None,
    ... )
    """

    backend: StorageBackend
    path: str
    compression: Optional[str] = "blosc"
    compression_level: int = 5
    read_only: bool = False
    create_if_missing: bool = True
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.backend == StorageBackend.MMAP and self.compression is not None:
            raise ValueError("Memory-mapped arrays do not support compression")

        if self.compression_level < 1 or self.compression_level > 9:
            raise ValueError(f"Compression level must be 1-9, got {self.compression_level}")

        if self.metadata is None:
            self.metadata = {}


class DataStore(ABC):
    """Abstract base class for data storage backends.

    Provides a unified interface for saving and loading JAX arrays,
    metadata, and checkpoints across different storage backends.
    """

    def __init__(self, config: StorageConfig):
        """Initialize data store.

        Parameters
        ----------
        config : StorageConfig
            Storage configuration
        """
        self.config = config
        self._validate_config()

    @abstractmethod
    def _validate_config(self) -> None:
        """Validate backend-specific configuration.

        Raises
        ------
        ValueError
            If configuration is invalid for this backend
        """
        pass

    @abstractmethod
    def save_array(
        self,
        key: str,
        array: Array,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save a JAX array to storage.

        Parameters
        ----------
        key : str
            Unique identifier for the array
        array : Array
            JAX array to save
        metadata : dict, optional
            Additional metadata to save with the array

        Examples
        --------
        >>> store.save_array("simulation_paths", paths, metadata={"n_paths": 100000})
        """
        pass

    @abstractmethod
    def load_array(self, key: str) -> Array:
        """Load a JAX array from storage.

        Parameters
        ----------
        key : str
            Unique identifier for the array

        Returns
        -------
        Array
            Loaded JAX array

        Raises
        ------
        KeyError
            If key not found in storage
        """
        pass

    @abstractmethod
    def save_metadata(self, key: str, metadata: Dict[str, Any]) -> None:
        """Save metadata (without associated array).

        Parameters
        ----------
        key : str
            Unique identifier
        metadata : dict
            Metadata to save (must be JSON-serializable)
        """
        pass

    @abstractmethod
    def load_metadata(self, key: str) -> Dict[str, Any]:
        """Load metadata.

        Parameters
        ----------
        key : str
            Unique identifier

        Returns
        -------
        dict
            Loaded metadata

        Raises
        ------
        KeyError
            If key not found
        """
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if a key exists in storage.

        Parameters
        ----------
        key : str
            Key to check

        Returns
        -------
        bool
            True if key exists
        """
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete a key from storage.

        Parameters
        ----------
        key : str
            Key to delete

        Raises
        ------
        KeyError
            If key not found
        """
        pass

    @abstractmethod
    def list_keys(self, prefix: Optional[str] = None) -> list[str]:
        """List all keys in storage.

        Parameters
        ----------
        prefix : str, optional
            Filter keys by prefix

        Returns
        -------
        list[str]
            List of keys
        """
        pass

    @abstractmethod
    def get_size_bytes(self, key: str) -> int:
        """Get storage size in bytes for a key.

        Parameters
        ----------
        key : str
            Key to query

        Returns
        -------
        int
            Size in bytes
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all data from storage.

        Warnings
        --------
        This is a destructive operation!
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the storage backend and release resources."""
        pass

    def __enter__(self) -> DataStore:
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.close()

    def get_path(self) -> Path:
        """Get storage path as Path object.

        Returns
        -------
        Path
            Storage path (local only, raises for cloud URIs)
        """
        if self.config.path.startswith(("s3://", "gs://", "http://", "https://")):
            raise ValueError(f"Cannot convert cloud URI to Path: {self.config.path}")
        return Path(self.config.path)
