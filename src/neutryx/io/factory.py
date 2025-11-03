"""Factory functions to create storage backends from configuration.

Provides a unified interface to instantiate the appropriate storage backend
based on StorageConfig, enabling easy switching between backends without
code changes.
"""

from __future__ import annotations

import os
import tempfile
from typing import Optional

from neutryx.io.base import DataStore, StorageBackend, StorageConfig
from neutryx.io.mmap_store import MMapStore
from neutryx.io.zarr_store import ZarrStore


def create_store(config: StorageConfig) -> DataStore:
    """Create a storage backend from configuration.

    Parameters
    ----------
    config : StorageConfig
        Storage configuration specifying backend type and parameters

    Returns
    -------
    DataStore
        Instantiated storage backend

    Raises
    ------
    ValueError
        If backend type is not supported

    Examples
    --------
    >>> # Create Zarr store for persistence
    >>> config = StorageConfig(
    ...     backend=StorageBackend.ZARR,
    ...     path="./data/simulations/",
    ...     compression="blosc",
    ... )
    >>> store = create_store(config)
    >>>
    >>> # Create MMap store for cache
    >>> config = StorageConfig(
    ...     backend=StorageBackend.MMAP,
    ...     path="/tmp/cache/",
    ...     compression=None,
    ... )
    >>> cache = create_store(config)
    """
    if config.backend == StorageBackend.ZARR:
        return ZarrStore(config)

    elif config.backend == StorageBackend.MMAP:
        return MMapStore(config)

    elif config.backend == StorageBackend.PICKLE:
        raise NotImplementedError(
            "Pickle backend is deprecated. Use Zarr for persistence or MMap for caching."
        )

    else:
        raise ValueError(f"Unsupported storage backend: {config.backend}")


def create_persistent_store(
    path: str,
    compression: str = "blosc",
    compression_level: int = 5,
    read_only: bool = False,
    **kwargs,
) -> DataStore:
    """Create a storage backend optimized for long-term persistence.

    Convenience function that creates a Zarr store with sensible defaults
    for permanent data storage.

    Parameters
    ----------
    path : str
        Storage path (local or cloud URI)
    compression : str, optional
        Compression algorithm (default: "blosc")
    compression_level : int, optional
        Compression level 1-9 (default: 5)
    read_only : bool, optional
        Open in read-only mode (default: False)
    **kwargs
        Additional metadata for StorageConfig

    Returns
    -------
    DataStore
        Zarr storage backend

    Examples
    --------
    >>> # Local persistent storage
    >>> store = create_persistent_store("./data/results/")
    >>>
    >>> # S3 storage with high compression
    >>> store = create_persistent_store(
    ...     "s3://my-bucket/simulations/",
    ...     compression="blosc",
    ...     compression_level=9,
    ...     fsspec={"anon": False},  # AWS credentials
    ... )
    """
    config = StorageConfig(
        backend=StorageBackend.ZARR,
        path=path,
        compression=compression,
        compression_level=compression_level,
        read_only=read_only,
        create_if_missing=True,
        metadata=kwargs,
    )
    return create_store(config)


def create_cache_store(
    path: Optional[str] = None,
    read_only: bool = False,
) -> DataStore:
    """Create a storage backend optimized for temporary caching.

    Convenience function that creates a memory-mapped store with sensible
    defaults for temporary data caching.

    Parameters
    ----------
    path : str, optional
        Local storage path. If None, uses system temp directory with
        "neutryx_cache" subdirectory. Can be overridden via
        NEUTRYX_CACHE_DIR environment variable.
    read_only : bool, optional
        Open in read-only mode (default: False)

    Returns
    -------
    DataStore
        Memory-mapped storage backend

    Examples
    --------
    >>> # Default temporary cache (uses system temp dir)
    >>> cache = create_cache_store()
    >>>
    >>> # Custom cache location
    >>> cache = create_cache_store("/scratch/neutryx/")
    """
    # Determine cache path
    if path is None:
        # Check environment variable first
        path = os.environ.get("NEUTRYX_CACHE_DIR")
        if path is None:
            # Use system temp directory
            temp_dir = tempfile.gettempdir()
            path = os.path.join(temp_dir, "neutryx_cache")

    config = StorageConfig(
        backend=StorageBackend.MMAP,
        path=path,
        compression=None,  # MMap doesn't support compression
        read_only=read_only,
        create_if_missing=True,
    )
    return create_store(config)


def create_store_from_string(
    backend: str,
    path: str,
    **kwargs,
) -> DataStore:
    """Create storage backend from string specification.

    Parameters
    ----------
    backend : {"zarr", "mmap", "pickle"}
        Backend type as string
    path : str
        Storage path
    **kwargs
        Additional configuration options

    Returns
    -------
    DataStore
        Storage backend

    Examples
    --------
    >>> store = create_store_from_string("zarr", "./data/", compression="blosc")
    """
    backend_enum = StorageBackend(backend.lower())
    config = StorageConfig(backend=backend_enum, path=path, **kwargs)
    return create_store(config)
