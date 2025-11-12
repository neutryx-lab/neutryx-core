"""Memory-mapped array storage for zero-copy temporary caching.

Memory-mapped arrays provide:
- Zero-copy reads/writes (OS page cache)
- Fastest I/O for local storage
- Handle arrays larger than memory
- No compression (raw performance)

Ideal for temporary caches, intermediate tensors, and gradient storage.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import jax.numpy as jnp
import numpy as np
from jax import Array

from neutryx.infrastructure.governance import record_artifact
from neutryx.io.base import DataStore, StorageConfig


class MMapStore(DataStore):
    """Memory-mapped array storage backend.

    Uses NumPy memory-mapped files for zero-copy array storage. Provides
    the fastest I/O for local storage but does not support compression or
    cloud storage.

    Examples
    --------
    >>> # Temporary cache for intermediate results
    >>> config = StorageConfig(
    ...     backend=StorageBackend.MMAP,
    ...     path="/tmp/neutryx_cache/",
    ...     compression=None,  # MMap doesn't support compression
    ... )
    >>> with MMapStore(config) as store:
    ...     # Save large gradient array
    ...     store.save_array("gradients", large_gradient_array)
    ...
    ...     # Load with zero-copy (just maps memory)
    ...     grads = store.load_array("gradients")
    """

    def __init__(self, config: StorageConfig):
        """Initialize memory-mapped store."""
        super().__init__(config)

        self._path = Path(config.path)

        if config.create_if_missing:
            self._path.mkdir(parents=True, exist_ok=True)

        if not self._path.exists():
            raise FileNotFoundError(f"Storage path does not exist: {self._path}")

        if not self._path.is_dir():
            raise NotADirectoryError(f"Storage path is not a directory: {self._path}")

        # Metadata directory
        self._metadata_dir = self._path / "__metadata__"
        if config.create_if_missing:
            self._metadata_dir.mkdir(exist_ok=True)

    def _validate_config(self) -> None:
        """Validate memory-mapped configuration."""
        if self.config.compression is not None:
            raise ValueError("Memory-mapped arrays do not support compression")

        if self.config.path.startswith(("s3://", "gs://", "http://", "https://")):
            raise ValueError("Memory-mapped arrays require local filesystem paths")

    def _get_array_path(self, key: str) -> Path:
        """Get file path for an array."""
        # Sanitize key to valid filename
        safe_key = key.replace("/", "_").replace("\\", "_")
        return self._path / f"{safe_key}.npy"

    def _get_metadata_path(self, key: str) -> Path:
        """Get file path for metadata."""
        safe_key = key.replace("/", "_").replace("\\", "_")
        return self._metadata_dir / f"{safe_key}.json"

    def save_array(
        self,
        key: str,
        array: Array,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save JAX array as memory-mapped file."""
        if self.config.read_only:
            raise PermissionError("Store is read-only")

        array_path = self._get_array_path(key)

        # Convert JAX array to NumPy
        np_array = np.array(array)

        # Create memory-mapped array file
        mmap_array = np.memmap(
            array_path,
            dtype=np_array.dtype,
            mode="w+",
            shape=np_array.shape,
        )

        # Write data
        mmap_array[:] = np_array[:]

        # Flush to disk
        mmap_array.flush()

        # Close the memmap
        del mmap_array

        metadata_path = self._get_metadata_path(key)
        base_metadata: Dict[str, Any] = dict(metadata or {})
        base_metadata.setdefault("storage_backend", "mmap")
        base_metadata.setdefault("array_shape", list(np_array.shape))
        base_metadata.setdefault("array_dtype", str(np_array.dtype))
        enriched_metadata = record_artifact(
            key,
            kind="array",
            metadata=base_metadata,
            extra_event_metadata={"storage_backend": "mmap", "array_shape": list(np_array.shape)},
        )
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(enriched_metadata, f, indent=2)

    def load_array(self, key: str) -> Array:
        """Load array from memory-mapped file.

        Returns a JAX array backed by memory-mapped storage. For very large
        arrays, consider using load_array_mmap() to work with the mmap directly.
        """
        array_path = self._get_array_path(key)

        if not array_path.exists():
            raise KeyError(f"Key not found: {key}")

        # Load memory-mapped array
        mmap_array = np.load(str(array_path), mmap_mode="r")

        # Convert to JAX array (will copy to device if GPU)
        return jnp.array(mmap_array)

    def load_array_mmap(self, key: str, mode: str = "r") -> np.memmap:
        """Load array as NumPy memmap without copying.

        Useful for very large arrays that don't fit in GPU memory.

        Parameters
        ----------
        key : str
            Array key
        mode : {"r", "r+", "c"}
            Access mode:
            - "r": Read-only
            - "r+": Read-write
            - "c": Copy-on-write

        Returns
        -------
        np.memmap
            Memory-mapped array (zero-copy)

        Examples
        --------
        >>> # Load large array without copying to GPU
        >>> mmap_array = store.load_array_mmap("huge_gradients", mode="r")
        >>> # Process in chunks
        >>> chunk = jnp.array(mmap_array[0:1000])  # Load first 1000 elements
        """
        array_path = self._get_array_path(key)

        if not array_path.exists():
            raise KeyError(f"Key not found: {key}")

        if self.config.read_only and mode != "r":
            raise PermissionError("Store is read-only")

        return np.load(str(array_path), mmap_mode=mode)

    def save_metadata(self, key: str, metadata: Dict[str, Any]) -> None:
        """Save metadata to JSON file."""
        if self.config.read_only:
            raise PermissionError("Store is read-only")

        metadata_path = self._get_metadata_path(key)

        enriched_metadata = record_artifact(
            key,
            kind="metadata",
            metadata={"storage_backend": "mmap", **metadata},
            extra_event_metadata={"storage_backend": "mmap"},
        )
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(enriched_metadata, f, indent=2)

    def load_metadata(self, key: str) -> Dict[str, Any]:
        """Load metadata from JSON file."""
        metadata_path = self._get_metadata_path(key)

        if not metadata_path.exists():
            raise KeyError(f"Metadata key not found: {key}")

        with open(metadata_path, "r") as f:
            return json.load(f)

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        return self._get_array_path(key).exists()

    def delete(self, key: str) -> None:
        """Delete array and associated metadata."""
        if self.config.read_only:
            raise PermissionError("Store is read-only")

        array_path = self._get_array_path(key)
        if not array_path.exists():
            raise KeyError(f"Key not found: {key}")

        # Delete array file
        array_path.unlink()

        # Delete metadata if exists
        metadata_path = self._get_metadata_path(key)
        if metadata_path.exists():
            metadata_path.unlink()

    def list_keys(self, prefix: Optional[str] = None) -> list[str]:
        """List all keys in storage."""
        keys = []

        for array_path in self._path.glob("*.npy"):
            key = array_path.stem  # Remove .npy extension

            if prefix is None or key.startswith(prefix):
                keys.append(key)

        return sorted(keys)

    def get_size_bytes(self, key: str) -> int:
        """Get file size in bytes."""
        array_path = self._get_array_path(key)

        if not array_path.exists():
            raise KeyError(f"Key not found: {key}")

        return array_path.stat().st_size

    def clear(self) -> None:
        """Clear all data from storage."""
        if self.config.read_only:
            raise PermissionError("Store is read-only")

        # Delete all .npy files
        for array_path in self._path.glob("*.npy"):
            array_path.unlink()

        # Delete all metadata
        if self._metadata_dir.exists():
            shutil.rmtree(self._metadata_dir)
            self._metadata_dir.mkdir()

    def close(self) -> None:
        """Close store (no-op for mmap)."""
        pass

    def get_total_size_bytes(self) -> int:
        """Get total size of all arrays in storage.

        Returns
        -------
        int
            Total size in bytes
        """
        total = 0
        for array_path in self._path.glob("*.npy"):
            total += array_path.stat().st_size
        return total

    def __repr__(self) -> str:
        """String representation."""
        n_arrays = len(list(self._path.glob("*.npy")))
        total_mb = self.get_total_size_bytes() / (1024**2)
        return f"MMapStore(path={self._path}, n_arrays={n_arrays}, size={total_mb:.1f}MB)"
