"""Zarr storage backend for long-term persistence with compression.

Zarr provides:
- High-performance compression (Blosc, LZ4, Gzip)
- Cloud storage support (S3, GCS) via fsspec
- Incremental writes and partial reads
- Metadata storage alongside arrays
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import jax.numpy as jnp
import numpy as np
from jax import Array

from neutryx.infrastructure.governance import record_artifact
from neutryx.io.base import DataStore, StorageConfig

try:
    import zarr
    from zarr.storage import DirectoryStore, FSStore

    ZARR_AVAILABLE = True
except ImportError:
    ZARR_AVAILABLE = False
    zarr = None
    DirectoryStore = None
    FSStore = None


class ZarrStore(DataStore):
    """Zarr-based storage backend with cloud support and compression.

    Provides efficient compressed storage with support for local filesystems
    and cloud object stores (S3, GCS). Ideal for long-term persistence of
    simulation results and large datasets.

    Examples
    --------
    >>> # Local storage with Blosc compression
    >>> config = StorageConfig(
    ...     backend=StorageBackend.ZARR,
    ...     path="./data/simulations/",
    ...     compression="blosc",
    ... )
    >>> with ZarrStore(config) as store:
    ...     store.save_array("paths", simulation_paths)
    ...     loaded_paths = store.load_array("paths")

    >>> # S3 storage
    >>> config = StorageConfig(
    ...     backend=StorageBackend.ZARR,
    ...     path="s3://my-bucket/neutryx/simulations/",
    ...     compression="blosc",
    ...     metadata={"s3": {"anon": False}},  # AWS credentials required
    ... )
    >>> store = ZarrStore(config)
    """

    def __init__(self, config: StorageConfig):
        """Initialize Zarr store."""
        if not ZARR_AVAILABLE:
            raise ImportError(
                "zarr is not installed. Install with: pip install zarr\n"
                "For cloud support also install: pip install fsspec s3fs gcsfs"
            )

        super().__init__(config)

        # Create storage backend
        if config.path.startswith(("s3://", "gs://", "http://", "https://")):
            # Cloud storage via fsspec
            try:
                import fsspec
            except ImportError as exc:
                raise ImportError(
                    "Cloud storage requires fsspec. Install with:\n"
                    "pip install fsspec s3fs gcsfs"
                ) from exc

            fs_options = config.metadata.get("fsspec", {})
            self._store = FSStore(config.path, **fs_options)
        else:
            # Local directory storage
            path = Path(config.path)
            if config.create_if_missing:
                path.mkdir(parents=True, exist_ok=True)
            self._store = DirectoryStore(str(path))

        # Open root group
        mode = "r" if config.read_only else "a"
        self._root = zarr.open_group(self._store, mode=mode)

        # Store compression settings
        self._compressor = self._get_compressor()

    def _validate_config(self) -> None:
        """Validate Zarr-specific configuration."""
        if self.config.compression not in {None, "blosc", "gzip", "lz4", "bz2", "lzma"}:
            raise ValueError(
                f"Unsupported compression for Zarr: {self.config.compression}. "
                f"Use: blosc, gzip, lz4, bz2, lzma, or None"
            )

    def _get_compressor(self) -> Optional[Any]:
        """Create Zarr compressor from config."""
        if self.config.compression is None:
            return None

        level = self.config.compression_level

        if self.config.compression == "blosc":
            return zarr.Blosc(cname="lz4", clevel=level, shuffle=zarr.Blosc.SHUFFLE)
        elif self.config.compression == "gzip":
            return zarr.GZip(level=level)
        elif self.config.compression == "lz4":
            return zarr.LZ4(level=level)
        elif self.config.compression == "bz2":
            return zarr.BZ2(level=level)
        elif self.config.compression == "lzma":
            return zarr.LZMA(preset=level)
        else:
            return None

    def save_array(
        self,
        key: str,
        array: Array,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save JAX array to Zarr."""
        if self.config.read_only:
            raise PermissionError("Store is read-only")

        # Convert JAX array to NumPy
        np_array = np.array(array)

        # Create or overwrite dataset
        self._root.array(
            key,
            np_array,
            chunks=True,  # Auto-chunking
            compressor=self._compressor,
            overwrite=True,
        )

        base_metadata: Dict[str, Any] = dict(metadata or {})
        base_metadata.setdefault("storage_backend", "zarr")
        base_metadata.setdefault("array_shape", list(np_array.shape))
        base_metadata.setdefault("array_dtype", str(np_array.dtype))
        enriched_metadata = record_artifact(
            key,
            kind="array",
            metadata=base_metadata,
            extra_event_metadata={"storage_backend": "zarr", "array_shape": list(np_array.shape)},
        )
        self._root[key].attrs.update(enriched_metadata)

    def load_array(self, key: str) -> Array:
        """Load array from Zarr."""
        if key not in self._root:
            raise KeyError(f"Key not found: {key}")

        # Load as NumPy array, convert to JAX
        np_array = self._root[key][:]
        return jnp.array(np_array)

    def save_metadata(self, key: str, metadata: Dict[str, Any]) -> None:
        """Save metadata to Zarr attributes."""
        if self.config.read_only:
            raise PermissionError("Store is read-only")

        # Store as JSON in a special metadata group
        metadata_group = self._root.require_group("__metadata__")
        enriched_metadata = record_artifact(
            key,
            kind="metadata",
            metadata={"storage_backend": "zarr", **metadata},
            extra_event_metadata={"storage_backend": "zarr"},
        )
        metadata_group.attrs[key] = json.dumps(enriched_metadata)

    def load_metadata(self, key: str) -> Dict[str, Any]:
        """Load metadata from Zarr attributes."""
        metadata_group = self._root.require_group("__metadata__")
        if key not in metadata_group.attrs:
            raise KeyError(f"Metadata key not found: {key}")

        return json.loads(metadata_group.attrs[key])

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        return key in self._root

    def delete(self, key: str) -> None:
        """Delete a key from Zarr."""
        if self.config.read_only:
            raise PermissionError("Store is read-only")

        if key not in self._root:
            raise KeyError(f"Key not found: {key}")

        del self._root[key]

    def list_keys(self, prefix: Optional[str] = None) -> list[str]:
        """List all keys in Zarr root."""
        keys = list(self._root.array_keys())

        if prefix is not None:
            keys = [k for k in keys if k.startswith(prefix)]

        return sorted(keys)

    def get_size_bytes(self, key: str) -> int:
        """Get compressed storage size."""
        if key not in self._root:
            raise KeyError(f"Key not found: {key}")

        dataset = self._root[key]
        # Return compressed size
        return int(dataset.nbytes_stored)

    def clear(self) -> None:
        """Clear all data from Zarr store."""
        if self.config.read_only:
            raise PermissionError("Store is read-only")

        for key in list(self._root.array_keys()):
            del self._root[key]

        for group in list(self._root.group_keys()):
            del self._root[group]

    def close(self) -> None:
        """Close Zarr store."""
        # Zarr stores don't require explicit closing
        pass

    def get_compression_ratio(self, key: str) -> float:
        """Get compression ratio for a dataset.

        Returns
        -------
        float
            Compression ratio (uncompressed / compressed)
        """
        if key not in self._root:
            raise KeyError(f"Key not found: {key}")

        dataset = self._root[key]
        uncompressed = dataset.nbytes
        compressed = dataset.nbytes_stored

        if compressed == 0:
            return 1.0

        return float(uncompressed) / float(compressed)

    def __repr__(self) -> str:
        """String representation."""
        n_arrays = len(list(self._root.array_keys()))
        return f"ZarrStore(path={self.config.path}, n_arrays={n_arrays})"
