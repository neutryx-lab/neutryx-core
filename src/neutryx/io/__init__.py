"""High-performance I/O backends for data persistence and caching.

This module provides storage backends optimized for different use cases:

1. **Zarr**: Compressed, cloud-native storage for long-term persistence
   - Blosc compression (5-10x size reduction)
   - Incremental writes
   - S3/GCS compatible
   - Use for: Simulation results, calibration outputs, historical data

2. **Memory-Mapped Arrays**: Zero-copy local storage for temporary caching
   - Fastest read/write (no serialization)
   - OS-level page caching
   - Use for: Intermediate tensors, gradient caches, large temporary arrays

3. **Unified Store API**: Switch between backends via configuration
   - Common interface for all backends
   - Easy testing and deployment

Design Principles
-----------------
- **Separation of concerns**: Persistence vs caching
- **Cloud-first**: S3/GCS support built-in
- **Performance**: Zero-copy when possible
- **Configurability**: Switch backends without code changes
"""

from neutryx.io.base import DataStore, StorageBackend, StorageConfig
from neutryx.io.factory import (
    create_cache_store,
    create_persistent_store,
    create_store,
    create_store_from_string,
)
from neutryx.io.mmap_store import MMapStore
from neutryx.io.zarr_store import ZarrStore

__all__ = [
    # Base interfaces
    "DataStore",
    "StorageBackend",
    "StorageConfig",
    # Factory functions
    "create_store",
    "create_persistent_store",
    "create_cache_store",
    "create_store_from_string",
    # Concrete implementations
    "ZarrStore",
    "MMapStore",
]
