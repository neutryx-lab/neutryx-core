"""Configuration utilities for Neutryx."""

from __future__ import annotations

import logging
import random
from copy import deepcopy
from typing import Any, Mapping, MutableMapping

import jax
import numpy as np

try:  # pragma: no cover - exercised indirectly in environments without ml-collections.
    from ml_collections import ConfigDict as _ConfigDict
except ModuleNotFoundError:  # pragma: no cover - fallback for limited environments.
    class _ConfigDict(dict):
        """Minimal stand-in for ``ml_collections.ConfigDict``."""

        def __getattr__(self, name: str) -> Any:
            if name in self:
                return self[name]
            raise AttributeError(name)

        def __setattr__(self, name: str, value: Any) -> None:
            self[name] = value

        def copy_and_resolve_references(self) -> "_ConfigDict":
            return _ConfigDict(deepcopy(self))

    ConfigDict = _ConfigDict
else:  # pragma: no cover - exercised when ml-collections is available.
    ConfigDict = _ConfigDict

__all__ = ["ConfigDict", "get_config", "get_default_config", "init_environment"]


def get_default_config() -> ConfigDict:
    """Return the canonical configuration for the Neutryx stack."""
    cfg = ConfigDict()
    cfg.seed = 0

    cfg.logging = ConfigDict()
    cfg.logging.level = "INFO"
    cfg.logging.format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    cfg.logging.datefmt = "%Y-%m-%d %H:%M:%S"
    cfg.logging.force = True

    cfg.jax = ConfigDict()
    cfg.jax.enable_x64 = True

    # Storage configuration
    cfg.storage = ConfigDict()
    cfg.storage.persistent = ConfigDict()
    cfg.storage.persistent.backend = "zarr"
    cfg.storage.persistent.path = "./data/persistent/"
    cfg.storage.persistent.compression = "blosc"
    cfg.storage.persistent.compression_level = 5

    cfg.storage.cache = ConfigDict()
    cfg.storage.cache.backend = "mmap"
    cfg.storage.cache.path = None  # Uses system temp dir (configurable via NEUTRYX_CACHE_DIR)
    cfg.storage.cache.compression = None

    # Compute configuration
    cfg.compute = ConfigDict()
    cfg.compute.backend = "local_cpu"  # local_cpu, local_gpu, ray_cluster
    cfg.compute.num_devices = None  # Auto-detect
    cfg.compute.chunk_size = 10000  # Paths per chunk for simulations
    cfg.compute.batch_size = None  # Auto-determine

    # Monte Carlo configuration
    cfg.monte_carlo = ConfigDict()
    cfg.monte_carlo.default_paths = 100000
    cfg.monte_carlo.default_steps = 600
    cfg.monte_carlo.dtype = "float32"
    cfg.monte_carlo.stream_to_storage = False  # Enable for >1M paths

    return cfg


def get_config(overrides: Mapping[str, Any] | None = None) -> ConfigDict:
    """Create a configuration, optionally applying ``overrides``."""
    cfg = get_default_config()
    if overrides:
        _deep_update(cfg, overrides)
    return cfg


def init_environment(config: ConfigDict | Mapping[str, Any] | None = None) -> ConfigDict:
    """Seed all libraries and configure logging based on ``config``."""
    if config is None:
        cfg = get_default_config()
    elif isinstance(config, ConfigDict):
        cfg = config.copy_and_resolve_references()
    else:
        cfg = ConfigDict(deepcopy(dict(config)))

    seed = int(cfg.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    cfg.runtime = ConfigDict()
    cfg.runtime.seed = seed
    cfg.runtime.jax_key = jax.random.PRNGKey(seed)

    logging_cfg = cfg.get("logging", {})
    level = logging_cfg.get("level", "INFO")
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format=logging_cfg.get("format", None),
        datefmt=logging_cfg.get("datefmt", None),
        force=logging_cfg.get("force", False),
    )

    jax_cfg = cfg.get("jax", {})
    enable_x64 = jax_cfg.get("enable_x64")
    if enable_x64 is not None:
        from jax import config as jax_config

        jax_config.update("jax_enable_x64", bool(enable_x64))

    return cfg


def _deep_update(target: MutableMapping[str, Any], updates: Mapping[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, Mapping):
            if key not in target or not isinstance(target[key], MutableMapping):
                target[key] = ConfigDict()
            _deep_update(target[key], value)
        else:
            target[key] = value
