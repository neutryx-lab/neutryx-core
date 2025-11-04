"""Utilities for configuring JAX distributed execution environments."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Sequence

import jax
import yaml

from neutryx.core.execution import MeshConfig

__all__ = [
    "ClusterConfig",
    "load_cluster_config",
]


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip():
        return int(value)
    return None


def _maybe_tuple(value: Sequence[int] | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    return tuple(int(v) for v in value)


@dataclass
class ClusterConfig:
    """Configuration for multi-process cluster deployments."""

    coordinator_address: str | None = None
    coordinator_port: int | None = None
    num_processes: int | None = None
    process_id: int | None = None
    local_device_ids: tuple[int, ...] | None = None
    mesh: MeshConfig | None = None

    def coordinator(self) -> str | None:
        if self.coordinator_address is None:
            return None
        if self.coordinator_port is None or ":" in self.coordinator_address:
            return self.coordinator_address
        return f"{self.coordinator_address}:{self.coordinator_port}"

    def initialize(self) -> None:
        """Initialise :mod:`jax.distributed` based on the configuration."""

        if self.coordinator_address is None:
            return

        if hasattr(jax, "distributed"):
            if getattr(jax.distributed, "is_initialized", lambda: False)():
                return

            kwargs: MutableMapping[str, Any] = {}
            if self.num_processes is not None:
                kwargs["num_processes"] = int(self.num_processes)
            if self.process_id is not None:
                kwargs["process_id"] = int(self.process_id)
            if self.local_device_ids is not None:
                kwargs["local_device_ids"] = tuple(int(v) for v in self.local_device_ids)

            jax.distributed.initialize(coordinator_address=self.coordinator(), **kwargs)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ClusterConfig":
        if not data:
            return cls()

        mesh_cfg = None
        mesh_data = data.get("mesh")
        if mesh_data:
            mesh_cfg = MeshConfig.from_mapping(mesh_data)

        local_ids = data.get("local_device_ids")
        local_ids_tuple = _maybe_tuple(local_ids)

        return cls(
            coordinator_address=data.get("coordinator_address"),
            coordinator_port=_maybe_int(data.get("coordinator_port")),
            num_processes=_maybe_int(data.get("num_processes")),
            process_id=_maybe_int(data.get("process_id")),
            local_device_ids=local_ids_tuple,
            mesh=mesh_cfg,
        )

    @classmethod
    def from_env(cls, prefix: str = "JAX_CLUSTER_") -> ClusterConfig | None:
        payload: dict[str, str] = {}
        for key, value in os.environ.items():
            if key.startswith(prefix):
                payload[key[len(prefix) :].lower()] = value

        if not payload:
            return None

        mesh_shape = payload.get("mesh_shape")
        mesh_axis_names = payload.get("mesh_axis_names")
        mesh_cfg = None
        if mesh_shape and mesh_axis_names:
            shape = tuple(int(v) for v in mesh_shape.split(",") if v)
            axis_names = tuple(name.strip() for name in mesh_axis_names.split(","))
            mesh_cfg = MeshConfig(shape=shape, axis_names=axis_names)

        local_ids = payload.get("local_device_ids")
        local_tuple = None
        if local_ids:
            local_tuple = tuple(int(v) for v in local_ids.split(",") if v)

        return cls(
            coordinator_address=payload.get("coordinator_address"),
            coordinator_port=_maybe_int(payload.get("coordinator_port")),
            num_processes=_maybe_int(payload.get("num_processes")),
            process_id=_maybe_int(payload.get("process_id")),
            local_device_ids=local_tuple,
            mesh=mesh_cfg,
        )


def load_cluster_config(path: str | os.PathLike[str], *, key: str = "cluster") -> ClusterConfig:
    """Load a :class:`ClusterConfig` from a YAML configuration file."""

    with open(path, "r", encoding="utf8") as fh:
        data = yaml.safe_load(fh) or {}

    cluster_data = data.get(key, data)
    return ClusterConfig.from_mapping(cluster_data)

