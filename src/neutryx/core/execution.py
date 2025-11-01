"""Mesh-aware execution utilities built on top of JAX SPMD primitives.

This module provides a thin abstraction layer around :func:`jax.experimental.pjit`
and :func:`jax.experimental.maps.xmap` to make it straightforward to construct
device meshes from configuration objects and bind compiled functions to those
meshes.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

import jax
from jax.experimental import pjit as pjit_lib
from jax.sharding import Mesh, NamedSharding, PartitionSpec

try:  # pragma: no cover - optional dependency that may be absent on CPU wheels.
    from jax.experimental import maps as _maps
except ImportError:  # pragma: no cover
    _maps = None

Array = jax.Array

__all__ = [
    "Array",
    "MeshConfig",
    "mesh_context",
    "mesh_named_sharding",
    "mesh_pjit",
    "mesh_xmap",
]


def _maybe_tuple(value: Sequence[int] | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    return tuple(int(v) for v in value)


def _maybe_axis_mapping(mapping: Mapping[str, int] | None) -> Mapping[str, int] | None:
    if mapping is None:
        return None
    return {str(k): int(v) for k, v in mapping.items()}


@dataclass(frozen=True)
class MeshConfig:
    """Configuration describing a logical device mesh."""

    axis_sizes: Mapping[str, int] | None = None
    shape: Sequence[int] | None = None
    axis_names: Sequence[str] | None = None
    devices: Sequence[jax.Device] | None = None

    def __post_init__(self) -> None:
        axis_sizes = _maybe_axis_mapping(self.axis_sizes)
        shape = _maybe_tuple(self.shape)
        axis_names = tuple(self.axis_names) if self.axis_names is not None else None

        object.__setattr__(self, "axis_sizes", axis_sizes)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "axis_names", axis_names)

        if self.axis_sizes is not None and (self.shape is not None or self.axis_names is not None):
            raise ValueError("Provide either axis_sizes or shape/axis_names, not both.")
        if self.axis_sizes is None:
            if self.shape is None or self.axis_names is None:
                raise ValueError("shape and axis_names must be provided when axis_sizes is omitted.")
            if len(self.shape) != len(self.axis_names):
                raise ValueError("shape and axis_names must have the same length.")
        if self.axis_sizes is not None and len(self.axis_sizes) == 0:
            raise ValueError("axis_sizes must contain at least one axis when provided.")
        if self.axis_sizes is None and self.shape and len(self.shape) == 0:
            raise ValueError("Mesh must contain at least one axis.")

    @property
    def axis_names_tuple(self) -> tuple[str, ...]:
        if self.axis_sizes is not None:
            return tuple(self.axis_sizes.keys())
        assert self.axis_names is not None
        return tuple(self.axis_names)

    @property
    def shape_tuple(self) -> tuple[int, ...]:
        if self.axis_sizes is not None:
            return tuple(self.axis_sizes[name] for name in self.axis_names_tuple)
        assert self.shape is not None
        return tuple(self.shape)

    def required_devices(self) -> int:
        shape = self.shape_tuple
        return int(np.prod(shape))

    def create_mesh(self) -> Mesh:
        """Create a :class:`jax.sharding.Mesh` from the configuration."""

        axis_names = self.axis_names_tuple
        shape = self.shape_tuple
        devices = list(self.devices) if self.devices is not None else list(jax.devices())
        required = self.required_devices()
        if required != len(devices):
            raise ValueError(
                f"Mesh requires {required} devices but {len(devices)} were provided."
            )
        device_array = np.array(devices, dtype=object).reshape(shape)
        return Mesh(device_array, axis_names)

    @contextmanager
    def context(self) -> Iterable[Mesh]:
        mesh = self.create_mesh()
        with mesh:
            yield mesh

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable dictionary representation."""

        return {
            "axis_names": list(self.axis_names_tuple),
            "shape": list(self.shape_tuple),
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "MeshConfig":
        """Instantiate from a dictionary (e.g. YAML configuration)."""

        if not data:
            raise ValueError("Mesh configuration mapping cannot be empty.")

        axis_sizes = data.get("axis_sizes")
        shape = data.get("shape")
        axis_names = data.get("axis_names")

        return cls(
            axis_sizes=_maybe_axis_mapping(axis_sizes),
            shape=_maybe_tuple(shape),
            axis_names=tuple(axis_names) if axis_names is not None else None,
        )


@contextmanager
def mesh_context(config: MeshConfig | None) -> Iterable[Mesh | None]:
    """Context manager that enters a device mesh if requested."""

    if config is None:
        yield None
        return

    with config.context() as mesh:
        yield mesh


def mesh_named_sharding(config: MeshConfig, partition: PartitionSpec) -> NamedSharding:
    """Helper creating a :class:`jax.sharding.NamedSharding` for a mesh."""

    mesh = config.create_mesh()
    return NamedSharding(mesh, partition)


def mesh_pjit(
    fn: Callable[..., Any],
    *,
    mesh: MeshConfig | None = None,
    in_shardings: Any = None,
    out_shardings: Any = None,
    donate_argnums: Any = (),
    static_argnums: Any = (),
    **pjit_kwargs: Any,
) -> Callable[..., Any]:
    """Compile ``fn`` with :func:`pjit` and bind it to ``mesh`` when provided."""

    compiled = pjit_lib.pjit(
        fn,
        in_shardings=in_shardings,
        out_shardings=out_shardings,
        donate_argnums=donate_argnums,
        static_argnums=static_argnums,
        **pjit_kwargs,
    )

    if mesh is None:
        return compiled

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with mesh_context(mesh):
            return compiled(*args, **kwargs)

    return wrapped


def mesh_xmap(
    fn: Callable[..., Any],
    *,
    mesh: MeshConfig | None,
    in_axes: Any,
    out_axes: Any,
    axis_resources: Mapping[str, str | tuple[str, ...]],
    donate_argnums: Iterable[int] | tuple[int, ...] = (),
    **xmap_kwargs: Any,
) -> Callable[..., Any]:
    """Wrap ``fn`` with :func:`xmap` and bind it to a mesh.

    When ``jax.experimental.maps`` is unavailable (e.g. on minimal CPU wheels) the
    implementation falls back to a ``vmap`` based approximation that supports
    single-axis mappings, ensuring that code paths remain testable in CI.
    """

    if _maps is None:  # pragma: no cover - executed on minimal builds.
        if len(axis_resources) != 1:
            raise NotImplementedError("xmap fallback only supports single axis resources.")
        axis_name, mesh_axis = next(iter(axis_resources.items()))
        if isinstance(mesh_axis, tuple):
            mesh_axis = mesh_axis[0]
        if mesh_axis != axis_name:
            raise NotImplementedError("Fallback expects axis_resources to match input axis names.")
        if not isinstance(in_axes, (tuple, list)) or len(in_axes) != 1 or in_axes[0] != axis_name:
            raise NotImplementedError("Fallback expects a single positional argument mapped on the declared axis.")
        if out_axes != axis_name:
            raise NotImplementedError("Fallback expects output axis to match input axis name.")

        def mapped(*args: Any, **kwargs: Any) -> Any:
            if kwargs:
                raise NotImplementedError("Fallback does not support keyword arguments.")
            (arg,) = args
            return jax.vmap(fn)(arg)

    else:
        mapped = _maps.xmap(
            fn,
            in_axes=in_axes,
            out_axes=out_axes,
            axis_resources=axis_resources,
            donate_argnums=tuple(donate_argnums),
            **xmap_kwargs,
        )

    if mesh is None:
        return mapped

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with mesh_context(mesh):
            return mapped(*args, **kwargs)

    return wrapped

