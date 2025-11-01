"""Utilities for managing JAX parallel execution.

This module provides a light-weight abstraction over JAX's :func:`pmap` and
``pjit`` APIs.  The project previously mixed both entry points directly which
made it difficult to reason about compilation behaviour and device placement.

The utilities exposed here aim to provide the following guarantees:

* A single access point (:func:`compile_parallel`) that selects the optimal
  execution strategy (``pjit`` when a mesh is available, otherwise ``pmap`` or
  ``jit``).
* Static-shape compilation by caching executables per-shape and optionally
  lowering with ``ShapeDtypeStruct`` templates.
* Convenience helpers for prefetching host data to devices to reduce stalls
  during multi-step workloads.

The helpers are intentionally conservative – they fall back to plain
``jax.jit`` on single device environments so the code path remains compatible
with CPU-only CI.
"""

from __future__ import annotations

import contextlib
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pjit
from jax.sharding import Mesh
from jax import tree_util


PyTree = Any


@dataclass(frozen=True)
class ParallelConfig:
    """Configuration parameters for :func:`compile_parallel`.

    Attributes
    ----------
    axis_name:
        Logical axis used when falling back to :func:`jax.pmap`.
    prefer_pjit:
        Whether to attempt compiling with :func:`jax.experimental.pjit` when a
        multi-device mesh is available.
    mesh_axes:
        Names assigned to the mesh dimensions when using ``pjit``.
    mesh_shape:
        Desired mesh shape.  When ``None`` a one-dimensional mesh matching the
        number of available devices is used.
    devices:
        Explicit devices to use.  When ``None`` the local devices are used.
    in_shardings / out_shardings:
        Optional shardings forwarded to ``pjit``.
    static_argnums / static_argnames:
        Forwarded to :func:`jax.jit` and :func:`jax.pmap` to ensure re-usable
        static compilation across calls.
    donate_argnums:
        Arguments that can be donated to the compiled computation.
    """

    axis_name: str = "devices"
    prefer_pjit: bool = True
    mesh_axes: tuple[str, ...] = ("devices",)
    mesh_shape: Optional[tuple[int, ...]] = None
    devices: Optional[Sequence[jax.Device]] = None
    in_shardings: Any = None
    out_shardings: Any = None
    static_argnums: tuple[int, ...] = ()
    static_argnames: tuple[str, ...] = ()
    donate_argnums: tuple[int, ...] = ()


class ParallelExecutable:
    """Wrapper around a compiled parallel function.

    ``pjit`` compiled callables must execute inside a mesh context.  The wrapper
    tracks the strategy that was chosen during compilation and ensures the mesh
    is activated transparently for callers.
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        strategy: str,
        mesh_factory: Optional[Callable[[], contextlib.AbstractContextManager[Mesh]]] = None,
    ) -> None:
        self._fn = fn
        self._strategy = strategy
        self._mesh_factory = mesh_factory

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._strategy == "pjit" and self._mesh_factory is not None:
            with self._mesh_factory():
                return self._fn(*args, **kwargs)
        return self._fn(*args, **kwargs)

    @property
    def strategy(self) -> str:
        """Return the parallel strategy used (``pjit``, ``pmap`` or ``jit``)."""

        return self._strategy


def _create_mesh_context(config: ParallelConfig) -> Callable[[], contextlib.AbstractContextManager[Mesh]]:
    devices = tuple(config.devices or jax.local_devices())
    mesh_shape = config.mesh_shape or (len(devices),)
    if np.prod(mesh_shape, dtype=int) != len(devices):
        raise ValueError(
            "mesh_shape must match the number of devices. "
            f"Got shape {mesh_shape} for {len(devices)} device(s)."
        )

    def _context() -> contextlib.AbstractContextManager[Mesh]:
        device_mesh = np.array(devices).reshape(mesh_shape)
        return Mesh(device_mesh, config.mesh_axes)

    return _context


def compile_parallel(
    fn: Callable[..., Any],
    config: ParallelConfig,
    *,
    example_args: Optional[tuple[Any, ...]] = None,
    example_kwargs: Optional[dict[str, Any]] = None,
) -> ParallelExecutable:
    """Compile ``fn`` with the best suited JAX parallel primitive.

    Parameters
    ----------
    fn:
        Callable to be compiled.
    config:
        Parallel configuration.
    example_args / example_kwargs:
        Optional sample inputs used to eagerly lower ``fn``.  When provided the
        arguments are only used to trigger compilation – they are not stored.

    Returns
    -------
    ParallelExecutable
        Callable that can be invoked with the same signature as ``fn``.
    """

    devices = tuple(config.devices or jax.local_devices())
    multi_device = len(devices) > 1

    if config.prefer_pjit and multi_device:
        mesh_factory = _create_mesh_context(config)
        compiled = pjit.pjit(
            fn,
            in_shardings=config.in_shardings,
            out_shardings=config.out_shardings,
            donate_argnums=config.donate_argnums,
            static_argnums=config.static_argnums,
            static_argnames=config.static_argnames,
        )
        if example_args is not None or example_kwargs is not None:
            compiled.lower(*(example_args or ()), **(example_kwargs or {}))
        return ParallelExecutable(compiled, "pjit", mesh_factory)

    if multi_device:
        compiled = jax.pmap(
            fn,
            axis_name=config.axis_name,
            static_broadcasted_argnums=config.static_argnums,
        )
        return ParallelExecutable(compiled, "pmap")

    compiled = jax.jit(
        fn,
        static_argnums=config.static_argnums,
        static_argnames=config.static_argnames,
        donate_argnums=config.donate_argnums,
    )
    return ParallelExecutable(compiled, "jit")


def static_struct_like(example: PyTree) -> PyTree:
    """Return a :class:`jax.ShapeDtypeStruct` tree matching ``example``.

    The helper is handy when lowering computations up-front to ensure the
    compiled executable is cached for a fixed shape.
    """

    def _to_struct(x: Any) -> jax.ShapeDtypeStruct:
        arr = jnp.asarray(x)
        return jax.ShapeDtypeStruct(arr.shape, arr.dtype)

    return tree_util.tree_map(_to_struct, example)


def _maybe_shard(batch: PyTree, devices: Sequence[jax.Device]) -> Optional[Sequence[PyTree]]:
    """Attempt to shard ``batch`` across ``devices``.

    Returns ``None`` when the leading dimension does not match the device count
    (in which case replication is generally cheaper).
    """

    device_count = len(devices)
    leaves, treedef = tree_util.tree_flatten(batch)
    if not leaves:
        return None

    shard_leaves: list[list[Any]] = []
    for leaf in leaves:
        arr = jnp.asarray(leaf)
        if arr.shape and arr.shape[0] == device_count:
            shard_leaves.append([arr[i] for i in range(device_count)])
        else:
            return None

    shards: list[PyTree] = []
    for i in range(device_count):
        shards.append(tree_util.tree_unflatten(treedef, [leaf[i] for leaf in shard_leaves]))
    return shards


def prefetch_to_device(
    iterable: Iterable[PyTree],
    *,
    size: int = 2,
    devices: Optional[Sequence[jax.Device]] = None,
) -> Iterator[PyTree]:
    """Prefetch items from ``iterable`` onto device(s).

    Parameters
    ----------
    iterable:
        Source iterable producing batches.
    size:
        Number of in-flight batches to maintain.
    devices:
        Target devices.  Defaults to local devices.
    """

    if size <= 0:
        raise ValueError("prefetch size must be positive")

    it = iter(iterable)
    queue: deque[PyTree] = deque()
    target_devices = tuple(devices or jax.local_devices())
    single_device = target_devices[0] if len(target_devices) == 1 else None

    def _device_put(batch: PyTree) -> PyTree:
        if len(target_devices) > 1:
            shards = _maybe_shard(batch, target_devices)
            if shards is not None:
                return jax.device_put_sharded(shards, target_devices)
        return jax.device_put(batch, single_device)

    def _fill_queue() -> None:
        while len(queue) < size:
            try:
                batch = next(it)
            except StopIteration:
                break
            queue.append(_device_put(batch))

    _fill_queue()
    while queue:
        batch = queue.popleft()
        _fill_queue()
        yield batch


__all__ = [
    "ParallelConfig",
    "ParallelExecutable",
    "compile_parallel",
    "prefetch_to_device",
    "static_struct_like",
]

