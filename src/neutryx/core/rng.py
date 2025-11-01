from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Tuple

import jax
import jax.numpy as jnp

from neutryx.core.utils.precision import canonicalize_dtype

try:  # pragma: no cover - compatibility shim for newer JAX releases
    KeyArray = jax.random.KeyArray
except AttributeError:  # JAX >= 0.8 removed the alias from jax.random
    KeyArray = jax.Array

__all__ = ["KeySeq", "split_key", "normal", "uniform"]


def split_key(key: KeyArray, n: int) -> Tuple[KeyArray, ...]:
    """Split a key into ``n`` sub-keys without mutating the original key."""
    if n < 1:
        return tuple()
    keys = jax.random.split(key, n + 1)
    return tuple(keys[1:])


def normal(key: KeyArray, shape: Iterable[int], dtype: Any = None) -> jnp.ndarray:
    """Standard normal samples for a given key and shape."""
    comp_dtype = canonicalize_dtype(dtype)
    return jax.random.normal(key, shape, dtype=comp_dtype)


def uniform(
    key: KeyArray,
    shape: Iterable[int],
    *,
    minval: float = 0.0,
    maxval: float = 1.0,
    dtype: Any = None,
) -> jnp.ndarray:
    """Uniform samples U[minval, maxval) for a given key and shape."""
    comp_dtype = canonicalize_dtype(dtype)
    return jax.random.uniform(
        key, shape, minval=minval, maxval=maxval, dtype=comp_dtype
    )


@dataclass
class KeySeq:
    """Stateful helper that manages a deterministic stream of PRNG keys."""

    seed: int = 0
    _key: KeyArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._key = jax.random.PRNGKey(self.seed)

    @classmethod
    def from_config(cls, config: Any) -> "KeySeq":
        """Instantiate a ``KeySeq`` using the global configuration ``seed``."""
        seed = getattr(config, "seed", None) if hasattr(config, "seed") else None
        if seed is None and isinstance(config, Mapping):
            seed = config.get("seed")  # type: ignore[arg-type]
        if seed is None:
            raise ValueError("Configuration does not define a 'seed' entry")
        return cls(seed=int(seed))

    @property
    def key(self) -> KeyArray:
        """Return the current master key (without consuming it)."""
        return self._key

    def next(self) -> KeyArray:
        """Return the next sub-key and update the internal state."""
        self._key, sub = jax.random.split(self._key)
        return sub

    def split(self, n: int) -> Tuple[KeyArray, ...]:
        """Return ``n`` sub-keys and update the internal state."""
        if n < 1:
            return tuple()
        keys = jax.random.split(self._key, n + 1)
        self._key = keys[0]
        return tuple(keys[1:])

    def fold_in(self, data: int) -> KeyArray:
        """Fold extra data into the stream to branch deterministically."""
        self._key = jax.random.fold_in(self._key, data)
        return self._key

    def normal(self, shape: Iterable[int], dtype: Any = None) -> jnp.ndarray:
        """Convenience wrapper for drawing standard normal samples."""
        return normal(self.next(), shape, dtype=dtype)

    def uniform(
        self,
        shape: Iterable[int],
        *,
        minval: float = 0.0,
        maxval: float = 1.0,
        dtype: Any = None,
    ) -> jnp.ndarray:
        """Convenience wrapper for drawing uniform samples."""
        return uniform(self.next(), shape, minval=minval, maxval=maxval, dtype=dtype)
