"""Barrier option payoffs."""
from dataclasses import dataclass

import jax.numpy as jnp

from .base import PathProduct


@dataclass
class UpAndOutCall(PathProduct):
    """Up-and-out barrier call option."""

    K: float
    T: float
    B: float

    def payoff_path(self, path: jnp.ndarray) -> jnp.ndarray:
        path = jnp.asarray(path)
        hit = path.max() >= self.B
        ST = path[-1]
        intrinsic = jnp.maximum(ST - self.K, 0.0)
        return jnp.where(hit, 0.0, intrinsic)

