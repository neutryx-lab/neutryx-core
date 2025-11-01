"""Lookback option payoffs."""
from dataclasses import dataclass

import jax.numpy as jnp

from .base import PathProduct


@dataclass
class LookbackFloatStrikeCall(PathProduct):
    """Floating-strike lookback call option."""

    T: float

    def payoff_path(self, path: jnp.ndarray) -> jnp.ndarray:
        path = jnp.asarray(path)
        return path[-1] - path.min()

