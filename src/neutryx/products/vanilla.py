"""Standard vanilla payoffs."""
from dataclasses import dataclass

import jax.numpy as jnp

from .base import Product


@dataclass
class European(Product):
    """European vanilla option payoff."""

    K: float
    T: float
    is_call: bool = True

    def payoff_terminal(self, spot: jnp.ndarray) -> jnp.ndarray:
        spot = jnp.asarray(spot)
        intrinsic = spot - self.K if self.is_call else self.K - spot
        return jnp.maximum(intrinsic, 0.0)

