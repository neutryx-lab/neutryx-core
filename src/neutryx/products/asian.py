"""Asian option payoffs."""
from dataclasses import dataclass

import jax.numpy as jnp

from .base import PathProduct


@dataclass
class AsianArithmetic(PathProduct):
    """Arithmetic-average Asian option."""

    K: float
    T: float
    is_call: bool = True

    def payoff_path(self, path: jnp.ndarray) -> jnp.ndarray:
        path = jnp.asarray(path)
        avg = path.mean()
        intrinsic = avg - self.K if self.is_call else self.K - avg
        return jnp.maximum(intrinsic, 0.0)

