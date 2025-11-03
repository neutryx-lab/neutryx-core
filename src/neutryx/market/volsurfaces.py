from dataclasses import dataclass

import jax.numpy as jnp


@dataclass
class VolSmile:
    K: jnp.ndarray
    iv: jnp.ndarray
    T: float
    def interp(self, k):
        # simple linear interpolation in strike
        return jnp.interp(k, self.K, self.iv)
