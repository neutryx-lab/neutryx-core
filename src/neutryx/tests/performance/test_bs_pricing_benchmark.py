import jax
import jax.numpy as jnp

from neutryx.models.bs import price


@jax.jit
def _price_call_spot_grid(spots: jnp.ndarray) -> jnp.ndarray:
    return jax.vmap(lambda s: price(s, 100.0, 1.0, 0.01, 0.00, 0.2, "call"))(spots)


def test_bs_pricing_benchmark(benchmark):
    spots = jnp.linspace(80.0, 120.0, 64)
    benchmark(_price_call_spot_grid, spots)
