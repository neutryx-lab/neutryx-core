import jax.numpy as jnp


def hazard_to_pd(lambda_t, times):
    return 1.0 - jnp.exp(-lambda_t * times)
