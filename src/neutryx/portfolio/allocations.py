import jax.numpy as jnp


def euler_allocation(portfolio_value, stand_alone_values):
    w = jnp.array(stand_alone_values) / (portfolio_value + 1e-12)
    return w * portfolio_value
