import jax.numpy as jnp

def epe(paths, K, is_call=True):
    ST = paths[:, -1]
    payoff = jnp.maximum(ST - K, 0.0) if is_call else jnp.maximum(K - ST, 0.0)
    return payoff.mean()
