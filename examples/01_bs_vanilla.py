import jax, jax.numpy as jnp
from neutryx.core.engine import MCConfig, simulate_gbm, present_value
from neutryx.models.bs import price as bs_price

key = jax.random.PRNGKey(42)
S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.01, 0.00, 0.2

cfg = MCConfig(steps=252, paths=100_000)
paths = simulate_gbm(key, S, r - q, sigma, T, cfg)
ST = paths[:, -1]
call_mc = present_value(jnp.maximum(ST - K, 0.0), jnp.array(T), r)
call_an = bs_price(S, K, T, r, q, sigma, "call")

print("Call (MC):", float(call_mc))
print("Call (BS):", float(call_an))
