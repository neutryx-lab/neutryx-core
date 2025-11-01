import jax, jax.numpy as jnp
from neutryx.core.engine import MCConfig, simulate_gbm, present_value
from neutryx.products.asian import AsianArithmetic

key = jax.random.PRNGKey(0)
S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.01, 0.00, 0.2
cfg = MCConfig(steps=64, paths=200_000)
paths = simulate_gbm(key, S, r - q, sigma, T, cfg)
prod = AsianArithmetic(K=K, T=T, is_call=True)

payoffs = prod.path_payoffs(paths)
pv = present_value(payoffs, jnp.array(T), r)
print("Asian arithmetic call (MC):", float(pv))
