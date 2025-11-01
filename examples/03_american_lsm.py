import jax, jax.numpy as jnp
from neutryx.core.engine import MCConfig, simulate_gbm
from neutryx.products.american import american_put_lsm

key = jax.random.PRNGKey(7)
S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.03, 0.00, 0.25
steps, paths = 100, 50_000
cfg = MCConfig(steps=steps, paths=paths)
paths = simulate_gbm(key, S, r - q, sigma, T, cfg)
dt = T / steps
pv = american_put_lsm(paths, K=K, r=r, dt=dt)
print("American put (LSM MC):", float(pv))
