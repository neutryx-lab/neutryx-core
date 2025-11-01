import jax
from neutryx.core.engine import MCConfig, simulate_gbm, present_value
import jax.numpy as jnp

S,K,T,r,q,sigma = 100.,100.,1.,0.01,0.0,0.2
cfg = MCConfig(steps=64, paths=50_000)
for seed in range(5):
    key = jax.random.PRNGKey(seed)
    paths = simulate_gbm(key, S, r-q, sigma, T, cfg)
    ST = paths[:,-1]
    pv = present_value(jnp.maximum(ST-K,0.0), jnp.array(T), r)
    print(seed, float(pv))
