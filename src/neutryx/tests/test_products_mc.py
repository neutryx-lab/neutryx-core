import jax, jax.numpy as jnp
from neutryx.core.engine import MCConfig, simulate_gbm, present_value
from neutryx.products.vanilla import European

def test_mc_vanilla_close_to_bs():
    key = jax.random.PRNGKey(0)
    S,K,T,r,q,sigma = 100.,100.,1.,0.01,0.0,0.2
    cfg = MCConfig(steps=64, paths=50_000)
    paths = simulate_gbm(key, S, r-q, sigma, T, cfg)
    ST = paths[:,-1]
    payoffs = jnp.maximum(ST-K, 0.0)
    # Discount and average manually
    pv = jnp.exp(-r * T) * jnp.mean(payoffs)
    assert 7.5 < float(pv) < 9.0
