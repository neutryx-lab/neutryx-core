"""Additional regression coverage for solver primitives."""
import jax
import jax.numpy as jnp
from neutryx.solver.sabr import SABRParams, hagan_implied_vol
from neutryx.solver.heston import HestonParams, heston_call_price


def test_sabr_vectorized_equals_scalar():
    params = SABRParams(alpha=0.25, beta=0.6, rho=-0.2, nu=0.5)
    F = 100.0
    strikes = jnp.array([90.0, 100.0, 110.0])
    maturity = 1.5

    vectorized = jax.vmap(lambda K: hagan_implied_vol(F, K, maturity, params))(strikes)
    scalar = jnp.array([hagan_implied_vol(F, float(K), maturity, params) for K in strikes])

    assert jnp.allclose(vectorized, scalar, atol=1e-8)


def test_sabr_negative_rho_creates_smile():
    params = SABRParams(alpha=0.2, beta=0.9, rho=-0.5, nu=0.4)
    F = 100.0
    maturity = 1.0

    low_strike = hagan_implied_vol(F, 90.0, maturity, params)
    atm_strike = hagan_implied_vol(F, 100.0, maturity, params)
    high_strike = hagan_implied_vol(F, 110.0, maturity, params)

    assert low_strike > atm_strike > high_strike


def test_heston_vectorized_equals_scalar():
    params = HestonParams(v0=0.04, kappa=1.5, theta=0.04, sigma=0.3, rho=-0.4)
    S0, r, q = 100.0, 0.03, 0.01
    strikes = jnp.array([90.0, 100.0, 110.0])
    maturities = jnp.array([0.5, 1.0, 1.5])

    vectorized = jax.vmap(lambda K, T: heston_call_price(S0, K, T, r, q, params))(strikes, maturities)
    scalar = jnp.array([
        heston_call_price(S0, float(K), float(T), r, q, params)
        for K, T in zip(strikes, maturities)
    ])

    assert jnp.allclose(vectorized, scalar, rtol=1e-5, atol=1e-5)


def test_heston_price_decreases_with_strike():
    params = HestonParams(v0=0.05, kappa=1.2, theta=0.04, sigma=0.4, rho=-0.5)
    S0, r, q, T = 100.0, 0.02, 0.0, 1.0
    strikes = jnp.array([80.0, 100.0, 120.0])

    prices = jax.vmap(lambda K: heston_call_price(S0, K, T, r, q, params))(strikes)

    assert prices[0] > prices[1] > prices[2]
