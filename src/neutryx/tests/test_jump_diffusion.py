"""Tests for jump-diffusion dynamics and local volatility utilities."""

import math

import jax
import jax.numpy as jnp

from neutryx.core.engine import (
    MCConfig,
    price_vanilla_jump_diffusion_mc,
)
from neutryx.models import bs as bs_model
from neutryx.models.jump_diffusion import merton_jump_price
from neutryx.solver.local_vol import dupire_local_volatility_surface


def _numpy_merton_call(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    lam: float,
    mu_jump: float,
    sigma_jump: float,
    *,
    n_terms: int = 128,
) -> float:
    if T <= 0:
        return max(S0 - K, 0.0)

    lt = lam * T
    kappa = math.exp(mu_jump + 0.5 * sigma_jump ** 2) - 1.0
    drift = (r - q - lam * kappa - 0.5 * sigma ** 2) * T
    total = 0.0
    weight = math.exp(-lt)

    logK = math.log(K)
    for n in range(n_terms):
        if n > 0:
            weight *= lt / n
        var = sigma ** 2 * T + n * sigma_jump ** 2
        mean = math.log(S0) + drift + n * mu_jump
        sqrt_var = math.sqrt(max(var, 1e-16))
        d1 = (mean - logK + var) / sqrt_var
        d2 = d1 - sqrt_var
        phi_d1 = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))
        phi_d2 = 0.5 * (1.0 + math.erf(d2 / math.sqrt(2.0)))
        payoff = math.exp(mean + 0.5 * var) * phi_d1 - K * phi_d2
        total += weight * payoff

    return math.exp(-r * T) * total


def test_merton_reduces_to_black_scholes():
    S0, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.03, 0.0, 0.2
    price_bs = bs_model.price(S0, K, T, r, q, sigma, kind="call")
    price_mjd = merton_jump_price(S0, K, T, r, q, sigma, 0.0, 0.0, 0.0)
    assert abs(float(price_bs - price_mjd)) < 1e-4


def test_merton_matches_numpy_benchmark():
    S0, K, T = 100.0, 95.0, 1.0
    r, q = 0.05, 0.0
    sigma = 0.15
    lam = 0.75
    mu_jump = -0.05
    sigma_jump = 0.3

    benchmark = _numpy_merton_call(S0, K, T, r, q, sigma, lam, mu_jump, sigma_jump, n_terms=256)
    price = float(merton_jump_price(S0, K, T, r, q, sigma, lam, mu_jump, sigma_jump, n_terms=128))

    assert math.isclose(price, benchmark, rel_tol=5e-5, abs_tol=5e-5)


def test_jump_diffusion_mc_matches_analytic():
    key = jax.random.PRNGKey(42)
    S0, K, T = 100.0, 100.0, 1.0
    r, q = 0.03, 0.0
    sigma = 0.2
    lam = 0.6
    mu_jump = -0.1
    sigma_jump = 0.25

    cfg = MCConfig(steps=128, paths=40_000, antithetic=True)
    mc_price = float(
        price_vanilla_jump_diffusion_mc(
            key,
            S0,
            K,
            T,
            r,
            q,
            sigma,
            lam,
            mu_jump,
            sigma_jump,
            cfg,
        )
    )
    analytic = float(merton_jump_price(S0, K, T, r, q, sigma, lam, mu_jump, sigma_jump, n_terms=128))

    assert abs(mc_price - analytic) / analytic < 0.07


def test_dupire_local_vol_constant_surface():
    S0 = 100.0
    strikes = jnp.linspace(80.0, 120.0, 5)
    maturities = jnp.linspace(0.25, 2.0, 4)
    implied_vol = jnp.full((maturities.shape[0], strikes.shape[0]), 0.2)

    surface = dupire_local_volatility_surface(S0, strikes, maturities, implied_vol, r=0.01, q=0.0)

    interior = surface.local_vol[1:-1, 1:-1]
    assert jnp.allclose(interior, 0.2, atol=5e-2)
    centre = surface.value(float(maturities[1]), float(strikes[2]))
    assert abs(centre - 0.2) < 2e-2
