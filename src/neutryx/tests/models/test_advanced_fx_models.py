"""Advanced FX model tests for jump-diffusion pricing."""

import jax
import jax.numpy as jnp

from neutryx.models.fx_models import FXBatesModel, FXHestonModel
from neutryx.models.jump_diffusion import merton_jump_price


def test_bates_reduces_to_heston_without_jumps():
    """When jump intensity is zero, Bates price should match Heston price."""
    heston = FXHestonModel(
        v0=0.04,
        kappa=2.0,
        theta=0.04,
        sigma=0.3,
        rho=-0.6,
        r_domestic=0.03,
        r_foreign=0.01,
    )

    bates = FXBatesModel(
        v0=heston.v0,
        kappa=heston.kappa,
        theta=heston.theta,
        sigma=heston.sigma,
        rho=heston.rho,
        r_domestic=heston.r_domestic,
        r_foreign=heston.r_foreign,
        lambda_jump=0.0,
        mu_jump=0.0,
        sigma_jump=0.0,
    )

    S = 1.12
    K = 1.05
    T = 1.5

    price_heston = float(heston.price(S, K, T, is_call=True, N=4096))
    price_bates = float(bates.price(S, K, T, is_call=True, N=4096))

    assert jnp.isfinite(price_bates)
    assert jnp.isfinite(price_heston)
    assert jnp.abs(price_bates - price_heston) < 1e-3


def test_bates_matches_merton_limit_and_mc():
    """Bates pricing converges to Merton analytic and Monte Carlo benchmarks."""
    # Parameters chosen to mimic constant volatility (≈ Merton jump diffusion)
    v0 = 0.0225  # 15% vol squared
    sigma_vol = 1e-6  # near-constant variance

    model = FXBatesModel(
        v0=v0,
        kappa=50.0,
        theta=v0,
        sigma=sigma_vol,
        rho=-0.2,
        r_domestic=0.02,
        r_foreign=0.01,
        lambda_jump=0.4,
        mu_jump=-0.05,
        sigma_jump=0.10,
    )

    S = 1.25
    K = 1.20
    T = 1.0

    price_bates = float(model.price(S, K, T, is_call=True, N=4096))

    # Analytic benchmark from Merton model
    sigma_bs = jnp.sqrt(v0)
    analytic_price = float(
        merton_jump_price(
            S0=S,
            K=K,
            T=T,
            r=model.r_domestic,
            q=model.r_foreign,
            sigma=sigma_bs,
            lam=model.lambda_jump,
            mu_jump=model.mu_jump,
            sigma_jump=model.sigma_jump,
            kind="call",
        )
    )

    # Monte Carlo benchmark using exact jump summation for constant variance case
    key = jax.random.PRNGKey(0)
    n_paths = 40_000
    vol = sigma_bs
    m = jnp.exp(model.mu_jump + 0.5 * model.sigma_jump ** 2) - 1.0

    key_z, key_n, key_jump = jax.random.split(key, 3)
    normals = jax.random.normal(key_z, (n_paths,))
    jump_counts = jax.random.poisson(key_n, model.lambda_jump * T, (n_paths,))

    # Sum of log jumps ~ Normal(n*mu, n*sigma^2)
    jump_means = jump_counts * model.mu_jump
    jump_stds = jnp.sqrt(jump_counts * (model.sigma_jump ** 2))
    jump_normals = jax.random.normal(key_jump, (n_paths,))
    jump_terms = jump_means + jump_stds * jump_normals

    log_ST = (
        jnp.log(S)
        + (model.r_domestic - model.r_foreign - model.lambda_jump * m - 0.5 * vol ** 2) * T
        + vol * jnp.sqrt(T) * normals
        + jump_terms
    )
    ST = jnp.exp(log_ST)
    payoffs = jnp.maximum(ST - K, 0.0)
    discounted_payoffs = jnp.exp(-model.r_domestic * T) * payoffs
    mc_price = float(jnp.mean(discounted_payoffs))
    mc_std = float(jnp.std(discounted_payoffs) / jnp.sqrt(n_paths))

    assert jnp.abs(price_bates - analytic_price) < 5e-3
    assert jnp.abs(price_bates - mc_price) < 4 * mc_std + 5e-3
