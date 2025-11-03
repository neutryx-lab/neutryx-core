"""Merton jump-diffusion dynamics and analytic pricing routines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln
from jax.scipy.stats import norm

from .sde import SDE

Array = jnp.ndarray


@dataclass
class MertonParams:
    """Model parameters for the Merton (lognormal) jump-diffusion."""

    sigma: float
    lam: float
    mu_jump: float
    sigma_jump: float

    def kappa(self) -> float:
        """Mean jump size minus one."""
        return float(jnp.exp(self.mu_jump + 0.5 * self.sigma_jump ** 2) - 1.0)


@dataclass
class MertonJumpDiffusion(SDE):
    """Jump-diffusion SDE following Merton (1976)."""

    mu: float
    params: MertonParams

    def drift(self, _t: float, x: float) -> float:
        """Instantaneous drift incorporating compensator for jumps."""
        kappa = self.params.kappa()
        return (self.mu - self.params.lam * kappa) * x

    def diffusion(self, _t: float, x: float) -> float:
        """Diffusion coefficient (diffusive volatility part)."""
        return self.params.sigma * x


def _poisson_weights(lt: Array, n_terms: int, dtype: jnp.dtype) -> Array:
    n = jnp.arange(n_terms, dtype=dtype)
    lt = jnp.asarray(lt, dtype=dtype)
    safe_lt = jnp.where(lt > 0, lt, 1.0)
    log_w = -lt + n * jnp.log(safe_lt) - gammaln(n + 1.0)
    weights = jnp.exp(log_w)
    return jnp.where(lt > 0, weights, jnp.where(n == 0.0, jnp.ones_like(weights), jnp.zeros_like(weights)))


def _lognormal_call_terms(
    log_s0: Array,
    log_strike: Array,
    r: float,
    T: float,
    mu: float,
    sigma: float,
    lam: float,
    mu_jump: float,
    sigma_jump: float,
    n_terms: int,
) -> Tuple[Array, Array, Array]:
    dtype = jnp.result_type(log_s0, log_strike, float(r))
    n = jnp.arange(n_terms, dtype=dtype)
    kappa = jnp.exp(mu_jump + 0.5 * sigma_jump ** 2) - 1.0
    drift = (mu - lam * kappa - 0.5 * sigma ** 2) * T
    log_mean = log_s0 + drift + n * mu_jump
    var = sigma ** 2 * T + n * (sigma_jump ** 2)
    sqrt_var = jnp.sqrt(jnp.maximum(var, 1e-16))
    d1 = (log_mean - log_strike + var) / sqrt_var
    d2 = d1 - sqrt_var
    exp_term = jnp.exp(log_mean + 0.5 * var)
    phi_d1 = norm.cdf(d1)
    phi_d2 = norm.cdf(d2)
    payoff = exp_term * phi_d1 - jnp.exp(log_strike) * phi_d2
    weights = _poisson_weights(lam * T, n_terms, dtype)
    discounted = jnp.exp(-r * T) * weights * payoff
    return discounted, phi_d1, phi_d2


def merton_jump_call(
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
    n_terms: int = 64,
) -> Array:
    """Closed-form European call price under Merton jump diffusion."""
    if T <= 0:
        return jnp.maximum(S0 - K, 0.0)
    mu = r - q
    log_s0 = jnp.log(jnp.asarray(S0))
    log_k = jnp.log(jnp.asarray(K))
    discounted, _, _ = _lognormal_call_terms(
        log_s0, log_k, r, T, mu, sigma, lam, mu_jump, sigma_jump, n_terms
    )
    return discounted.sum()


def merton_jump_price(
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
    kind: str = "call",
    n_terms: int = 64,
) -> Array:
    call_price = merton_jump_call(S0, K, T, r, q, sigma, lam, mu_jump, sigma_jump, n_terms=n_terms)
    if kind == "call":
        return call_price
    parity = S0 * jnp.exp(-q * T) - K * jnp.exp(-r * T)
    return call_price - parity


def characteristic_exponent(u: Array, params: MertonParams) -> Array:
    """Characteristic exponent of log-returns for Merton jump diffusion."""
    u = jnp.asarray(u)
    return -0.5 * params.sigma ** 2 * u ** 2 + params.lam * (
        jnp.exp(1j * u * params.mu_jump - 0.5 * params.sigma_jump ** 2 * u ** 2) - 1.0
    )


def calibrate_merton(
    S0: float,
    strikes: Array,
    maturities: Array,
    market_prices: Array,
    *,
    r: float,
    q: float,
    initial: MertonParams | None = None,
    n_terms: int = 64,
    lr: float = 5e-2,
    n_iterations: int = 200,
) -> MertonParams:
    """Least squares calibration of Merton jump parameters to call prices."""

    import optax

    if initial is None:
        initial = MertonParams(sigma=0.2, lam=0.1, mu_jump=-0.05, sigma_jump=0.2)

    strikes = jnp.asarray(strikes)
    maturities = jnp.asarray(maturities)
    targets = jnp.asarray(market_prices)

    params_dict = {
        "sigma": jnp.asarray(initial.sigma),
        "lam": jnp.asarray(initial.lam),
        "mu_jump": jnp.asarray(initial.mu_jump),
        "sigma_jump": jnp.asarray(initial.sigma_jump),
    }

    opt = optax.adam(lr)
    opt_state = opt.init(params_dict)

    def loss(pdict):
        price_fn = jax.vmap(
            lambda K, T: merton_jump_price(
                S0,
                K,
                T,
                r,
                q,
                pdict["sigma"],
                pdict["lam"],
                pdict["mu_jump"],
                pdict["sigma_jump"],
                n_terms=n_terms,
            )
        )
        model_prices = price_fn(strikes, maturities)
        return jnp.mean((model_prices - targets) ** 2)

    for _ in range(n_iterations):
        _, grads = jax.value_and_grad(loss)(params_dict)
        updates, opt_state_new = opt.update(grads, opt_state, params_dict)
        params_dict = optax.apply_updates(params_dict, updates)
        opt_state = opt_state_new

    return MertonParams(
        sigma=float(params_dict["sigma"]),
        lam=float(params_dict["lam"]),
        mu_jump=float(params_dict["mu_jump"]),
        sigma_jump=float(params_dict["sigma_jump"]),
    )


__all__ = [
    "MertonJumpDiffusion",
    "MertonParams",
    "calibrate_merton",
    "characteristic_exponent",
    "merton_jump_call",
    "merton_jump_price",
]

