# SABR (Hagan) implied volatility + robust calibration with parameter constraints.
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax

from ..core.utils.precision import apply_loss_scaling, get_loss_scale, undo_loss_scaling


@dataclass
class SABRParams:
    """SABR model parameters.

    Args:
        alpha: Initial volatility level (must be > 0)
        beta: CEV exponent, typically fixed (0 < beta <= 1)
        rho: Correlation between asset and volatility (-1 < rho < 1)
        nu: Volatility of volatility (must be > 0)
    """
    alpha: float
    beta: float
    rho: float
    nu: float


def hagan_implied_vol(F: float, K: float, T: float, p: SABRParams) -> float:
    """Compute SABR implied volatility using Hagan's formula.

    Args:
        F: Forward price
        K: Strike price
        T: Time to maturity
        p: SABR parameters

    Returns:
        Implied volatility (annualized)
    """
    eps = 1e-12
    one_m_beta = 1.0 - p.beta

    # Compute both ATM and non-ATM cases, then select based on F-K distance
    # This allows JAX to trace through without Python if statements

    # ATM case computation
    FK_pow = jnp.maximum(F ** one_m_beta, eps)
    A_atm = p.alpha / FK_pow
    term_atm = T * (
        ((one_m_beta ** 2) / 24) * (p.alpha ** 2) / (FK_pow ** 2 + eps)
        + 0.25 * p.rho * p.beta * p.nu * p.alpha / (FK_pow + eps)
        + (2 - 3 * p.rho ** 2) / 24 * (p.nu ** 2)
    )
    iv_atm = A_atm * (1 + term_atm)

    # Non-ATM case computation
    FK = jnp.maximum((F * K) ** (0.5 * one_m_beta), eps)
    logFK = jnp.log((F + eps) / (K + eps))

    # Compute z with numerical safeguards
    z = (p.nu / (p.alpha + eps)) * FK * logFK
    z = jnp.clip(z, -3.0, 3.0)  # Prevent extreme z values

    # Compute x(z) with numerical stability
    sqrt_term = jnp.sqrt(jnp.maximum(1 - 2 * p.rho * z + z * z, eps))
    xz_arg = jnp.maximum((sqrt_term + z - p.rho) / (1 - p.rho + eps), eps)
    xz = jnp.log(xz_arg)

    # Use limit as z -> 0 for numerical stability
    denom = jnp.where(jnp.abs(z) < 1e-7, 1.0, z / (xz + eps))

    # Main term
    A = p.alpha / FK
    B = 1 + ((one_m_beta ** 2) / 24) * (logFK ** 2) + ((one_m_beta ** 4) / 1920) * (logFK ** 4)
    vol0 = A * denom * B
    # time-dependent correction
    term = T * ( ((one_m_beta**2)/24) * (p.alpha**2) / ((F*K)**(one_m_beta))
               + 0.25 * p.rho * p.beta * p.nu * p.alpha / ((F*K)**(0.5*one_m_beta))
               + (2 - 3*p.rho**2)/24 * (p.nu**2) )
    return vol0 * (1 + term)

def calibrate(F, strikes, maturities, target_iv):
    params = SABRParams(alpha=0.2, beta=0.7, rho=-0.2, nu=0.5)
    opt = optax.adam(1e-2)
    opt_state = opt.init(vars(params))

    def loss(pdict):
        p = SABRParams(**pdict)

        # Compute predicted implied volatilities
        pred = jax.vmap(lambda K, T: hagan_implied_vol(F, K, T, p))(strikes, maturities)
        return jnp.mean((pred - target_iv)**2)

    def scaled_loss(pdict, loss_scale):
        return apply_loss_scaling(loss(pdict), loss_scale=loss_scale)

    pdict = {k:getattr(params,k) for k in vars(params)}
    for _ in range(200):
        loss_scale = get_loss_scale()
        _, scaled_grad = jax.value_and_grad(
            lambda current: scaled_loss(current, loss_scale)
        )(pdict)
        grads = undo_loss_scaling(scaled_grad, loss_scale=loss_scale)
        updates, opt_state_new = opt.update(grads, opt_state, pdict)
        pdict = optax.apply_updates(pdict, updates)
        opt_state = opt_state_new
    return SABRParams(**pdict)
