# Heston model calibration using characteristic function and Optax optimizer.
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import jax
import jax.numpy as jnp
import optax

from ..core.infrastructure.tracking import (
    BaseTracker,
    TrackingConfig,
    calibration_metric_template,
    calibration_param_template,
    resolve_tracker,
)
from ..models.heston_cf import characteristic_function


@dataclass
class HestonParams:
    """Heston model parameters.

    Attributes:
        v0: Initial variance
        kappa: Mean reversion speed
        theta: Long-term variance
        sigma: Volatility of variance (vol-of-vol)
        rho: Correlation between asset and variance
    """
    v0: float
    kappa: float
    theta: float
    sigma: float
    rho: float


def heston_call_price(S0, K, T, r, q, params: HestonParams):
    """Calculate European call option price using Heston characteristic function.

    Uses the standard Heston (1993) semi-analytical formula with Fourier inversion.
    Implements the formulation from Albrecher et al. (2007).

    Args:
        S0: Current spot price
        K: Strike price
        T: Time to maturity
        r: Risk-free rate
        q: Dividend yield
        params: Heston model parameters

    Returns:
        Call option price
    """
    # Standard Heston formula: C = S*exp(-qT)*P1 - K*exp(-rT)*P2
    # where P1, P2 are probabilities computed via characteristic function

    def integrand(u, j):
        """Integrand for computing probabilities P1 (j=1) and P2 (j=2)."""
        # Compute characteristic function with appropriate modification
        if j == 1:
            # For P1, use modified cf with shift
            cf_arg = u - 1j
        else:
            # For P2, use standard cf
            cf_arg = u

        cf = characteristic_function(
            cf_arg, T, params.v0, params.kappa, params.theta,
            params.sigma, params.rho, r, q
        )

        # Forward price
        F = S0 * jnp.exp((r - q) * T)

        # Compute integrand Re[(exp(-iu*log(K/F)) * cf / cf(0)) / (iu)]
        if j == 1:
            # P1 integrand
            cf0 = characteristic_function(
                -1j, T, params.v0, params.kappa, params.theta,
                params.sigma, params.rho, r, q
            )
            numerator = jnp.exp(-1j * u * jnp.log(K / F)) * cf / cf0
        else:
            # P2 integrand
            numerator = jnp.exp(-1j * u * jnp.log(K / F)) * cf

        denominator = 1j * u

        return jnp.real(numerator / denominator)

    # Numerical integration parameters
    N = 256
    u_max = 100.0
    u_vals = jnp.linspace(0.0001, u_max, N)
    du = u_vals[1] - u_vals[0]

    # Compute P1 and P2 via numerical integration
    integrand1 = jax.vmap(lambda u: integrand(u, 1))(u_vals)
    integrand2 = jax.vmap(lambda u: integrand(u, 2))(u_vals)

    P1 = 0.5 + (1.0 / jnp.pi) * du * jnp.sum(integrand1)
    P2 = 0.5 + (1.0 / jnp.pi) * du * jnp.sum(integrand2)

    # Option price formula
    call_price = S0 * jnp.exp(-q * T) * P1 - K * jnp.exp(-r * T) * P2

    return jnp.maximum(call_price, 0.0)


def calibrate(
    S0,
    strikes,
    maturities,
    target_prices,
    r: float = 0.0,
    q: float = 0.0,
    learning_rate: float = 1e-3,
    n_iterations: int = 300,
    *,
    tracker: Optional[BaseTracker] = None,
    tracking_config: Optional[TrackingConfig | Mapping[str, Any]] = None,
    log_every: Optional[int] = None,
):
    """Calibrate Heston model parameters to market option prices.

    Uses Adam optimizer from Optax to minimize MSE between model and market prices.

    Args:
        S0: Current spot price
        strikes: Array of strike prices
        maturities: Array of times to maturity
        target_prices: Array of market option prices
        r: Risk-free rate (default: 0.0)
        q: Dividend yield (default: 0.0)
        learning_rate: Learning rate for Adam optimizer (default: 1e-3)
        n_iterations: Number of optimization iterations (default: 300)

    Returns:
        Calibrated HestonParams object
    """
    # Initial parameter guess (reasonable starting point)
    params = HestonParams(
        v0=0.04,      # Initial variance ~20% vol
        kappa=2.0,    # Mean reversion speed
        theta=0.04,   # Long-term variance ~20% vol
        sigma=0.3,    # Vol-of-vol
        rho=-0.5      # Negative correlation (leverage effect)
    )

    # Setup optimizer
    opt = optax.adam(learning_rate)
    opt_state = opt.init(vars(params))

    def loss(pdict):
        """MSE loss between model and market prices."""
        p = HestonParams(**pdict)

        # Compute model prices for all strikes/maturities
        pred_prices = jax.vmap(
            lambda K, T, price: heston_call_price(S0, K, T, r, q, p)
        )(strikes, maturities, target_prices)

        # Mean squared error
        return jnp.mean((pred_prices - target_prices)**2)

    def scaled_loss(pdict, loss_scale):
        return apply_loss_scaling(loss(pdict), loss_scale=loss_scale)

    # Optimization loop
    pdict = {k: getattr(params, k) for k in vars(params)}

    tracking_cfg: Optional[TrackingConfig]
    if isinstance(tracking_config, TrackingConfig):
        tracking_cfg = tracking_config
    elif tracking_config is not None:
        tracking_cfg = TrackingConfig(**dict(tracking_config))
    else:
        tracking_cfg = None

    log_cadence = log_every or (tracking_cfg.log_every if tracking_cfg else 25)

    with resolve_tracker(tracker, tracking_cfg) as active_tracker:
        active_tracker.log_params(
            calibration_param_template(pdict, prefix="heston")
        )

        for i in range(n_iterations):
            l, g = jax.value_and_grad(loss)(pdict)
            updates, opt_state_new = opt.update(g, opt_state, pdict)
            pdict = optax.apply_updates(pdict, updates)
            opt_state = opt_state_new

            # Apply constraints to ensure valid parameters
            # v0, theta > 0 (positive variance)
            pdict['v0'] = jnp.maximum(pdict['v0'], 1e-6)
            pdict['theta'] = jnp.maximum(pdict['theta'], 1e-6)
            # kappa > 0 (positive mean reversion)
            pdict['kappa'] = jnp.maximum(pdict['kappa'], 0.01)
            # sigma > 0 (positive vol-of-vol)
            pdict['sigma'] = jnp.maximum(pdict['sigma'], 0.01)
            # -1 < rho < 1 (correlation bounds)
            pdict['rho'] = jnp.clip(pdict['rho'], -0.99, 0.99)

            if i % log_cadence == 0 or i == n_iterations - 1:
                active_tracker.log_metrics(
                    calibration_metric_template(l, pdict, prefix="heston"),
                    step=i,
                )

    return HestonParams(**pdict)
