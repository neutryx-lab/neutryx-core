"""Extended Heston stochastic volatility model.

This module provides a complete implementation of the Heston (1993) model
including simulation, pricing via characteristic function, and calibration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional

import jax
import jax.numpy as jnp
import optax
from jax.scipy.stats import norm

from neutryx.core.infrastructure.tracking import (
    BaseTracker,
    TrackingConfig,
    calibration_metric_template,
    calibration_param_template,
    resolve_tracker,
)
from .heston_cf import characteristic_function

Array = jnp.ndarray


@dataclass
class HestonParams:
    """Heston model parameters.

    The Heston model:
        dS/S = (r - q) dt + sqrt(v) dW_S
        dv = kappa * (theta - v) dt + sigma * sqrt(v) dW_v
        corr(dW_S, dW_v) = rho

    Attributes
    ----------
    v0 : float
        Initial variance
    kappa : float
        Mean reversion speed
    theta : float
        Long-term mean variance
    sigma : float
        Volatility of variance (vol-of-vol)
    rho : float
        Correlation between asset and variance processes
    """

    v0: float
    kappa: float
    theta: float
    sigma: float
    rho: float

    def feller_condition(self) -> bool:
        """Check if Feller condition is satisfied (ensures v stays positive).

        Returns
        -------
        bool
            True if 2 * kappa * theta >= sigma^2
        """
        return 2.0 * self.kappa * self.theta >= self.sigma ** 2


def simulate_heston(
    key: jax.random.KeyArray,
    S0: float,
    params: HestonParams,
    r: float,
    q: float,
    T: float,
    n_steps: int,
    n_paths: int,
    scheme: Literal["euler", "milstein", "qe"] = "euler",
) -> tuple[Array, Array]:
    """Simulate Heston model paths.

    Parameters
    ----------
    key : jax.random.KeyArray
        PRNG key
    S0 : float
        Initial spot price
    params : HestonParams
        Heston parameters
    r : float
        Risk-free rate
    q : float
        Dividend yield
    T : float
        Time to maturity
    n_steps : int
        Number of time steps
    n_paths : int
        Number of paths
    scheme : str
        Discretization scheme: 'euler', 'milstein', or 'qe' (quadratic-exponential)

    Returns
    -------
    S_paths : Array
        Asset price paths, shape (n_paths, n_steps + 1)
    v_paths : Array
        Variance paths, shape (n_paths, n_steps + 1)
    """
    dt = T / n_steps
    sqrt_dt = jnp.sqrt(dt)

    # Initialize
    S_paths = jnp.zeros((n_paths, n_steps + 1))
    v_paths = jnp.zeros((n_paths, n_steps + 1))
    S_paths = S_paths.at[:, 0].set(S0)
    v_paths = v_paths.at[:, 0].set(params.v0)

    # Generate correlated Brownian motions
    key_s, key_v = jax.random.split(key)
    Z1 = jax.random.normal(key_s, (n_paths, n_steps))
    Z2_indep = jax.random.normal(key_v, (n_paths, n_steps))

    # Correlate Z2 with Z1
    Z2 = params.rho * Z1 + jnp.sqrt(1.0 - params.rho ** 2) * Z2_indep

    if scheme == "euler":
        # Simple Euler scheme with reflection
        for i in range(n_steps):
            v_curr = v_paths[:, i]
            S_curr = S_paths[:, i]

            # Ensure variance stays positive (reflection)
            v_curr_pos = jnp.maximum(v_curr, 0.0)

            # Variance update
            dv = (
                params.kappa * (params.theta - v_curr_pos) * dt
                + params.sigma * jnp.sqrt(v_curr_pos) * sqrt_dt * Z2[:, i]
            )
            v_next = jnp.maximum(v_curr + dv, 0.0)

            # Asset update
            dS = (
                (r - q) * S_curr * dt
                + S_curr * jnp.sqrt(v_curr_pos) * sqrt_dt * Z1[:, i]
            )
            S_next = S_curr + dS

            S_paths = S_paths.at[:, i + 1].set(S_next)
            v_paths = v_paths.at[:, i + 1].set(v_next)

    elif scheme == "qe":
        # Quadratic-Exponential (QE) scheme (Andersen, 2008)
        psi_c = 1.5  # Critical threshold

        for i in range(n_steps):
            v_curr = v_paths[:, i]
            S_curr = S_paths[:, i]

            # Variance update using QE scheme
            m = params.theta + (v_curr - params.theta) * jnp.exp(-params.kappa * dt)
            s2 = (
                v_curr * params.sigma ** 2 * jnp.exp(-params.kappa * dt) / params.kappa * (1.0 - jnp.exp(-params.kappa * dt))
                + params.theta * params.sigma ** 2 / (2.0 * params.kappa) * (1.0 - jnp.exp(-params.kappa * dt)) ** 2
            )

            psi = s2 / (m ** 2)

            # QE scheme branches
            def qe_large(Z):
                """QE scheme for psi <= psi_c."""
                b2 = 2.0 / psi - 1.0 + jnp.sqrt(2.0 / psi) * jnp.sqrt(2.0 / psi - 1.0)
                a = m / (1.0 + b2)
                return a * (jnp.sqrt(b2) + Z) ** 2

            def qe_small(Z):
                """QE scheme for psi > psi_c."""
                p = (psi - 1.0) / (psi + 1.0)
                beta = (1.0 - p) / m
                u = norm.cdf(Z)
                return jnp.where(u <= p, 0.0, jnp.log((1.0 - p) / (1.0 - u)) / beta)

            v_next = jnp.where(
                psi <= psi_c,
                qe_large(Z2[:, i]),
                qe_small(Z2[:, i]),
            )
            v_next = jnp.maximum(v_next, 0.0)

            # Asset update with integrated variance
            K0 = (r - q - params.rho / params.sigma * params.kappa * params.theta) * dt
            K1 = (
                params.rho * params.kappa / params.sigma - 0.5
            ) * dt - params.rho / params.sigma
            K2 = params.rho / params.sigma

            log_S_next = (
                jnp.log(S_curr)
                + K0
                + K1 * v_curr
                + K2 * v_next
                + jnp.sqrt((1.0 - params.rho ** 2) * v_curr) * sqrt_dt * Z1[:, i]
            )
            S_next = jnp.exp(log_S_next)

            S_paths = S_paths.at[:, i + 1].set(S_next)
            v_paths = v_paths.at[:, i + 1].set(v_next)

    else:
        raise ValueError(f"Unknown scheme: {scheme}")

    return S_paths, v_paths


def heston_call_price_cf(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    params: HestonParams,
    n_points: int = 1000,
) -> float:
    """Price European call using Heston characteristic function (FFT).

    Uses the Carr-Madan FFT approach.

    Parameters
    ----------
    S0 : float
        Spot price
    K : float
        Strike
    T : float
        Maturity
    r : float
        Risk-free rate
    q : float
        Dividend yield
    params : HestonParams
        Heston parameters
    n_points : int
        Number of FFT points

    Returns
    -------
    float
        Call option price
    """
    # Carr-Madan damping parameter
    alpha = 1.5

    # FFT parameters
    eta = 0.25
    lambda_val = 2.0 * jnp.pi / (n_points * eta)
    b = n_points * lambda_val / 2.0

    # Strike in log terms
    log_K = jnp.log(K)

    # Integration points
    v = jnp.arange(n_points) * eta

    # Modified characteristic function for call
    def psi(u):
        cf = characteristic_function(
            u - (alpha + 1) * 1j, T, params.v0, params.kappa, params.theta,
            params.sigma, params.rho, r, q
        )
        return jnp.exp(-r * T) * cf / (alpha ** 2 + alpha - u ** 2 + 1j * (2 * alpha + 1) * u)

    # FFT integration
    x = jnp.exp(1j * b * v) * psi(v) * eta * (3 + (-1) ** jnp.arange(n_points))
    x = x.at[0].set(x[0] / 2.0)

    # FFT
    fft_result = jnp.fft.fft(x)

    # Extract price
    k_values = -b + lambda_val * jnp.arange(n_points)
    call_values = jnp.exp(-alpha * k_values) / jnp.pi * jnp.real(fft_result)

    # Interpolate to strike
    call_price = jnp.interp(log_K, k_values, call_values)

    return float(call_price)


def heston_call_price_semi_analytical(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    params: HestonParams,
    n_points: int = 256,
) -> float:
    """Price European call using semi-analytical Heston formula.

    Uses the standard Heston (1993) formula with Fourier inversion.
    Implements the formulation from Albrecher et al. (2007).

    Parameters
    ----------
    S0 : float
        Current spot price
    K : float
        Strike price
    T : float
        Time to maturity
    r : float
        Risk-free rate
    q : float
        Dividend yield
    params : HestonParams
        Heston model parameters
    n_points : int
        Number of integration points (default: 256)

    Returns
    -------
    float
        Call option price

    Notes
    -----
    Formula: C = S*exp(-qT)*P1 - K*exp(-rT)*P2
    where P1, P2 are probabilities computed via characteristic function.
    """
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
    u_max = 100.0
    u_vals = jnp.linspace(0.0001, u_max, n_points)
    du = u_vals[1] - u_vals[0]

    # Compute P1 and P2 via numerical integration
    integrand1 = jax.vmap(lambda u: integrand(u, 1))(u_vals)
    integrand2 = jax.vmap(lambda u: integrand(u, 2))(u_vals)

    P1 = 0.5 + (1.0 / jnp.pi) * du * jnp.sum(integrand1)
    P2 = 0.5 + (1.0 / jnp.pi) * du * jnp.sum(integrand2)

    # Option price formula
    call_price = S0 * jnp.exp(-q * T) * P1 - K * jnp.exp(-r * T) * P2

    return jnp.maximum(call_price, 0.0)


def calibrate_heston(
    S0: float,
    strikes: Array,
    maturities: Array,
    market_prices: Array,
    r: float,
    q: float,
    initial_params: HestonParams | None = None,
    n_iterations: int = 300,
    lr: float = 1e-2,
    use_transform: bool = True,
    pricing_method: Literal["fft", "semi_analytical"] = "fft",
    tracker: Optional[BaseTracker] = None,
    tracking_config: Optional[TrackingConfig | Mapping[str, Any]] = None,
    log_every: Optional[int] = None,
) -> HestonParams:
    """Calibrate Heston parameters to market option prices.

    Parameters
    ----------
    S0 : float
        Spot price
    strikes : Array
        Strike prices
    maturities : Array
        Maturities
    market_prices : Array
        Market call prices
    r : float
        Risk-free rate
    q : float
        Dividend yield
    initial_params : HestonParams | None
        Initial parameter guess
    n_iterations : int
        Number of optimization iterations
    lr : float
        Learning rate
    use_transform : bool
        Use log/tanh transforms for unconstrained optimization (default: True)
    pricing_method : str
        Pricing method: 'fft' (Carr-Madan) or 'semi_analytical' (direct integration)
    tracker : BaseTracker | None
        Tracking backend for logging calibration progress
    tracking_config : TrackingConfig | Mapping | None
        Configuration for tracking
    log_every : int | None
        Log metrics every N iterations (overrides tracking_config.log_every)

    Returns
    -------
    HestonParams
        Calibrated parameters
    """
    if initial_params is None:
        initial_params = HestonParams(
            v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5
        )

    strikes = jnp.asarray(strikes)
    maturities = jnp.asarray(maturities)
    market_prices = jnp.asarray(market_prices)

    # Choose pricing function
    if pricing_method == "fft":
        price_fn = heston_call_price_cf
    elif pricing_method == "semi_analytical":
        price_fn = heston_call_price_semi_analytical
    else:
        raise ValueError(f"Unknown pricing_method: {pricing_method}")

    if use_transform:
        # Transform parameters for unconstrained optimization
        params = {
            "v0": jnp.log(jnp.array(initial_params.v0)),
            "kappa": jnp.log(jnp.array(initial_params.kappa)),
            "theta": jnp.log(jnp.array(initial_params.theta)),
            "sigma": jnp.log(jnp.array(initial_params.sigma)),
            "rho": jnp.arctanh(jnp.array(initial_params.rho * 0.99)),
        }

        def transform_params(p):
            """Transform to valid Heston parameters."""
            return HestonParams(
                v0=jnp.exp(p["v0"]),
                kappa=jnp.exp(p["kappa"]),
                theta=jnp.exp(p["theta"]),
                sigma=jnp.exp(p["sigma"]),
                rho=jnp.tanh(p["rho"]),
            )
    else:
        # Direct optimization with constraints applied post-update
        params = {
            "v0": jnp.array(initial_params.v0),
            "kappa": jnp.array(initial_params.kappa),
            "theta": jnp.array(initial_params.theta),
            "sigma": jnp.array(initial_params.sigma),
            "rho": jnp.array(initial_params.rho),
        }

        def transform_params(p):
            """Apply constraints to ensure valid parameters."""
            return HestonParams(
                v0=jnp.maximum(p["v0"], 1e-6),
                kappa=jnp.maximum(p["kappa"], 0.01),
                theta=jnp.maximum(p["theta"], 1e-6),
                sigma=jnp.maximum(p["sigma"], 0.01),
                rho=jnp.clip(p["rho"], -0.99, 0.99),
            )

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    def loss_fn(p):
        """MSE loss."""
        hp = transform_params(p)

        def price_single(K, T):
            return price_fn(S0, K, T, r, q, hp)

        model_prices = jax.vmap(price_single)(strikes, maturities)
        return jnp.mean((model_prices - market_prices) ** 2)

    # Setup tracking
    tracking_cfg: Optional[TrackingConfig]
    if isinstance(tracking_config, TrackingConfig):
        tracking_cfg = tracking_config
    elif tracking_config is not None:
        tracking_cfg = TrackingConfig(**dict(tracking_config))
    else:
        tracking_cfg = None

    log_cadence = log_every or (tracking_cfg.log_every if tracking_cfg else 25)

    # Optimization loop with tracking
    with resolve_tracker(tracker, tracking_cfg) as active_tracker:
        # Log initial parameters
        initial_dict = {k: float(v) for k, v in params.items()}
        active_tracker.log_params(
            calibration_param_template(initial_dict, prefix="heston")
        )

        for i in range(n_iterations):
            loss_val, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

            # Apply constraints for non-transformed case
            if not use_transform:
                params['v0'] = jnp.maximum(params['v0'], 1e-6)
                params['theta'] = jnp.maximum(params['theta'], 1e-6)
                params['kappa'] = jnp.maximum(params['kappa'], 0.01)
                params['sigma'] = jnp.maximum(params['sigma'], 0.01)
                params['rho'] = jnp.clip(params['rho'], -0.99, 0.99)

            # Log metrics
            if i % log_cadence == 0 or i == n_iterations - 1:
                param_dict = {k: float(v) for k, v in params.items()}
                active_tracker.log_metrics(
                    calibration_metric_template(float(loss_val), param_dict, prefix="heston"),
                    step=i,
                )

    return transform_params(params)


# Compatibility alias for legacy code
heston_call_price = heston_call_price_semi_analytical

__all__ = [
    "HestonParams",
    "simulate_heston",
    "heston_call_price_cf",
    "heston_call_price_semi_analytical",
    "heston_call_price",  # compatibility alias
    "calibrate_heston",
]
