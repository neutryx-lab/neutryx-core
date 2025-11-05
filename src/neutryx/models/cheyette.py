"""Cheyette model implementation with JAX.

The Cheyette model is a powerful extension of the Heath-Jarrow-Morton (HJM)
framework that provides a tractable finite-dimensional representation while
maintaining rich dynamics.

The model describes the evolution of:
1. The instantaneous forward rate f(t, T)
2. A state variable x(t) that captures the stochastic part
3. An auxiliary variance process y(t)

The system of SDEs is:

    dx(t) = -κ * x(t) dt + σ(t) dW₁(t)
    dy(t) = σ²(t) dt
    r(t) = f(0, t) + x(t)

where:
    - x(t): stochastic state variable (mean-reverting)
    - y(t): integrated variance (auxiliary process)
    - κ: mean reversion speed
    - σ(t): time-dependent volatility function
    - f(0, t): initial forward curve
    - W₁(t): standard Brownian motion

For the multi-factor version:
    dx_i(t) = -κᵢ * x_i(t) dt + σᵢ(t) dWᵢ(t)
    r(t) = f(0, t) + Σᵢ xᵢ(t)

Key advantages:
1. Exact fit to initial term structure
2. Analytical bond option formulas
3. Flexible volatility term structure
4. Markovian representation (finite state)
5. Efficient simulation and calibration

The Cheyette model is widely used in practice for pricing and risk management
of interest rate derivatives, especially swaptions and exotic structures.

References
----------
Cheyette, O. (1992). "Markov representation of the Heath-Jarrow-Morton model."
Working paper, BARRA.

Piterbarg, V. (2005). "A multi-currency model with FX volatility skew."
Working paper, Barclays Capital.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax import lax


@dataclass
class CheyetteParams:
    """Parameters for the Cheyette model.

    Attributes
    ----------
    kappa : float or Array
        Mean reversion speed(s). Scalar for single-factor, array for multi-factor.
    sigma_fn : Callable
        Time-dependent volatility function σ(t). Returns scalar or array.
    forward_curve_fn : Callable
        Initial forward curve f(0, t)
    r0 : float
        Initial short rate
    n_factors : int, optional
        Number of factors (default: 1)
    rho : Optional[Array]
        Correlation matrix between factors (n_factors × n_factors).
        If None, assumes independent factors.
    """
    kappa: float | jnp.ndarray
    sigma_fn: Callable[[float], float | jnp.ndarray]
    forward_curve_fn: Callable[[float], float]
    r0: float
    n_factors: int = 1
    rho: Optional[jnp.ndarray] = None

    def __post_init__(self):
        """Validate parameters."""
        # Convert kappa to array if multi-factor
        if self.n_factors > 1:
            if isinstance(self.kappa, (int, float)):
                # Broadcast scalar to all factors
                self.kappa = jnp.ones(self.n_factors) * self.kappa
            else:
                self.kappa = jnp.array(self.kappa)
                if self.kappa.shape[0] != self.n_factors:
                    raise ValueError(
                        f"kappa length {self.kappa.shape[0]} != n_factors {self.n_factors}"
                    )

            # Validate kappa values
            if jnp.any(self.kappa <= 0):
                raise ValueError("All mean reversion speeds must be positive")

            # Set up correlation matrix
            if self.rho is None:
                # Independent factors
                self.rho = jnp.eye(self.n_factors)
            else:
                self.rho = jnp.array(self.rho)
                if self.rho.shape != (self.n_factors, self.n_factors):
                    raise ValueError(
                        f"Correlation matrix shape {self.rho.shape} must be "
                        f"({self.n_factors}, {self.n_factors})"
                    )
        else:
            # Single factor
            if isinstance(self.kappa, jnp.ndarray):
                self.kappa = float(self.kappa[0])
            if self.kappa <= 0:
                raise ValueError(f"Mean reversion speed must be positive, got {self.kappa}")

            self.rho = jnp.array([[1.0]])


def zero_coupon_bond_price(
    params: CheyetteParams,
    T: float,
    x_t: float | jnp.ndarray,
    y_t: float | jnp.ndarray,
    t: float = 0.0,
) -> float:
    """Calculate zero-coupon bond price under Cheyette model.

    The bond price has the affine form:
        P(t, T) = P_market(t, T) * exp(-B(t, T) · x(t) - 0.5 * C(t, T) · y(t))

    where:
        - B(t, T) = (1 - exp(-κ(T-t))) / κ
        - C(t, T) = B(t, T)²
        - x(t), y(t) are the state variables

    Parameters
    ----------
    params : CheyetteParams
        Cheyette model parameters
    T : float
        Bond maturity time
    x_t : float or Array
        Current state variable(s). Scalar for single-factor, array for multi-factor.
    y_t : float or Array
        Current variance process value(s)
    t : float, optional
        Current time (default: 0)

    Returns
    -------
    float
        Zero-coupon bond price
    """
    tau = T - t

    if params.n_factors == 1:
        # Single factor case
        kappa = params.kappa
        B_tau = (1.0 - jnp.exp(-kappa * tau)) / kappa
        C_tau = B_tau * B_tau

        # Market forward bond price (from forward curve)
        # P_market(t, T) = exp(-∫ₜᵀ f(0, s) ds)
        # Simplified: use average forward rate
        avg_fwd = params.forward_curve_fn((t + T) / 2.0)
        P_market = jnp.exp(-avg_fwd * tau)

        # Affine adjustment
        adjustment = jnp.exp(-B_tau * x_t - 0.5 * C_tau * y_t)

        bond_price = P_market * adjustment

    else:
        # Multi-factor case
        kappas = params.kappa
        x_t = jnp.atleast_1d(x_t)
        y_t = jnp.atleast_1d(y_t)

        # B_i(τ) for each factor
        B_tau = (1.0 - jnp.exp(-kappas * tau)) / kappas

        # Market bond price
        avg_fwd = params.forward_curve_fn((t + T) / 2.0)
        P_market = jnp.exp(-avg_fwd * tau)

        # Affine adjustment: sum over factors
        x_adjustment = jnp.dot(B_tau, x_t)
        y_adjustment = 0.5 * jnp.dot(B_tau * B_tau, y_t)

        adjustment = jnp.exp(-x_adjustment - y_adjustment)

        bond_price = P_market * adjustment

    return float(bond_price)


def simulate_path(
    params: CheyetteParams,
    T: float,
    n_steps: int,
    key: jax.random.KeyArray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Simulate a single path of the Cheyette process.

    Parameters
    ----------
    params : CheyetteParams
        Cheyette model parameters
    T : float
        Total simulation time
    n_steps : int
        Number of time steps
    key : jax.random.KeyArray
        JAX random key

    Returns
    -------
    tuple[Array, Array, Array]
        Tuple of (r_path, x_path, y_path) where:
        - r_path: short rate path of shape [n_steps + 1]
        - x_path: state variable path(s) of shape [n_steps + 1] or [n_steps + 1, n_factors]
        - y_path: variance process path(s) of shape [n_steps + 1] or [n_steps + 1, n_factors]

    Notes
    -----
    The discretization uses:
        x(t+dt) = x(t) * exp(-κ dt) + σ(t) * sqrt((1 - exp(-2κ dt)) / (2κ)) * Z
        y(t+dt) = y(t) + σ²(t) * dt
        r(t) = f(0, t) + x(t)
    """
    dt = T / n_steps

    if params.n_factors == 1:
        # Single factor
        kappa = params.kappa
        x0 = 0.0
        y0 = 0.0

        # Generate random normals
        Z = jax.random.normal(key, shape=(n_steps,))

        exp_neg_kappa_dt = jnp.exp(-kappa * dt)
        vol_scaling = jnp.sqrt((1.0 - jnp.exp(-2.0 * kappa * dt)) / (2.0 * kappa))

        def step_fn(carry, inputs):
            x_t, y_t, t = carry
            z = inputs

            # Get volatility at time t
            sigma_t = params.sigma_fn(t)

            # Update x (mean-reverting OU process)
            x_next = x_t * exp_neg_kappa_dt + sigma_t * vol_scaling * z

            # Update y (integrated variance)
            y_next = y_t + sigma_t * sigma_t * dt

            # Short rate
            r_t = params.forward_curve_fn(t) + x_t

            t_next = t + dt

            return (x_next, y_next, t_next), (r_t, x_next, y_next)

        _, (r_path, x_path, y_path) = lax.scan(step_fn, (x0, y0, 0.0), Z)

        # Prepend initial values
        r0 = params.r0
        r_path_full = jnp.concatenate([jnp.array([r0]), r_path])
        x_path_full = jnp.concatenate([jnp.array([x0]), x_path])
        y_path_full = jnp.concatenate([jnp.array([y0]), y_path])

    else:
        # Multi-factor case
        kappas = params.kappa
        x0 = jnp.zeros(params.n_factors)
        y0 = jnp.zeros(params.n_factors)

        # Generate correlated random normals
        # First generate independent normals
        Z_indep = jax.random.normal(key, shape=(n_steps, params.n_factors))

        # Apply Cholesky decomposition of correlation matrix
        L = jnp.linalg.cholesky(params.rho)
        Z_corr = Z_indep @ L.T

        exp_neg_kappa_dt = jnp.exp(-kappas * dt)
        vol_scaling = jnp.sqrt((1.0 - jnp.exp(-2.0 * kappas * dt)) / (2.0 * kappas))

        def step_fn(carry, inputs):
            x_t, y_t, t = carry
            z = inputs  # Shape: [n_factors]

            # Get volatilities at time t (can be vector)
            sigma_t = params.sigma_fn(t)
            if jnp.isscalar(sigma_t):
                sigma_t = jnp.ones(params.n_factors) * sigma_t
            else:
                sigma_t = jnp.atleast_1d(sigma_t)

            # Update each factor
            x_next = x_t * exp_neg_kappa_dt + sigma_t * vol_scaling * z

            # Update integrated variance
            y_next = y_t + sigma_t * sigma_t * dt

            # Short rate is sum of forward rate and all factors
            r_t = params.forward_curve_fn(t) + jnp.sum(x_t)

            t_next = t + dt

            return (x_next, y_next, t_next), (r_t, x_next, y_next)

        _, (r_path, x_path, y_path) = lax.scan(step_fn, (x0, y0, 0.0), Z_corr)

        # Prepend initial values
        r0 = params.r0
        r_path_full = jnp.concatenate([jnp.array([r0]), r_path])
        x_path_full = jnp.concatenate([x0[None, :], x_path], axis=0)
        y_path_full = jnp.concatenate([y0[None, :], y_path], axis=0)

    return r_path_full, x_path_full, y_path_full


def simulate_paths(
    params: CheyetteParams,
    T: float,
    n_steps: int,
    n_paths: int,
    key: jax.random.KeyArray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Simulate multiple paths of the Cheyette process.

    Parameters
    ----------
    params : CheyetteParams
        Cheyette model parameters
    T : float
        Total simulation time
    n_steps : int
        Number of time steps
    n_paths : int
        Number of paths to simulate
    key : jax.random.KeyArray
        JAX random key

    Returns
    -------
    tuple[Array, Array, Array]
        Tuple of (r_paths, x_paths, y_paths) with appropriate shapes
    """
    keys = jax.random.split(key, n_paths)

    def sim_single_path(k):
        return simulate_path(params, T, n_steps, k)

    r_paths, x_paths, y_paths = jax.vmap(sim_single_path)(keys)
    return r_paths, x_paths, y_paths


def swaption_price_analytical(
    params: CheyetteParams,
    strike: float,
    option_expiry: float,
    swap_tenor: float,
    payment_frequency: float = 0.5,
    is_payer: bool = True,
) -> float:
    """Price a European swaption using semi-analytical formula.

    The Cheyette model admits semi-analytical formulas for swaptions
    similar to the Hull-White model, using Jamshidian's decomposition.

    Parameters
    ----------
    params : CheyetteParams
        Cheyette model parameters
    strike : float
        Swap fixed rate (strike)
    option_expiry : float
        Time to swaption expiration
    swap_tenor : float
        Length of underlying swap
    payment_frequency : float, optional
        Payment frequency in years (default: 0.5 for semi-annual)
    is_payer : bool, optional
        If True, price payer swaption. If False, price receiver swaption.

    Returns
    -------
    float
        Swaption price

    Notes
    -----
    This is a simplified implementation. Full production implementation
    would use Jamshidian's decomposition with numerical root finding.
    """
    from scipy.stats import norm

    T_opt = option_expiry
    T_swap = swap_tenor

    # Payment dates
    n_payments = int(T_swap / payment_frequency)
    payment_dates = jnp.array(
        [T_opt + (i + 1) * payment_frequency for i in range(n_payments)]
    )

    # For single-factor, we can use analytical approach
    if params.n_factors == 1:
        kappa = params.kappa

        # Bond prices at option expiry
        x_0 = 0.0  # At t=0, x=0
        y_0 = 0.0

        # Compute variance of x(T_opt)
        # Var[x(T)] = ∫₀ᵀ σ²(s) * exp(-2κ(T-s)) ds
        # Simplified: assume constant σ
        sigma_avg = params.sigma_fn(T_opt / 2.0)
        var_x_T = (sigma_avg * sigma_avg / (2.0 * kappa)) * (
            1.0 - jnp.exp(-2.0 * kappa * T_opt)
        )
        std_x_T = jnp.sqrt(var_x_T)

        # Simplified Black-like formula
        # This is an approximation; full implementation requires Jamshidian

        # ATM forward swap rate (simplified)
        bond_prices = jnp.array([
            zero_coupon_bond_price(params, t, x_0, y_0, 0.0) for t in payment_dates
        ])
        P_float = zero_coupon_bond_price(params, T_opt, x_0, y_0, 0.0)
        P_last = bond_prices[-1]

        annuity = jnp.sum(bond_prices) * payment_frequency

        # Forward swap rate
        forward_swap_rate = (P_float - P_last) / annuity

        # Volatility approximation
        # This is a rough approximation of the swap rate volatility
        swap_vol = std_x_T / jnp.sqrt(T_opt) * 0.5  # Scaling factor

        # Black formula
        if swap_vol > 1e-10:
            d1 = (jnp.log(forward_swap_rate / strike) + 0.5 * swap_vol * swap_vol * T_opt) / \
                 (swap_vol * jnp.sqrt(T_opt))
            d2 = d1 - swap_vol * jnp.sqrt(T_opt)

            if is_payer:
                value = annuity * (
                    forward_swap_rate * norm.cdf(float(d1)) - strike * norm.cdf(float(d2))
                )
            else:
                value = annuity * (
                    strike * norm.cdf(float(-d2)) - forward_swap_rate * norm.cdf(float(-d1))
                )
        else:
            # Intrinsic value
            if is_payer:
                value = annuity * jnp.maximum(forward_swap_rate - strike, 0.0)
            else:
                value = annuity * jnp.maximum(strike - forward_swap_rate, 0.0)

        return float(value)

    else:
        # Multi-factor: use Monte Carlo or numerical methods
        raise NotImplementedError(
            "Analytical swaption pricing not yet implemented for multi-factor Cheyette"
        )


__all__ = [
    "CheyetteParams",
    "zero_coupon_bond_price",
    "simulate_path",
    "simulate_paths",
    "swaption_price_analytical",
]
