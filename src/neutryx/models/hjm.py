"""Heath-Jarrow-Morton (HJM) framework implementation with JAX.

The Heath-Jarrow-Morton (HJM) framework is the most general continuous-time
framework for modeling the evolution of the entire forward rate curve. It
provides a no-arbitrage condition that any interest rate model must satisfy.

The framework models the instantaneous forward rate f(t, T) for all maturities T:

    df(t, T) = α(t, T) dt + Σᵢ σᵢ(t, T) dWᵢ(t)

where:
    - f(t, T): instantaneous forward rate at time t for maturity T
    - α(t, T): drift of forward rate
    - σᵢ(t, T): i-th volatility function
    - Wᵢ(t): i-th Brownian motion

The HJM no-arbitrage condition (drift restriction):

    α(t, T) = Σᵢ σᵢ(t, T) ∫ₜᵀ σᵢ(t, s) ds

This ensures that the forward rate dynamics are arbitrage-free under the
risk-neutral measure.

The short rate is:
    r(t) = f(t, t) = f(0, t) + ∫₀ᵗ α(s, t) ds + Σᵢ ∫₀ᵗ σᵢ(s, t) dWᵢ(s)

Key features:
1. Most general framework for term structure modeling
2. No-arbitrage by construction (via drift restriction)
3. Can accommodate any volatility structure
4. Many models are special cases (Vasicek, Hull-White, LMM)
5. Requires specifying entire volatility surface σ(t, T)

Relationship to other models:
- Vasicek/Hull-White: σ(t, T) = σ exp(-a(T-t))
- LMM: Discrete-tenor version of HJM
- Cheyette: Markovian reduction of HJM

The HJM framework is fundamental for:
- Theoretical understanding of interest rate dynamics
- Developing new models
- Exotic derivative pricing
- Model calibration and risk management

References
----------
Heath, D., Jarrow, R., & Morton, A. (1992). "Bond pricing and the term
structure of interest rates: A new methodology for contingent claims valuation."
Econometrica, 60(1), 77-105.

Musiela, M., & Rutkowski, M. (2005). "Martingale Methods in Financial Modelling."
Springer. (Chapter 12: HJM Models)

Brigo, D., & Mercurio, F. (2006). "Interest Rate Models - Theory and Practice."
Springer. (Chapter 5: The Heath-Jarrow-Morton Framework)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax import lax


@dataclass
class HJMParams:
    """Parameters for the Heath-Jarrow-Morton framework.

    Attributes
    ----------
    forward_curve_fn : Callable
        Initial forward curve f(0, T)
    volatility_fns : list[Callable]
        List of volatility functions σᵢ(t, T). Each function takes
        (t, T) and returns volatility of forward rate f(t, T).
    r0 : float
        Initial short rate r(0) = f(0, 0)
    n_factors : int, optional
        Number of Brownian factors (default: 1)
    rho : Optional[Array]
        Correlation matrix between Brownian motions. If None, assumes independent.
    tenor_grid : Optional[Array]
        Maturity grid for discretization. If None, creates uniform grid.
    max_maturity : float, optional
        Maximum maturity to simulate (default: 30.0 years)
    n_maturities : int, optional
        Number of maturity points in discretization (default: 60)
    """
    forward_curve_fn: Callable[[float], float]
    volatility_fns: list[Callable[[float, float], float]]
    r0: float
    n_factors: int = 1
    rho: Optional[jnp.ndarray] = None
    tenor_grid: Optional[jnp.ndarray] = None
    max_maturity: float = 30.0
    n_maturities: int = 60

    def __post_init__(self):
        """Validate parameters and set up grids."""
        if len(self.volatility_fns) != self.n_factors:
            raise ValueError(
                f"Number of volatility functions {len(self.volatility_fns)} must equal "
                f"n_factors {self.n_factors}"
            )

        # Set up correlation matrix
        if self.rho is None:
            self.rho = jnp.eye(self.n_factors)
        else:
            self.rho = jnp.array(self.rho)
            if self.rho.shape != (self.n_factors, self.n_factors):
                raise ValueError(
                    f"Correlation matrix shape {self.rho.shape} must be "
                    f"({self.n_factors}, {self.n_factors})"
                )

        # Set up tenor grid for discretization
        if self.tenor_grid is None:
            self.tenor_grid = jnp.linspace(0, self.max_maturity, self.n_maturities)
        else:
            self.tenor_grid = jnp.array(self.tenor_grid)


def compute_hjm_drift(
    params: HJMParams,
    t: float,
    T: float,
) -> float:
    """Compute the HJM drift α(t, T) satisfying no-arbitrage condition.

    The no-arbitrage drift is:
        α(t, T) = Σᵢ σᵢ(t, T) ∫ₜᵀ σᵢ(t, s) ds

    Parameters
    ----------
    params : HJMParams
        HJM model parameters
    t : float
        Current time
    T : float
        Maturity time

    Returns
    -------
    float
        Drift α(t, T)

    Notes
    -----
    This requires integrating the volatility functions from t to T.
    For general σ(t, T), we use numerical integration.
    """
    drift = 0.0

    for i in range(params.n_factors):
        sigma_i_t_T = params.volatility_fns[i](t, T)

        # Integrate σᵢ(t, s) from t to T
        # Use trapezoidal rule
        n_steps = 20
        s_grid = jnp.linspace(t, T, n_steps + 1)
        ds = (T - t) / n_steps

        integral = 0.0
        for j in range(n_steps):
            s = s_grid[j]
            sigma_i_t_s = params.volatility_fns[i](t, s)
            sigma_i_t_s_next = params.volatility_fns[i](t, s_grid[j + 1])
            integral += 0.5 * (sigma_i_t_s + sigma_i_t_s_next) * ds

        drift += sigma_i_t_T * integral

    return drift


def simulate_forward_curve_path(
    params: HJMParams,
    T_horizon: float,
    n_steps: int,
    key: jax.random.KeyArray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Simulate evolution of the entire forward curve.

    Parameters
    ----------
    params : HJMParams
        HJM model parameters
    T_horizon : float
        Simulation horizon
    n_steps : int
        Number of time steps
    key : jax.random.KeyArray
        JAX random key

    Returns
    -------
    tuple[Array, Array]
        Tuple of (time_grid, forward_curves) where:
        - time_grid: time points of shape [n_steps + 1]
        - forward_curves: forward curves at each time, shape [n_steps + 1, n_maturities]

    Notes
    -----
    We discretize both time and maturity:
    - Time discretization: [0, dt, 2dt, ..., T_horizon]
    - Maturity discretization: tenor_grid

    For each maturity T_j, we simulate:
        df(tᵢ, T_j) = α(tᵢ, T_j) dt + Σₖ σₖ(tᵢ, T_j) dWₖ(tᵢ)
    """
    dt = T_horizon / n_steps
    time_grid = jnp.linspace(0, T_horizon, n_steps + 1)

    # Generate correlated Brownian increments
    # Shape: [n_steps, n_factors]
    Z_indep = jax.random.normal(key, shape=(n_steps, params.n_factors))
    L_chol = jnp.linalg.cholesky(params.rho)
    dW = jnp.sqrt(dt) * (Z_indep @ L_chol.T)

    # Initial forward curve
    # Shape: [n_maturities]
    initial_curve = jnp.array([params.forward_curve_fn(T) for T in params.tenor_grid])

    def step_fn(carry, inputs):
        f_t, t = carry  # f_t shape: [n_maturities]
        dW_t = inputs  # shape: [n_factors]

        # Vectorized update for all maturities
        # For maturities that have passed (T_j <= t), keep current value
        # For active maturities (T_j > t), update with HJM dynamics

        def update_single_maturity(j):
            T_j = params.tenor_grid[j]

            # Compute drift
            alpha_t_Tj = compute_hjm_drift(params, t, T_j)

            # Compute diffusion
            diffusion = 0.0
            for k in range(params.n_factors):
                sigma_k_t_Tj = params.volatility_fns[k](t, T_j)
                diffusion += sigma_k_t_Tj * dW_t[k]

            # Update forward rate
            df = alpha_t_Tj * dt + diffusion
            f_new = f_t[j] + df

            # Use where to conditionally update based on whether maturity has passed
            # If T_j > t (active), use f_new; otherwise keep f_t[j]
            return jnp.where(T_j > t, f_new, f_t[j])

        # Vectorize over all maturities
        f_next = jax.vmap(update_single_maturity)(jnp.arange(params.n_maturities))

        t_next = t + dt

        return (f_next, t_next), f_next

    _, forward_curves = lax.scan(step_fn, (initial_curve, 0.0), dW)

    # Prepend initial curve
    forward_curves_full = jnp.concatenate([initial_curve[None, :], forward_curves], axis=0)

    return time_grid, forward_curves_full


def simulate_short_rate_path(
    params: HJMParams,
    T: float,
    n_steps: int,
    key: jax.random.KeyArray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Simulate the short rate path r(t) = f(t, t).

    Parameters
    ----------
    params : HJMParams
        HJM model parameters
    T : float
        Simulation horizon
    n_steps : int
        Number of time steps
    key : jax.random.KeyArray
        JAX random key

    Returns
    -------
    tuple[Array, Array]
        Tuple of (time_grid, short_rate_path) where:
        - time_grid: time points of shape [n_steps + 1]
        - short_rate_path: short rates of shape [n_steps + 1]

    Notes
    -----
    The short rate r(t) = f(t, t) can be tracked by simulating f(t, T)
    and interpolating to T = t, or by directly simulating r(t) dynamics.

    For simplicity, we simulate the forward curve and extract r(t).
    """
    time_grid, forward_curves = simulate_forward_curve_path(params, T, n_steps, key)

    # Extract short rate r(t) = f(t, t)
    # At each time t, find f(t, t) by interpolating the forward curve
    short_rates = jnp.zeros(n_steps + 1)

    for i in range(n_steps + 1):
        t = time_grid[i]
        f_t = forward_curves[i]

        # Find f(t, t) by interpolating at maturity = t
        # Use linear interpolation
        idx = jnp.searchsorted(params.tenor_grid, t)

        if idx == 0:
            short_rates = short_rates.at[i].set(f_t[0])
        elif idx >= params.n_maturities:
            short_rates = short_rates.at[i].set(f_t[-1])
        else:
            # Linear interpolation
            T1, T2 = params.tenor_grid[idx - 1], params.tenor_grid[idx]
            f1, f2 = f_t[idx - 1], f_t[idx]
            weight = (t - T1) / (T2 - T1)
            r_t = f1 + weight * (f2 - f1)
            short_rates = short_rates.at[i].set(r_t)

    return time_grid, short_rates


def zero_coupon_bond_price(
    params: HJMParams,
    forward_curve_t: jnp.ndarray,
    T_maturity: float,
    t: float = 0.0,
) -> float:
    """Calculate zero-coupon bond price from forward curve.

    The bond price is:
        P(t, T) = exp(-∫ₜᵀ f(t, s) ds)

    Parameters
    ----------
    params : HJMParams
        HJM model parameters
    forward_curve_t : Array
        Forward curve f(t, T) for all maturities T. Shape: [n_maturities]
    T_maturity : float
        Bond maturity
    t : float, optional
        Current time (default: 0)

    Returns
    -------
    float
        Zero-coupon bond price

    Notes
    -----
    We integrate the forward curve from t to T_maturity using trapezoidal rule.
    """
    # Find relevant part of forward curve from t to T_maturity
    mask = (params.tenor_grid >= t) & (params.tenor_grid <= T_maturity)
    relevant_tenors = params.tenor_grid[mask]
    relevant_forwards = forward_curve_t[mask]

    if len(relevant_tenors) < 2:
        # Not enough points for integration
        return 1.0

    # Integrate using trapezoidal rule
    integral = jnp.trapz(relevant_forwards, relevant_tenors)

    # Bond price
    P = jnp.exp(-integral)

    return float(P)


def gaussian_hjm_volatility(
    sigma: float,
    kappa: float,
) -> Callable[[float, float], float]:
    """Create a Gaussian HJM volatility function (Hull-White type).

    The volatility function is:
        σ(t, T) = σ exp(-κ(T - t))

    This corresponds to the Hull-White/Vasicek model.

    Parameters
    ----------
    sigma : float
        Base volatility
    kappa : float
        Mean reversion speed

    Returns
    -------
    Callable
        Volatility function σ(t, T)
    """
    def vol_fn(t: float, T: float) -> float:
        return sigma * jnp.exp(-kappa * (T - t))

    return vol_fn


def exponential_hjm_volatility(
    sigma0: float,
    decay_rate: float,
    term_structure_decay: float = 0.1,
) -> Callable[[float, float], float]:
    """Create an exponential HJM volatility function.

    The volatility function is:
        σ(t, T) = σ₀ exp(-α t) exp(-β(T - t))

    This captures both time decay and term structure effects.

    Parameters
    ----------
    sigma0 : float
        Initial volatility level
    decay_rate : float
        Time decay parameter α
    term_structure_decay : float, optional
        Term structure decay parameter β (default: 0.1)

    Returns
    -------
    Callable
        Volatility function σ(t, T)
    """
    def vol_fn(t: float, T: float) -> float:
        time_decay = jnp.exp(-decay_rate * t)
        term_decay = jnp.exp(-term_structure_decay * (T - t))
        return sigma0 * time_decay * term_decay

    return vol_fn


def simulate_paths(
    params: HJMParams,
    T: float,
    n_steps: int,
    n_paths: int,
    key: jax.random.KeyArray,
    output_type: str = "short_rate",
) -> jnp.ndarray:
    """Simulate multiple paths under HJM framework.

    Parameters
    ----------
    params : HJMParams
        HJM model parameters
    T : float
        Simulation horizon
    n_steps : int
        Number of time steps
    n_paths : int
        Number of paths to simulate
    key : jax.random.KeyArray
        JAX random key
    output_type : str, optional
        Type of output: "short_rate" returns r(t), "forward_curve" returns full curve.
        Default: "short_rate"

    Returns
    -------
    Array
        Simulated paths. Shape depends on output_type:
        - "short_rate": [n_paths, n_steps + 1]
        - "forward_curve": [n_paths, n_steps + 1, n_maturities]
    """
    keys = jax.random.split(key, n_paths)

    if output_type == "short_rate":
        def sim_fn(k):
            _, r_path = simulate_short_rate_path(params, T, n_steps, k)
            return r_path

        return jax.vmap(sim_fn)(keys)

    elif output_type == "forward_curve":
        def sim_fn(k):
            _, curves = simulate_forward_curve_path(params, T, n_steps, k)
            return curves

        return jax.vmap(sim_fn)(keys)

    else:
        raise ValueError(f"Unknown output_type: {output_type}. Use 'short_rate' or 'forward_curve'.")


__all__ = [
    "HJMParams",
    "compute_hjm_drift",
    "simulate_forward_curve_path",
    "simulate_short_rate_path",
    "zero_coupon_bond_price",
    "gaussian_hjm_volatility",
    "exponential_hjm_volatility",
    "simulate_paths",
]
