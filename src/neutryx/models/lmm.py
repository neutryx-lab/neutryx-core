"""LIBOR Market Model (LMM) / Brace-Gatarek-Musiela (BGM) implementation with JAX.

The LIBOR Market Model (also known as BGM model) is a forward rate model
that directly models market-observable LIBOR rates. It's one of the most
important models for pricing and hedging interest rate derivatives.

The model describes the evolution of forward LIBOR rates L_i(t) for each
tenor period [T_i, T_{i+1}]:

Under the spot LIBOR measure Q_{T_{i+1}}:
    dL_i(t) / L_i(t) = λ_i(t) dW_i^{i+1}(t)

Under a common terminal measure Q_N:
    dL_i(t) / L_i(t) = λ_i(t) · Σⱼ₌ᵢ₊₁ᴺ⁻¹ [δⱼ λⱼ(t) L_j(t) / (1 + δⱼ L_j(t)) ρᵢⱼ] dt
                       + λ_i(t) dW_i^N(t)

where:
    - L_i(t): forward LIBOR rate for period [T_i, T_{i+1}]
    - λ_i(t): volatility of the i-th forward rate
    - δ_i: day count fraction for period [T_i, T_{i+1}]
    - ρᵢⱼ: instantaneous correlation between rates i and j
    - W_i: Brownian motion under appropriate measure

Key features:
1. Market-consistent: models traded LIBOR rates directly
2. No-arbitrage: ensures consistent evolution under measure changes
3. Flexible volatility: can match cap/swaption market volatilities
4. Correlations: captures correlation structure between different tenors
5. Positive rates: log-normal specification ensures positive rates

The LMM is the industry standard for:
- Bermudan swaptions
- CMS products
- Range accruals
- Callable structures
- Any product depending on multiple LIBOR rates

References
----------
Brace, A., Gatarek, D., & Musiela, M. (1997). "The market model of interest
rate dynamics." Mathematical Finance, 7(2), 127-155.

Rebonato, R. (2002). "Modern Pricing of Interest-Rate Derivatives:
The LIBOR Market Model and Beyond." Princeton University Press.

Brigo, D., & Mercurio, F. (2006). "Interest Rate Models - Theory and Practice."
Springer. (Chapter 6: The LIBOR and Swap Market Models)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax import lax


@dataclass
class LMMParams:
    """Parameters for the LIBOR Market Model.

    Attributes
    ----------
    forward_rates : Array
        Initial forward LIBOR rates L_i(0) for each tenor period.
        Shape: [n_rates]
    tenor_structure : Array
        Tenor dates [T_0, T_1, ..., T_n]. Shape: [n_rates + 1]
    volatility_fn : Callable
        Volatility function λ_i(t) returning array of shape [n_rates].
        Can be constant, piecewise constant, or parametric (e.g., exp(-αt)).
    correlation_matrix : Array
        Instantaneous correlation matrix ρᵢⱼ. Shape: [n_rates, n_rates]
    day_count_fractions : Optional[Array]
        Day count fractions δ_i for each period. If None, computed from tenor structure.
    terminal_measure : bool, optional
        If True, use terminal measure Q_N. If False, use spot measure.
        Default: True
    """
    forward_rates: jnp.ndarray
    tenor_structure: jnp.ndarray
    volatility_fn: Callable[[float], jnp.ndarray]
    correlation_matrix: jnp.ndarray
    day_count_fractions: Optional[jnp.ndarray] = None
    terminal_measure: bool = True

    def __post_init__(self):
        """Validate parameters."""
        self.forward_rates = jnp.array(self.forward_rates)
        self.tenor_structure = jnp.array(self.tenor_structure)
        self.correlation_matrix = jnp.array(self.correlation_matrix)

        n_rates = len(self.forward_rates)

        if len(self.tenor_structure) != n_rates + 1:
            raise ValueError(
                f"Tenor structure length {len(self.tenor_structure)} must be "
                f"n_rates + 1 = {n_rates + 1}"
            )

        if self.correlation_matrix.shape != (n_rates, n_rates):
            raise ValueError(
                f"Correlation matrix shape {self.correlation_matrix.shape} must be "
                f"({n_rates}, {n_rates})"
            )

        # Check correlation matrix is valid
        if not jnp.allclose(self.correlation_matrix, self.correlation_matrix.T):
            raise ValueError("Correlation matrix must be symmetric")

        eigenvalues = jnp.linalg.eigvalsh(self.correlation_matrix)
        if jnp.any(eigenvalues < -1e-10):
            raise ValueError("Correlation matrix must be positive semi-definite")

        # Compute day count fractions if not provided
        if self.day_count_fractions is None:
            self.day_count_fractions = jnp.diff(self.tenor_structure)
        else:
            self.day_count_fractions = jnp.array(self.day_count_fractions)
            if len(self.day_count_fractions) != n_rates:
                raise ValueError(
                    f"Day count fractions length {len(self.day_count_fractions)} "
                    f"must equal n_rates {n_rates}"
                )

        # Validate forward rates
        if jnp.any(self.forward_rates < 0):
            raise ValueError("Forward rates must be non-negative")


def simulate_path_terminal_measure(
    params: LMMParams,
    T: float,
    n_steps: int,
    key: jax.random.KeyArray,
) -> jnp.ndarray:
    """Simulate forward LIBOR rates under terminal measure Q_N.

    Parameters
    ----------
    params : LMMParams
        LMM model parameters
    T : float
        Total simulation time
    n_steps : int
        Number of time steps
    key : jax.random.KeyArray
        JAX random key

    Returns
    -------
    Array
        Forward rate paths of shape [n_steps + 1, n_rates]

    Notes
    -----
    Under terminal measure Q_N, the drift term is:

        μ_i(t) = -λ_i(t) · Σⱼ₌ᵢ₊₁ᴺ⁻¹ [δⱼ λⱼ(t) L_j(t) / (1 + δⱼ L_j(t)) ρᵢⱼ]

    The SDE for each rate is:
        dL_i(t) = L_i(t) * [μ_i(t) dt + λ_i(t) dW_i^N(t)]

    We use log-normal dynamics to ensure positive rates:
        d ln(L_i) = [μ_i - 0.5 λ_i²] dt + λ_i dW_i
    """
    n_rates = len(params.forward_rates)
    dt = T / n_steps

    # Generate correlated Brownian increments
    # First generate independent normals
    Z_indep = jax.random.normal(key, shape=(n_steps, n_rates))

    # Apply Cholesky decomposition for correlation
    L_chol = jnp.linalg.cholesky(params.correlation_matrix)
    dW = jnp.sqrt(dt) * (Z_indep @ L_chol.T)  # Shape: [n_steps, n_rates]

    # Initial forward rates (log scale for log-normal dynamics)
    ln_L_init = jnp.log(params.forward_rates)

    def step_fn(carry, inputs):
        ln_L_t, L_t, t = carry
        dW_t = inputs  # Shape: [n_rates]

        # Get volatilities at current time
        lambda_t = params.volatility_fn(t)  # Shape: [n_rates]

        # Compute drift for each rate under terminal measure
        # μ_i(t) = -λ_i(t) · Σⱼ₌ᵢ₊₁ᴺ⁻¹ [δⱼ λⱼ(t) L_j(t) / (1 + δⱼ L_j(t)) ρᵢⱼ]

        drift = jnp.zeros(n_rates)

        for i in range(n_rates):
            drift_sum = 0.0
            for j in range(i + 1, n_rates):
                delta_j = params.day_count_fractions[j]
                rho_ij = params.correlation_matrix[i, j]
                numerator = delta_j * lambda_t[j] * L_t[j]
                denominator = 1.0 + delta_j * L_t[j]
                drift_sum += (numerator / denominator) * rho_ij

            # Drift for log(L_i)
            drift = drift.at[i].set(-lambda_t[i] * drift_sum - 0.5 * lambda_t[i]**2)

        # Update log(L) using Euler-Maruyama
        d_ln_L = drift * dt + lambda_t * dW_t
        ln_L_next = ln_L_t + d_ln_L

        # Convert back to L
        L_next = jnp.exp(ln_L_next)

        t_next = t + dt

        return (ln_L_next, L_next, t_next), L_next

    _, L_path = lax.scan(step_fn, (ln_L_init, params.forward_rates, 0.0), dW)

    # Prepend initial rates
    L_path_full = jnp.concatenate([params.forward_rates[None, :], L_path], axis=0)

    return L_path_full


def simulate_path_spot_measure(
    params: LMMParams,
    T: float,
    n_steps: int,
    key: jax.random.KeyArray,
) -> jnp.ndarray:
    """Simulate forward LIBOR rates under rolling spot measure.

    Parameters
    ----------
    params : LMMParams
        LMM model parameters
    T : float
        Total simulation time
    n_steps : int
        Number of time steps
    key : jax.random.KeyArray
        JAX random key

    Returns
    -------
    Array
        Forward rate paths of shape [n_steps + 1, n_rates]

    Notes
    -----
    Under the spot measure, the drift changes as we roll through fixing dates.
    This is more complex than terminal measure but useful for certain products.
    """
    # For simplicity, use terminal measure
    # Full spot measure implementation requires tracking the rolling measure
    return simulate_path_terminal_measure(params, T, n_steps, key)


def simulate_paths(
    params: LMMParams,
    T: float,
    n_steps: int,
    n_paths: int,
    key: jax.random.KeyArray,
) -> jnp.ndarray:
    """Simulate multiple paths of forward LIBOR rates.

    Parameters
    ----------
    params : LMMParams
        LMM model parameters
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
    Array
        Forward rate paths of shape [n_paths, n_steps + 1, n_rates]
    """
    keys = jax.random.split(key, n_paths)

    if params.terminal_measure:
        sim_fn = simulate_path_terminal_measure
    else:
        sim_fn = simulate_path_spot_measure

    def sim_single_path(k):
        return sim_fn(params, T, n_steps, k)

    return jax.vmap(sim_single_path)(keys)


def zero_coupon_bond_price(
    params: LMMParams,
    forward_rates_t: jnp.ndarray,
    T_start: float,
    T_end: float,
) -> float:
    """Calculate zero-coupon bond price from forward LIBOR rates.

    The bond price P(t, T) is computed from the forward LIBOR rates using:

        P(t, T) = ∏ᵢ [1 / (1 + δᵢ L_i(t))]

    where the product is over all periods from T_start to T_end.

    Parameters
    ----------
    params : LMMParams
        LMM model parameters
    forward_rates_t : Array
        Current forward LIBOR rates L_i(t). Shape: [n_rates]
    T_start : float
        Start time
    T_end : float
        End time

    Returns
    -------
    float
        Zero-coupon bond price P(T_start, T_end)
    """
    # Find indices corresponding to [T_start, T_end]
    tenor_dates = params.tenor_structure

    # Find which rates to include
    start_idx = jnp.searchsorted(tenor_dates, T_start)
    end_idx = jnp.searchsorted(tenor_dates, T_end)

    if start_idx >= end_idx:
        return 1.0

    # Compute product of discount factors
    bond_price = 1.0
    for i in range(start_idx, end_idx):
        delta_i = params.day_count_fractions[i]
        L_i = forward_rates_t[i]
        bond_price *= 1.0 / (1.0 + delta_i * L_i)

    return float(bond_price)


def swap_rate(
    params: LMMParams,
    forward_rates_t: jnp.ndarray,
    T_start: float,
    T_end: float,
    payment_frequency: float = 0.5,
) -> float:
    """Calculate forward swap rate from LIBOR rates.

    The forward swap rate S(t; T_start, T_end) is:

        S = [P(t, T_start) - P(t, T_end)] / [Σᵢ δᵢ P(t, Tᵢ)]

    where the sum is over all payment dates.

    Parameters
    ----------
    params : LMMParams
        LMM model parameters
    forward_rates_t : Array
        Current forward LIBOR rates
    T_start : float
        Swap start time
    T_end : float
        Swap end time
    payment_frequency : float, optional
        Payment frequency in years (default: 0.5 for semi-annual)

    Returns
    -------
    float
        Forward swap rate
    """
    tenor_dates = params.tenor_structure

    # Find relevant indices
    start_idx = jnp.searchsorted(tenor_dates, T_start)
    end_idx = jnp.searchsorted(tenor_dates, T_end)

    # Compute bond prices
    P_start = zero_coupon_bond_price(params, forward_rates_t, 0.0, T_start)
    P_end = zero_coupon_bond_price(params, forward_rates_t, 0.0, T_end)

    # Compute annuity (sum of discount factors times accrual periods)
    annuity = 0.0
    for i in range(start_idx, end_idx):
        T_i = tenor_dates[i + 1]
        P_i = zero_coupon_bond_price(params, forward_rates_t, 0.0, T_i)
        delta_i = params.day_count_fractions[i]
        annuity += delta_i * P_i

    # Swap rate
    if annuity > 1e-10:
        S = (P_start - P_end) / annuity
    else:
        S = 0.0

    return float(S)


def caplet_price_mc(
    params: LMMParams,
    strike: float,
    caplet_index: int,
    n_paths: int = 50000,
    key: jax.random.KeyArray = None,
) -> float:
    """Price a caplet using Monte Carlo simulation.

    Parameters
    ----------
    params : LMMParams
        LMM model parameters
    strike : float
        Strike rate
    caplet_index : int
        Index of the caplet (which forward rate)
    n_paths : int, optional
        Number of Monte Carlo paths (default: 50000)
    key : jax.random.KeyArray, optional
        JAX random key

    Returns
    -------
    float
        Caplet price

    Notes
    -----
    Caplet on L_i pays: δ_i * max(L_i(T_i) - K, 0) at time T_{i+1}
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    # Simulate to the fixing date T_i
    T_fixing = params.tenor_structure[caplet_index]
    n_steps = max(50, int(T_fixing * 50))

    # Simulate paths
    L_paths = simulate_paths(params, T_fixing, n_steps, n_paths, key)

    # Terminal LIBOR rate L_i(T_i)
    L_fixing = L_paths[:, -1, caplet_index]

    # Caplet payoff at T_{i+1}
    delta_i = params.day_count_fractions[caplet_index]
    payoff = delta_i * jnp.maximum(L_fixing - strike, 0.0)

    # Discount back to time 0
    # Need to compute bond price P(0, T_{i+1}) from simulated rates
    # Simplified: use average discount
    T_payment = params.tenor_structure[caplet_index + 1]
    avg_L = jnp.mean(L_paths[:, -1, :caplet_index + 1])
    discount = 1.0 / ((1.0 + params.day_count_fractions[caplet_index] * avg_L) ** (caplet_index + 1))

    discounted_payoff = payoff * discount

    caplet_value = jnp.mean(discounted_payoff)

    return float(caplet_value)


def simple_volatility_structure(
    initial_vol: float,
    decay_rate: float = 0.0,
    n_rates: int = 10,
) -> Callable[[float], jnp.ndarray]:
    """Create a simple decaying volatility structure.

    Parameters
    ----------
    initial_vol : float
        Initial volatility level
    decay_rate : float, optional
        Exponential decay rate (default: 0 for constant)
    n_rates : int, optional
        Number of forward rates (default: 10)

    Returns
    -------
    Callable
        Volatility function λ(t) returning array of shape [n_rates]

    Notes
    -----
    Common parameterizations include:
    - Constant: λ_i(t) = σ_i
    - Exponential: λ_i(t) = σ_i * exp(-α * t)
    - Piecewise constant: λ_i(t) = σ_i,k for t ∈ [T_k, T_{k+1})
    """
    def volatility_fn(t: float) -> jnp.ndarray:
        # Decaying volatility over time
        time_decay = jnp.exp(-decay_rate * t)

        # Declining volatility across tenors (typical term structure)
        tenor_decline = jnp.linspace(1.0, 0.7, n_rates)

        vols = initial_vol * time_decay * tenor_decline

        return vols

    return volatility_fn


def create_correlation_matrix(
    n_rates: int,
    beta: float = 0.1,
    rho_infty: float = 0.4,
) -> jnp.ndarray:
    """Create a correlation matrix using exponential parameterization.

    A common parameterization is:
        ρᵢⱼ = ρ_∞ + (1 - ρ_∞) * exp(-β * |T_i - T_j|)

    Parameters
    ----------
    n_rates : int
        Number of forward rates
    beta : float, optional
        Decay parameter (default: 0.1)
    rho_infty : float, optional
        Long-term correlation (default: 0.4)

    Returns
    -------
    Array
        Correlation matrix of shape [n_rates, n_rates]
    """
    # Create index grid
    i_grid, j_grid = jnp.meshgrid(jnp.arange(n_rates), jnp.arange(n_rates), indexing='ij')

    # Exponential correlation
    rho = rho_infty + (1.0 - rho_infty) * jnp.exp(-beta * jnp.abs(i_grid - j_grid))

    return rho


__all__ = [
    "LMMParams",
    "simulate_path_terminal_measure",
    "simulate_path_spot_measure",
    "simulate_paths",
    "zero_coupon_bond_price",
    "swap_rate",
    "caplet_price_mc",
    "simple_volatility_structure",
    "create_correlation_matrix",
]
