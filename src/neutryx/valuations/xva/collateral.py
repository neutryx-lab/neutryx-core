"""Collateral-adjusted exposure calculations for CVA/DVA.

This module integrates CSA terms with exposure simulations to compute
collateralized exposure profiles for XVA calculations.
"""
from __future__ import annotations

from typing import Optional

import jax.numpy as jnp

from neutryx.contracts.csa import CSA


def calculate_collateralized_exposure(
    uncollateralized_exposure: jnp.ndarray,
    threshold: float = 0.0,
    independent_amount: float = 0.0,
    mta: float = 0.0,
) -> jnp.ndarray:
    """Calculate collateralized exposure profile.

    Parameters
    ----------
    uncollateralized_exposure : jnp.ndarray
        Uncollateralized exposure profile, shape [n_time_steps] or [n_time_steps, n_paths]
    threshold : float
        CSA threshold amount
    independent_amount : float
        Independent amount (always at risk)
    mta : float
        Minimum transfer amount (approximation for path-level)

    Returns
    -------
    jnp.ndarray
        Collateralized exposure profile, same shape as input

    Notes
    -----
    Simplified formula:
    collateralized_exposure = max(0, uncollateralized_exposure - threshold) + IA

    For more accurate modeling, MTA introduces a buffer zone around the threshold.
    """
    # Exposure exceeding threshold
    excess_exposure = jnp.maximum(0.0, uncollateralized_exposure - threshold - mta)

    # Add independent amount (always at risk)
    collateralized_exposure = excess_exposure + independent_amount

    return collateralized_exposure


def calculate_epe_with_collateral(
    exposure_paths: jnp.ndarray,
    threshold: float = 0.0,
    independent_amount: float = 0.0,
) -> float:
    """Calculate Expected Positive Exposure with collateral.

    Parameters
    ----------
    exposure_paths : jnp.ndarray
        Exposure paths from Monte Carlo simulation, shape [n_time_steps, n_paths]
    threshold : float
        CSA threshold
    independent_amount : float
        Independent amount

    Returns
    -------
    float
        Collateralized EPE (time-averaged)

    Notes
    -----
    EPE_collateralized = E[max(0, exposure - threshold) + IA]
    """
    # Apply collateral adjustment to each path
    collateralized_paths = calculate_collateralized_exposure(
        uncollateralized_exposure=exposure_paths,
        threshold=threshold,
        independent_amount=independent_amount,
    )

    # Take positive part (should already be positive after max operation)
    positive_exposure = jnp.maximum(collateralized_paths, 0.0)

    # Time average, then path average
    epe_profile = jnp.mean(positive_exposure, axis=1)  # Average across paths at each time
    epe = float(jnp.mean(epe_profile))  # Average across time

    return epe


def calculate_cva_with_collateral(
    exposure_paths: jnp.ndarray,
    time_grid: jnp.ndarray,
    hazard_rates: jnp.ndarray,
    discount_factors: jnp.ndarray,
    lgd: float = 0.6,
    threshold: float = 0.0,
    independent_amount: float = 0.0,
) -> float:
    """Calculate CVA with collateral adjustment.

    Parameters
    ----------
    exposure_paths : jnp.ndarray
        Exposure paths, shape [n_time_steps, n_paths]
    time_grid : jnp.ndarray
        Time points in years, shape [n_time_steps]
    hazard_rates : jnp.ndarray
        Constant hazard rates over each period, shape [n_time_steps]
    discount_factors : jnp.ndarray
        Discount factors, shape [n_time_steps]
    lgd : float
        Loss Given Default (default 0.6)
    threshold : float
        CSA threshold
    independent_amount : float
        Independent amount

    Returns
    -------
    float
        CVA amount

    Notes
    -----
    CVA = LGD * sum_t [ DF(t) * EPE_collateralized(t) * dPD(t) ]
    where dPD(t) = survival(t-1) * (1 - exp(-hazard_rate * dt))
    """
    # Apply collateral adjustment
    collateralized_paths = calculate_collateralized_exposure(
        uncollateralized_exposure=exposure_paths,
        threshold=threshold,
        independent_amount=independent_amount,
    )

    # Calculate EPE at each time point
    epe_profile = jnp.mean(collateralized_paths, axis=1)  # Shape: [n_time_steps]

    # Calculate time increments
    dt = jnp.diff(time_grid, prepend=0.0)

    # Calculate marginal default probabilities
    # Survival probability: S(t) = exp(-integral_0^t hazard(s) ds)
    cumulative_hazard = jnp.cumsum(hazard_rates * dt)
    survival_prob = jnp.exp(-cumulative_hazard)

    # Marginal PD: dPD(t) = S(t-1) * [1 - exp(-hazard(t) * dt)]
    survival_prev = jnp.concatenate([jnp.array([1.0]), survival_prob[:-1]])
    marginal_pd = survival_prev * (1.0 - jnp.exp(-hazard_rates * dt))

    # CVA integral
    cva = lgd * jnp.sum(discount_factors * epe_profile * marginal_pd)

    return float(cva)


def calculate_collateral_benefit(
    uncollateralized_cva: float,
    collateralized_cva: float,
) -> float:
    """Calculate the benefit of collateralization on CVA.

    Parameters
    ----------
    uncollateralized_cva : float
        CVA without collateral
    collateralized_cva : float
        CVA with collateral

    Returns
    -------
    float
        CVA reduction due to collateral (positive value = benefit)

    Notes
    -----
    Collateral benefit = uncollateralized_CVA - collateralized_CVA
    """
    return uncollateralized_cva - collateralized_cva


def simulate_collateral_evolution(
    exposure_paths: jnp.ndarray,
    threshold: float = 0.0,
    mta: float = 0.0,
    rounding: float = 0.0,
) -> jnp.ndarray:
    """Simulate evolution of posted collateral over time.

    Parameters
    ----------
    exposure_paths : jnp.ndarray
        Exposure paths, shape [n_time_steps, n_paths]
    threshold : float
        CSA threshold
    mta : float
        Minimum transfer amount
    rounding : float
        Rounding increment

    Returns
    -------
    jnp.ndarray
        Simulated collateral balance over time, shape [n_time_steps, n_paths]

    Notes
    -----
    This is a simplified simulation. A full simulation would track:
    - Previous collateral balance
    - Margin calls and returns
    - Time delays in posting
    - Disputes and reconciliation breaks
    """
    n_steps, n_paths = exposure_paths.shape
    collateral_balance = jnp.zeros((n_steps, n_paths))

    # Simplified: collateral = max(0, exposure - threshold) with rounding
    required_collateral = jnp.maximum(0.0, exposure_paths - threshold)

    # Apply rounding if specified
    if rounding > 0:
        required_collateral = jnp.round(required_collateral / rounding) * rounding

    # In practice, we'd track collateral balance evolution with MTA filtering
    # For now, assume instantaneous collateral posting
    collateral_balance = required_collateral

    return collateral_balance


def calculate_margin_valuation_adjustment(
    initial_margin_profile: jnp.ndarray,
    time_grid: jnp.ndarray,
    discount_factors: jnp.ndarray,
    funding_spread: float,
) -> float:
    """Calculate Margin Valuation Adjustment (MVA).

    Parameters
    ----------
    initial_margin_profile : jnp.ndarray
        Initial margin requirements over time, shape [n_time_steps]
    time_grid : jnp.ndarray
        Time points in years, shape [n_time_steps]
    discount_factors : jnp.ndarray
        Discount factors, shape [n_time_steps]
    funding_spread : float
        Funding spread for posting IM (as decimal, e.g., 0.01 = 1%)

    Returns
    -------
    float
        MVA amount

    Notes
    -----
    MVA = funding_spread * sum_t [ DF(t) * IM(t) * dt ]

    MVA captures the cost of funding initial margin requirements.
    """
    # Calculate time increments
    dt = jnp.diff(time_grid, prepend=0.0)

    # MVA integral
    mva = funding_spread * jnp.sum(discount_factors * initial_margin_profile * dt)

    return float(mva)


def calculate_collateral_gap(
    exposure: float,
    posted_collateral: float,
    threshold: float = 0.0,
) -> float:
    """Calculate collateral gap (uncollateralized exposure).

    Parameters
    ----------
    exposure : float
        Current exposure
    posted_collateral : float
        Currently posted collateral
    threshold : float
        CSA threshold

    Returns
    -------
    float
        Collateral gap (positive = under-collateralized)

    Notes
    -----
    Gap = max(0, exposure - threshold) - posted_collateral
    Positive gap means additional collateral is needed.
    """
    required_collateral = max(0.0, exposure - threshold)
    gap = required_collateral - posted_collateral
    return gap
