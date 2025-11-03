"""Initial Margin (IM) calculations for uncleared OTC derivatives.

Initial margin is risk-based collateral posted to cover potential future exposure
in the event of counterparty default.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

import jax.numpy as jnp


class InitialMarginModel(str, Enum):
    """Initial margin calculation methodology."""

    SIMM = "SIMM"  # ISDA Standard Initial Margin Model
    SCHEDULE = "Schedule"  # BCBS/IOSCO Schedule IM
    GRID = "Grid"  # Simple grid-based IM
    CUSTOM = "Custom"  # Custom internal model


def calculate_grid_im(
    gross_notional: float,
    asset_class_multiplier: float = 0.15,
) -> float:
    """Calculate initial margin using simple grid approach.

    Parameters
    ----------
    gross_notional : float
        Gross notional of the portfolio
    asset_class_multiplier : float
        Asset class-specific percentage (default 15%)

    Returns
    -------
    float
        Initial margin amount

    Notes
    -----
    Grid IM is a simple percentage of gross notional.
    Typical multipliers:
    - Interest Rate: 2-4%
    - Credit: 4-6%
    - Equity: 15-18%
    - FX: 6-8%
    - Commodity: 15-25%
    """
    return gross_notional * asset_class_multiplier


def calculate_schedule_im(
    notional_by_asset_class: dict[str, float],
    net_replacement_cost: float = 0.0,
) -> float:
    """Calculate initial margin using BCBS/IOSCO Schedule IM.

    Parameters
    ----------
    notional_by_asset_class : dict[str, float]
        Gross notional amounts by asset class
        Keys: "IR", "FX", "Credit", "Equity", "Commodity"
    net_replacement_cost : float
        Net replacement cost (positive MTM), default 0

    Returns
    -------
    float
        Schedule IM amount

    Notes
    -----
    BCBS/IOSCO Schedule IM formula:
    IM = 0.4 * sum(notional * multiplier) + 0.6 * NGR + NRC

    Where:
    - NGR (Net to Gross Ratio) = net_replacement_cost / gross_notional
    - NRC (Net Replacement Cost) = max(MTM, 0)
    - Multipliers by asset class per regulatory schedule
    """
    # BCBS/IOSCO Schedule multipliers
    multipliers = {
        "IR": 0.005,  # 0.5% for interest rate
        "FX": 0.04,  # 4% for FX
        "Credit": 0.02,  # 2% for credit (AAA-A)
        "Equity": 0.15,  # 15% for equity
        "Commodity": 0.15,  # 15% for commodity
    }

    # Calculate gross IM component
    gross_im = 0.0
    gross_notional_total = 0.0

    for asset_class, notional in notional_by_asset_class.items():
        multiplier = multipliers.get(asset_class, 0.10)  # Default 10%
        gross_im += notional * multiplier
        gross_notional_total += notional

    # Calculate net-to-gross ratio component
    if gross_notional_total > 0:
        ngr = net_replacement_cost / gross_notional_total
    else:
        ngr = 0.0

    # Schedule IM formula
    schedule_im = 0.4 * gross_im + 0.6 * ngr * gross_notional_total + net_replacement_cost

    return max(0.0, schedule_im)


def calculate_im_from_simm(
    simm_result: float,
    simm_multiplier: float = 1.0,
) -> float:
    """Calculate initial margin from SIMM result.

    Parameters
    ----------
    simm_result : float
        SIMM calculation result (from SIMM module)
    simm_multiplier : float
        Multiplier applied to SIMM (e.g., for regulatory add-ons), default 1.0

    Returns
    -------
    float
        Initial margin amount

    Notes
    -----
    SIMM produces a risk-based IM estimate. Regulatory rules may require
    multipliers or add-ons (e.g., 1.1x for certain jurisdictions).
    """
    return simm_result * simm_multiplier


def calculate_im_time_profile(
    im_values: jnp.ndarray,
    time_points: jnp.ndarray,
) -> tuple[jnp.ndarray, float]:
    """Calculate initial margin time profile statistics.

    Parameters
    ----------
    im_values : jnp.ndarray
        IM values across time points or scenarios, shape [n_points]
    time_points : jnp.ndarray
        Time points in years, shape [n_points]

    Returns
    -------
    tuple[jnp.ndarray, float]
        (im_profile, expected_im)
        - im_profile: IM values over time
        - expected_im: Time-weighted average IM

    Notes
    -----
    For MVA calculations, we need the time profile of IM requirements.
    """
    # Expected IM (simple average for now)
    expected_im = float(jnp.mean(im_values))

    return im_values, expected_im


def calculate_dynamic_im(
    exposure_profile: jnp.ndarray,
    confidence_level: float = 0.99,
    margin_period_of_risk: int = 10,
) -> jnp.ndarray:
    """Calculate dynamic initial margin based on exposure distribution.

    Parameters
    ----------
    exposure_profile : jnp.ndarray
        Simulated exposure paths, shape [n_time_steps, n_paths]
    confidence_level : float
        Confidence level for VaR-based IM (default 99%)
    margin_period_of_risk : int
        Margin period of risk in days (default 10 days)

    Returns
    -------
    jnp.ndarray
        Dynamic IM profile over time, shape [n_time_steps]

    Notes
    -----
    Dynamic IM = VaR(exposure, confidence_level) at each time step.
    This represents the potential loss over the margin period of risk.
    """
    # Calculate quantile at each time step
    quantile = confidence_level
    im_profile = jnp.quantile(exposure_profile, quantile, axis=1)

    return im_profile


def calculate_two_way_im(
    our_im: float,
    their_im: float,
    asymmetric: bool = False,
) -> tuple[float, float]:
    """Calculate two-way (bilateral) initial margin posting.

    Parameters
    ----------
    our_im : float
        IM we should post to them
    their_im : float
        IM they should post to us
    asymmetric : bool
        Whether to apply asymmetric IM rules (e.g., only higher-risk party posts)

    Returns
    -------
    tuple[float, float]
        (we_post, they_post) - IM amounts posted by each party

    Notes
    -----
    Standard two-way IM: both parties post independently.
    Asymmetric IM: only the party with higher IM requirement posts.
    """
    if asymmetric:
        # Asymmetric: only higher IM posts the difference
        if our_im > their_im:
            return our_im - their_im, 0.0
        else:
            return 0.0, their_im - our_im
    else:
        # Symmetric: both parties post independently
        return our_im, their_im


def calculate_im_with_threshold(
    im_amount: float,
    threshold: float = 0.0,
    minimum_transfer_amount: float = 0.0,
) -> float:
    """Apply threshold and MTA to initial margin.

    Parameters
    ----------
    im_amount : float
        Calculated IM amount
    threshold : float
        IM threshold (no posting required below this)
    minimum_transfer_amount : float
        Minimum IM transfer amount

    Returns
    -------
    float
        Adjusted IM after threshold and MTA

    Notes
    -----
    Some IM CSAs have thresholds and MTAs similar to VM CSAs.
    """
    if im_amount < threshold:
        return 0.0

    if im_amount < minimum_transfer_amount:
        return 0.0

    return im_amount
