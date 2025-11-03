"""Netting calculations for bilateral and payment netting.

This module provides functions for:
- Close-out netting (bilateral netting of MTM values)
- Payment netting (netting of cash flows by currency and date)
- Collateral-adjusted exposure calculations
"""
from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional

import jax.numpy as jnp


def net_exposure(payoffs_matrix):
    """Net exposure across trades (legacy function for backward compatibility).

    Parameters
    ----------
    payoffs_matrix : array
        Payoffs matrix with shape [trades, paths]

    Returns
    -------
    array
        Net exposure across trades for each path, shape [paths]

    Notes
    -----
    This is a simplified toy demo for common-maturity trades.
    """
    # payoffs_matrix: [trades, paths] common maturity for toy demo
    return payoffs_matrix.sum(axis=0)


def calculate_close_out_amount(
    mtm_values: List[float],
    include_termination_costs: bool = False,
    termination_costs: Optional[List[float]] = None,
) -> float:
    """Calculate close-out amount under bilateral netting.

    Parameters
    ----------
    mtm_values : list[float]
        Mark-to-market values for each trade (positive = owed to us)
    include_termination_costs : bool
        Whether to include termination costs in calculation
    termination_costs : list[float], optional
        Termination costs for each trade (if applicable)

    Returns
    -------
    float
        Net close-out amount (positive = owed to us, negative = we owe)

    Notes
    -----
    Under ISDA close-out netting, all transactions under the master agreement
    are netted to a single payment amount.
    """
    net_mtm = sum(mtm_values)

    if include_termination_costs and termination_costs:
        if len(termination_costs) != len(mtm_values):
            raise ValueError("termination_costs must match length of mtm_values")
        # Add termination costs (typically positive, increasing amount owed)
        net_mtm += sum(termination_costs)

    return net_mtm


def calculate_net_exposure_by_currency(
    mtm_by_currency: Dict[str, float],
) -> Dict[str, float]:
    """Calculate net exposure grouped by currency.

    Parameters
    ----------
    mtm_by_currency : dict[str, float]
        Map of currency -> net MTM in that currency

    Returns
    -------
    dict[str, float]
        Net exposure by currency (same as input for bilateral netting)

    Notes
    -----
    This function performs netting within each currency. For cross-currency
    netting, FX conversion would be needed (not implemented here).
    """
    # For bilateral netting within currency, net exposure is just the MTM
    return mtm_by_currency.copy()


def calculate_payment_netting(
    cash_flows: List[Dict[str, any]],
    netting_by_currency: bool = True,
) -> List[Dict[str, any]]:
    """Perform payment netting on scheduled cash flows.

    Parameters
    ----------
    cash_flows : list[dict]
        List of cash flows, each with keys:
        - 'date': payment date
        - 'currency': currency code
        - 'amount': payment amount (positive = receive, negative = pay)
    netting_by_currency : bool
        Whether to net separately by currency (default True)

    Returns
    -------
    list[dict]
        Netted cash flows by date (and currency if netting_by_currency=True)

    Notes
    -----
    Payment netting reduces the number of payments by netting all payments
    due on the same date (and in the same currency if applicable).
    """
    if not cash_flows:
        return []

    # Group by (date, currency) if netting by currency, else by date only
    netted: Dict[tuple, float] = {}

    for cf in cash_flows:
        cf_date = cf["date"]
        cf_currency = cf["currency"]
        cf_amount = cf["amount"]

        if netting_by_currency:
            key = (cf_date, cf_currency)
        else:
            key = (cf_date,)

        netted[key] = netted.get(key, 0.0) + cf_amount

    # Convert back to list format
    result = []
    for key, amount in netted.items():
        if abs(amount) < 1e-10:  # Skip near-zero netted amounts
            continue

        if netting_by_currency:
            cf_date, cf_currency = key
            result.append({"date": cf_date, "currency": cf_currency, "amount": amount})
        else:
            cf_date = key[0]
            result.append({"date": cf_date, "amount": amount})

    # Sort by date
    result.sort(key=lambda x: x["date"])
    return result


def calculate_collateral_adjusted_exposure(
    exposure: float,
    threshold: float = 0.0,
    posted_collateral: float = 0.0,
    independent_amount: float = 0.0,
) -> float:
    """Calculate exposure adjusted for posted collateral and CSA terms.

    Parameters
    ----------
    exposure : float
        Gross exposure before collateral (MTM or EPE)
    threshold : float
        Threshold amount (uncollateralized exposure allowed)
    posted_collateral : float
        Amount of collateral currently posted
    independent_amount : float
        Independent amount (minimum collateral required)

    Returns
    -------
    float
        Net exposure after collateral adjustment

    Notes
    -----
    Formula:
        net_exposure = max(0, exposure - threshold - posted_collateral) + independent_amount

    For CVA calculations, this represents the collateralized exposure.
    """
    # Exposure exceeding threshold, reduced by posted collateral
    net_exposure = max(0.0, exposure - threshold - posted_collateral)

    # Add independent amount (which is not tied to exposure)
    net_exposure += independent_amount

    return net_exposure


def apply_bilateral_netting_to_paths(
    trade_values: jnp.ndarray,
) -> jnp.ndarray:
    """Apply bilateral netting to pathwise trade values.

    Parameters
    ----------
    trade_values : jnp.ndarray
        Array of shape [n_trades, n_paths] with trade values per path

    Returns
    -------
    jnp.ndarray
        Net portfolio value per path, shape [n_paths]

    Notes
    -----
    This is the JAX-compatible version for Monte Carlo exposure simulations.
    """
    return jnp.sum(trade_values, axis=0)


def calculate_epe_with_netting(
    trade_values: jnp.ndarray,
    discount_factors: Optional[jnp.ndarray] = None,
) -> float:
    """Calculate Expected Positive Exposure (EPE) with netting.

    Parameters
    ----------
    trade_values : jnp.ndarray
        Array of shape [n_trades, n_paths] with trade values per path
    discount_factors : jnp.ndarray, optional
        Discount factors for each path (shape [n_paths]), default all 1.0

    Returns
    -------
    float
        Expected Positive Exposure with netting benefit

    Notes
    -----
    EPE = E[max(net_exposure, 0)]
    where net_exposure = sum of all trade values on each path
    """
    # Net across trades for each path
    net_values = apply_bilateral_netting_to_paths(trade_values)

    # Take positive part
    positive_exposure = jnp.maximum(net_values, 0.0)

    # Apply discount factors if provided
    if discount_factors is not None:
        positive_exposure = positive_exposure * discount_factors

    # Expected value across paths
    return float(jnp.mean(positive_exposure))


def calculate_netting_factor(
    gross_epe: float,
    net_epe: float,
) -> float:
    """Calculate netting factor (ratio of net to gross EPE).

    Parameters
    ----------
    gross_epe : float
        Gross EPE without netting (sum of individual trade EPEs)
    net_epe : float
        Net EPE with netting benefit

    Returns
    -------
    float
        Netting factor (0-1), where lower values indicate greater netting benefit

    Notes
    -----
    Netting factor = net_epe / gross_epe

    A factor of 0.3 means netting provides a 70% reduction in exposure.
    """
    if gross_epe <= 0.0:
        return 0.0
    return net_epe / gross_epe
