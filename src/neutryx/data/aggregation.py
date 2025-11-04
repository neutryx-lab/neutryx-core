"""Sparse aggregation for efficient counterparty and netting set grouping.

Provides high-performance aggregation functions to sum trade-level exposures
to counterparty/netting set level using sparse matrix operations.
"""

from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.experimental import sparse

from neutryx.data.portfolio_batch import PortfolioBatch


@jax.jit
def aggregate_by_index(
    values: Array,
    indices: Array,
    num_groups: int,
) -> Array:
    """Aggregate values by group indices using segment_sum.

    Parameters
    ----------
    values : Array
        Values to aggregate [n_items]
    indices : Array
        Group index for each value [n_items]
    num_groups : int
        Total number of groups

    Returns
    -------
    Array
        Aggregated values [num_groups]

    Examples
    --------
    >>> # Aggregate 5 trade MTMs to 2 counterparties
    >>> mtms = jnp.array([100.0, 200.0, 150.0, 300.0, 50.0])
    >>> cp_indices = jnp.array([0, 1, 0, 1, 0])  # CP assignments
    >>> cp_exposures = aggregate_by_index(mtms, cp_indices, num_groups=2)
    >>> cp_exposures
    Array([300.0, 500.0], dtype=float32)  # CP0: 100+150+50, CP1: 200+300

    Notes
    -----
    Uses JAX's efficient segment_sum operation, which is optimized for
    both CPU (parallel reduction) and GPU (parallel scan).
    """
    return jax.ops.segment_sum(values, indices, num_segments=num_groups)


@jax.jit
def aggregate_to_counterparties(
    portfolio: PortfolioBatch,
    trade_values: Array,
) -> Array:
    """Aggregate trade-level values to counterparty level.

    Parameters
    ----------
    portfolio : PortfolioBatch
        Portfolio containing counterparty mappings
    trade_values : Array
        Values for each trade (e.g., MTM, delta) [n_trades]

    Returns
    -------
    Array
        Aggregated values by counterparty [n_counterparties]

    Examples
    --------
    >>> # Aggregate MTMs to counterparties
    >>> mtms = price_portfolio_batch(portfolio, market_grid)
    >>> cp_exposures = aggregate_to_counterparties(portfolio, mtms)
    >>> cp_exposures.shape
    (1000,)  # 1K counterparties
    """
    return aggregate_by_index(
        values=trade_values,
        indices=portfolio.counterparty_idx,
        num_groups=portfolio.n_counterparties,
    )


@jax.jit
def compute_counterparty_exposures(
    portfolio: PortfolioBatch,
    trade_mtms: Array,
) -> Tuple[Array, Array]:
    """Compute positive and negative exposures by counterparty.

    Parameters
    ----------
    portfolio : PortfolioBatch
        Portfolio batch
    trade_mtms : Array
        Mark-to-market values [n_trades]

    Returns
    -------
    positive_exposures : Array
        Sum of positive MTMs by counterparty [n_counterparties]
    negative_exposures : Array
        Sum of negative MTMs by counterparty [n_counterparties]

    Examples
    --------
    >>> mtms = price_portfolio_batch(portfolio, market_grid)
    >>> pos_exp, neg_exp = compute_counterparty_exposures(portfolio, mtms)
    >>> net_exp = pos_exp + neg_exp  # Negative values in neg_exp
    """
    # Separate positive and negative MTMs
    positive_mtms = jnp.maximum(trade_mtms, 0.0)
    negative_mtms = jnp.minimum(trade_mtms, 0.0)

    # Aggregate to counterparty level
    positive_exposures = aggregate_to_counterparties(portfolio, positive_mtms)
    negative_exposures = aggregate_to_counterparties(portfolio, negative_mtms)

    return positive_exposures, negative_exposures


@jax.jit
def compute_netting_factors(
    positive_exposures: Array,
    negative_exposures: Array,
) -> Array:
    """Compute netting benefit factors for each counterparty.

    The netting factor measures how much bilateral netting reduces gross exposure.
    Factor = Net Exposure / Gross Exposure, where:
    - Net Exposure = max(Positive + Negative, 0)
    - Gross Exposure = Positive + |Negative|

    Parameters
    ----------
    positive_exposures : Array
        Positive exposures by counterparty [n_counterparties]
    negative_exposures : Array
        Negative exposures by counterparty [n_counterparties]

    Returns
    -------
    Array
        Netting factors [n_counterparties], range [0, 1]

    Notes
    -----
    A factor of:
    - 1.0 = No netting benefit (all one-sided)
    - 0.5 = 50% netting benefit
    - 0.0 = Perfect netting (net exposure = 0)
    """
    net_exposure = jnp.maximum(positive_exposures + negative_exposures, 0.0)
    gross_exposure = positive_exposures + jnp.abs(negative_exposures)

    # Avoid division by zero
    netting_factor = jnp.where(
        gross_exposure > 0,
        net_exposure / gross_exposure,
        0.0,
    )

    return netting_factor


@jax.jit
def compute_gross_exposure(
    portfolio: PortfolioBatch,
    trade_mtms: Array,
) -> Array:
    """Compute gross exposure (sum of absolute values) by counterparty.

    Parameters
    ----------
    portfolio : PortfolioBatch
        Portfolio batch
    trade_mtms : Array
        Trade MTMs [n_trades]

    Returns
    -------
    Array
        Gross exposures [n_counterparties]

    Examples
    --------
    >>> gross_exp = compute_gross_exposure(portfolio, mtms)
    >>> total_gross = jnp.sum(gross_exp)
    """
    abs_mtms = jnp.abs(trade_mtms)
    return aggregate_to_counterparties(portfolio, abs_mtms)


@jax.jit
def compute_top_n_counterparties(
    portfolio: PortfolioBatch,
    trade_values: Array,
    n: int = 10,
    by_absolute: bool = True,
) -> Tuple[Array, Array]:
    """Identify top N counterparties by exposure.

    Parameters
    ----------
    portfolio : PortfolioBatch
        Portfolio batch
    trade_values : Array
        Trade-level values (MTMs, deltas, etc.)
    n : int, optional
        Number of top counterparties to return (default: 10)
    by_absolute : bool, optional
        If True, rank by absolute values. If False, rank by signed values

    Returns
    -------
    top_indices : Array
        Counterparty indices of top N [n]
    top_values : Array
        Exposure values of top N [n]

    Examples
    --------
    >>> # Find top 10 counterparties by gross exposure
    >>> top_cp_idx, top_exposures = compute_top_n_counterparties(
    ...     portfolio, mtms, n=10
    ... )
    >>> # Decode counterparty IDs
    >>> top_cp_ids = portfolio.counterparty_mapping.decode(top_cp_idx)
    """
    # Aggregate to counterparty level
    cp_exposures = aggregate_to_counterparties(portfolio, trade_values)

    # Rank by absolute or signed values
    if by_absolute:
        rank_values = jnp.abs(cp_exposures)
    else:
        rank_values = cp_exposures

    # Get top N indices
    # Note: jnp.argsort sorts in ascending order, so take last N
    all_indices = jnp.argsort(rank_values)
    top_indices = all_indices[-n:][::-1]  # Reverse to descending order

    top_values = cp_exposures[top_indices]

    return top_indices, top_values


@jax.jit
def compute_concentration_metrics(
    portfolio: PortfolioBatch,
    trade_values: Array,
) -> dict:
    """Compute portfolio concentration metrics.

    Parameters
    ----------
    portfolio : PortfolioBatch
        Portfolio batch
    trade_values : Array
        Trade-level values

    Returns
    -------
    dict
        Concentration metrics:
        - herfindahl_index: Sum of squared market shares
        - top_5_concentration: Fraction of total in top 5 counterparties
        - top_10_concentration: Fraction of total in top 10 counterparties
        - max_concentration: Largest single counterparty fraction

    Notes
    -----
    Herfindahl Index ranges from 1/N (perfectly diversified) to 1 (single counterparty).
    """
    # Aggregate to counterparties
    cp_exposures = jnp.abs(aggregate_to_counterparties(portfolio, trade_values))
    total_exposure = jnp.sum(cp_exposures)

    # Market shares
    shares = cp_exposures / (total_exposure + 1e-10)

    # Herfindahl Index
    herfindahl = jnp.sum(shares**2)

    # Top N concentrations
    sorted_shares = jnp.sort(shares)[::-1]  # Descending
    top_5 = jnp.sum(sorted_shares[:5])
    top_10 = jnp.sum(sorted_shares[:10])
    max_concentration = sorted_shares[0]

    return {
        "herfindahl_index": float(herfindahl),
        "top_5_concentration": float(top_5),
        "top_10_concentration": float(top_10),
        "max_concentration": float(max_concentration),
    }


@jax.jit
def filter_portfolio_by_counterparty_indices(
    portfolio: PortfolioBatch,
    cp_indices: Array,
) -> Tuple[Array, Array]:
    """Get mask and indices for trades belonging to specified counterparties.

    Parameters
    ----------
    portfolio : PortfolioBatch
        Portfolio batch
    cp_indices : Array
        Counterparty indices to filter [n_selected_cps]

    Returns
    -------
    mask : Array
        Boolean mask for trades [n_trades]
    trade_indices : Array
        Indices of matching trades

    Examples
    --------
    >>> # Get all trades for top 10 counterparties
    >>> top_cp_idx, _ = compute_top_n_counterparties(portfolio, mtms, n=10)
    >>> mask, trade_idx = filter_portfolio_by_counterparty_indices(
    ...     portfolio, top_cp_idx
    ... )
    >>> # Slice portfolio
    >>> top_10_portfolio = portfolio.slice_trades(
    ...     int(trade_idx[0]), int(trade_idx[-1]) + 1
    ... )
    """
    # Create mask: True for trades belonging to any of the specified counterparties
    mask = jnp.isin(portfolio.counterparty_idx, cp_indices)
    trade_indices = jnp.where(mask)[0]

    return mask, trade_indices
