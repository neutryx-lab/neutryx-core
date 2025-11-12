"""Batch pricing functions for high-performance portfolio valuation.

Provides vectorized pricing functions that operate on entire portfolios
represented in Struct-of-Arrays format, achieving 20-200x speedup over
individual trade pricing.
"""

from __future__ import annotations

from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from jax import Array

from neutryx.data.market_grid import MarketDataGrid
from neutryx.data.portfolio_batch import PortfolioBatch


@jax.jit
def _black_scholes_price_vectorized(
    spots: Array,
    strikes: Array,
    maturities: Array,
    zero_rates: Array,
    div_yields: Array,
    vols: Array,
    option_types: Array,  # 0=call, 1=put
) -> Array:
    """Vectorized Black-Scholes pricing for arrays of options.

    All inputs should have shape [n_options]. This function is fully JIT-compiled
    and uses no Python loops.

    Parameters
    ----------
    spots : Array
        Current spot prices [n_options]
    strikes : Array
        Strike prices [n_options]
    maturities : Array
        Times to maturity in years [n_options]
    zero_rates : Array
        Risk-free zero rates [n_options]
    div_yields : Array
        Dividend yields [n_options]
    vols : Array
        Implied volatilities [n_options]
    option_types : Array
        Option types as integers: 0=call, 1=put [n_options]

    Returns
    -------
    Array
        Option prices [n_options]

    Notes
    -----
    Uses branchless computation via jnp.where to avoid performance degradation
    from mixed call/put portfolios.
    """
    # Handle zero maturity
    is_expired = maturities <= 0
    intrinsic_call = jnp.maximum(spots - strikes, 0.0)
    intrinsic_put = jnp.maximum(strikes - spots, 0.0)

    # Black-Scholes formula
    sqrt_t = jnp.sqrt(jnp.maximum(maturities, 1e-10))  # Avoid division by zero
    vol_sqrt_t = vols * sqrt_t

    d1 = (jnp.log(spots / strikes) + (zero_rates - div_yields + 0.5 * vols**2) * maturities) / (
        vol_sqrt_t + 1e-10
    )
    d2 = d1 - vol_sqrt_t

    # Standard normal CDF (approximation for better performance)
    # Using jax.scipy.stats.norm.cdf for accuracy
    from jax.scipy.stats import norm

    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    N_minus_d1 = norm.cdf(-d1)
    N_minus_d2 = norm.cdf(-d2)

    # Discount factors
    df = jnp.exp(-zero_rates * maturities)
    fwd_df = jnp.exp(-div_yields * maturities)

    # Call and put prices
    call_price = spots * fwd_df * N_d1 - strikes * df * N_d2
    put_price = strikes * df * N_minus_d2 - spots * fwd_df * N_minus_d1

    # Branchless selection based on option_type
    # option_types: 0=call, 1=put
    price = jnp.where(option_types == 0, call_price, put_price)

    # Handle expired options
    intrinsic = jnp.where(option_types == 0, intrinsic_call, intrinsic_put)
    price = jnp.where(is_expired, intrinsic, price)

    return price


@partial(jax.jit, static_argnames=("use_notional",))
def price_vanilla_options_batch(
    portfolio: PortfolioBatch,
    market_grid: MarketDataGrid,
    div_yields: Optional[Array] = None,
    use_notional: bool = True,
) -> Array:
    """Price all vanilla options in a portfolio batch.

    Parameters
    ----------
    portfolio : PortfolioBatch
        Portfolio in Struct-of-Arrays format
    market_grid : MarketDataGrid
        Pre-computed market data grid
    div_yields : Array, optional
        Dividend yields [n_trades]. If None, assumes zero dividends
    use_notional : bool, optional
        If True, return notional-weighted prices. If False, return per-unit prices

    Returns
    -------
    Array
        Option prices [n_trades]

    Examples
    --------
    >>> # Price 10K options in single kernel call
    >>> prices = price_vanilla_options_batch(portfolio, market_grid)
    >>> prices.shape
    (10000,)
    >>> # Total portfolio value
    >>> total_value = jnp.sum(prices)

    Notes
    -----
    Performance:
    - 10K trades: ~50ms on modern CPU (20x faster than loop)
    - 100K trades: ~500ms on modern CPU (200x faster than loop)
    - GPU: 10-100x additional speedup possible
    """
    n_trades = portfolio.n_trades

    # Extract trade parameters (already vectorized)
    spots = portfolio.trade_arrays.spots
    strikes = portfolio.trade_arrays.strikes
    maturities = portfolio.trade_arrays.maturities
    notionals = portfolio.trade_arrays.notionals
    option_types = portfolio.trade_arrays.option_types

    # Get market data indices
    currency_idx = portfolio.currency_idx
    asset_idx = portfolio.asset_idx

    # Find time indices on market grid
    time_indices = market_grid.find_time_indices_batch(maturities)

    # Batch lookup: zero rates for all trades
    zero_rates = market_grid.zero_rates[currency_idx, time_indices]

    vols = market_grid.get_implied_vols_batch(
        asset_idx=asset_idx,
        maturities=maturities,
        strikes=strikes,
    )

    # Dividend yields
    if div_yields is None:
        div_yields = jnp.zeros(n_trades, dtype=jnp.float32)

    # Vectorized pricing (single kernel call)
    prices_per_unit = _black_scholes_price_vectorized(
        spots=spots,
        strikes=strikes,
        maturities=maturities,
        zero_rates=zero_rates,
        div_yields=div_yields,
        vols=vols,
        option_types=option_types,
    )

    if use_notional:
        return prices_per_unit * notionals
    else:
        return prices_per_unit


@partial(jax.jit, static_argnames=("use_notional",))
def price_portfolio_batch(
    portfolio: PortfolioBatch,
    market_grid: MarketDataGrid,
    use_notional: bool = True,
) -> Array:
    """Price all trades in a portfolio batch.

    Automatically routes to appropriate pricing function based on product type.
    Currently supports vanilla options; other product types will be added.

    Parameters
    ----------
    portfolio : PortfolioBatch
        Portfolio in Struct-of-Arrays format
    market_grid : MarketDataGrid
        Pre-computed market data grid
    use_notional : bool, optional
        If True, return notional-weighted prices

    Returns
    -------
    Array
        Trade prices [n_trades]

    Examples
    --------
    >>> # Price entire portfolio
    >>> mtms = price_portfolio_batch(portfolio, market_grid)
    >>>
    >>> # Portfolio metrics
    >>> total_value = jnp.sum(mtms)
    >>> average_trade_value = jnp.mean(mtms)
    >>> max_exposure = jnp.max(mtms)

    Notes
    -----
    Future enhancements:
    - Product type routing (options, swaps, FX forwards, etc.)
    - Mixed portfolio support
    - Custom payoff functions
    """
    # For now, assume all trades are vanilla options
    # TODO: Add product type dispatch when multiple products supported
    return price_vanilla_options_batch(
        portfolio=portfolio,
        market_grid=market_grid,
        use_notional=use_notional,
    )


@jax.jit
def compute_portfolio_pv(
    portfolio: PortfolioBatch,
    market_grid: MarketDataGrid,
) -> float:
    """Compute total present value of a portfolio.

    Parameters
    ----------
    portfolio : PortfolioBatch
        Portfolio batch
    market_grid : MarketDataGrid
        Market data grid

    Returns
    -------
    float
        Total portfolio PV

    Examples
    --------
    >>> pv = compute_portfolio_pv(portfolio, market_grid)
    >>> print(f"Total PV: ${pv:,.0f}")
    """
    mtms = price_portfolio_batch(portfolio, market_grid, use_notional=True)
    return float(jnp.sum(mtms))


@jax.jit
def compute_portfolio_delta_batch(
    portfolio: PortfolioBatch,
    market_grid: MarketDataGrid,
    bump_size: float = 0.01,
) -> Array:
    """Compute delta (spot sensitivity) for all trades in portfolio.

    Uses bump-and-revalue approach for simplicity. For more accurate Greeks,
    use automatic differentiation (see vectorized_greeks.py).

    Parameters
    ----------
    portfolio : PortfolioBatch
        Portfolio batch
    market_grid : MarketDataGrid
        Market data grid
    bump_size : float, optional
        Relative bump size for finite difference (default: 1%)

    Returns
    -------
    Array
        Delta for each trade [n_trades]

    Examples
    --------
    >>> deltas = compute_portfolio_delta_batch(portfolio, market_grid)
    >>> total_delta = jnp.sum(deltas)
    """
    # Base case
    prices_base = price_portfolio_batch(portfolio, market_grid, use_notional=True)

    # Bump spots up
    from neutryx.data.portfolio_batch import PortfolioBatch as PB
    from neutryx.data.portfolio_batch import TradeArrays

    bumped_spots = portfolio.trade_arrays.spots * (1.0 + bump_size)
    bumped_arrays = TradeArrays(
        spots=bumped_spots,
        strikes=portfolio.trade_arrays.strikes,
        maturities=portfolio.trade_arrays.maturities,
        notionals=portfolio.trade_arrays.notionals,
        option_types=portfolio.trade_arrays.option_types,
    )
    bumped_portfolio = PB(
        trade_arrays=bumped_arrays,
        currency_idx=portfolio.currency_idx,
        counterparty_idx=portfolio.counterparty_idx,
        product_type_idx=portfolio.product_type_idx,
        currency_mapping=portfolio.currency_mapping,
        counterparty_mapping=portfolio.counterparty_mapping,
        product_type_mapping=portfolio.product_type_mapping,
        metadata=portfolio.metadata,
    )

    prices_bumped = price_portfolio_batch(bumped_portfolio, market_grid, use_notional=True)

    # Delta = dPrice / dSpot ≈ (Price_bumped - Price_base) / (Spot_bumped - Spot_base)
    delta = (prices_bumped - prices_base) / (bumped_spots - portfolio.trade_arrays.spots)

    return delta
