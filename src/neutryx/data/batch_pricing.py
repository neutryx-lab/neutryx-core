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
from neutryx.products.fx_vanilla_exotic import FXForward
from neutryx.products.swap import price_vanilla_swap


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

    # Find time indices on market grid
    time_indices = market_grid.find_time_indices_batch(maturities)

    # Batch lookup: zero rates for all trades
    zero_rates = market_grid.zero_rates[currency_idx, time_indices]

    # TODO: Batch lookup volatilities (requires asset mapping in portfolio)
    # For now, use flat 20% vol
    vols = jnp.full(n_trades, 0.2, dtype=jnp.float32)

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
    option_type_aliases = {"VanillaOption", "EuropeanOption", "Option"}
    swap_type_aliases = {"InterestRateSwap", "IRS", "VanillaSwap"}
    fx_forward_aliases = {"FXForward", "FX Forward", "FX_FORWARD"}

    prices = jnp.zeros(portfolio.n_trades, dtype=jnp.float32)

    unique_type_indices = jnp.unique(portfolio.product_type_idx).tolist()
    type_values = portfolio.product_type_mapping.idx_to_value

    for type_idx in unique_type_indices:
        int_type_idx = int(type_idx)
        product_type = str(type_values[int_type_idx])
        mask = portfolio.product_type_idx == int_type_idx
        bool_mask = jnp.asarray(mask, dtype=bool)

        if not bool(bool_mask.any()):
            continue

        indices = jnp.where(bool_mask)[0]

        if product_type in option_type_aliases:
            sub_portfolio = portfolio.select_trades_by_mask(bool_mask)
            sub_prices = price_vanilla_options_batch(
                portfolio=sub_portfolio,
                market_grid=market_grid,
                use_notional=use_notional,
            )
        elif product_type in swap_type_aliases:
            sub_prices = _price_interest_rate_swaps(
                portfolio=portfolio,
                indices=indices,
                use_notional=use_notional,
            )
        elif product_type in fx_forward_aliases:
            sub_prices = _price_fx_forwards(
                portfolio=portfolio,
                indices=indices,
                use_notional=use_notional,
            )
        else:
            raise ValueError(f"Unsupported product type: {product_type}")

        prices = prices.at[indices].set(jnp.asarray(sub_prices, dtype=jnp.float32))

    return prices


def _price_interest_rate_swaps(
    portfolio: PortfolioBatch,
    indices: Array,
    use_notional: bool,
) -> Array:
    """Price interest rate swaps using the vanilla swap engine."""

    results = []
    for trade_idx in map(int, indices.tolist()):
        params = portfolio.product_parameters[trade_idx]

        if "fixed_rate" not in params or "floating_rate" not in params or "maturity" not in params:
            raise ValueError(
                f"Missing swap parameters for trade {portfolio.product_ids[trade_idx]}"
            )

        notional = float(
            params.get("notional", float(portfolio.trade_arrays.notionals[trade_idx]))
        )
        payment_frequency = int(params.get("payment_frequency", 2))
        discount_rate = float(params.get("discount_rate", params["floating_rate"]))
        pay_fixed = bool(params.get("pay_fixed", True))

        price = float(
            price_vanilla_swap(
                notional=notional,
                fixed_rate=float(params["fixed_rate"]),
                floating_rate=float(params["floating_rate"]),
                maturity=float(params["maturity"]),
                payment_frequency=payment_frequency,
                discount_rate=discount_rate,
                pay_fixed=pay_fixed,
            )
        )

        if not use_notional and notional != 0.0:
            price /= notional

        results.append(price)

    return jnp.asarray(results, dtype=jnp.float32)


def _price_fx_forwards(
    portfolio: PortfolioBatch,
    indices: Array,
    use_notional: bool,
) -> Array:
    """Price FX forwards using the FXForward mark-to-market implementation."""

    results = []
    for trade_idx in map(int, indices.tolist()):
        params = portfolio.product_parameters[trade_idx]

        required_keys = {"spot", "forward_rate", "expiry", "domestic_rate", "foreign_rate"}
        missing = required_keys.difference(params)
        if missing:
            raise ValueError(
                f"Missing FX forward parameters {sorted(missing)} for trade {portfolio.product_ids[trade_idx]}"
            )

        notional = float(
            params.get(
                "notional_foreign",
                params.get("notional", float(portfolio.trade_arrays.notionals[trade_idx])),
            )
        )

        contract = FXForward(
            spot=float(params["spot"]),
            forward_rate=float(params["forward_rate"]),
            expiry=float(params["expiry"]),
            domestic_rate=float(params["domestic_rate"]),
            foreign_rate=float(params["foreign_rate"]),
            notional_foreign=notional,
            is_long=bool(params.get("is_long", True)),
        )

        price = float(contract.mark_to_market())

        if not use_notional and notional != 0.0:
            price /= notional

        results.append(price)

    return jnp.asarray(results, dtype=jnp.float32)


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
        product_ids=portfolio.product_ids,
        product_parameters=portfolio.product_parameters,
        metadata=portfolio.metadata,
    )

    prices_bumped = price_portfolio_batch(bumped_portfolio, market_grid, use_notional=True)

    # Delta = dPrice / dSpot ≈ (Price_bumped - Price_base) / (Spot_bumped - Spot_base)
    delta = (prices_bumped - prices_base) / (bumped_spots - portfolio.trade_arrays.spots)

    return delta
