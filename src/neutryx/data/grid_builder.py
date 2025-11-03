"""Utilities to build MarketDataGrid from various sources.

Provides functions to convert legacy MarketDataEnvironment and other formats
into pre-computed grids for high-performance batch operations.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import jax.numpy as jnp
from jax import Array

from neutryx.data.indices import IndexMapping, build_index_mapping
from neutryx.data.market_grid import MarketDataGrid


def build_time_grid(
    max_maturity: float,
    n_steps: int = 600,
    min_time: float = 0.0,
    spacing: str = "linear",
) -> Array:
    """Build a time grid for market data pre-computation.

    Parameters
    ----------
    max_maturity : float
        Maximum time horizon in years
    n_steps : int, optional
        Number of time steps (default: 600)
    min_time : float, optional
        Starting time (default: 0.0)
    spacing : {"linear", "sqrt", "log"}, optional
        Grid spacing strategy:
        - "linear": Uniform spacing
        - "sqrt": Denser near origin (sqrt spacing)
        - "log": Logarithmic spacing (requires min_time > 0)

    Returns
    -------
    Array
        Time grid [n_steps]

    Examples
    --------
    >>> # Linear grid for 5 years, 600 steps
    >>> time_grid = build_time_grid(5.0, n_steps=600)
    >>> time_grid.shape
    (600,)
    >>> float(time_grid[-1])
    5.0

    >>> # Square-root spacing (denser near origin)
    >>> time_grid = build_time_grid(5.0, n_steps=600, spacing="sqrt")
    """
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}")
    if max_maturity <= min_time:
        raise ValueError(f"max_maturity {max_maturity} must be > min_time {min_time}")

    if spacing == "linear":
        return jnp.linspace(min_time, max_maturity, n_steps, dtype=jnp.float32)

    elif spacing == "sqrt":
        # Square-root spacing: denser near origin
        sqrt_max = jnp.sqrt(max_maturity - min_time)
        sqrt_grid = jnp.linspace(0, sqrt_max, n_steps, dtype=jnp.float32)
        return min_time + sqrt_grid**2

    elif spacing == "log":
        # Logarithmic spacing
        if min_time <= 0:
            raise ValueError("log spacing requires min_time > 0")
        return jnp.logspace(
            jnp.log10(min_time),
            jnp.log10(max_maturity),
            n_steps,
            dtype=jnp.float32,
        )

    else:
        raise ValueError(f"Unknown spacing: {spacing}")


def build_market_data_grid_simple(
    time_grid: Array,
    currencies: Sequence[str],
    assets: Sequence[str],
    flat_rate: float = 0.05,
    flat_vol: float = 0.2,
    n_strikes: int = 21,
    strike_range: tuple[float, float] = (50.0, 150.0),
) -> MarketDataGrid:
    """Build a simple market data grid with flat curves and surfaces.

    Useful for testing and prototyping.

    Parameters
    ----------
    time_grid : Array
        Time grid [n_times]
    currencies : Sequence[str]
        List of currency codes
    assets : Sequence[str]
        List of asset identifiers
    flat_rate : float, optional
        Constant interest rate for all currencies (default: 5%)
    flat_vol : float, optional
        Constant volatility for all assets (default: 20%)
    n_strikes : int, optional
        Number of strike points (default: 21)
    strike_range : tuple[float, float], optional
        Strike grid range (default: [50, 150])

    Returns
    -------
    MarketDataGrid
        Pre-computed market data grid

    Examples
    --------
    >>> time_grid = build_time_grid(5.0, n_steps=600)
    >>> grid = build_market_data_grid_simple(
    ...     time_grid,
    ...     currencies=["USD", "EUR"],
    ...     assets=["SPX", "SX5E"],
    ... )
    >>> grid.n_currencies
    2
    >>> grid.n_assets
    2
    """
    n_times = time_grid.shape[0]
    n_currencies = len(currencies)
    n_assets = len(assets)

    # Build mappings
    currency_mapping = build_index_mapping(currencies, name="currency")
    asset_mapping = build_index_mapping(assets, name="asset")
    fx_pair_mapping = IndexMapping.from_values([])  # Empty for simplicity

    # Build strike grid
    strike_grid = jnp.linspace(strike_range[0], strike_range[1], n_strikes, dtype=jnp.float32)

    # Build flat discount factors: DF(t) = exp(-r * t)
    discount_factors = jnp.exp(-flat_rate * time_grid[None, :])  # [1, n_times]
    discount_factors = jnp.broadcast_to(discount_factors, (n_currencies, n_times))

    # Flat zero rates
    zero_rates = jnp.full((n_currencies, n_times), flat_rate, dtype=jnp.float32)

    # Flat forward rates
    forward_rates = jnp.full((n_currencies, n_times), flat_rate, dtype=jnp.float32)

    # Flat volatilities
    implied_vols = jnp.full((n_assets, n_times, n_strikes), flat_vol, dtype=jnp.float32)

    # Empty FX spots
    fx_spots = jnp.array([], dtype=jnp.float32)

    return MarketDataGrid(
        time_grid=time_grid,
        discount_factors=discount_factors,
        zero_rates=zero_rates,
        forward_rates=forward_rates,
        implied_vols=implied_vols,
        strike_grid=strike_grid,
        fx_spots=fx_spots,
        currency_mapping=currency_mapping,
        asset_mapping=asset_mapping,
        fx_pair_mapping=fx_pair_mapping,
        metadata={
            "source": "simple_flat",
            "flat_rate": flat_rate,
            "flat_vol": flat_vol,
        },
    )


def build_market_data_grid_from_environment(
    market_env: Any,
    time_grid: Array,
    strike_grid: Array,
) -> MarketDataGrid:
    """Convert MarketDataEnvironment to pre-computed MarketDataGrid.

    Parameters
    ----------
    market_env : MarketDataEnvironment
        Legacy market data environment
    time_grid : Array
        Time grid for pre-computation [n_times]
    strike_grid : Array
        Strike grid for volatility surfaces [n_strikes]

    Returns
    -------
    MarketDataGrid
        Pre-computed market data grid

    Notes
    -----
    This function batch-evaluates all curves and surfaces from the legacy
    MarketDataEnvironment onto the specified grids, eliminating runtime
    interpolation overhead.

    Examples
    --------
    >>> from neutryx.market.environment import MarketDataEnvironment
    >>> from neutryx.market.curves import FlatCurve
    >>>
    >>> # Legacy environment
    >>> env = MarketDataEnvironment(
    ...     discount_curves={"USD": FlatCurve(r=0.05)},
    ...     vol_surfaces={},
    ... )
    >>>
    >>> # Convert to grid
    >>> time_grid = build_time_grid(5.0, n_steps=600)
    >>> strike_grid = jnp.linspace(80, 120, 21)
    >>> grid = build_market_data_grid_from_environment(env, time_grid, strike_grid)
    """
    n_times = time_grid.shape[0]
    n_strikes = strike_grid.shape[0]

    # Extract currencies
    currencies = sorted(market_env.discount_curves.keys())
    currency_mapping = build_index_mapping(currencies, name="currency")
    n_currencies = len(currencies)

    # Pre-compute discount factors
    discount_factors_list = []
    zero_rates_list = []

    for currency in currencies:
        curve = market_env.discount_curves[currency]

        # Batch evaluate discount factors
        dfs = jnp.array([curve.discount_factor(float(t)) for t in time_grid], dtype=jnp.float32)
        discount_factors_list.append(dfs)

        # Compute zero rates from discount factors: r(t) = -ln(DF(t)) / t
        # Handle t=0 case
        zero_rates = jnp.where(
            time_grid > 0,
            -jnp.log(dfs) / time_grid,
            0.0,  # Convention: zero rate at t=0 is 0
        )
        zero_rates_list.append(zero_rates)

    discount_factors = jnp.stack(discount_factors_list, axis=0)  # [n_currencies, n_times]
    zero_rates = jnp.stack(zero_rates_list, axis=0)

    # Compute forward rates (finite difference approximation)
    # f(t) ≈ -d(ln DF)/dt
    forward_rates = jnp.zeros_like(zero_rates)
    for i in range(n_currencies):
        # Forward difference for interior points
        dt = jnp.diff(time_grid)
        dln_df = jnp.diff(jnp.log(discount_factors[i, :]))
        fwd = -dln_df / dt
        # Extrapolate to last point
        forward_rates = forward_rates.at[i, :-1].set(fwd)
        forward_rates = forward_rates.at[i, -1].set(fwd[-1])

    # Pre-compute volatilities
    assets = sorted(market_env.vol_surfaces.keys()) if market_env.vol_surfaces else []
    asset_mapping = build_index_mapping(assets, name="asset") if assets else IndexMapping.from_values([])
    n_assets = len(assets)

    if n_assets > 0:
        implied_vols_list = []

        for asset in assets:
            surface = market_env.vol_surfaces[asset]
            vols_2d = []

            for t in time_grid:
                vols_1d = []
                for k in strike_grid:
                    try:
                        vol = surface.implied_vol(float(t), float(k))
                    except Exception:
                        # Fallback to flat vol if surface evaluation fails
                        vol = 0.2
                    vols_1d.append(vol)
                vols_2d.append(vols_1d)

            implied_vols_list.append(vols_2d)

        implied_vols = jnp.array(implied_vols_list, dtype=jnp.float32)  # [n_assets, n_times, n_strikes]
    else:
        # No assets: create empty vol array
        implied_vols = jnp.zeros((0, n_times, n_strikes), dtype=jnp.float32)

    # Extract FX spots
    fx_pairs = sorted(market_env.fx_spots.keys()) if hasattr(market_env, "fx_spots") else []
    fx_pair_mapping = build_index_mapping(fx_pairs, name="fx_pair") if fx_pairs else IndexMapping.from_values([])

    if fx_pairs:
        fx_spots = jnp.array([market_env.fx_spots[pair] for pair in fx_pairs], dtype=jnp.float32)
    else:
        fx_spots = jnp.array([], dtype=jnp.float32)

    return MarketDataGrid(
        time_grid=time_grid,
        discount_factors=discount_factors,
        zero_rates=zero_rates,
        forward_rates=forward_rates,
        implied_vols=implied_vols,
        strike_grid=strike_grid,
        fx_spots=fx_spots,
        currency_mapping=currency_mapping,
        asset_mapping=asset_mapping,
        fx_pair_mapping=fx_pair_mapping,
        metadata={
            "source": "market_environment_conversion",
            "n_original_curves": len(currencies),
            "n_original_surfaces": len(assets),
        },
    )


def estimate_grid_memory_size(
    n_times: int,
    n_currencies: int,
    n_assets: int,
    n_strikes: int,
    dtype_size: int = 4,  # float32
) -> dict[str, float]:
    """Estimate memory footprint of a MarketDataGrid.

    Parameters
    ----------
    n_times : int
        Number of time points
    n_currencies : int
        Number of currencies
    n_assets : int
        Number of assets
    n_strikes : int
        Number of strike points
    dtype_size : int, optional
        Bytes per float (4 for float32, 8 for float64)

    Returns
    -------
    dict[str, float]
        Dictionary with memory estimates in KB

    Examples
    --------
    >>> estimate_grid_memory_size(
    ...     n_times=600, n_currencies=10, n_assets=5, n_strikes=21
    ... )
    {'time_grid': 2.34, 'discount_factors': 23.44, 'zero_rates': 23.44, ...}
    """
    mem = {}
    mem["time_grid_kb"] = n_times * dtype_size / 1024
    mem["discount_factors_kb"] = n_currencies * n_times * dtype_size / 1024
    mem["zero_rates_kb"] = n_currencies * n_times * dtype_size / 1024
    mem["forward_rates_kb"] = n_currencies * n_times * dtype_size / 1024
    mem["implied_vols_kb"] = n_assets * n_times * n_strikes * dtype_size / 1024
    mem["strike_grid_kb"] = n_strikes * dtype_size / 1024
    mem["total_kb"] = sum(mem.values())
    mem["total_mb"] = mem["total_kb"] / 1024

    return mem
