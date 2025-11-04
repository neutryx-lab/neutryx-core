"""Tests for batch pricing functionality."""

import jax.numpy as jnp
import pytest

from neutryx.compute import price_vanilla_options_batch
from neutryx.data import (
    TradeArrays,
    PortfolioBatch,
    IndexMapping,
    build_market_data_grid_simple,
    build_time_grid,
)


def test_price_vanilla_options_batch_basic():
    """Test basic batch pricing functionality."""
    # Create simple portfolio
    trade_arrays = TradeArrays(
        spots=jnp.array([100.0, 100.0]),
        strikes=jnp.array([100.0, 100.0]),
        maturities=jnp.array([1.0, 1.0]),
        notionals=jnp.array([1.0, 1.0]),
        option_types=jnp.array([0, 1]),  # ATM call, ATM put
    )

    currency_mapping = IndexMapping.from_values(["USD"])
    cp_mapping = IndexMapping.from_values(["CP1"])
    product_mapping = IndexMapping.from_values(["VanillaOption"])

    portfolio = PortfolioBatch(
        trade_arrays=trade_arrays,
        currency_idx=jnp.array([0, 0]),
        counterparty_idx=jnp.array([0, 0]),
        product_type_idx=jnp.array([0, 0]),
        currency_mapping=currency_mapping,
        counterparty_mapping=cp_mapping,
        product_type_mapping=product_mapping,
    )

    # Create market grid
    time_grid = build_time_grid(5.0, n_steps=100)
    market_grid = build_market_data_grid_simple(
        time_grid,
        currencies=["USD"],
        assets=["SPX"],
        flat_rate=0.05,
        flat_vol=0.20,
    )

    # Price portfolio
    prices = price_vanilla_options_batch(
        portfolio, market_grid, use_notional=False
    )

    assert prices.shape == (2,)
    # Check put-call parity: C - P = S*e^(-qT) - K*e^(-rT)
    # For S=K=100, r=0.05, q=0, T=1: C - P ≈ 4.88
    expected_diff = 100.0 * jnp.exp(-0.0 * 1.0) - 100.0 * jnp.exp(-0.05 * 1.0)
    assert jnp.abs((prices[0] - prices[1]) - expected_diff) < 0.5  # Within $0.50


def test_batch_pricing_multiple_currencies():
    """Test batch pricing with multiple currencies."""
    # Create portfolio with different currencies
    trade_arrays = TradeArrays(
        spots=jnp.array([100.0, 100.0, 100.0]),
        strikes=jnp.array([110.0, 110.0, 110.0]),
        maturities=jnp.array([1.0, 1.0, 1.0]),
        notionals=jnp.array([1e6, 1e6, 1e6]),
        option_types=jnp.array([0, 0, 0]),  # All calls
    )

    currency_mapping = IndexMapping.from_values(["EUR", "GBP", "USD"])
    cp_mapping = IndexMapping.from_values(["CP1"])
    product_mapping = IndexMapping.from_values(["VanillaOption"])

    portfolio = PortfolioBatch(
        trade_arrays=trade_arrays,
        currency_idx=jnp.array([0, 1, 2]),  # Different currencies
        counterparty_idx=jnp.array([0, 0, 0]),
        product_type_idx=jnp.array([0, 0, 0]),
        currency_mapping=currency_mapping,
        counterparty_mapping=cp_mapping,
        product_type_mapping=product_mapping,
    )

    # Create market grid
    time_grid = build_time_grid(5.0, n_steps=100)
    market_grid = build_market_data_grid_simple(
        time_grid,
        currencies=["EUR", "GBP", "USD"],
        assets=["SPX"],
        flat_rate=0.05,
        flat_vol=0.20,
    )

    # Price portfolio
    prices = price_vanilla_options_batch(portfolio, market_grid, use_notional=True)

    assert prices.shape == (3,)
    # All prices should be positive
    assert jnp.all(prices > 0)


def test_batch_pricing_large_portfolio():
    """Test batch pricing scales to large portfolios."""
    n_trades = 1000

    # Create large portfolio
    trade_arrays = TradeArrays(
        spots=jnp.full(n_trades, 100.0),
        strikes=jnp.full(n_trades, 110.0),
        maturities=jnp.linspace(0.25, 5.0, n_trades),
        notionals=jnp.full(n_trades, 1e6),
        option_types=jnp.zeros(n_trades, dtype=jnp.int32),
    )

    currency_mapping = IndexMapping.from_values(["USD"])
    cp_mapping = IndexMapping.from_values([f"CP{i}" for i in range(100)])
    product_mapping = IndexMapping.from_values(["VanillaOption"])

    portfolio = PortfolioBatch(
        trade_arrays=trade_arrays,
        currency_idx=jnp.zeros(n_trades, dtype=jnp.int32),
        counterparty_idx=jnp.arange(n_trades, dtype=jnp.int32) % 100,
        product_type_idx=jnp.zeros(n_trades, dtype=jnp.int32),
        currency_mapping=currency_mapping,
        counterparty_mapping=cp_mapping,
        product_type_mapping=product_mapping,
    )

    # Create market grid
    time_grid = build_time_grid(5.0, n_steps=600)
    market_grid = build_market_data_grid_simple(
        time_grid,
        currencies=["USD"],
        assets=["SPX"],
        flat_rate=0.05,
        flat_vol=0.20,
    )

    # Price portfolio (should complete without error)
    prices = price_vanilla_options_batch(portfolio, market_grid)

    assert prices.shape == (n_trades,)
    assert jnp.all(prices > 0)
