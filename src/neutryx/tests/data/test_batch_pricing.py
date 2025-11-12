"""Tests for batch pricing functionality."""

import jax.numpy as jnp
import pytest

from neutryx.data import (
    IndexMapping,
    PortfolioBatch,
    TradeArrays,
    build_market_data_grid_simple,
    build_time_grid,
    price_portfolio_batch,
    price_vanilla_options_batch,
)
from neutryx.products.fx_vanilla_exotic import FXForward
from neutryx.products.swap import price_vanilla_swap


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
    asset_mapping = IndexMapping.from_values(["SPX"])

    portfolio = PortfolioBatch(
        trade_arrays=trade_arrays,
        currency_idx=jnp.array([0, 0]),
        counterparty_idx=jnp.array([0, 0]),
        product_type_idx=jnp.array([0, 0]),
        asset_idx=jnp.array([0, 0]),
        currency_mapping=currency_mapping,
        counterparty_mapping=cp_mapping,
        product_type_mapping=product_mapping,
        asset_mapping=asset_mapping,
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
    asset_mapping = IndexMapping.from_values(["SPX"])

    portfolio = PortfolioBatch(
        trade_arrays=trade_arrays,
        currency_idx=jnp.array([0, 1, 2]),  # Different currencies
        counterparty_idx=jnp.array([0, 0, 0]),
        product_type_idx=jnp.array([0, 0, 0]),
        asset_idx=jnp.array([0, 0, 0]),
        currency_mapping=currency_mapping,
        counterparty_mapping=cp_mapping,
        product_type_mapping=product_mapping,
        asset_mapping=asset_mapping,
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
    asset_mapping = IndexMapping.from_values(["SPX"])

    portfolio = PortfolioBatch(
        trade_arrays=trade_arrays,
        currency_idx=jnp.zeros(n_trades, dtype=jnp.int32),
        counterparty_idx=jnp.arange(n_trades, dtype=jnp.int32) % 100,
        product_type_idx=jnp.zeros(n_trades, dtype=jnp.int32),
        asset_idx=jnp.zeros(n_trades, dtype=jnp.int32),
        currency_mapping=currency_mapping,
        counterparty_mapping=cp_mapping,
        product_type_mapping=product_mapping,
        asset_mapping=asset_mapping,
    )

    time_grid = build_time_grid(5.0, n_steps=600)
    market_grid = build_market_data_grid_simple(
        time_grid,
        currencies=["USD"],
        assets=["SPX"],
        flat_rate=0.05,
        flat_vol=0.20,
    )

    prices = price_vanilla_options_batch(portfolio, market_grid)

    assert prices.shape == (n_trades,)
    assert jnp.all(prices > 0)


def test_batch_pricing_responds_to_vol_surface_changes():
    """Verify that pricing responds to changes in market volatility inputs."""

    trade_arrays = TradeArrays(
        spots=jnp.array([100.0]),
        strikes=jnp.array([105.0]),
        maturities=jnp.array([1.25]),
        notionals=jnp.array([1.0]),
        option_types=jnp.array([0]),
    )

    currency_mapping = IndexMapping.from_values(["USD"])
    cp_mapping = IndexMapping.from_values(["CP1"])
    product_mapping = IndexMapping.from_values(["VanillaOption"])
    asset_mapping = IndexMapping.from_values(["SPX"])

    portfolio = PortfolioBatch(
        trade_arrays=trade_arrays,
        currency_idx=jnp.array([0]),
        counterparty_idx=jnp.array([0]),
        product_type_idx=jnp.array([0]),
        asset_idx=jnp.array([0]),
        currency_mapping=currency_mapping,
        counterparty_mapping=cp_mapping,
        product_type_mapping=product_mapping,
        asset_mapping=asset_mapping,
    )

    time_grid = build_time_grid(2.0, n_steps=10)

    grid_low_vol = build_market_data_grid_simple(
        time_grid,
        currencies=["USD"],
        assets=["SPX"],
        flat_rate=0.02,
        flat_vol=0.15,
    )

    grid_high_vol = build_market_data_grid_simple(
        time_grid,
        currencies=["USD"],
        assets=["SPX"],
        flat_rate=0.02,
        flat_vol=0.35,
    )

    price_low = price_vanilla_options_batch(portfolio, grid_low_vol, use_notional=False)
    price_high = price_vanilla_options_batch(portfolio, grid_high_vol, use_notional=False)

    assert price_high.shape == (1,)
    assert float(price_high[0]) > float(price_low[0])

    # Ensure that the interpolated volatility feeds directly into pricing
    assert float(price_low[0]) > 0
def test_price_portfolio_batch_mixed_products():
    """Mixed portfolio should route to the appropriate product engines."""

    time_grid = build_time_grid(5.0, n_steps=128)
    market_grid = build_market_data_grid_simple(
        time_grid,
        currencies=["USD"],
        assets=["SPX"],
        flat_rate=0.05,
        flat_vol=0.20,
    )

    trade_arrays = TradeArrays(
        spots=jnp.array([100.0, 0.0, 1.10], dtype=jnp.float32),
        strikes=jnp.array([105.0, 0.0, 1.12], dtype=jnp.float32),
        maturities=jnp.array([1.0, 5.0, 1.0], dtype=jnp.float32),
        notionals=jnp.array([1e6, 5e6, 1e6], dtype=jnp.float32),
        option_types=jnp.array([0, -1, -1], dtype=jnp.int32),
    )

    currency_mapping = IndexMapping.from_values(["USD"])
    cp_mapping = IndexMapping.from_values(["CP1", "CP2"])
    product_mapping = IndexMapping.from_values(
        ["VanillaOption", "InterestRateSwap", "FXForward"]
    )

    product_parameters = (
        {},
        {
            "notional": 5_000_000.0,
            "fixed_rate": 0.05,
            "floating_rate": 0.045,
            "maturity": 5.0,
            "payment_frequency": 2,
            "discount_rate": 0.05,
            "pay_fixed": True,
        },
        {
            "spot": 1.10,
            "forward_rate": 1.12,
            "expiry": 1.0,
            "domestic_rate": 0.05,
            "foreign_rate": 0.02,
            "notional_foreign": 1_000_000.0,
            "is_long": True,
        },
    )

    portfolio = PortfolioBatch(
        trade_arrays=trade_arrays,
        currency_idx=jnp.array([0, 0, 0], dtype=jnp.int32),
        counterparty_idx=jnp.array([0, 1, 1], dtype=jnp.int32),
        product_type_idx=jnp.array([0, 1, 2], dtype=jnp.int32),
        currency_mapping=currency_mapping,
        counterparty_mapping=cp_mapping,
        product_type_mapping=product_mapping,
        product_ids=("OPT-1", "IRS-1", "FXFWD-1"),
        product_parameters=product_parameters,
    )

    prices = price_portfolio_batch(portfolio, market_grid, use_notional=True)
    assert prices.shape == (3,)

    option_portfolio = portfolio.select_trades_by_mask(jnp.array([True, False, False]))
    expected_option = price_vanilla_options_batch(
        option_portfolio, market_grid, use_notional=True
    )[0]
    expected_swap = price_vanilla_swap(**product_parameters[1])
    expected_fx = FXForward(**product_parameters[2]).mark_to_market()

    assert jnp.isclose(prices[0], expected_option, rtol=1e-5)
    assert jnp.isclose(prices[1], expected_swap, rtol=1e-5)
    assert jnp.isclose(prices[2], expected_fx, rtol=1e-5)

    per_unit = price_portfolio_batch(portfolio, market_grid, use_notional=False)
    assert jnp.isclose(per_unit[0], expected_option / 1e6, rtol=1e-5)
    assert jnp.isclose(per_unit[1], expected_swap / product_parameters[1]["notional"], rtol=1e-5)
    assert jnp.isclose(
        per_unit[2], expected_fx / product_parameters[2]["notional_foreign"], rtol=1e-5
    )
