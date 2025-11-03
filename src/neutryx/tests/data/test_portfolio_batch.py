"""Tests for PortfolioBatch and related data structures."""

import jax.numpy as jnp
import pytest

from neutryx.data import (
    IndexMapping,
    PortfolioBatch,
    TradeArrays,
    trades_to_portfolio_batch,
)


def test_index_mapping_basic():
    """Test basic IndexMapping functionality."""
    mapping = IndexMapping.from_values(["USD", "EUR", "GBP"])

    assert len(mapping) == 3
    assert mapping.encode("USD") == 0
    assert mapping.encode("EUR") == 1
    assert mapping.encode("GBP") == 2

    indices = mapping.encode_batch(["EUR", "USD", "GBP"])
    assert jnp.array_equal(indices, jnp.array([1, 0, 2]))

    decoded = mapping.decode([2, 0, 1])
    assert decoded == ["GBP", "USD", "EUR"]


def test_trade_arrays_creation():
    """Test TradeArrays creation and validation."""
    arrays = TradeArrays(
        spots=jnp.array([100.0, 105.0]),
        strikes=jnp.array([110.0, 115.0]),
        maturities=jnp.array([1.0, 2.0]),
        notionals=jnp.array([1e6, 2e6]),
        option_types=jnp.array([0, 1]),  # call, put
    )

    assert arrays.n_trades == 2
    assert len(arrays) == 2


def test_trade_arrays_validation():
    """Test TradeArrays shape validation."""
    with pytest.raises(ValueError, match="Inconsistent array shapes"):
        TradeArrays(
            spots=jnp.array([100.0, 105.0]),
            strikes=jnp.array([110.0]),  # Wrong shape
            maturities=jnp.array([1.0, 2.0]),
            notionals=jnp.array([1e6, 2e6]),
            option_types=jnp.array([0, 1]),
        )


def test_portfolio_batch_creation():
    """Test PortfolioBatch creation."""
    trade_arrays = TradeArrays(
        spots=jnp.array([100.0, 105.0, 110.0]),
        strikes=jnp.array([110.0, 115.0, 120.0]),
        maturities=jnp.array([1.0, 2.0, 0.5]),
        notionals=jnp.array([1e6, 2e6, 5e5]),
        option_types=jnp.array([0, 0, 1]),
    )

    currency_mapping = IndexMapping.from_values(["EUR", "USD"])
    cp_mapping = IndexMapping.from_values(["CP1", "CP2"])
    product_mapping = IndexMapping.from_values(["VanillaOption"])

    portfolio = PortfolioBatch(
        trade_arrays=trade_arrays,
        currency_idx=jnp.array([0, 1, 0]),  # EUR, USD, EUR
        counterparty_idx=jnp.array([0, 0, 1]),  # CP1, CP1, CP2
        product_type_idx=jnp.array([0, 0, 0]),
        currency_mapping=currency_mapping,
        counterparty_mapping=cp_mapping,
        product_type_mapping=product_mapping,
    )

    assert portfolio.n_trades == 3
    assert portfolio.n_currencies == 2
    assert portfolio.n_counterparties == 2
    assert len(portfolio) == 3


def test_trades_to_portfolio_batch():
    """Test conversion from trade dicts to PortfolioBatch."""
    trades = [
        {
            "spot": 100.0,
            "strike": 110.0,
            "maturity": 1.0,
            "notional": 1e6,
            "option_type": "call",
            "currency": "USD",
            "counterparty_id": "CP1",
            "product_type": "VanillaOption",
        },
        {
            "spot": 105.0,
            "strike": 115.0,
            "maturity": 2.0,
            "notional": 2e6,
            "option_type": "put",
            "currency": "EUR",
            "counterparty_id": "CP1",
            "product_type": "VanillaOption",
        },
    ]

    portfolio = trades_to_portfolio_batch(trades)

    assert portfolio.n_trades == 2
    assert portfolio.n_currencies == 2  # EUR, USD
    assert portfolio.n_counterparties == 1  # CP1
    assert float(portfolio.trade_arrays.spots[0]) == 100.0
    assert float(portfolio.trade_arrays.strikes[1]) == 115.0


def test_portfolio_slice():
    """Test portfolio slicing."""
    trades = [
        {
            "spot": 100.0 + i,
            "strike": 110.0,
            "maturity": 1.0,
            "notional": 1e6,
            "option_type": "call",
            "currency": "USD",
            "counterparty_id": "CP1",
            "product_type": "VanillaOption",
        }
        for i in range(10)
    ]

    portfolio = trades_to_portfolio_batch(trades)
    sliced = portfolio.slice_trades(0, 5)

    assert sliced.n_trades == 5
    assert float(sliced.trade_arrays.spots[0]) == 100.0
    assert float(sliced.trade_arrays.spots[4]) == 104.0


def test_portfolio_filter_by_counterparty():
    """Test filtering by counterparty."""
    trades = [
        {
            "spot": 100.0,
            "strike": 110.0,
            "maturity": 1.0,
            "notional": 1e6,
            "option_type": "call",
            "currency": "USD",
            "counterparty_id": f"CP{i % 2}",
            "product_type": "VanillaOption",
        }
        for i in range(10)
    ]

    portfolio = trades_to_portfolio_batch(trades)
    cp0_portfolio = portfolio.filter_by_counterparty("CP0")

    assert cp0_portfolio.n_trades == 5  # Half the trades
    assert cp0_portfolio.n_counterparties == 1
