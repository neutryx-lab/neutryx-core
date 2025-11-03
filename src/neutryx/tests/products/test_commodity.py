"""Tests for commodity products."""
import jax.numpy as jnp

from neutryx.products.commodity import (
    asian_commodity_price,
    commodity_forward_price,
    commodity_forward_value,
    commodity_option_price,
    commodity_swap_value,
    multi_period_commodity_swap_value,
    spread_option_price,
)


def test_commodity_forward_price():
    """Test commodity forward pricing with convenience yield."""
    spot = 50.0
    maturity = 1.0
    risk_free_rate = 0.05
    storage_cost = 0.02
    convenience_yield = 0.03

    forward = commodity_forward_price(spot, maturity, risk_free_rate, storage_cost, convenience_yield)

    # Expected: 50 * exp((0.05 + 0.02 - 0.03) * 1) ≈ 52.04
    expected = 52.04
    assert abs(forward - expected) < 0.05


def test_commodity_forward_value():
    """Test commodity forward valuation."""
    spot = 50.0
    strike = 48.0
    maturity = 1.0
    risk_free_rate = 0.05
    storage_cost = 0.02
    convenience_yield = 0.03

    value_long = commodity_forward_value(
        spot, strike, maturity, risk_free_rate, storage_cost, convenience_yield, "long"
    )
    value_short = commodity_forward_value(
        spot, strike, maturity, risk_free_rate, storage_cost, convenience_yield, "short"
    )

    # Long and short should be opposites
    assert abs(value_long + value_short) < 1e-6

    # Long forward should have positive value when forward > strike
    assert value_long > 0


def test_commodity_option_price():
    """Test commodity option pricing."""
    spot = 50.0
    strike = 52.0
    maturity = 1.0
    risk_free_rate = 0.05
    volatility = 0.30
    storage_cost = 0.02
    convenience_yield = 0.03

    call_price = commodity_option_price(
        spot, strike, maturity, risk_free_rate, volatility, storage_cost, convenience_yield, "call"
    )

    put_price = commodity_option_price(
        spot, strike, maturity, risk_free_rate, volatility, storage_cost, convenience_yield, "put"
    )

    # Both should be positive
    assert call_price > 0
    assert put_price > 0

    # Put-call parity check (approximate due to cost of carry)
    # Call - Put ≈ PV(Forward - Strike)
    forward = commodity_forward_price(spot, maturity, risk_free_rate, storage_cost, convenience_yield)
    discount_factor = jnp.exp(-risk_free_rate * maturity)
    parity_diff = (forward - strike) * discount_factor

    assert abs((call_price - put_price) - parity_diff) < 0.5


def test_commodity_swap_value():
    """Test commodity swap valuation."""
    quantity = 1000.0
    fixed_price = 50.0
    floating_price = 55.0
    discount_factor = 0.95

    value_fixed_payer = commodity_swap_value(
        quantity, fixed_price, floating_price, discount_factor, "fixed_payer"
    )

    value_fixed_receiver = commodity_swap_value(
        quantity, fixed_price, floating_price, discount_factor, "fixed_receiver"
    )

    # Expected: 1000 * (55 - 50) * 0.95 = 4750
    expected = 4750.0
    assert abs(value_fixed_payer - expected) < 0.01

    # Fixed payer and receiver should be opposites
    assert abs(value_fixed_payer + value_fixed_receiver) < 1e-6


def test_multi_period_commodity_swap():
    """Test multi-period commodity swap."""
    quantity = 1000.0
    fixed_price = 50.0
    floating_prices = jnp.array([52.0, 54.0, 53.0])
    discount_factors = jnp.array([0.98, 0.96, 0.94])

    value = multi_period_commodity_swap_value(
        quantity, fixed_price, floating_prices, discount_factors, "fixed_payer"
    )

    # Period 1: 1000 * (52-50) * 0.98 = 1960
    # Period 2: 1000 * (54-50) * 0.96 = 3840
    # Period 3: 1000 * (53-50) * 0.94 = 2820
    # Total: 8620
    expected = 8620.0
    assert abs(value - expected) < 1.0


def test_spread_option_price():
    """Test spread option pricing."""
    spot1 = 50.0
    spot2 = 45.0
    strike = 3.0
    maturity = 1.0
    risk_free_rate = 0.05
    vol1 = 0.25
    vol2 = 0.30
    correlation = 0.6

    call_price = spread_option_price(
        spot1, spot2, strike, maturity, risk_free_rate, vol1, vol2, correlation, "call"
    )

    put_price = spread_option_price(
        spot1, spot2, strike, maturity, risk_free_rate, vol1, vol2, correlation, "put"
    )

    # Both should be positive
    assert call_price > 0
    assert put_price > 0

    # Call should be more valuable than put when spot spread > strike
    # Current spread: 50 - 45 = 5, strike = 3
    assert call_price > put_price


def test_asian_commodity_price():
    """Test Asian commodity option pricing."""
    spot = 50.0
    strike = 50.0
    maturity = 1.0
    risk_free_rate = 0.05
    volatility = 0.30
    num_fixings = 12

    call_price = asian_commodity_price(
        spot, strike, maturity, risk_free_rate, volatility, num_fixings, "call"
    )

    put_price = asian_commodity_price(
        spot, strike, maturity, risk_free_rate, volatility, num_fixings, "put"
    )

    # Both should be positive for ATM options
    assert call_price > 0
    assert put_price > 0

    # Asian options should be cheaper than vanilla due to reduced volatility
    # (We'd need to compare with vanilla option to verify this)
    assert call_price < 10.0  # Reasonable upper bound
