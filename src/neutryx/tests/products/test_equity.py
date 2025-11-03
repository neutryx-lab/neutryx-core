"""Tests for equity products."""
import jax.numpy as jnp
import pytest

from neutryx.products.equity import (
    dividend_swap_value,
    equity_forward_price,
    equity_forward_value,
    equity_linked_note_price,
    total_return_swap_value,
    variance_swap_value,
    volatility_swap_convexity_adjustment,
)


def test_equity_forward_price():
    """Test equity forward pricing."""
    spot = 100.0
    maturity = 1.0
    risk_free_rate = 0.05
    dividend_yield = 0.02

    forward = equity_forward_price(spot, maturity, risk_free_rate, dividend_yield)

    # Expected: 100 * exp((0.05 - 0.02) * 1) = 103.045
    expected = 103.045
    assert abs(forward - expected) < 0.01


def test_equity_forward_value():
    """Test equity forward valuation."""
    spot = 100.0
    strike = 98.0
    maturity = 1.0
    risk_free_rate = 0.05
    dividend_yield = 0.02

    value_long = equity_forward_value(spot, strike, maturity, risk_free_rate, dividend_yield, "long")
    value_short = equity_forward_value(
        spot, strike, maturity, risk_free_rate, dividend_yield, "short"
    )

    # Long and short should be opposites
    assert abs(value_long + value_short) < 1e-6

    # Long forward should have positive value when forward > strike
    assert value_long > 0


def test_dividend_swap_value():
    """Test dividend swap valuation."""
    notional = 1000.0
    strike = 5.0
    expected_dividends = 6.5
    discount_factor = 0.95

    value = dividend_swap_value(notional, strike, expected_dividends, discount_factor)

    # Expected: 1000 * (6.5 - 5.0) * 0.95 = 1425
    expected = 1425.0
    assert abs(value - expected) < 0.01


def test_variance_swap_value():
    """Test variance swap valuation."""
    notional = 10_000.0
    strike = 0.04  # 20% vol squared
    realized_variance = 0.035
    expected_variance = 0.050  # Changed to make it non-zero
    discount_factor = 0.95

    value = variance_swap_value(
        notional, strike, realized_variance, expected_variance, discount_factor
    )

    # Average variance: (0.035 + 0.050) / 2 = 0.0425
    # Value: 10000 * (0.0425 - 0.04) * 0.95 = 23.75
    expected = 23.75
    assert abs(value - expected) < 0.1


def test_volatility_swap_convexity_adjustment():
    """Test volatility swap convexity adjustment."""
    variance_strike = 0.04  # 20% vol squared

    vol_strike = volatility_swap_convexity_adjustment(variance_strike)

    # sqrt(0.04) = 0.2, adjustment = 0.04/(8*0.2) = 0.025
    # Result: 0.2 - 0.025 = 0.175
    # Should be less than sqrt(variance_strike) due to convexity
    assert vol_strike < jnp.sqrt(variance_strike)
    assert vol_strike > 0.17  # Approximately 17.5%
    assert vol_strike < 0.18


def test_total_return_swap_value():
    """Test total return swap valuation."""
    notional = 1_000_000
    spot_initial = 100.0
    spot_current = 105.0
    dividends = 3.0
    funding_rate = 0.03
    time_elapsed = 0.5
    discount_factor = 0.985

    value = total_return_swap_value(
        notional,
        spot_initial,
        spot_current,
        dividends,
        funding_rate,
        time_elapsed,
        discount_factor,
    )

    # Equity return: (105/100 - 1) + 3/100 = 0.05 + 0.03 = 0.08
    # Funding cost: 0.03 * 0.5 = 0.015
    # Net: 0.08 - 0.015 = 0.065
    # Value: 1M * 0.065 * 0.985 = 64,025
    assert value > 60_000
    assert value < 70_000


def test_equity_linked_note_price():
    """Test equity-linked note pricing."""
    principal = 100_000
    participation = 0.8
    spot_initial = 100.0
    spot_current = 120.0
    maturity = 1.0
    risk_free_rate = 0.05

    # With floor=0 (principal protection)
    value = equity_linked_note_price(
        principal, participation, spot_initial, spot_current, maturity, risk_free_rate, floor=0.0
    )

    # Equity return: 20%, participated: 16%, principal protected
    # Should be approximately 116,000 discounted
    assert value > 110_000
    assert value < 120_000


def test_equity_linked_note_with_cap():
    """Test equity-linked note with cap."""
    principal = 100_000
    participation = 1.0
    spot_initial = 100.0
    spot_current = 150.0  # 50% return
    maturity = 1.0
    risk_free_rate = 0.05
    cap = 0.25  # Cap at 25%

    value = equity_linked_note_price(
        principal,
        participation,
        spot_initial,
        spot_current,
        maturity,
        risk_free_rate,
        floor=0.0,
        cap=cap,
    )

    # Return capped at 25%, so terminal value = 125,000
    # Discounted: 125,000 * exp(-0.05) ≈ 118,940
    assert value < 120_000
    assert value > 115_000
