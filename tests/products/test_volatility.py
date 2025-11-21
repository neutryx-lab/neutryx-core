"""Tests for volatility products."""
import jax.numpy as jnp

from neutryx.products.volatility import (
    conditional_variance_swap_value,
    gamma_swap_payoff,
    realized_variance,
    variance_swap_payoff,
    vix_futures_price,
    vix_option_price,
    volatility_swap_payoff,
    vvix_index,
)


def test_vix_futures_price_simple():
    """Test VIX futures pricing without mean reversion."""
    vix_spot = 20.0
    maturity = 0.5

    futures_price = vix_futures_price(vix_spot, maturity, 0.0, 0.0, 0.0)

    # Without mean reversion, futures = spot
    assert abs(futures_price - vix_spot) < 1e-6


def test_vix_futures_price_mean_reversion():
    """Test VIX futures pricing with mean reversion."""
    vix_spot = 25.0
    maturity = 0.5
    mean_reversion = 2.0
    long_term_vol = 20.0

    futures_price = vix_futures_price(vix_spot, maturity, 0.0, mean_reversion, long_term_vol)

    # With mean reversion, futures should be between spot and long-term
    assert futures_price < vix_spot
    assert futures_price > long_term_vol


def test_vix_option_price():
    """Test VIX option pricing."""
    vix_futures = 20.0
    strike = 22.0
    maturity = 0.5
    vix_volatility = 0.80

    call_price = vix_option_price(vix_futures, strike, maturity, vix_volatility, "call")
    put_price = vix_option_price(vix_futures, strike, maturity, vix_volatility, "put")

    # Both should be positive
    assert call_price > 0
    assert put_price > 0

    # Put should be more valuable when strike > futures
    assert put_price > call_price


def test_realized_variance():
    """Test realized variance calculation."""
    prices = jnp.array([100.0, 101.0, 99.5, 102.0, 101.5, 103.0])

    realized_var = realized_variance(prices, annualization_factor=252.0)

    # Should be positive
    assert realized_var > 0

    # Should be reasonable (not extreme)
    assert realized_var < 1.0  # Less than 100% variance


def test_variance_swap_payoff():
    """Test variance swap payoff calculation."""
    notional = 10_000
    strike = 0.04  # 20% vol
    realized_var = 0.045  # 21.2% vol

    payoff = variance_swap_payoff(notional, strike, realized_var)

    # Expected: 10,000 * (0.045 - 0.04) = 50
    expected = 50.0
    assert abs(payoff - expected) < 0.01


def test_volatility_swap_payoff():
    """Test volatility swap payoff calculation."""
    notional = 1_000_000
    strike = 0.20  # 20% vol
    realized_vol = 0.25  # 25% vol

    payoff = volatility_swap_payoff(notional, strike, realized_vol)

    # Expected: 1M * (0.25 - 0.20) = 50,000
    expected = 50_000.0
    assert abs(payoff - expected) < 0.01


def test_conditional_variance_swap_value():
    """Test conditional variance swap valuation."""
    notional = 10_000
    strike = 0.04
    expected_variance = 0.045
    time_in_corridor = 0.75
    discount_factor = 0.95

    value = conditional_variance_swap_value(
        notional, strike, expected_variance, time_in_corridor, discount_factor
    )

    # Expected: 10,000 * 0.75 * (0.045 - 0.04) * 0.95 = 35.625
    expected = 35.625
    assert abs(value - expected) < 0.01


def test_gamma_swap_payoff():
    """Test gamma swap payoff calculation."""
    notional = 1_000_000
    prices = jnp.array([100.0, 102.0, 99.0, 101.0, 103.0])

    payoff = gamma_swap_payoff(notional, prices)

    # Sum of absolute log returns
    # |ln(102/100)| + |ln(99/102)| + |ln(101/99)| + |ln(103/101)|
    # Should be positive
    assert payoff > 0
    assert payoff < 100_000  # Reasonable upper bound


def test_vvix_index():
    """Test VVIX calculation."""
    strikes = jnp.array([15.0, 18.0, 20.0, 22.0, 25.0])
    calls = jnp.array([7.0, 4.5, 3.0, 2.0, 1.0])
    puts = jnp.array([1.0, 2.0, 3.0, 4.5, 7.0])
    vix_spot = 20.0
    maturity = 1.0 / 12.0  # 1 month

    vvix = vvix_index(strikes, calls, puts, vix_spot, maturity)

    # VVIX should be positive and typically 60-120
    assert vvix > 0
    assert vvix < 200


def test_realized_variance_constant_price():
    """Test realized variance with constant prices."""
    prices = jnp.array([100.0, 100.0, 100.0, 100.0])

    realized_var = realized_variance(prices, annualization_factor=252.0)

    # Should be zero (or very close) for constant prices
    assert realized_var < 1e-6


def test_realized_variance_increases_with_volatility():
    """Test that realized variance increases with price volatility."""
    # Low volatility
    prices_low = jnp.array([100.0, 100.5, 99.8, 100.2, 99.9])

    # High volatility
    prices_high = jnp.array([100.0, 105.0, 95.0, 102.0, 98.0])

    var_low = realized_variance(prices_low, 252.0)
    var_high = realized_variance(prices_high, 252.0)

    # High volatility prices should have higher realized variance
    assert var_high > var_low


def test_variance_swap_payoff_symmetry():
    """Test that variance swap payoff is symmetric for long/short."""
    notional = 10_000
    strike = 0.04
    realized_var = 0.045

    payoff_long = variance_swap_payoff(notional, strike, realized_var)
    payoff_short = variance_swap_payoff(-notional, strike, realized_var)

    # Long and short should be opposites
    assert abs(payoff_long + payoff_short) < 1e-6
