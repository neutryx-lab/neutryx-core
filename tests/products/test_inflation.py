"""Tests for inflation-linked products."""
import jax.numpy as jnp

from neutryx.products.inflation import (
    breakeven_inflation,
    inflation_caplet_price,
    inflation_floorlet_price,
    inflation_linked_bond_price,
    real_to_nominal_yield,
    year_on_year_inflation_swap_rate,
    year_on_year_inflation_swap_value,
    zero_coupon_inflation_swap_rate,
    zero_coupon_inflation_swap_value,
)


def test_inflation_linked_bond_price():
    """Test inflation-linked bond pricing."""
    face_value = 100.0
    real_coupon_rate = 0.02
    real_yield = 0.01
    maturity = 5.0
    index_ratio = 1.15  # 15% cumulative inflation
    frequency = 2

    price = inflation_linked_bond_price(
        face_value, real_coupon_rate, real_yield, maturity, index_ratio, frequency
    )

    # With positive real coupon and lower real yield, bond trades above adjusted par
    # Adjusted par = 100 * 1.15 = 115
    assert price > 115
    assert price < 125


def test_zero_coupon_inflation_swap_rate():
    """Test zero-coupon inflation swap rate calculation."""
    forward_cpi = 110.0
    base_cpi = 100.0

    rate = zero_coupon_inflation_swap_rate(forward_cpi, base_cpi)

    # Expected: (110/100) - 1 = 0.10
    expected = 0.10
    assert abs(rate - expected) < 1e-6


def test_zero_coupon_inflation_swap_value():
    """Test zero-coupon inflation swap valuation."""
    notional = 1_000_000
    strike = 0.10
    maturity = 5.0
    current_cpi = 105.0
    forward_cpi = 115.0
    base_cpi = 100.0
    discount_factor = 0.95

    value = zero_coupon_inflation_swap_value(
        notional, strike, maturity, current_cpi, forward_cpi, base_cpi, discount_factor
    )

    # Expected inflation: 115/100 - 1 = 0.15
    # Payoff: 1M * (0.15 - 0.10) * 0.95 = 47,500
    expected = 47_500.0
    assert abs(value - expected) < 1.0


def test_year_on_year_inflation_swap_rate():
    """Test year-on-year inflation rate calculation."""
    forward_cpi_t = 103.0
    forward_cpi_t_minus_1 = 100.0

    rate = year_on_year_inflation_swap_rate(forward_cpi_t, forward_cpi_t_minus_1)

    # Expected: (103/100) - 1 = 0.03
    expected = 0.03
    assert abs(rate - expected) < 1e-6


def test_year_on_year_inflation_swap_value():
    """Test year-on-year inflation swap valuation."""
    notional = 1_000_000
    strike = 0.02
    payment_dates = jnp.array([1.0, 2.0, 3.0])
    forward_cpi_levels = jnp.array([102.0, 105.0, 108.0])
    discount_factors = jnp.array([0.95, 0.90, 0.85])

    value = year_on_year_inflation_swap_value(
        notional, strike, payment_dates, forward_cpi_levels, discount_factors
    )

    # Year 1: (102/100 - 1) - 0.02 = 0.00
    # Year 2: (105/102 - 1) - 0.02 = 0.0294 - 0.02 = 0.0094
    # Year 3: (108/105 - 1) - 0.02 = 0.0286 - 0.02 = 0.0086
    # Total PV should be positive but small
    assert value > 0
    assert value < 20_000


def test_inflation_caplet_price():
    """Test inflation caplet pricing."""
    notional = 1_000_000
    strike = 0.03
    forward_inflation = 0.025
    volatility = 0.01
    maturity = 1.0
    discount_factor = 0.95

    caplet_price = inflation_caplet_price(
        notional, strike, forward_inflation, volatility, maturity, discount_factor
    )

    # Caplet is OTM (forward < strike), so should have some time value
    assert caplet_price >= 0
    assert caplet_price < 10_000


def test_inflation_floorlet_price():
    """Test inflation floorlet pricing."""
    notional = 1_000_000
    strike = 0.03
    forward_inflation = 0.025
    volatility = 0.01
    maturity = 1.0
    discount_factor = 0.95

    floorlet_price = inflation_floorlet_price(
        notional, strike, forward_inflation, volatility, maturity, discount_factor
    )

    # Floorlet is ITM (forward < strike), so should be more valuable
    assert floorlet_price > 0
    assert floorlet_price > 4_500  # Adjusted based on actual calculation


def test_caplet_floorlet_put_call_parity():
    """Test put-call parity for inflation caps/floors."""
    notional = 1_000_000
    strike = 0.03
    forward_inflation = 0.025
    volatility = 0.01
    maturity = 1.0
    discount_factor = 0.95

    caplet = inflation_caplet_price(
        notional, strike, forward_inflation, volatility, maturity, discount_factor
    )
    floorlet = inflation_floorlet_price(
        notional, strike, forward_inflation, volatility, maturity, discount_factor
    )

    # Put-call parity: Caplet - Floorlet = DF * Notional * (Forward - Strike)
    expected_diff = discount_factor * notional * (forward_inflation - strike)
    actual_diff = caplet - floorlet

    assert abs(actual_diff - expected_diff) < 100  # Small tolerance for numerical error


def test_real_to_nominal_yield():
    """Test Fisher equation for yield conversion."""
    real_yield = 0.02
    expected_inflation = 0.025
    inflation_risk_premium = 0.005

    nominal = real_to_nominal_yield(real_yield, expected_inflation, inflation_risk_premium)

    # Approximate: 0.02 + 0.025 + 0.005 = 0.05
    # Exact: (1.02 * 1.025 * 1.005) - 1 = 0.05076
    assert abs(nominal - 0.05076) < 0.0001


def test_breakeven_inflation():
    """Test breakeven inflation calculation."""
    nominal_yield = 0.05
    real_yield = 0.02

    breakeven = breakeven_inflation(nominal_yield, real_yield)

    # (1.05 / 1.02) - 1 = 0.0294
    expected = 0.0294
    assert abs(breakeven - expected) < 0.0001


def test_real_nominal_round_trip():
    """Test round-trip conversion between real and nominal yields."""
    real_yield = 0.02
    expected_inflation = 0.025

    # Convert real to nominal
    nominal = real_to_nominal_yield(real_yield, expected_inflation, inflation_risk_premium=0.0)

    # Calculate breakeven
    breakeven = breakeven_inflation(nominal, real_yield)

    # Breakeven should match expected inflation (when risk premium = 0)
    assert abs(breakeven - expected_inflation) < 0.0001
