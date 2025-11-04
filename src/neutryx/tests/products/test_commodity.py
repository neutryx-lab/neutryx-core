"""Tests for commodity products."""
import jax.numpy as jnp
import pytest

from neutryx.products.commodity import (
    AgricultureFuture,
    CommodityAsianOption,
    CommodityForward,
    CommoditySwap,
    CommodityType,
    EnergyFuture,
    MetalFuture,
    asian_commodity_price,
    basis_swap_value,
    commodity_forward_price,
    commodity_forward_value,
    commodity_option_price,
    commodity_swap_value,
    multi_period_commodity_swap_value,
    price_commodity_asian_arithmetic,
    price_commodity_asian_geometric,
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


# ========== Tests for Sector-Specific Futures ==========


def test_energy_future_wti():
    """Test WTI crude oil future."""
    future = EnergyFuture(
        spot=75.0,
        strike=72.0,
        maturity=0.5,
        risk_free_rate=0.04,
        energy_type=CommodityType.WTI_CRUDE,
    )

    forward_price = future.forward_price()

    # With default WTI parameters: storage_cost=0.02, convenience_yield=0.03
    # Forward = 75 * exp((0.04 + 0.02 - 0.03) * 0.5)
    expected = 75.0 * jnp.exp(0.03 * 0.5)
    assert abs(forward_price - expected) < 0.1


def test_energy_future_natural_gas():
    """Test natural gas future."""
    future = EnergyFuture(
        spot=3.5,
        strike=3.2,
        maturity=1.0,
        risk_free_rate=0.04,
        energy_type=CommodityType.NATURAL_GAS,
    )

    forward_price = future.forward_price()

    # Natural gas has higher storage costs and convenience yields
    # storage_cost=0.05, convenience_yield=0.08
    # Forward = 3.5 * exp((0.04 + 0.05 - 0.08) * 1.0)
    expected = 3.5 * jnp.exp(0.01 * 1.0)
    assert abs(forward_price - expected) < 0.01


def test_energy_future_invalid_type():
    """Test that non-energy commodity type raises error."""
    with pytest.raises(ValueError):
        EnergyFuture(
            spot=1500.0,
            strike=1480.0,
            maturity=1.0,
            risk_free_rate=0.03,
            energy_type=CommodityType.GOLD,  # Invalid: gold is not energy
        )


def test_metal_future_gold():
    """Test gold future."""
    future = MetalFuture(
        spot=1950.0,
        strike=1900.0,
        maturity=1.0,
        risk_free_rate=0.03,
        metal_type=CommodityType.GOLD,
    )

    forward_price = future.forward_price()

    # Gold has low storage costs and minimal convenience yield
    # storage_cost=0.005, convenience_yield=0.001
    # Forward = 1950 * exp((0.03 + 0.005 - 0.001) * 1.0)
    expected = 1950.0 * jnp.exp(0.034 * 1.0)
    assert abs(forward_price - expected) < 1.0


def test_metal_future_copper():
    """Test copper future."""
    future = MetalFuture(
        spot=4.2,
        strike=4.0,
        maturity=0.5,
        risk_free_rate=0.04,
        metal_type=CommodityType.COPPER,
    )

    forward_price = future.forward_price()

    # Copper: storage_cost=0.02, convenience_yield=0.03
    expected = 4.2 * jnp.exp((0.04 + 0.02 - 0.03) * 0.5)
    assert abs(forward_price - expected) < 0.01


def test_metal_future_invalid_type():
    """Test that non-metal commodity type raises error."""
    with pytest.raises(ValueError):
        MetalFuture(
            spot=50.0,
            strike=48.0,
            maturity=1.0,
            risk_free_rate=0.04,
            metal_type=CommodityType.WTI_CRUDE,  # Invalid: crude is not metal
        )


def test_agriculture_future_corn():
    """Test corn future."""
    future = AgricultureFuture(
        spot=5.8,
        strike=5.5,
        maturity=0.5,
        risk_free_rate=0.04,
        agriculture_type=CommodityType.CORN,
    )

    forward_price = future.forward_price()

    # Corn: storage_cost=0.06, convenience_yield=0.04
    # Forward = 5.8 * exp((0.04 + 0.06 - 0.04) * 0.5) * seasonal_factor
    expected = 5.8 * jnp.exp(0.06 * 0.5) * 1.0  # seasonal_factor=1.0
    assert abs(forward_price - expected) < 0.05


def test_agriculture_future_with_seasonal_factor():
    """Test agriculture future with seasonal adjustment."""
    future = AgricultureFuture(
        spot=5.8,
        strike=5.5,
        maturity=0.5,
        risk_free_rate=0.04,
        agriculture_type=CommodityType.WHEAT,
        seasonal_factor=1.15,  # 15% seasonal premium
    )

    forward_price = future.forward_price()

    # Wheat: storage_cost=0.05, convenience_yield=0.04
    base_forward = 5.8 * jnp.exp((0.04 + 0.05 - 0.04) * 0.5)
    expected = base_forward * 1.15
    assert abs(forward_price - expected) < 0.05


def test_agriculture_future_invalid_type():
    """Test that non-agriculture commodity type raises error."""
    with pytest.raises(ValueError):
        AgricultureFuture(
            spot=1950.0,
            strike=1900.0,
            maturity=1.0,
            risk_free_rate=0.03,
            agriculture_type=CommodityType.SILVER,  # Invalid: silver is not agriculture
        )


# ========== Tests for Enhanced CommodityForward ==========


def test_commodity_forward_with_type():
    """Test CommodityForward with commodity_type auto-fill."""
    forward = CommodityForward(
        spot=75.0,
        strike=72.0,
        maturity=1.0,
        risk_free_rate=0.04,
        commodity_type=CommodityType.WTI_CRUDE,
    )

    # Should auto-fill storage_cost and convenience_yield from defaults
    assert forward.storage_cost == 0.02  # WTI default
    assert forward.convenience_yield == 0.03  # WTI default


def test_commodity_forward_override_defaults():
    """Test that explicit parameters override defaults."""
    forward = CommodityForward(
        spot=75.0,
        strike=72.0,
        maturity=1.0,
        risk_free_rate=0.04,
        commodity_type=CommodityType.WTI_CRUDE,
        storage_cost=0.05,  # Override default 0.02
        convenience_yield=0.01,  # Override default 0.03
    )

    assert forward.storage_cost == 0.05
    assert forward.convenience_yield == 0.01


def test_commodity_forward_no_type():
    """Test CommodityForward without commodity_type."""
    forward = CommodityForward(
        spot=100.0,
        strike=95.0,
        maturity=1.0,
        risk_free_rate=0.05,
        storage_cost=0.03,
        convenience_yield=0.02,
    )

    # Should keep explicit values
    assert forward.storage_cost == 0.03
    assert forward.convenience_yield == 0.02


# ========== Tests for Enhanced CommoditySwap ==========


def test_commodity_swap_with_type():
    """Test CommoditySwap with commodity_type."""
    swap = CommoditySwap(
        notional=1000.0,
        fixed_price=75.0,
        payment_dates=[0.5, 1.0, 1.5],
        commodity_type=CommodityType.BRENT_CRUDE,
        swap_type="fixed_floating",
    )

    assert swap.commodity_type == CommodityType.BRENT_CRUDE
    assert swap.swap_type == "fixed_floating"


def test_commodity_swap_basis():
    """Test basis swap specification."""
    swap = CommoditySwap(
        notional=5000.0,
        fixed_price=2.0,  # Fixed spread
        payment_dates=[0.25, 0.5, 0.75, 1.0],
        swap_type="basis",
    )

    assert swap.swap_type == "basis"


# ========== Tests for Basis Swap ==========


def test_basis_swap_value():
    """Test commodity basis swap valuation."""
    # WTI vs Brent spread
    quantity = 10000.0  # 10,000 barrels
    wti_price = 75.0
    brent_price = 78.0
    strike_spread = 2.5  # Expected spread (Brent - WTI)
    discount_factor = 0.98

    # Long spread: long Brent, short WTI
    value_long = basis_swap_value(
        quantity, brent_price, wti_price, strike_spread, discount_factor, "long_spread"
    )

    # Actual spread: 78 - 75 = 3.0
    # Payoff: 10000 * (3.0 - 2.5) = 5000
    # PV: 5000 * 0.98 = 4900
    expected = 4900.0
    assert abs(value_long - expected) < 0.1

    # Short spread should be opposite
    value_short = basis_swap_value(
        quantity, brent_price, wti_price, strike_spread, discount_factor, "short_spread"
    )
    assert abs(value_long + value_short) < 1e-6


# ========== Tests for CommodityAsianOption ==========


def test_commodity_asian_option_with_type():
    """Test CommodityAsianOption with auto-fill from commodity type."""
    asian = CommodityAsianOption(
        spot=50.0,
        strike=50.0,
        maturity=1.0,
        risk_free_rate=0.05,
        volatility=0.0,  # Will be auto-filled
        commodity_type=CommodityType.GOLD,
        averaging_type="arithmetic",
        num_fixings=12,
    )

    # Should auto-fill volatility from GOLD defaults
    assert asian.volatility == 0.15  # Gold default volatility
    assert asian.storage_cost == 0.005  # Gold default storage
    assert asian.convenience_yield == 0.001  # Gold default convenience


def test_commodity_asian_option_geometric():
    """Test geometric averaging Asian option."""
    asian = CommodityAsianOption(
        spot=75.0,
        strike=72.0,
        maturity=1.0,
        risk_free_rate=0.04,
        volatility=0.35,
        commodity_type=CommodityType.WTI_CRUDE,
        averaging_type="geometric",
        num_fixings=12,
    )

    assert asian.averaging_type == "geometric"


# ========== Tests for Asian Pricing Functions ==========


def test_price_commodity_asian_arithmetic():
    """Test arithmetic Asian option pricing."""
    price = price_commodity_asian_arithmetic(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        risk_free_rate=0.05,
        volatility=0.30,
        num_fixings=12,
        storage_cost=0.02,
        convenience_yield=0.01,
        is_call=True,
    )

    # ATM call should be positive
    assert price > 0
    assert price < 20.0  # Reasonable upper bound


def test_price_commodity_asian_geometric():
    """Test geometric Asian option pricing."""
    price = price_commodity_asian_geometric(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        risk_free_rate=0.05,
        volatility=0.30,
        num_fixings=12,
        storage_cost=0.02,
        convenience_yield=0.01,
        is_call=True,
    )

    # ATM call should be positive
    assert price > 0
    assert price < 20.0  # Reasonable upper bound


def test_asian_arithmetic_vs_geometric():
    """Test that arithmetic Asian is more expensive than geometric."""
    spot = 100.0
    strike = 100.0
    maturity = 1.0
    risk_free_rate = 0.05
    volatility = 0.30
    num_fixings = 12

    arith_price = price_commodity_asian_arithmetic(
        spot, strike, maturity, risk_free_rate, volatility, num_fixings, is_call=True
    )

    geo_price = price_commodity_asian_geometric(
        spot, strike, maturity, risk_free_rate, volatility, num_fixings, is_call=True
    )

    # Arithmetic average Asian should be more expensive than geometric
    # (geometric mean <= arithmetic mean)
    assert arith_price >= geo_price


def test_asian_commodity_put_call_relationship():
    """Test put-call relationship for Asian options."""
    spot = 50.0
    strike = 50.0
    maturity = 1.0
    risk_free_rate = 0.05
    volatility = 0.30
    num_fixings = 12

    call_price = price_commodity_asian_arithmetic(
        spot, strike, maturity, risk_free_rate, volatility, num_fixings, is_call=True
    )

    put_price = price_commodity_asian_arithmetic(
        spot, strike, maturity, risk_free_rate, volatility, num_fixings, is_call=False
    )

    # For ATM options, call and put should be similar in value
    # (not exact due to cost of carry effects)
    assert abs(call_price - put_price) < 5.0


def test_asian_commodity_with_convenience_yield():
    """Test Asian option with significant convenience yield."""
    # High convenience yield should reduce forward price and affect option value
    spot = 75.0
    strike = 75.0
    maturity = 1.0
    risk_free_rate = 0.04
    volatility = 0.35

    # Case 1: No convenience yield
    price_no_cy = price_commodity_asian_arithmetic(
        spot,
        strike,
        maturity,
        risk_free_rate,
        volatility,
        12,
        storage_cost=0.0,
        convenience_yield=0.0,
        is_call=True,
    )

    # Case 2: High convenience yield (reduces forward price)
    price_with_cy = price_commodity_asian_arithmetic(
        spot,
        strike,
        maturity,
        risk_free_rate,
        volatility,
        12,
        storage_cost=0.0,
        convenience_yield=0.05,
        is_call=True,
    )

    # High convenience yield should reduce call option value (lowers forward price)
    assert price_with_cy < price_no_cy
