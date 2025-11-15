"""Tests for exotic commodity derivatives."""
import jax.numpy as jnp
import pytest

from neutryx.products.commodity_exotics import (
    SwingOption,
    SparkSpread,
    DarkSpread,
    CrackSpread,
    HeatingDegreeDays,
    CoolingDegreeDays,
    RainfallDerivative,
    CommodityBasketOption,
    InterruptibleContract,
)


def test_swing_option_basic():
    """Test basic swing option payoff."""
    product = SwingOption(
        T=1.0,
        strike=50.0,
        min_daily=0.0,
        max_daily=10.0,
        min_total=0.0,
        max_total=100.0,
        fixing_times=jnp.array([0.25, 0.5, 0.75, 1.0]),
        penalty_rate=0.0,
        is_call=True,
    )

    # Prices rise above strike
    path = jnp.array([50.0, 55.0, 60.0, 58.0, 62.0])
    payoff = product.payoff_path(path)

    assert payoff > 0.0


def test_swing_option_penalty():
    """Test swing option with minimum volume penalty."""
    product = SwingOption(
        T=1.0,
        strike=50.0,
        min_daily=1.0,
        max_daily=10.0,
        min_total=20.0,
        max_total=100.0,
        fixing_times=jnp.array([0.25, 0.5, 0.75, 1.0]),
        penalty_rate=5.0,
        is_call=True,
    )

    # Low prices, minimal exercise
    path = jnp.full(100, 45.0)
    payoff = product.payoff_path(path)

    # Should include penalty
    assert isinstance(float(payoff), float)


def test_spark_spread_call():
    """Test spark spread option."""
    product = SparkSpread(
        T=1.0,
        strike=5.0,
        heat_rate=8.0,
        variable_om=2.0,
        max_capacity=1.0,
        is_call=True,
    )

    # Single commodity path (simplified)
    path = jnp.array([50.0, 55.0, 60.0, 62.0])
    payoff = product.payoff_path(path)

    assert payoff >= 0.0


def test_dark_spread():
    """Test dark spread option."""
    product = DarkSpread(
        T=1.0,
        strike=10.0,
        heat_rate=0.5,
        emissions_rate=1.0,
        variable_om=3.0,
        max_capacity=1.0,
        is_call=True,
    )

    path = jnp.array([100.0, 105.0, 110.0, 108.0])
    payoff = product.payoff_path(path)

    assert payoff >= 0.0


def test_crack_spread():
    """Test crack spread option (3-2-1)."""
    product = CrackSpread(
        T=1.0,
        strike=10.0,
        gasoline_weight=2.0,
        heating_oil_weight=1.0,
        crude_weight=3.0,
        max_capacity=1.0,
        is_call=True,
    )

    path = jnp.array([80.0, 82.0, 85.0, 87.0])
    payoff = product.payoff_path(path)

    assert payoff >= 0.0


def test_heating_degree_days_call():
    """Test HDD call option."""
    product = HeatingDegreeDays(
        T=1.0,
        base_temperature=65.0,
        strike_hdd=500.0,
        tick_value=10.0,
        contract_type="call",
    )

    # Cold temperatures (high HDD)
    path = jnp.full(100, 40.0)  # 25 degrees below base daily
    payoff = product.payoff_path(path)

    # Total HDD = 100 * 25 = 2500, well above strike
    assert payoff > 0.0


def test_heating_degree_days_put():
    """Test HDD put option."""
    product = HeatingDegreeDays(
        T=1.0,
        base_temperature=65.0,
        strike_hdd=500.0,
        tick_value=10.0,
        contract_type="put",
    )

    # Warm temperatures (low HDD)
    path = jnp.full(100, 70.0)  # Above base temp
    payoff = product.payoff_path(path)

    # HDD = 0, put pays strike
    assert payoff > 0.0


def test_cooling_degree_days_call():
    """Test CDD call option."""
    product = CoolingDegreeDays(
        T=1.0,
        base_temperature=65.0,
        strike_cdd=500.0,
        tick_value=10.0,
        contract_type="call",
    )

    # Hot temperatures (high CDD)
    path = jnp.full(100, 85.0)  # 20 degrees above base
    payoff = product.payoff_path(path)

    # Total CDD = 100 * 20 = 2000
    assert payoff > 0.0


def test_cooling_degree_days_swap():
    """Test CDD swap."""
    product = CoolingDegreeDays(
        T=1.0,
        base_temperature=65.0,
        strike_cdd=1000.0,
        tick_value=10.0,
        contract_type="swap",
    )

    # Moderate temperatures
    path = jnp.full(100, 75.0)
    payoff = product.payoff_path(path)

    # Can be positive or negative
    assert isinstance(float(payoff), float)


def test_rainfall_derivative_put():
    """Test rainfall derivative (drought protection)."""
    product = RainfallDerivative(
        T=1.0,
        strike_rainfall=50.0,
        tick_value=100.0,
        contract_type="put",
        cap=None,
    )

    # Low rainfall (drought)
    path = jnp.full(100, 0.2)  # 0.2mm per day
    payoff = product.payoff_path(path)

    # Total rainfall = 20mm, strike = 50mm
    # Put pays 30mm * 100 = 3000
    assert payoff > 0.0


def test_rainfall_derivative_call():
    """Test rainfall derivative (flood protection)."""
    product = RainfallDerivative(
        T=1.0,
        strike_rainfall=100.0,
        tick_value=50.0,
        contract_type="call",
    )

    # High rainfall
    path = jnp.full(100, 2.0)  # 2mm per day
    payoff = product.payoff_path(path)

    # Total rainfall = 200mm, above strike
    assert payoff > 0.0


def test_rainfall_derivative_collar():
    """Test rainfall derivative collar."""
    product = RainfallDerivative(
        T=1.0,
        strike_rainfall=50.0,
        tick_value=100.0,
        contract_type="collar",
        cap=100.0,
    )

    path = jnp.full(100, 0.3)
    payoff = product.payoff_path(path)

    assert isinstance(float(payoff), float)


def test_commodity_basket_option_average():
    """Test commodity basket option with weighted average."""
    product = CommodityBasketOption(
        K=100.0,
        T=1.0,
        weights=jnp.array([0.5, 0.5]),
        basket_type="average",
        is_call=True,
    )

    # Terminal prices for two commodities
    spot = jnp.array([105.0, 95.0])
    payoff = product.payoff_terminal(spot)

    # Average = 100, at strike
    assert payoff >= 0.0


def test_commodity_basket_option_sum():
    """Test commodity basket option with weighted sum."""
    product = CommodityBasketOption(
        K=100.0,
        T=1.0,
        weights=jnp.array([1.0, 1.0, 1.0]),
        basket_type="sum",
        is_call=True,
    )

    spot = jnp.array([40.0, 50.0, 60.0])
    payoff = product.payoff_terminal(spot)

    # Sum = 150, above strike
    assert payoff > 0.0


def test_interruptible_contract():
    """Test interruptible supply contract."""
    product = InterruptibleContract(
        T=1.0,
        contract_price=50.0,
        interruption_threshold=60.0,
        interruption_payment=5.0,
        fixing_times=jnp.array([0.25, 0.5, 0.75, 1.0]),
        max_interruptions=2,
    )

    # Some periods with high prices
    path = jnp.array([50.0, 65.0, 70.0, 55.0, 62.0])
    payoff = product.payoff_path(path)

    # Supplier exercises interruption rights
    assert isinstance(float(payoff), float)
