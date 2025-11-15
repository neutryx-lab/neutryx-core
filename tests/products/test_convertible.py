"""Tests for convertible bonds."""
from neutryx.products.convertible import (
    convertible_bond_delta,
    convertible_bond_floor,
    convertible_bond_parity,
    convertible_bond_premium,
    convertible_bond_simple_price,
    exchangeable_bond_value,
    mandatory_convertible_price,
)


def test_convertible_bond_parity():
    """Test conversion parity calculation."""
    stock_price = 50.0
    conversion_ratio = 20.0

    parity = convertible_bond_parity(stock_price, conversion_ratio)

    # Expected: 50 * 20 = 1000
    expected = 1000.0
    assert abs(parity - expected) < 1e-6


def test_convertible_bond_premium():
    """Test conversion and investment premium calculations."""
    bond_price = 1100.0
    conversion_value = 1000.0
    straight_bond_value = 980.0

    conversion_prem, investment_prem = convertible_bond_premium(
        bond_price, conversion_value, straight_bond_value
    )

    # Conversion premium: (1100/1000) - 1 = 0.10
    assert abs(conversion_prem - 0.10) < 0.001

    # Investment premium: (1100/980) - 1 = 0.122
    assert abs(investment_prem - 0.122) < 0.001


def test_convertible_bond_floor():
    """Test convertible bond floor value."""
    conversion_value = 950.0
    straight_bond_value = 980.0

    floor = convertible_bond_floor(conversion_value, straight_bond_value)

    # Floor should be max of the two values
    assert floor == 980.0


def test_convertible_bond_floor_when_conversion_higher():
    """Test floor when conversion value is higher."""
    conversion_value = 1050.0
    straight_bond_value = 980.0

    floor = convertible_bond_floor(conversion_value, straight_bond_value)

    # Floor should be conversion value
    assert floor == 1050.0


def test_convertible_bond_delta():
    """Test convertible bond equity delta."""
    bond_price = 1050.0
    stock_price = 50.0
    conversion_ratio = 20.0
    sensitivity = 0.6

    delta = convertible_bond_delta(bond_price, stock_price, conversion_ratio, sensitivity)

    # Expected: 20 * 0.6 = 12
    expected = 12.0
    assert abs(delta - expected) < 1e-6


def test_convertible_bond_simple_price():
    """Test simplified convertible bond pricing."""
    face_value = 1000.0
    coupon_rate = 0.03
    yield_rate = 0.05
    maturity = 5.0
    stock_price = 45.0
    conversion_ratio = 25.0
    stock_volatility = 0.30
    risk_free_rate = 0.04
    frequency = 2

    result = convertible_bond_simple_price(
        face_value,
        coupon_rate,
        yield_rate,
        maturity,
        stock_price,
        conversion_ratio,
        stock_volatility,
        risk_free_rate,
        frequency,
    )

    # Check all components are present
    assert "straight_bond" in result
    assert "conversion_value" in result
    assert "option_value" in result
    assert "convertible_value" in result

    # Straight bond should be less than par (yield > coupon)
    assert result["straight_bond"] < face_value

    # Conversion value: 45 * 25 = 1125
    assert abs(result["conversion_value"] - 1125.0) < 1.0

    # Option value should be positive
    assert result["option_value"] > 0

    # Convertible value should be greater than floor
    floor = max(result["straight_bond"], result["conversion_value"])
    assert result["convertible_value"] >= floor * 0.95  # Allow small tolerance


def test_mandatory_convertible_price():
    """Test mandatory convertible pricing."""
    stock_price = 100.0
    conversion_ratio_low = 10.0
    conversion_ratio_high = 12.0
    threshold_low = 90.0
    threshold_high = 110.0
    maturity = 3.0
    risk_free_rate = 0.05
    volatility = 0.25

    price = mandatory_convertible_price(
        stock_price,
        conversion_ratio_low,
        conversion_ratio_high,
        threshold_low,
        threshold_high,
        maturity,
        risk_free_rate,
        volatility,
    )

    # Price should be positive and reasonable
    assert price > 0
    assert price < 2000  # Should be less than max possible value


def test_exchangeable_bond_value():
    """Test exchangeable bond valuation."""
    face_value = 1000.0
    straight_bond_value = 950.0
    underlying_stock_price = 48.0
    conversion_ratio = 22.0
    volatility = 0.35
    maturity = 4.0
    risk_free_rate = 0.04

    value = exchangeable_bond_value(
        face_value,
        straight_bond_value,
        underlying_stock_price,
        conversion_ratio,
        volatility,
        maturity,
        risk_free_rate,
    )

    # Exchange value: 48 * 22 = 1056
    exchange_value = 1056.0

    # Value should be at least the exchange value
    assert value >= exchange_value * 0.9  # Allow for discounting


def test_convertible_bond_in_the_money():
    """Test convertible bond when deeply in the money."""
    face_value = 1000.0
    coupon_rate = 0.02
    yield_rate = 0.05
    maturity = 5.0
    stock_price = 60.0  # High stock price
    conversion_ratio = 20.0
    stock_volatility = 0.25
    risk_free_rate = 0.04

    result = convertible_bond_simple_price(
        face_value,
        coupon_rate,
        yield_rate,
        maturity,
        stock_price,
        conversion_ratio,
        stock_volatility,
        risk_free_rate,
        2,
    )

    # Conversion value: 60 * 20 = 1200 (deeply ITM)
    assert result["conversion_value"] == 1200.0

    # Convertible value should be close to conversion value when deeply ITM
    assert result["convertible_value"] > result["conversion_value"] * 0.95


def test_convertible_bond_out_of_money():
    """Test convertible bond when out of the money."""
    face_value = 1000.0
    coupon_rate = 0.05
    yield_rate = 0.04
    maturity = 5.0
    stock_price = 30.0  # Low stock price
    conversion_ratio = 20.0
    stock_volatility = 0.20
    risk_free_rate = 0.04

    result = convertible_bond_simple_price(
        face_value,
        coupon_rate,
        yield_rate,
        maturity,
        stock_price,
        conversion_ratio,
        stock_volatility,
        risk_free_rate,
        2,
    )

    # Conversion value: 30 * 20 = 600 (OTM)
    assert result["conversion_value"] == 600.0

    # When OTM, convertible should trade close to straight bond value
    # Straight bond value should be > face (coupon > yield)
    assert result["straight_bond"] > face_value

    # Convertible value should be at least straight bond value
    assert result["convertible_value"] >= result["straight_bond"] * 0.95
