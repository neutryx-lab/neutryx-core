"""Tests for bond pricing and analytics."""

import jax.numpy as jnp
import pytest

from neutryx.products.bonds import (
    bond_price_from_curve,
    bond_yield_to_maturity,
    convexity,
    coupon_bond_price,
    floating_rate_note_price,
    macaulay_duration,
    modified_duration,
    zero_coupon_bond_price,
)


def test_zero_coupon_bond_price():
    """Test zero-coupon bond pricing."""
    # At 5% yield, 2 years, $100 face value
    price = zero_coupon_bond_price(100.0, 0.05, 2.0, frequency=2)
    expected = 100.0 / (1.025**4)  # Semiannual compounding
    assert jnp.isclose(price, expected, atol=1e-4)

    # Zero coupon bond with 1 year maturity should be worth more
    price_1y = zero_coupon_bond_price(100.0, 0.05, 1.0, frequency=2)
    assert price_1y > price


def test_zero_coupon_bond_continuous():
    """Test zero-coupon bond with continuous compounding."""
    # Continuous compounding (frequency = 0)
    price = zero_coupon_bond_price(100.0, 0.05, 2.0, frequency=0)
    expected = 100.0 * jnp.exp(-0.05 * 2.0)
    assert jnp.isclose(price, expected, atol=1e-6)


def test_coupon_bond_at_par():
    """Test that bond trades at par when coupon = yield."""
    # When coupon rate equals yield, bond should trade at par
    price = coupon_bond_price(100.0, 0.05, 0.05, 5.0, frequency=2)
    assert jnp.isclose(price, 100.0, atol=1e-4)


def test_coupon_bond_premium_discount():
    """Test premium and discount bond pricing."""
    # Premium bond: coupon > yield
    premium_price = coupon_bond_price(100.0, 0.06, 0.04, 3.0, frequency=2)
    assert premium_price > 100.0

    # Discount bond: coupon < yield
    discount_price = coupon_bond_price(100.0, 0.04, 0.06, 3.0, frequency=2)
    assert discount_price < 100.0


def test_coupon_bond_maturity_effect():
    """Test that longer maturity bonds have more price sensitivity."""
    # Short maturity
    price_short = coupon_bond_price(100.0, 0.05, 0.06, 1.0, frequency=2)

    # Long maturity
    price_long = coupon_bond_price(100.0, 0.05, 0.06, 10.0, frequency=2)

    # Both are discount bonds, long maturity should be cheaper
    assert price_long < price_short


def test_bond_yield_to_maturity():
    """Test yield to maturity calculation."""
    # Create a bond at a known yield
    known_yield = 0.05
    price = coupon_bond_price(100.0, 0.06, known_yield, 5.0, frequency=2)

    # Calculate YTM from price
    calculated_ytm = bond_yield_to_maturity(price, 100.0, 0.06, 5.0, frequency=2)

    # Should recover the original yield
    assert jnp.isclose(calculated_ytm, known_yield, atol=1e-4)


def test_bond_yield_discount_bond():
    """Test YTM for a discount bond."""
    # Bond trading below par should have yield > coupon
    price = 95.0
    coupon = 0.04
    ytm = bond_yield_to_maturity(price, 100.0, coupon, 5.0, frequency=2)

    assert ytm > coupon


def test_macaulay_duration_zero_coupon():
    """Test that zero-coupon bond duration equals maturity."""
    maturity = 5.0
    duration = macaulay_duration(100.0, 0.0, 0.05, maturity, frequency=2)

    # For zero-coupon bond, Macaulay duration = maturity
    assert jnp.isclose(duration, maturity, atol=1e-4)


def test_macaulay_duration_properties():
    """Test Macaulay duration properties."""
    # Duration should be positive
    duration = macaulay_duration(100.0, 0.05, 0.05, 5.0, frequency=2)
    assert duration > 0

    # Duration should be less than maturity for coupon bonds
    assert duration < 5.0

    # Higher coupon => lower duration
    duration_high_coupon = macaulay_duration(100.0, 0.08, 0.05, 5.0, frequency=2)
    duration_low_coupon = macaulay_duration(100.0, 0.02, 0.05, 5.0, frequency=2)
    assert duration_high_coupon < duration_low_coupon


def test_modified_duration():
    """Test modified duration calculation."""
    mac_dur = macaulay_duration(100.0, 0.05, 0.05, 5.0, frequency=2)
    mod_dur = modified_duration(100.0, 0.05, 0.05, 5.0, frequency=2)

    # ModD = MacD / (1 + y/m)
    expected_mod_dur = mac_dur / (1.0 + 0.05 / 2)
    assert jnp.isclose(mod_dur, expected_mod_dur, atol=1e-6)

    # Modified duration should be less than Macaulay duration
    assert mod_dur < mac_dur


def test_modified_duration_price_sensitivity():
    """Test that modified duration predicts price changes."""
    face = 100.0
    coupon = 0.05
    ytm = 0.05
    maturity = 5.0
    frequency = 2

    # Calculate modified duration
    mod_dur = modified_duration(face, coupon, ytm, maturity, frequency)

    # Small yield change
    dy = 0.0001  # 1 basis point

    # Price at original yield
    P0 = coupon_bond_price(face, coupon, ytm, maturity, frequency)

    # Price at new yield
    P1 = coupon_bond_price(face, coupon, ytm + dy, maturity, frequency)

    # Actual price change
    actual_dp = (P1 - P0) / P0

    # Predicted price change using modified duration
    predicted_dp = -mod_dur * dy

    # Should be close for small yield changes
    assert jnp.isclose(actual_dp, predicted_dp, atol=1e-4)


def test_convexity_positive():
    """Test that convexity is positive."""
    conv = convexity(100.0, 0.05, 0.05, 5.0, frequency=2)
    assert conv > 0


def test_convexity_price_prediction():
    """Test that convexity improves price change prediction."""
    face = 100.0
    coupon = 0.05
    ytm = 0.05
    maturity = 5.0
    frequency = 2

    # Calculate duration and convexity
    mod_dur = modified_duration(face, coupon, ytm, maturity, frequency)
    conv = convexity(face, coupon, ytm, maturity, frequency)

    # Larger yield change
    dy = 0.01  # 100 basis points

    # Price at original yield
    P0 = coupon_bond_price(face, coupon, ytm, maturity, frequency)

    # Price at new yield
    P1 = coupon_bond_price(face, coupon, ytm + dy, maturity, frequency)

    # Actual price change
    actual_dp = (P1 - P0) / P0

    # Prediction with duration only
    predicted_dp_duration = -mod_dur * dy

    # Prediction with duration + convexity
    predicted_dp_both = -mod_dur * dy + 0.5 * conv * dy * dy

    # Convexity should improve prediction
    error_duration = jnp.abs(actual_dp - predicted_dp_duration)
    error_both = jnp.abs(actual_dp - predicted_dp_both)
    assert error_both < error_duration


def test_floating_rate_note_at_par():
    """Test FRN pricing when reference rate = discount rate and spread = 0."""
    # When reference rate = discount rate and spread = 0, FRN should trade near par
    price = floating_rate_note_price(100.0, 0.05, 0.0, 0.05, 2.0, frequency=4)
    assert jnp.isclose(price, 100.0, atol=1e-4)


def test_floating_rate_note_with_spread():
    """Test FRN pricing with credit spread."""
    # With positive spread, FRN should trade above par if spread > 0
    # and discount_rate < reference_rate + spread
    price = floating_rate_note_price(100.0, 0.03, 0.01, 0.03, 2.0, frequency=4)
    assert price > 99.0  # Should be above discount price


def test_bond_price_from_curve():
    """Test bond pricing using a discount curve."""

    # Flat discount curve at 5% with discrete compounding matching frequency=2
    def flat_curve(t):
        return 1.0 / ((1.0 + 0.05 / 2.0) ** (2.0 * t))

    price_curve = bond_price_from_curve(100.0, 0.05, 3.0, 2, flat_curve)

    # Should match price from flat yield (both using discrete semiannual compounding)
    price_flat = coupon_bond_price(100.0, 0.05, 0.05, 3.0, frequency=2)

    assert jnp.isclose(price_curve, price_flat, atol=1e-2)


def test_bond_price_upward_sloping_curve():
    """Test bond pricing with upward sloping curve."""

    # Upward sloping curve: rates increase with maturity
    def upward_curve(t):
        rate = 0.03 + 0.01 * t  # 3% + 1% per year
        return jnp.exp(-rate * t)

    # Downward sloping curve
    def downward_curve(t):
        rate = 0.05 - 0.005 * t  # 5% - 0.5% per year
        return jnp.exp(-rate * t)

    price_up = bond_price_from_curve(100.0, 0.04, 5.0, 2, upward_curve)
    price_down = bond_price_from_curve(100.0, 0.04, 5.0, 2, downward_curve)

    # Upward sloping curve (higher rates for later payments) => lower price
    assert price_up < price_down


def test_bond_frequencies():
    """Test different payment frequencies."""
    # Same bond priced with different frequencies
    annual = coupon_bond_price(100.0, 0.05, 0.05, 5.0, frequency=1)
    semiannual = coupon_bond_price(100.0, 0.05, 0.05, 5.0, frequency=2)
    quarterly = coupon_bond_price(100.0, 0.05, 0.05, 5.0, frequency=4)

    # All should be close to par when coupon = yield
    assert jnp.isclose(annual, 100.0, atol=0.1)
    assert jnp.isclose(semiannual, 100.0, atol=0.1)
    assert jnp.isclose(quarterly, 100.0, atol=0.1)


def test_zero_coupon_vs_coupon_bond():
    """Test relationship between zero-coupon and coupon bonds."""
    # Zero-coupon bond
    zero_price = zero_coupon_bond_price(100.0, 0.05, 5.0, frequency=2)

    # Very low coupon bond (almost zero-coupon)
    low_coupon_price = coupon_bond_price(100.0, 0.001, 0.05, 5.0, frequency=2)

    # Should be close
    assert jnp.isclose(zero_price, low_coupon_price, atol=2.0)


def test_jit_compilation():
    """Test that bond functions are JIT-compilable."""
    import jax

    # JIT compile key functions
    jitted_zero = jax.jit(lambda: zero_coupon_bond_price(100.0, 0.05, 5.0, frequency=2))
    jitted_coupon = jax.jit(lambda: coupon_bond_price(100.0, 0.05, 0.05, 5.0, frequency=2))
    jitted_duration = jax.jit(lambda: macaulay_duration(100.0, 0.05, 0.05, 5.0, frequency=2))

    # Execute
    price_zero = jitted_zero()
    price_coupon = jitted_coupon()
    dur = jitted_duration()

    # Should produce valid results
    assert jnp.isfinite(price_zero)
    assert jnp.isfinite(price_coupon)
    assert jnp.isfinite(dur)
    assert dur > 0
