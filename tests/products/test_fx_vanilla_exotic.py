"""Comprehensive tests for FX vanilla and exotic derivatives."""
from __future__ import annotations

import jax.numpy as jnp
import pytest

from neutryx.products.fx_vanilla_exotic import (
    FXForward,
    FXAmericanOption,
    FXDigitalAssetOrNothing,
    FXAsianArithmetic,
    FXAsianGeometric,
    FXAsianArithmeticFloatingStrike,
    FXAsianGeometricFloatingStrike,
    FXLookbackFloatingStrikeCall,
    FXLookbackFloatingStrikePut,
    FXLookbackFixedStrikeCall,
    FXLookbackFixedStrikePut,
    FXLookbackPartialFixedStrikeCall,
    FXLookbackPartialFixedStrikePut,
)


# ============================================================================
# FX Forward Tests
# ============================================================================


class TestFXForward:
    """Test cases for FX forward contracts."""

    def test_fair_forward_rate(self):
        """Test fair forward rate calculation using interest rate parity."""
        forward = FXForward(
            spot=1.10,
            forward_rate=1.12,
            expiry=1.0,
            domestic_rate=0.05,
            foreign_rate=0.02,
            notional_foreign=1_000_000.0,
            is_long=True
        )

        fair_rate = forward.fair_forward_rate()
        # F = S * exp((r_d - r_f) * T) = 1.10 * exp((0.05 - 0.02) * 1.0)
        expected = 1.10 * jnp.exp(0.03)
        assert abs(fair_rate - expected) < 1e-6, "Fair forward rate calculation incorrect"

    def test_mark_to_market_long(self):
        """Test mark-to-market for long forward position."""
        forward = FXForward(
            spot=1.10,
            forward_rate=1.12,
            expiry=1.0,
            domestic_rate=0.05,
            foreign_rate=0.02,
            notional_foreign=1_000_000.0,
            is_long=True
        )

        mtm = forward.mark_to_market()
        # If fair rate < contract rate, long position has negative value
        fair_rate = forward.fair_forward_rate()
        expected_pnl = (fair_rate - 1.12) * 1_000_000.0 * jnp.exp(-0.05 * 1.0)
        assert abs(mtm - expected_pnl) < 1e-4, "MTM calculation for long forward incorrect"

    def test_mark_to_market_short(self):
        """Test mark-to-market for short forward position."""
        forward = FXForward(
            spot=1.10,
            forward_rate=1.12,
            expiry=1.0,
            domestic_rate=0.05,
            foreign_rate=0.02,
            notional_foreign=1_000_000.0,
            is_long=False
        )

        mtm = forward.mark_to_market()
        # Short benefits when fair rate < contract rate
        fair_rate = forward.fair_forward_rate()
        expected_pnl = (1.12 - fair_rate) * 1_000_000.0 * jnp.exp(-0.05 * 1.0)
        assert abs(mtm - expected_pnl) < 1e-4, "MTM calculation for short forward incorrect"

    def test_settlement_payoff_long(self):
        """Test settlement payoff for long position."""
        forward = FXForward(
            spot=1.10,
            forward_rate=1.12,
            expiry=1.0,
            domestic_rate=0.05,
            foreign_rate=0.02,
            notional_foreign=1_000_000.0,
            is_long=True
        )

        # If spot at expiry is 1.15, long makes profit
        settlement = forward.settlement_payoff(1.15)
        expected = (1.15 - 1.12) * 1_000_000.0
        assert abs(settlement - expected) < 1e-4, "Settlement payoff for long incorrect"

    def test_settlement_payoff_short(self):
        """Test settlement payoff for short position."""
        forward = FXForward(
            spot=1.10,
            forward_rate=1.12,
            expiry=1.0,
            domestic_rate=0.05,
            foreign_rate=0.02,
            notional_foreign=1_000_000.0,
            is_long=False
        )

        # If spot at expiry is 1.08, short makes profit
        settlement = forward.settlement_payoff(1.08)
        expected = (1.12 - 1.08) * 1_000_000.0
        assert abs(settlement - expected) < 1e-4, "Settlement payoff for short incorrect"


# ============================================================================
# FX American Option Tests
# ============================================================================


class TestFXAmericanOption:
    """Test cases for FX American options."""

    def test_american_call_immediate_exercise(self):
        """Test immediate exercise value for FX American call."""
        option = FXAmericanOption(
            strike=1.12,
            T=1.0,
            domestic_rate=0.05,
            foreign_rate=0.02,
            is_call=True,
            notional=1_000_000.0
        )

        # ITM spot
        spot_itm = jnp.array([1.15])
        exercise_value = option.immediate_exercise(spot_itm)
        expected = (1.15 - 1.12) * 1_000_000.0
        assert abs(exercise_value - expected) < 1.0, "ITM call exercise value incorrect"

        # OTM spot
        spot_otm = jnp.array([1.10])
        exercise_value_otm = option.immediate_exercise(spot_otm)
        assert abs(exercise_value_otm - 0.0) < 1e-6, "OTM call should have zero exercise value"

    def test_american_put_immediate_exercise(self):
        """Test immediate exercise value for FX American put."""
        option = FXAmericanOption(
            strike=1.12,
            T=1.0,
            domestic_rate=0.05,
            foreign_rate=0.02,
            is_call=False,
            notional=1_000_000.0
        )

        # ITM spot
        spot_itm = jnp.array([1.08])
        exercise_value = option.immediate_exercise(spot_itm)
        expected = (1.12 - 1.08) * 1_000_000.0
        assert abs(exercise_value - expected) < 1.0, "ITM put exercise value incorrect"

        # OTM spot
        spot_otm = jnp.array([1.15])
        exercise_value_otm = option.immediate_exercise(spot_otm)
        assert abs(exercise_value_otm - 0.0) < 1e-6, "OTM put should have zero exercise value"

    def test_american_call_path_payoff(self):
        """Test path payoff for FX American call."""
        option = FXAmericanOption(
            strike=1.12,
            T=1.0,
            domestic_rate=0.05,
            foreign_rate=0.02,
            is_call=True,
            notional=1_000_000.0
        )

        path = jnp.array([1.10, 1.12, 1.14, 1.16])
        payoff = option.payoff_path(path)
        expected = (1.16 - 1.12) * 1_000_000.0
        assert abs(payoff - expected) < 1.0, "Path payoff incorrect"


# ============================================================================
# FX Digital Asset-or-Nothing Tests
# ============================================================================


class TestFXDigitalAssetOrNothing:
    """Test cases for FX digital asset-or-nothing options."""

    def test_digital_call_pricing(self):
        """Test asset-or-nothing call pricing."""
        digital = FXDigitalAssetOrNothing(
            spot=1.10,
            strike=1.12,
            expiry=1.0,
            domestic_rate=0.05,
            foreign_rate=0.02,
            volatility=0.10,
            is_call=True,
            notional=1.0
        )

        price = digital.price()
        # Price should be positive for call
        assert price > 0, "Digital call price should be positive"
        # Price should be less than spot discounted
        assert price < 1.10 * jnp.exp(-0.02), "Digital call price should be reasonable"

    def test_digital_put_pricing(self):
        """Test asset-or-nothing put pricing."""
        digital = FXDigitalAssetOrNothing(
            spot=1.10,
            strike=1.12,
            expiry=1.0,
            domestic_rate=0.05,
            foreign_rate=0.02,
            volatility=0.10,
            is_call=False,
            notional=1.0
        )

        price = digital.price()
        # Price should be positive for put
        assert price > 0, "Digital put price should be positive"

    def test_digital_call_put_parity(self):
        """Test that call + put = discounted spot."""
        digital_call = FXDigitalAssetOrNothing(
            spot=1.10,
            strike=1.12,
            expiry=1.0,
            domestic_rate=0.05,
            foreign_rate=0.02,
            volatility=0.10,
            is_call=True,
            notional=1.0
        )

        digital_put = FXDigitalAssetOrNothing(
            spot=1.10,
            strike=1.12,
            expiry=1.0,
            domestic_rate=0.05,
            foreign_rate=0.02,
            volatility=0.10,
            is_call=False,
            notional=1.0
        )

        call_price = digital_call.price()
        put_price = digital_put.price()
        expected_sum = 1.10 * jnp.exp(-0.02 * 1.0)

        assert abs(call_price + put_price - expected_sum) < 1e-6, \
            "Asset-or-nothing call + put should equal discounted spot"

    def test_digital_zero_expiry(self):
        """Test digital at expiry."""
        # Call ITM at expiry
        digital_call = FXDigitalAssetOrNothing(
            spot=1.15,
            strike=1.12,
            expiry=0.0,
            domestic_rate=0.05,
            foreign_rate=0.02,
            volatility=0.10,
            is_call=True,
            notional=1.0
        )
        price = digital_call.price()
        assert abs(price - 1.15) < 1e-6, "ITM call at expiry should pay spot"

        # Call OTM at expiry
        digital_call_otm = FXDigitalAssetOrNothing(
            spot=1.10,
            strike=1.12,
            expiry=0.0,
            domestic_rate=0.05,
            foreign_rate=0.02,
            volatility=0.10,
            is_call=True,
            notional=1.0
        )
        price_otm = digital_call_otm.price()
        assert abs(price_otm - 0.0) < 1e-6, "OTM call at expiry should pay zero"


# ============================================================================
# FX Asian Option Tests
# ============================================================================


class TestFXAsianOptions:
    """Test cases for FX Asian options."""

    def test_asian_arithmetic_call(self):
        """Test arithmetic average Asian call."""
        asian = FXAsianArithmetic(
            strike=1.12,
            T=1.0,
            domestic_rate=0.05,
            foreign_rate=0.02,
            is_call=True,
            notional=1_000_000.0
        )

        # Path with average above strike
        path = jnp.array([1.10, 1.12, 1.14, 1.16])
        payoff = asian.payoff_path(path)
        avg = path.mean()
        expected = max(avg - 1.12, 0.0) * 1_000_000.0
        assert abs(payoff - expected) < 1e-4, "Arithmetic Asian call payoff incorrect"

    def test_asian_arithmetic_put(self):
        """Test arithmetic average Asian put."""
        asian = FXAsianArithmetic(
            strike=1.12,
            T=1.0,
            domestic_rate=0.05,
            foreign_rate=0.02,
            is_call=False,
            notional=1_000_000.0
        )

        # Path with average below strike
        path = jnp.array([1.08, 1.10, 1.11, 1.09])
        payoff = asian.payoff_path(path)
        avg = path.mean()
        expected = max(1.12 - avg, 0.0) * 1_000_000.0
        assert abs(payoff - expected) < 1e-4, "Arithmetic Asian put payoff incorrect"

    def test_asian_geometric_call(self):
        """Test geometric average Asian call."""
        asian = FXAsianGeometric(
            strike=1.12,
            T=1.0,
            domestic_rate=0.05,
            foreign_rate=0.02,
            is_call=True,
            notional=1_000_000.0
        )

        path = jnp.array([1.10, 1.12, 1.14, 1.16])
        payoff = asian.payoff_path(path)
        geometric_avg = jnp.exp(jnp.log(path).mean())
        expected = max(geometric_avg - 1.12, 0.0) * 1_000_000.0
        assert abs(payoff - expected) < 1e-4, "Geometric Asian call payoff incorrect"

    def test_asian_arithmetic_floating_strike_call(self):
        """Test arithmetic floating strike Asian call."""
        asian = FXAsianArithmeticFloatingStrike(
            T=1.0,
            domestic_rate=0.05,
            foreign_rate=0.02,
            is_call=True,
            notional=1_000_000.0
        )

        path = jnp.array([1.10, 1.12, 1.14, 1.16])
        payoff = asian.payoff_path(path)
        avg = path.mean()
        terminal = path[-1]
        expected = max(terminal - avg, 0.0) * 1_000_000.0
        assert abs(payoff - expected) < 1e-4, "Floating strike Asian call payoff incorrect"

    def test_asian_geometric_less_than_arithmetic(self):
        """Test that geometric average <= arithmetic average."""
        path = jnp.array([1.10, 1.15, 1.20, 1.12])

        arithmetic_avg = path.mean()
        geometric_avg = jnp.exp(jnp.log(path).mean())

        assert geometric_avg <= arithmetic_avg + 1e-10, \
            "Geometric average should be <= arithmetic average"


# ============================================================================
# FX Lookback Option Tests
# ============================================================================


class TestFXLookbackOptions:
    """Test cases for FX lookback options."""

    def test_lookback_floating_strike_call(self):
        """Test floating strike lookback call."""
        lookback = FXLookbackFloatingStrikeCall(
            T=1.0,
            domestic_rate=0.05,
            foreign_rate=0.02,
            notional=1_000_000.0
        )

        path = jnp.array([1.10, 1.08, 1.15, 1.12])
        payoff = lookback.payoff_path(path)
        expected = (1.12 - 1.08) * 1_000_000.0  # terminal - min
        assert abs(payoff - expected) < 1.0, "Floating strike call payoff incorrect"

    def test_lookback_floating_strike_put(self):
        """Test floating strike lookback put."""
        lookback = FXLookbackFloatingStrikePut(
            T=1.0,
            domestic_rate=0.05,
            foreign_rate=0.02,
            notional=1_000_000.0
        )

        path = jnp.array([1.10, 1.15, 1.08, 1.12])
        payoff = lookback.payoff_path(path)
        expected = (1.15 - 1.12) * 1_000_000.0  # max - terminal
        assert abs(payoff - expected) < 1.0, "Floating strike put payoff incorrect"

    def test_lookback_fixed_strike_call(self):
        """Test fixed strike lookback call."""
        lookback = FXLookbackFixedStrikeCall(
            strike=1.12,
            T=1.0,
            domestic_rate=0.05,
            foreign_rate=0.02,
            notional=1_000_000.0
        )

        path = jnp.array([1.10, 1.15, 1.18, 1.14])
        payoff = lookback.payoff_path(path)
        expected = max(1.18 - 1.12, 0.0) * 1_000_000.0  # max(max - K, 0)
        assert abs(payoff - expected) < 1.0, "Fixed strike call payoff incorrect"

    def test_lookback_fixed_strike_put(self):
        """Test fixed strike lookback put."""
        lookback = FXLookbackFixedStrikePut(
            strike=1.12,
            T=1.0,
            domestic_rate=0.05,
            foreign_rate=0.02,
            notional=1_000_000.0
        )

        path = jnp.array([1.15, 1.10, 1.08, 1.14])
        payoff = lookback.payoff_path(path)
        expected = max(1.12 - 1.08, 0.0) * 1_000_000.0  # max(K - min, 0)
        assert abs(payoff - expected) < 1.0, "Fixed strike put payoff incorrect"

    def test_lookback_partial_call(self):
        """Test partial lookback call with observation window."""
        lookback = FXLookbackPartialFixedStrikeCall(
            strike=1.12,
            T=1.0,
            domestic_rate=0.05,
            foreign_rate=0.02,
            observation_start=0.5,
            notional=1_000_000.0
        )

        # Path where first half has lower max than second half
        path = jnp.array([1.10, 1.11, 1.15, 1.16])  # Observe from index 2 onward
        payoff = lookback.payoff_path(path)

        # Should only observe second half: [1.15, 1.16]
        # Max is 1.16, payoff = max(1.16 - 1.12, 0) = 0.04
        expected = max(1.16 - 1.12, 0.0) * 1_000_000.0
        assert abs(payoff - expected) < 1.0, "Partial lookback call payoff incorrect"

    def test_lookback_always_nonnegative(self):
        """Test that lookback options always have non-negative payoffs."""
        call = FXLookbackFloatingStrikeCall(
            T=1.0,
            domestic_rate=0.05,
            foreign_rate=0.02,
            notional=1.0
        )

        put = FXLookbackFloatingStrikePut(
            T=1.0,
            domestic_rate=0.05,
            foreign_rate=0.02,
            notional=1.0
        )

        # Various paths
        paths = [
            jnp.array([1.10, 1.15, 1.20]),
            jnp.array([1.20, 1.15, 1.10]),
            jnp.array([1.15, 1.10, 1.15]),
        ]

        for path in paths:
            call_payoff = call.payoff_path(path)
            put_payoff = put.payoff_path(path)
            assert call_payoff >= -1e-10, "Lookback call payoff should be non-negative"
            assert put_payoff >= -1e-10, "Lookback put payoff should be non-negative"


# ============================================================================
# Property-Based Tests
# ============================================================================


class TestFXDerivativesProperties:
    """Test mathematical properties of FX derivatives."""

    def test_forward_zero_rates_parity(self):
        """Test that with zero rates, forward rate equals spot."""
        forward = FXForward(
            spot=1.10,
            forward_rate=1.10,
            expiry=1.0,
            domestic_rate=0.0,
            foreign_rate=0.0,
            notional_foreign=1.0,
            is_long=True
        )

        fair_rate = forward.fair_forward_rate()
        assert abs(fair_rate - 1.10) < 1e-6, "With zero rates, forward should equal spot"

    def test_american_option_nonnegative(self):
        """Test that American option payoffs are always non-negative."""
        option = FXAmericanOption(
            strike=1.12,
            T=1.0,
            domestic_rate=0.05,
            foreign_rate=0.02,
            is_call=True,
            notional=1.0
        )

        spots = jnp.linspace(0.80, 1.50, 50)
        for spot in spots:
            payoff = option.immediate_exercise(jnp.array([spot]))
            assert payoff >= -1e-10, f"Payoff should be non-negative for spot={spot}"

    def test_asian_reduces_volatility(self):
        """Test that Asian options reduce effective volatility."""
        # For the same strike and paths, Asian should have lower dispersion
        # This is a simplified test checking that averaging dampens extremes
        path_high_vol = jnp.array([1.00, 1.20, 0.90, 1.30, 0.80, 1.40])

        asian = FXAsianArithmetic(
            strike=1.10,
            T=1.0,
            domestic_rate=0.05,
            foreign_rate=0.02,
            is_call=True,
            notional=1.0
        )

        payoff = asian.payoff_path(path_high_vol)
        avg = path_high_vol.mean()

        # Average should dampen extremes
        assert avg < path_high_vol.max(), "Average should be less than max"
        assert avg > path_high_vol.min(), "Average should be more than min"


# ============================================================================
# Edge Cases
# ============================================================================


class TestFXDerivativesEdgeCases:
    """Test edge cases for FX derivatives."""

    def test_forward_at_expiry(self):
        """Test forward with zero time to expiry."""
        forward = FXForward(
            spot=1.10,
            forward_rate=1.12,
            expiry=0.001,  # Very short time
            domestic_rate=0.05,
            foreign_rate=0.02,
            notional_foreign=1.0,
            is_long=True
        )

        fair_rate = forward.fair_forward_rate()
        # Should be very close to spot
        assert abs(fair_rate - 1.10) < 0.01, "Forward rate should approach spot near expiry"

    def test_asian_constant_path(self):
        """Test Asian option with constant path."""
        asian = FXAsianArithmetic(
            strike=1.12,
            T=1.0,
            domestic_rate=0.05,
            foreign_rate=0.02,
            is_call=True,
            notional=1.0
        )

        constant_path = jnp.array([1.15, 1.15, 1.15, 1.15])
        payoff = asian.payoff_path(constant_path)
        expected = max(1.15 - 1.12, 0.0)
        assert abs(payoff - expected) < 1e-6, "Asian with constant path should equal vanilla"

    def test_lookback_constant_path(self):
        """Test lookback with constant path."""
        lookback = FXLookbackFloatingStrikeCall(
            T=1.0,
            domestic_rate=0.05,
            foreign_rate=0.02,
            notional=1.0
        )

        constant_path = jnp.array([1.15, 1.15, 1.15, 1.15])
        payoff = lookback.payoff_path(constant_path)
        # terminal - min = 1.15 - 1.15 = 0
        assert abs(payoff - 0.0) < 1e-6, "Lookback with constant path should be zero"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
