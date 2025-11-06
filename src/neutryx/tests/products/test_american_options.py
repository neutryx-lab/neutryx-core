"""Tests for American option products."""
from __future__ import annotations

import jax.numpy as jnp
import pytest

from neutryx.products.vanilla import American, European


class TestAmericanOption:
    """Test cases for American options."""

    def test_american_call_immediate_exercise(self):
        """Test immediate exercise value for American call."""
        K = 100.0
        american_call = American(K=K, T=1.0, is_call=True)

        # Test ITM spot
        spot_itm = jnp.array([110.0])
        exercise_value = american_call.immediate_exercise(spot_itm)
        assert abs(exercise_value - 10.0) < 1e-6, "ITM call should have intrinsic value"

        # Test ATM spot
        spot_atm = jnp.array([100.0])
        exercise_value = american_call.immediate_exercise(spot_atm)
        assert abs(exercise_value - 0.0) < 1e-6, "ATM call should have zero intrinsic"

        # Test OTM spot
        spot_otm = jnp.array([90.0])
        exercise_value = american_call.immediate_exercise(spot_otm)
        assert abs(exercise_value - 0.0) < 1e-6, "OTM call should have zero intrinsic"

    def test_american_put_immediate_exercise(self):
        """Test immediate exercise value for American put."""
        K = 100.0
        american_put = American(K=K, T=1.0, is_call=False)

        # Test ITM spot
        spot_itm = jnp.array([90.0])
        exercise_value = american_put.immediate_exercise(spot_itm)
        assert abs(exercise_value - 10.0) < 1e-6, "ITM put should have intrinsic value"

        # Test ATM spot
        spot_atm = jnp.array([100.0])
        exercise_value = american_put.immediate_exercise(spot_atm)
        assert abs(exercise_value - 0.0) < 1e-6, "ATM put should have zero intrinsic"

        # Test OTM spot
        spot_otm = jnp.array([110.0])
        exercise_value = american_put.immediate_exercise(spot_otm)
        assert abs(exercise_value - 0.0) < 1e-6, "OTM put should have zero intrinsic"

    def test_american_call_path_payoff(self):
        """Test path payoff for American call."""
        K = 100.0
        american_call = American(K=K, T=1.0, is_call=True)

        # Path ending ITM
        path = jnp.array([100.0, 105.0, 110.0, 115.0])
        payoff = american_call.payoff_path(path)
        assert abs(payoff - 15.0) < 1e-6, "Should return terminal intrinsic value"

        # Path ending OTM
        path_otm = jnp.array([100.0, 95.0, 90.0, 85.0])
        payoff_otm = american_call.payoff_path(path_otm)
        assert abs(payoff_otm - 0.0) < 1e-6, "Should return zero for OTM"

    def test_american_put_path_payoff(self):
        """Test path payoff for American put."""
        K = 100.0
        american_put = American(K=K, T=1.0, is_call=False)

        # Path ending ITM
        path = jnp.array([100.0, 95.0, 90.0, 85.0])
        payoff = american_put.payoff_path(path)
        assert abs(payoff - 15.0) < 1e-6, "Should return terminal intrinsic value"

        # Path ending OTM
        path_otm = jnp.array([100.0, 105.0, 110.0, 115.0])
        payoff_otm = american_put.payoff_path(path_otm)
        assert abs(payoff_otm - 0.0) < 1e-6, "Should return zero for OTM"

    def test_immediate_exercise_array(self):
        """Test immediate exercise with array of spots."""
        K = 100.0
        american_call = American(K=K, T=1.0, is_call=True)

        spots = jnp.array([90.0, 95.0, 100.0, 105.0, 110.0])
        exercise_values = american_call.immediate_exercise(spots)

        expected = jnp.array([0.0, 0.0, 0.0, 5.0, 10.0])
        assert jnp.allclose(exercise_values, expected), "Should compute exercise values for all spots"

    def test_american_vs_european_parity(self):
        """Test that American and European have same terminal payoff."""
        K = 100.0
        american = American(K=K, T=1.0, is_call=True)
        european = European(K=K, T=1.0, is_call=True)

        spots = jnp.array([80.0, 90.0, 100.0, 110.0, 120.0])

        for spot in spots:
            american_payoff = american.payoff_path(jnp.array([spot]))
            european_payoff = european.payoff_terminal(spot)
            assert abs(american_payoff - european_payoff) < 1e-6, \
                f"Terminal payoff should match for spot={spot}"


class TestAmericanPutProperties:
    """Test mathematical properties of American puts."""

    def test_american_put_always_nonnegative(self):
        """Test that American put payoff is always non-negative."""
        K = 100.0
        american_put = American(K=K, T=1.0, is_call=False)

        # Test various spot prices
        spots = jnp.linspace(50.0, 150.0, 50)
        for spot in spots:
            payoff = american_put.immediate_exercise(jnp.array([spot]))
            assert payoff >= -1e-10, f"Payoff should be non-negative for spot={spot}"

    def test_american_put_monotonic(self):
        """Test that American put value increases as spot decreases."""
        K = 100.0
        american_put = American(K=K, T=1.0, is_call=False)

        spot_high = jnp.array([110.0])
        spot_low = jnp.array([90.0])

        value_high = american_put.immediate_exercise(spot_high)
        value_low = american_put.immediate_exercise(spot_low)

        assert value_low > value_high, "Put value should increase as spot decreases"

    def test_american_put_bounded(self):
        """Test that American put is bounded by strike."""
        K = 100.0
        american_put = American(K=K, T=1.0, is_call=False)

        # Test at spot = 0 (maximum payoff scenario)
        spot_zero = jnp.array([0.01])
        max_payoff = american_put.immediate_exercise(spot_zero)

        assert max_payoff <= K, "Put payoff should be bounded by strike"


class TestAmericanCallProperties:
    """Test mathematical properties of American calls."""

    def test_american_call_always_nonnegative(self):
        """Test that American call payoff is always non-negative."""
        K = 100.0
        american_call = American(K=K, T=1.0, is_call=True)

        # Test various spot prices
        spots = jnp.linspace(50.0, 150.0, 50)
        for spot in spots:
            payoff = american_call.immediate_exercise(jnp.array([spot]))
            assert payoff >= -1e-10, f"Payoff should be non-negative for spot={spot}"

    def test_american_call_monotonic(self):
        """Test that American call value increases as spot increases."""
        K = 100.0
        american_call = American(K=K, T=1.0, is_call=True)

        spot_low = jnp.array([90.0])
        spot_high = jnp.array([110.0])

        value_low = american_call.immediate_exercise(spot_low)
        value_high = american_call.immediate_exercise(spot_high)

        assert value_high > value_low, "Call value should increase as spot increases"

    def test_american_call_unbounded(self):
        """Test that American call is unbounded (can grow with spot)."""
        K = 100.0
        american_call = American(K=K, T=1.0, is_call=True)

        # Very high spot should give very high payoff
        spot_high = jnp.array([1000.0])
        payoff = american_call.immediate_exercise(spot_high)

        assert payoff >= 900.0, "Call payoff should scale with spot price"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
