"""Tests for ladder options (profit lock-in features)."""
import jax.numpy as jnp
import pytest

from neutryx.products.ladder import (
    LadderCall,
    LadderPut,
    PercentageLadderCall,
    PercentageLadderPut,
)


def test_ladder_call_no_rungs_hit():
    """Test ladder call when no rungs are hit (vanilla behavior)."""
    product = LadderCall(
        K=100.0,
        T=1.0,
        rungs=jnp.array([110.0, 120.0, 130.0]),
    )

    # Path stays below first rung
    path = jnp.array([100.0, 102.0, 105.0, 103.0])
    payoff = product.payoff_path(path)

    # Should behave like vanilla call: max(S_T - K, 0)
    expected = max(103.0 - 100.0, 0.0)
    assert jnp.isclose(payoff, expected, atol=1e-6)


def test_ladder_call_all_rungs_hit():
    """Test ladder call when all rungs are hit."""
    product = LadderCall(
        K=100.0,
        T=1.0,
        rungs=jnp.array([110.0, 120.0, 130.0]),
    )

    # Path hits all rungs then falls back
    path = jnp.array([100.0, 115.0, 125.0, 135.0, 105.0])
    payoff = product.payoff_path(path)

    # Should lock in highest rung (130) - K = 30
    # Even though terminal is only 105
    expected = 130.0 - 100.0
    assert jnp.isclose(payoff, expected, atol=1e-6)


def test_ladder_call_partial_rungs_hit():
    """Test ladder call with partial rung hits."""
    product = LadderCall(
        K=100.0,
        T=1.0,
        rungs=jnp.array([110.0, 120.0, 130.0]),
    )

    # Path hits first two rungs but terminal is lower
    path = jnp.array([100.0, 115.0, 125.0, 115.0])
    payoff = product.payoff_path(path)

    # Should lock in 120 - 100 = 20 (terminal is only 115, so use locked)
    expected = 120.0 - 100.0
    assert jnp.isclose(payoff, expected, atol=1e-6)


def test_ladder_call_terminal_exceeds_locked():
    """Test ladder call when terminal exceeds locked profit."""
    product = LadderCall(
        K=100.0,
        T=1.0,
        rungs=jnp.array([110.0, 120.0, 130.0]),
    )

    # Hits first rung, then goes much higher at terminal
    path = jnp.array([100.0, 115.0, 105.0, 140.0])
    payoff = product.payoff_path(path)

    # Should take terminal payoff (140 - 100 = 40) over locked (110 - 100 = 10)
    expected = 140.0 - 100.0
    assert jnp.isclose(payoff, expected, atol=1e-6)


def test_ladder_put_no_rungs_hit():
    """Test ladder put when no rungs are hit."""
    product = LadderPut(
        K=100.0,
        T=1.0,
        rungs=jnp.array([90.0, 80.0, 70.0]),  # Descending
    )

    # Path stays above first rung
    path = jnp.array([100.0, 98.0, 95.0, 97.0])
    payoff = product.payoff_path(path)

    # Should behave like vanilla put: max(K - S_T, 0)
    expected = max(100.0 - 97.0, 0.0)
    assert jnp.isclose(payoff, expected, atol=1e-6)


def test_ladder_put_all_rungs_hit():
    """Test ladder put when all rungs are hit."""
    product = LadderPut(
        K=100.0,
        T=1.0,
        rungs=jnp.array([90.0, 80.0, 70.0]),
    )

    # Path hits all rungs then recovers
    path = jnp.array([100.0, 85.0, 75.0, 65.0, 95.0])
    payoff = product.payoff_path(path)

    # Should lock in K - lowest rung = 100 - 70 = 30
    expected = 100.0 - 70.0
    assert jnp.isclose(payoff, expected, atol=1e-6)


def test_ladder_put_terminal_exceeds_locked():
    """Test ladder put when terminal exceeds locked profit."""
    product = LadderPut(
        K=100.0,
        T=1.0,
        rungs=jnp.array([90.0, 80.0, 70.0]),
    )

    # Hits first rung, then crashes at terminal
    path = jnp.array([100.0, 85.0, 95.0, 60.0])
    payoff = product.payoff_path(path)

    # Should take terminal (100 - 60 = 40) over locked (100 - 90 = 10)
    expected = 100.0 - 60.0
    assert jnp.isclose(payoff, expected, atol=1e-6)


def test_percentage_ladder_call():
    """Test percentage ladder call."""
    product = PercentageLadderCall(
        K=100.0,
        T=1.0,
        rung_percentages=jnp.array([0.10, 0.20, 0.30]),  # 110%, 120%, 130% of strike
    )

    # Verify rungs are calculated correctly
    expected_rungs = jnp.array([110.0, 120.0, 130.0])
    assert jnp.allclose(product.rungs, expected_rungs, atol=1e-6)

    # Path hits first two rungs
    path = jnp.array([100.0, 115.0, 125.0, 118.0])
    payoff = product.payoff_path(path)

    # Should lock in 120 - 100 = 20
    expected = 20.0
    assert jnp.isclose(payoff, expected, atol=1e-6)


def test_percentage_ladder_put():
    """Test percentage ladder put."""
    product = PercentageLadderPut(
        K=100.0,
        T=1.0,
        rung_percentages=jnp.array([-0.10, -0.20, -0.30]),  # 90%, 80%, 70% of strike
    )

    # Verify rungs are calculated correctly
    expected_rungs = jnp.array([90.0, 80.0, 70.0])
    assert jnp.allclose(product.rungs, expected_rungs, atol=1e-6)

    # Path hits first two rungs
    path = jnp.array([100.0, 85.0, 75.0, 82.0])
    payoff = product.payoff_path(path)

    # Should lock in 100 - 80 = 20
    expected = 20.0
    assert jnp.isclose(payoff, expected, atol=1e-6)


def test_ladder_call_out_of_money():
    """Test ladder call that expires out of the money."""
    product = LadderCall(
        K=100.0,
        T=1.0,
        rungs=jnp.array([110.0, 120.0, 130.0]),
    )

    # Path never hits any rungs and ends below strike
    path = jnp.array([100.0, 105.0, 102.0, 95.0])
    payoff = product.payoff_path(path)

    # Should be zero (out of the money)
    assert jnp.isclose(payoff, 0.0, atol=1e-6)


def test_ladder_put_out_of_money():
    """Test ladder put that expires out of the money."""
    product = LadderPut(
        K=100.0,
        T=1.0,
        rungs=jnp.array([90.0, 80.0, 70.0]),
    )

    # Path never hits any rungs and ends above strike
    path = jnp.array([100.0, 95.0, 98.0, 105.0])
    payoff = product.payoff_path(path)

    # Should be zero (out of the money)
    assert jnp.isclose(payoff, 0.0, atol=1e-6)


def test_ladder_call_single_rung():
    """Test ladder call with single rung."""
    product = LadderCall(
        K=100.0,
        T=1.0,
        rungs=jnp.array([110.0]),
    )

    # Path hits the single rung
    path = jnp.array([100.0, 112.0, 105.0])
    payoff = product.payoff_path(path)

    # Should lock in 110 - 100 = 10
    expected = 10.0
    assert jnp.isclose(payoff, expected, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
