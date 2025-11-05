"""Tests for ladder option products."""
from __future__ import annotations

import jax.numpy as jnp
import pytest

from neutryx.products.ladder import (
    LadderCall,
    LadderPut,
    PercentageLadderCall,
    PercentageLadderPut,
)


class TestLadderCall:
    """Test cases for Ladder Call options."""

    def test_no_rungs_hit_vanilla_payoff(self):
        """Test ladder call behaves like vanilla when no rungs are hit."""
        K = 100.0
        rungs = jnp.array([110.0, 120.0, 130.0])
        ladder = LadderCall(K=K, T=1.0, rungs=rungs)

        # Path that never hits any rungs, ends at 105
        path = jnp.array([100.0, 102.0, 105.0, 103.0, 105.0])
        payoff = ladder.payoff_path(path)

        # Should be max(105 - 100, 0) = 5
        assert abs(payoff - 5.0) < 1e-6, "Should equal vanilla call payoff"

    def test_rung_locks_in_profit(self):
        """Test that hitting a rung locks in profit."""
        K = 100.0
        rungs = jnp.array([110.0, 120.0, 130.0])
        ladder = LadderCall(K=K, T=1.0, rungs=rungs)

        # Path hits 120 (second rung), then falls to 105
        path = jnp.array([100.0, 115.0, 122.0, 110.0, 105.0])
        payoff = ladder.payoff_path(path)

        # Locked in at 120, so payoff = max(120 - 100, 105 - 100) = 20
        assert abs(payoff - 20.0) < 1e-6, "Should lock in rung at 120"

    def test_terminal_exceeds_locked_rung(self):
        """Test terminal value exceeding locked rung."""
        K = 100.0
        rungs = jnp.array([110.0, 120.0, 130.0])
        ladder = LadderCall(K=K, T=1.0, rungs=rungs)

        # Path hits 115 (first rung), then rises to 135
        path = jnp.array([100.0, 112.0, 115.0, 125.0, 135.0])
        payoff = ladder.payoff_path(path)

        # Locked in at 110, but terminal is 135 - 100 = 35 > 10
        assert abs(payoff - 35.0) < 1e-6, "Should take higher terminal value"

    def test_all_rungs_hit(self):
        """Test when all rungs are hit."""
        K = 100.0
        rungs = jnp.array([110.0, 120.0, 130.0])
        ladder = LadderCall(K=K, T=1.0, rungs=rungs)

        # Path hits all rungs, ends at 125
        path = jnp.array([100.0, 115.0, 125.0, 135.0, 125.0])
        payoff = ladder.payoff_path(path)

        # Highest rung hit is 130, terminal is 125 - 100 = 25
        # Should take max(130 - 100, 25) = 30
        assert abs(payoff - 30.0) < 1e-6, "Should lock in highest rung hit"

    def test_out_of_money_with_rungs_hit(self):
        """Test OTM terminal with rungs hit still pays locked profit."""
        K = 100.0
        rungs = jnp.array([110.0, 120.0, 130.0])
        ladder = LadderCall(K=K, T=1.0, rungs=rungs)

        # Path hits rungs, ends below strike
        path = jnp.array([100.0, 115.0, 122.0, 110.0, 95.0])
        payoff = ladder.payoff_path(path)

        # Locked in at 120, terminal is 95 - 100 = -5 (but floored at 0)
        # Payoff = max(120 - 100, 0) = 20
        assert abs(payoff - 20.0) < 1e-6, "Should still pay locked profit"


class TestLadderPut:
    """Test cases for Ladder Put options."""

    def test_no_rungs_hit_vanilla_payoff(self):
        """Test ladder put behaves like vanilla when no rungs are hit."""
        K = 100.0
        rungs = jnp.array([90.0, 80.0, 70.0])
        ladder = LadderPut(K=K, T=1.0, rungs=rungs)

        # Path never hits any rungs, ends at 95
        path = jnp.array([100.0, 98.0, 95.0, 97.0, 95.0])
        payoff = ladder.payoff_path(path)

        # Should be max(100 - 95, 0) = 5
        assert abs(payoff - 5.0) < 1e-6, "Should equal vanilla put payoff"

    def test_rung_locks_in_profit(self):
        """Test that hitting a rung locks in profit for put."""
        K = 100.0
        rungs = jnp.array([90.0, 80.0, 70.0])
        ladder = LadderPut(K=K, T=1.0, rungs=rungs)

        # Path hits 80 (second rung), then rises to 95
        path = jnp.array([100.0, 85.0, 78.0, 90.0, 95.0])
        payoff = ladder.payoff_path(path)

        # Locked in at 80, so payoff = max(100 - 80, 100 - 95) = 20
        assert abs(payoff - 20.0) < 1e-6, "Should lock in rung at 80"

    def test_terminal_exceeds_locked_rung(self):
        """Test terminal value exceeding locked rung for put."""
        K = 100.0
        rungs = jnp.array([90.0, 80.0, 70.0])
        ladder = LadderPut(K=K, T=1.0, rungs=rungs)

        # Path hits 85 (first rung), then falls to 65
        path = jnp.array([100.0, 88.0, 85.0, 75.0, 65.0])
        payoff = ladder.payoff_path(path)

        # Locked in at 90, but terminal is 100 - 65 = 35 > 10
        assert abs(payoff - 35.0) < 1e-6, "Should take higher terminal value"


class TestPercentageLadderCall:
    """Test cases for Percentage Ladder Call options."""

    def test_percentage_conversion(self):
        """Test that percentage rungs are converted correctly."""
        K = 100.0
        rung_percentages = jnp.array([0.10, 0.20, 0.30])
        ladder = PercentageLadderCall(K=K, T=1.0, rung_percentages=rung_percentages)

        # Rungs should be at 110, 120, 130
        expected_rungs = jnp.array([110.0, 120.0, 130.0])
        assert jnp.allclose(ladder.rungs, expected_rungs), "Rungs should match expected values"

    def test_percentage_ladder_payoff(self):
        """Test percentage ladder call payoff."""
        K = 100.0
        rung_percentages = jnp.array([0.10, 0.20, 0.30])
        ladder = PercentageLadderCall(K=K, T=1.0, rung_percentages=rung_percentages)

        # Path hits 115 (10% rung), ends at 105
        path = jnp.array([100.0, 112.0, 115.0, 110.0, 105.0])
        payoff = ladder.payoff_path(path)

        # Locked in at 110 (10% rung), terminal is 5
        assert abs(payoff - 10.0) < 1e-6, "Should lock in 10% rung"


class TestPercentageLadderPut:
    """Test cases for Percentage Ladder Put options."""

    def test_percentage_conversion(self):
        """Test that percentage rungs are converted correctly for put."""
        K = 100.0
        rung_percentages = jnp.array([-0.10, -0.20, -0.30])
        ladder = PercentageLadderPut(K=K, T=1.0, rung_percentages=rung_percentages)

        # Rungs should be at 90, 80, 70
        expected_rungs = jnp.array([90.0, 80.0, 70.0])
        assert jnp.allclose(ladder.rungs, expected_rungs), "Rungs should match expected values"

    def test_percentage_ladder_put_payoff(self):
        """Test percentage ladder put payoff."""
        K = 100.0
        rung_percentages = jnp.array([-0.10, -0.20, -0.30])
        ladder = PercentageLadderPut(K=K, T=1.0, rung_percentages=rung_percentages)

        # Path hits 85 (10% rung), ends at 95
        path = jnp.array([100.0, 88.0, 85.0, 90.0, 95.0])
        payoff = ladder.payoff_path(path)

        # Locked in at 90 (10% rung), terminal is 5
        assert abs(payoff - 10.0) < 1e-6, "Should lock in 10% rung"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
