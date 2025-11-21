"""Tests for equity structured products."""
from __future__ import annotations

import jax.numpy as jnp
import pytest

from neutryx.products.structured import (
    AthenaAutocallable,
    CliquetOption,
    NapoleonOption,
)


class TestAthenaAutocallable:
    """Test cases for Athena (non-memory) Autocallable."""

    def test_autocall_at_first_observation(self):
        """Test autocall at first observation date."""
        K = 100.0
        athena = AthenaAutocallable(
            K=K,
            T=1.0,
            autocall_barrier=1.0,
            coupon_barrier=0.75,
            coupon_rate=0.05,
            observation_times=jnp.array([0.25, 0.5, 0.75, 1.0]),
            put_strike=1.0,
        )

        # Path autocalls at first observation (spot >= 100)
        path = jnp.linspace(100.0, 105.0, 100)
        payoff = athena.payoff_path(path)

        # Should return principal + 1 coupon = 1.05
        assert abs(payoff - 1.05) < 1e-6, "Should autocall with principal + coupon"

    def test_no_memory_coupon_lost(self):
        """Test that unpaid coupons are lost (no memory)."""
        K = 100.0
        athena = AthenaAutocallable(
            K=K,
            T=1.0,
            autocall_barrier=1.1,  # 110% - won't autocall
            coupon_barrier=0.80,
            coupon_rate=0.05,
            observation_times=jnp.array([0.25, 0.5, 0.75, 1.0]),
            put_strike=1.0,
        )

        # Path: 100 -> 75 (miss) -> 85 (hit) -> 75 (miss) -> 90 (hit at maturity)
        path = jnp.array([100.0, 75.0, 85.0, 75.0, 90.0])
        payoff = athena.payoff_path(path)

        # Coupons paid: periods 1, 2, 4 (above 80%)
        # Total coupons = 0.05 * 3 = 0.15
        # Payoff = 1.0 + 0.15 = 1.15
        expected = 1.0 + 0.15
        assert abs(payoff - expected) < 1e-2, "Should only pay hit coupons (no memory)"

    def test_maturity_below_barrier(self):
        """Test maturity payoff below coupon barrier."""
        K = 100.0
        athena = AthenaAutocallable(
            K=K,
            T=1.0,
            autocall_barrier=1.1,
            coupon_barrier=0.80,
            coupon_rate=0.05,
            observation_times=jnp.array([0.25, 0.5, 0.75, 1.0]),
            put_strike=1.0,
        )

        # Path ends at 70 (below barrier)
        path = jnp.array([100.0, 85.0, 80.0, 75.0, 70.0])
        payoff = athena.payoff_path(path)

        # Coupons paid at periods 1, 2 (above 80%), not at 3 and 4
        # At maturity: 70% < barrier, so put protection applies
        # Payoff = 0.70 + coupons paid
        # Coupons = 0.05 * 2 = 0.10
        # Total = 0.70 + 0.10 = 0.80
        assert 0.75 < payoff < 0.85, "Should apply put protection with paid coupons"

    def test_autocall_mid_life(self):
        """Test autocall in middle of life."""
        K = 100.0
        athena = AthenaAutocallable(
            K=K,
            T=1.0,
            autocall_barrier=1.0,
            coupon_barrier=0.75,
            coupon_rate=0.05,
            observation_times=jnp.array([0.25, 0.5, 0.75, 1.0]),
            put_strike=1.0,
        )

        # Path: below barrier initially, then hits autocall at period 3
        # 100 -> 80 -> 90 -> 105 (autocall here)
        n_steps = 100
        path = jnp.concatenate([
            jnp.linspace(100, 80, 25),  # First quarter
            jnp.linspace(80, 90, 25),  # Second quarter
            jnp.linspace(90, 105, 25),  # Third quarter (autocall)
            jnp.full(25, 105.0)  # Rest (won't be reached)
        ])

        payoff = athena.payoff_path(path)

        # Should autocall at third observation
        # Coupons paid at obs 1, 2, 3 (all above 75%)
        # Payoff = 1.0 + 0.05 + 0.05 + 0.05 = 1.15
        assert 1.10 < payoff < 1.20, "Should autocall with accumulated coupons"


class TestCliquetOption:
    """Test cases for Cliquet (Ratchet) options."""

    def test_local_cap_applied(self):
        """Test that local caps are applied to period returns."""
        K = 100.0
        cliquet = CliquetOption(
            K=K,
            T=1.0,
            reset_times=jnp.array([0.25, 0.5, 0.75, 1.0]),
            local_floor=0.0,
            local_cap=0.10,
            global_floor=0.0,
            global_cap=None,
        )

        # Path: 100 -> 120 (+20%, capped at 10%) -> 130 -> 140 -> 150
        # Period returns: 20%, 8.33%, 7.69%, 7.14%
        # After caps: 10%, 8.33%, 7.69%, 7.14%
        n_steps = 100
        path = jnp.linspace(100.0, 150.0, n_steps)
        payoff = cliquet.payoff_path(path)

        # Total return should be capped at 10% for first period
        # Approximate total: 10% + 8.33% + 7.69% + 7.14% ≈ 33%
        # Payoff ≈ 100 * 0.33 = 33, but could be slightly higher
        assert 30 < payoff < 42, "Should apply local caps to returns"

    def test_local_floor_applied(self):
        """Test that local floors protect against negative returns."""
        K = 100.0
        cliquet = CliquetOption(
            K=K,
            T=1.0,
            reset_times=jnp.array([0.25, 0.5, 0.75, 1.0]),
            local_floor=0.0,
            local_cap=0.20,
            global_floor=0.0,
            global_cap=None,
        )

        # Path: 100 -> 120 -> 100 (-16.67%, floored at 0%) -> 110 -> 115
        path = jnp.concatenate([
            jnp.linspace(100, 120, 25),
            jnp.linspace(120, 100, 25),
            jnp.linspace(100, 110, 25),
            jnp.linspace(110, 115, 25)
        ])

        payoff = cliquet.payoff_path(path)

        # Period 1: +20% (capped at 20%)
        # Period 2: -16.67% (floored at 0%)
        # Period 3: +10%
        # Period 4: +4.5%
        # Total: 20% + 0% + 10% + 4.5% = 34.5%
        # Payoff ≈ 100 * 0.345 = 34.5
        assert 30 < payoff < 40, "Should apply local floor to negative returns"

    def test_global_cap_applied(self):
        """Test that global cap limits total return."""
        K = 100.0
        cliquet = CliquetOption(
            K=K,
            T=1.0,
            reset_times=jnp.array([0.25, 0.5, 0.75, 1.0]),
            local_floor=0.0,
            local_cap=0.15,
            global_floor=0.0,
            global_cap=0.40,  # 40% global cap
        )

        # Path with strong returns that would exceed global cap
        path = jnp.array([100.0, 115.0, 132.0, 152.0, 175.0])
        payoff = cliquet.payoff_path(path)

        # Without global cap, returns would be higher
        # With 40% global cap: payoff = 100 * 0.40 = 40
        assert abs(payoff - 40.0) < 5.0, "Should apply global cap"

    def test_global_floor_applied(self):
        """Test that global floor provides minimum return."""
        K = 100.0
        cliquet = CliquetOption(
            K=K,
            T=1.0,
            reset_times=jnp.array([0.25, 0.5, 0.75, 1.0]),
            local_floor=0.0,
            local_cap=0.10,
            global_floor=0.10,  # 10% minimum return
            global_cap=None,
        )

        # Path with all negative returns (but floored locally at 0%)
        path = jnp.array([100.0, 95.0, 90.0, 85.0, 80.0])
        payoff = cliquet.payoff_path(path)

        # All local returns negative -> floored at 0%
        # Total accumulated = 0%
        # Global floor = 10%
        # Payoff = 100 * 0.10 = 10
        assert abs(payoff - 10.0) < 1.0, "Should apply global floor"


class TestNapoleonOption:
    """Test cases for Napoleon options."""

    def test_guaranteed_return_applied(self):
        """Test that guaranteed return is paid when market underperforms."""
        K = 100.0
        napoleon = NapoleonOption(
            K=K,
            T=1.0,
            reset_times=jnp.array([0.25, 0.5, 0.75, 1.0]),
            local_floor=0.0,
            local_cap=0.10,
            guaranteed_return=0.05,  # 5% guaranteed
        )

        # Path with weak returns
        path = jnp.array([100.0, 101.0, 102.0, 103.0, 104.0])
        payoff = napoleon.payoff_path(path)

        # Accumulated return ≈ 4% (less than 5% guaranteed)
        # Should pay guaranteed return: 100 * 0.05 = 5
        assert abs(payoff - 5.0) < 1.0, "Should pay guaranteed minimum return"

    def test_market_return_exceeds_guaranteed(self):
        """Test that market return is paid when it exceeds guaranteed return."""
        K = 100.0
        napoleon = NapoleonOption(
            K=K,
            T=1.0,
            reset_times=jnp.array([0.25, 0.5, 0.75, 1.0]),
            local_floor=0.0,
            local_cap=0.10,
            guaranteed_return=0.05,
        )

        # Path with strong returns
        path = jnp.array([100.0, 110.0, 121.0, 133.0, 146.0])
        payoff = napoleon.payoff_path(path)

        # Each period +10% (capped at 10%)
        # Total: 40% > 5% guaranteed
        # Payoff = 100 * 0.40 = 40
        assert abs(payoff - 40.0) < 2.0, "Should pay market return when higher"

    def test_negative_returns_with_guaranteed(self):
        """Test guaranteed return with negative market returns."""
        K = 100.0
        napoleon = NapoleonOption(
            K=K,
            T=1.0,
            reset_times=jnp.array([0.25, 0.5, 0.75, 1.0]),
            local_floor=0.0,
            local_cap=0.10,
            guaranteed_return=0.03,
        )

        # Path with losses
        path = jnp.array([100.0, 95.0, 90.0, 85.0, 80.0])
        payoff = napoleon.payoff_path(path)

        # All returns negative, floored at 0%, total = 0%
        # Guaranteed return = 3%
        # Payoff = 100 * 0.03 = 3
        assert abs(payoff - 3.0) < 0.5, "Should pay guaranteed return on losses"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
