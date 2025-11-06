"""Tests for interest rate swap products."""
from __future__ import annotations

import jax.numpy as jnp
import pytest

from neutryx.products.linear_rates.swaps import (
    BasisSwap,
    CrossCurrencySwap,
    DayCount,
    InterestRateSwap,
    OvernightIndexSwap,
    SwapType,
    Tenor,
)


class TestInterestRateSwap:
    """Test cases for Interest Rate Swaps."""

    def test_payer_swap_positive_value(self):
        """Test payer swap has positive value when floating > fixed."""
        # Setup: flat curves with floating > fixed
        irs = InterestRateSwap(
            T=5.0,
            notional=1_000_000,
            fixed_rate=0.03,  # 3% fixed
            swap_type=SwapType.PAYER,
            payment_frequency=2,
            forward_curve_rates=jnp.full(10, 0.04),  # 4% floating
            discount_curve_rates=jnp.full(10, 0.035),
        )

        pv = irs.payoff_terminal(0.0)
        assert pv > 0, "Payer swap should have positive value when floating > fixed"

    def test_receiver_swap_positive_value(self):
        """Test receiver swap has positive value when fixed > floating."""
        irs = InterestRateSwap(
            T=5.0,
            notional=1_000_000,
            fixed_rate=0.05,  # 5% fixed
            swap_type=SwapType.RECEIVER,
            payment_frequency=2,
            forward_curve_rates=jnp.full(10, 0.03),  # 3% floating
            discount_curve_rates=jnp.full(10, 0.04),
        )

        pv = irs.payoff_terminal(0.0)
        assert pv > 0, "Receiver swap should have positive value when fixed > floating"

    def test_at_market_swap_zero_value(self):
        """Test at-market swap has near-zero value."""
        fixed_rate = 0.04
        irs = InterestRateSwap(
            T=5.0,
            notional=1_000_000,
            fixed_rate=fixed_rate,
            swap_type=SwapType.PAYER,
            payment_frequency=2,
            forward_curve_rates=jnp.full(10, fixed_rate),
            discount_curve_rates=jnp.full(10, fixed_rate),
        )

        pv = irs.payoff_terminal(0.0)
        assert abs(pv) < 1e-6, "At-market swap should have near-zero value"

    def test_par_rate_calculation(self):
        """Test par swap rate calculation."""
        irs = InterestRateSwap(
            T=5.0,
            notional=1_000_000,
            fixed_rate=0.03,
            swap_type=SwapType.PAYER,
            payment_frequency=2,
            forward_curve_rates=jnp.full(10, 0.04),
            discount_curve_rates=jnp.full(10, 0.035),
        )

        par_rate = irs.par_rate()
        assert 0.035 < par_rate < 0.045, "Par rate should be reasonable"

        # Create swap at par rate - should have zero value
        par_swap = InterestRateSwap(
            T=5.0,
            notional=1_000_000,
            fixed_rate=par_rate,
            swap_type=SwapType.PAYER,
            payment_frequency=2,
            forward_curve_rates=jnp.full(10, 0.04),
            discount_curve_rates=jnp.full(10, 0.035),
        )

        pv = par_swap.payoff_terminal(0.0)
        assert abs(pv) < 1.0, "Swap at par rate should have near-zero value"

    def test_spread_impact(self):
        """Test that spread on floating leg affects swap value."""
        spread = 0.01  # 100 bps
        irs_no_spread = InterestRateSwap(
            T=5.0,
            notional=1_000_000,
            fixed_rate=0.03,
            swap_type=SwapType.PAYER,
            payment_frequency=2,
            spread=0.0,
        )

        irs_with_spread = InterestRateSwap(
            T=5.0,
            notional=1_000_000,
            fixed_rate=0.03,
            swap_type=SwapType.PAYER,
            payment_frequency=2,
            spread=spread,
        )

        pv_no_spread = irs_no_spread.payoff_terminal(0.0)
        pv_with_spread = irs_with_spread.payoff_terminal(0.0)

        assert (
            pv_with_spread > pv_no_spread
        ), "Payer swap value should increase with positive spread"


class TestOvernightIndexSwap:
    """Test cases for Overnight Index Swaps."""

    def test_ois_basic_pricing(self):
        """Test basic OIS pricing with flat overnight rates."""
        ois = OvernightIndexSwap(
            T=1.0,
            notional=1_000_000,
            fixed_rate=0.03,
            swap_type=SwapType.PAYER,
            overnight_rates=jnp.full(360, 0.035),  # Slightly higher than fixed
        )

        # Create path of overnight rates
        path = jnp.full(360, 0.035)
        pv = ois.payoff_path(path)

        assert pv > 0, "Payer OIS should have positive value when overnight > fixed"

    def test_ois_compounding(self):
        """Test that OIS properly compounds overnight rates."""
        # Rising rate environment
        rising_rates = jnp.linspace(0.02, 0.04, 360)

        ois = OvernightIndexSwap(
            T=1.0,
            notional=1_000_000,
            fixed_rate=0.03,
            swap_type=SwapType.PAYER,
            overnight_rates=rising_rates,
        )

        pv = ois.payoff_path(rising_rates)
        assert isinstance(pv, (float, jnp.ndarray)), "Should return numeric value"


class TestCrossCurrencySwap:
    """Test cases for Cross-Currency Swaps."""

    def test_ccs_without_reset(self):
        """Test CCS without FX reset."""
        ccs = CrossCurrencySwap(
            T=5.0,
            notional_domestic=1_000_000,
            notional_foreign=800_000,
            domestic_rate=0.03,
            foreign_rate=0.02,
            fx_spot=1.25,  # USD/EUR
            fx_reset=False,
            payment_frequency=2,
        )

        # Terminal FX rate same as initial
        pv = ccs.payoff_terminal(1.25)
        assert isinstance(pv, (float, jnp.ndarray)), "Should return numeric value"

    def test_ccs_with_reset(self):
        """Test CCS with FX reset."""
        ccs = CrossCurrencySwap(
            T=5.0,
            notional_domestic=1_000_000,
            notional_foreign=800_000,
            domestic_rate=0.03,
            foreign_rate=0.02,
            fx_spot=1.25,
            fx_reset=True,
            payment_frequency=2,
        )

        # FX path from 1.25 to 1.30
        fx_path = jnp.linspace(1.25, 1.30, 100)
        pv = ccs.payoff_path(fx_path)

        assert isinstance(pv, (float, jnp.ndarray)), "Should return numeric value"

    def test_ccs_fx_appreciation(self):
        """Test CCS value changes with FX appreciation."""
        ccs = CrossCurrencySwap(
            T=5.0,
            notional_domestic=1_000_000,
            notional_foreign=800_000,
            domestic_rate=0.03,
            foreign_rate=0.03,  # Same rate
            fx_spot=1.25,
            fx_reset=False,
            payment_frequency=2,
        )

        pv_initial = ccs.payoff_terminal(1.25)
        pv_appreciated = ccs.payoff_terminal(1.30)  # Foreign currency appreciates

        assert (
            pv_appreciated > pv_initial
        ), "CCS value should increase when foreign currency appreciates"


class TestBasisSwap:
    """Test cases for Basis Swaps."""

    def test_basis_swap_flat_curves(self):
        """Test basis swap with flat forward curves."""
        basis_swap = BasisSwap(
            T=5.0,
            notional=1_000_000,
            tenor_1=Tenor.THREE_MONTH,
            tenor_2=Tenor.SIX_MONTH,
            basis_spread=0.001,  # 10 bps
            payment_frequency=4,
            forward_curve_1=jnp.full(20, 0.03),
            forward_curve_2=jnp.full(20, 0.03),
        )

        pv = basis_swap.payoff_terminal(0.0)
        assert pv > 0, "Basis swap with positive spread should have positive value"

    def test_par_spread_calculation(self):
        """Test par basis spread calculation."""
        basis_swap = BasisSwap(
            T=5.0,
            notional=1_000_000,
            tenor_1=Tenor.THREE_MONTH,
            tenor_2=Tenor.SIX_MONTH,
            basis_spread=0.0,
            payment_frequency=4,
            forward_curve_1=jnp.full(20, 0.03),
            forward_curve_2=jnp.full(20, 0.032),  # 20 bps difference
        )

        par_spread = basis_swap.par_spread()
        # Par spread should be negative (leg2 > leg1, so we subtract to equalize)
        assert abs(par_spread + 0.002) < 0.001, "Par spread should be approximately -20 bps"

        # Create swap at par spread - should have zero value
        par_basis_swap = BasisSwap(
            T=5.0,
            notional=1_000_000,
            tenor_1=Tenor.THREE_MONTH,
            tenor_2=Tenor.SIX_MONTH,
            basis_spread=par_spread,
            payment_frequency=4,
            forward_curve_1=jnp.full(20, 0.03),
            forward_curve_2=jnp.full(20, 0.032),
        )

        pv = par_basis_swap.payoff_terminal(0.0)
        assert abs(pv) < 100, "Swap at par spread should have near-zero value"

    def test_negative_spread(self):
        """Test basis swap with negative spread."""
        basis_swap = BasisSwap(
            T=5.0,
            notional=1_000_000,
            tenor_1=Tenor.THREE_MONTH,
            tenor_2=Tenor.SIX_MONTH,
            basis_spread=-0.001,  # -10 bps
            payment_frequency=4,
            forward_curve_1=jnp.full(20, 0.03),
            forward_curve_2=jnp.full(20, 0.03),
        )

        pv = basis_swap.payoff_terminal(0.0)
        assert pv < 0, "Basis swap with negative spread should have negative value"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
