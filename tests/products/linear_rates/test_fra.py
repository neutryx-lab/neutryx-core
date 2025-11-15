"""Tests for Forward Rate Agreements."""
from __future__ import annotations

import jax.numpy as jnp
import pytest

from neutryx.products.linear_rates.fra import (
    ForwardRateAgreement,
    SettlementType,
    fra_forward_rate,
    fra_settlement_amount,
)


class TestForwardRateAgreement:
    """Test cases for Forward Rate Agreements."""

    def test_payer_fra_profit(self):
        """Test payer FRA profits when reference rate > fixed rate."""
        fra = ForwardRateAgreement(
            T=0.25,  # 3 months to settlement
            notional=1_000_000,
            fixed_rate=0.03,  # 3% agreed rate
            period_length=0.25,  # 3-month rate
            is_payer=True,
        )

        # Reference rate at settlement is 4%
        reference_rate = 0.04
        pv = fra.payoff_terminal(reference_rate)

        assert pv > 0, "Payer FRA should profit when reference rate > fixed rate"

    def test_receiver_fra_profit(self):
        """Test receiver FRA profits when fixed rate > reference rate."""
        fra = ForwardRateAgreement(
            T=0.25,
            notional=1_000_000,
            fixed_rate=0.05,  # 5% agreed rate
            period_length=0.25,
            is_payer=False,  # Receiver
        )

        # Reference rate at settlement is 3%
        reference_rate = 0.03
        pv = fra.payoff_terminal(reference_rate)

        assert pv > 0, "Receiver FRA should profit when fixed rate > reference rate"

    def test_fra_at_market_zero_value(self):
        """Test FRA has zero value when rates match."""
        fixed_rate = 0.04
        fra = ForwardRateAgreement(
            T=0.25,
            notional=1_000_000,
            fixed_rate=fixed_rate,
            period_length=0.25,
            is_payer=True,
        )

        pv = fra.payoff_terminal(fixed_rate)
        assert abs(pv) < 0.01, "FRA should have near-zero value when rates match"

    def test_advance_vs_arrears_settlement(self):
        """Test difference between advance and arrears settlement."""
        base_params = {
            "T": 0.25,
            "notional": 1_000_000,
            "fixed_rate": 0.03,
            "period_length": 0.25,
            "is_payer": True,
        }

        fra_advance = ForwardRateAgreement(
            **base_params, settlement_type=SettlementType.ADVANCE
        )

        fra_arrears = ForwardRateAgreement(
            **base_params, settlement_type=SettlementType.ARREARS
        )

        reference_rate = 0.04

        pv_advance = fra_advance.payoff_terminal(reference_rate)
        pv_arrears = fra_arrears.payoff_terminal(reference_rate)

        # Advance settlement should be slightly less due to discounting
        assert (
            pv_advance < pv_arrears
        ), "Advance settlement should have lower PV due to discounting"

    def test_fra_settlement_amount_calculation(self):
        """Test standalone settlement amount calculation."""
        notional = 1_000_000
        fixed_rate = 0.03
        reference_rate = 0.04
        day_count_factor = 0.25

        settlement = fra_settlement_amount(
            notional=notional,
            fixed_rate=fixed_rate,
            reference_rate=reference_rate,
            day_count_factor=day_count_factor,
            is_payer=True,
        )

        # Expected: 1M × (0.04 - 0.03) × 0.25 / (1 + 0.04 × 0.25)
        # = 1M × 0.01 × 0.25 / 1.01 = 2,500 / 1.01 ≈ 2,475
        expected = notional * (reference_rate - fixed_rate) * day_count_factor / (
            1 + reference_rate * day_count_factor
        )

        assert abs(settlement - expected) < 1, "Settlement amount should match formula"

    def test_fra_forward_rate_calculation(self):
        """Test implied forward rate from discount factors."""
        # Example: 3-month rate, discount factors imply 4% forward rate
        period_length = 0.25
        rate = 0.04

        df_start = jnp.exp(-rate * 0.25)  # Discount to 3 months
        df_end = jnp.exp(-rate * 0.50)  # Discount to 6 months

        forward_rate = fra_forward_rate(df_start, df_end, period_length)

        # Forward rate should be approximately 4%
        assert (
            abs(forward_rate - 0.04) < 0.001
        ), "Forward rate should match implied rate"

    def test_fra_dv01(self):
        """Test DV01 calculation for FRA."""
        fra = ForwardRateAgreement(
            T=0.25,
            notional=1_000_000,
            fixed_rate=0.03,
            period_length=0.25,
            is_payer=True,
        )

        reference_rate = 0.04
        dv01 = fra.dv01(reference_rate)

        # DV01 should be positive for payer FRA
        assert dv01 > 0, "DV01 should be positive for payer FRA"

        # Rough estimate: Notional × Period × DF ≈ 1M × 0.25 × 0.99 / 10000 ≈ 24.75
        expected_dv01 = (
            fra.notional
            * fra.period_length
            * jnp.exp(-fra.discount_rate * fra.T)
            / 10000
        )
        assert (
            abs(dv01 - expected_dv01) < 10
        ), "DV01 should be approximately notional × period × DF / 10000"

    def test_3x6_fra_notation(self):
        """Test 3x6 FRA (3 months to settlement, 3-month period)."""
        # 3x6 FRA: settles in 3 months, covers 3-month period
        fra_3x6 = ForwardRateAgreement(
            T=0.25,  # 3 months to settlement
            notional=10_000_000,
            fixed_rate=0.035,
            period_length=0.25,  # 3-month period
            is_payer=True,
        )

        reference_rate = 0.04
        pv = fra_3x6.payoff_terminal(reference_rate)

        assert pv > 0, "3x6 payer FRA should profit when rate rises"

    def test_6x12_fra_notation(self):
        """Test 6x12 FRA (6 months to settlement, 6-month period)."""
        # 6x12 FRA: settles in 6 months, covers 6-month period
        fra_6x12 = ForwardRateAgreement(
            T=0.5,  # 6 months to settlement
            notional=10_000_000,
            fixed_rate=0.04,
            period_length=0.5,  # 6-month period
            is_payer=True,
        )

        reference_rate = 0.045
        pv = fra_6x12.payoff_terminal(reference_rate)

        assert pv > 0, "6x12 payer FRA should profit when rate rises"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
