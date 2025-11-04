"""Tests for advanced interest rate products."""
import jax.numpy as jnp
import pytest

from neutryx.products.advanced_rates import (
    BermudanSwaption,
    CallablePutableBond,
    CMSSpreadRangeAccrual,
    ConstantMaturitySwap,
    RangeAccrualSwap,
    TargetRedemptionNote,
    YieldCurveOption,
)


class TestBermudanSwaption:
    """Test Bermudan swaption implementation."""

    def test_bermudan_swaption_creation(self):
        """Test basic Bermudan swaption instantiation."""
        exercise_dates = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        swaption = BermudanSwaption(
            T=5.0,
            K=0.05,
            notional=1_000_000,
            exercise_dates=exercise_dates,
            option_type='payer',
        )
        assert swaption.T == 5.0
        assert swaption.K == 0.05
        assert len(swaption.exercise_dates) == 5

    def test_intrinsic_value_payer(self):
        """Test intrinsic value for payer swaption."""
        exercise_dates = jnp.array([1.0, 2.0, 3.0])
        swaption = BermudanSwaption(
            T=3.0,
            K=0.05,
            notional=1_000_000,
            exercise_dates=exercise_dates,
            option_type='payer',
            tenor=10.0,
        )
        # Swap rate above strike - positive intrinsic
        intrinsic = swaption.intrinsic_value(0.06, 1.0)
        assert intrinsic > 0

        # Swap rate below strike - zero intrinsic
        intrinsic = swaption.intrinsic_value(0.04, 1.0)
        assert jnp.allclose(intrinsic, 0.0)

    def test_intrinsic_value_receiver(self):
        """Test intrinsic value for receiver swaption."""
        exercise_dates = jnp.array([1.0, 2.0, 3.0])
        swaption = BermudanSwaption(
            T=3.0,
            K=0.05,
            notional=1_000_000,
            exercise_dates=exercise_dates,
            option_type='receiver',
        )
        # Swap rate below strike - positive intrinsic
        intrinsic = swaption.intrinsic_value(0.04, 1.0)
        assert intrinsic > 0

    def test_payoff_path(self):
        """Test payoff calculation along a path."""
        exercise_dates = jnp.array([0.5, 1.0, 1.5])
        swaption = BermudanSwaption(
            T=2.0,
            K=0.05,
            notional=1_000_000,
            exercise_dates=exercise_dates,
        )
        # Simplified path
        path = jnp.linspace(0.04, 0.06, 20)
        payoff = swaption.payoff_path(path)
        assert payoff >= 0


class TestCallablePutableBond:
    """Test callable/putable bond."""

    def test_callable_bond_creation(self):
        """Test callable bond instantiation."""
        call_dates = jnp.array([3.0, 4.0, 5.0])
        call_prices = jnp.array([102.0, 101.0, 100.0])
        bond = CallablePutableBond(
            T=5.0,
            face_value=100.0,
            coupon_rate=0.05,
            call_dates=call_dates,
            call_prices=call_prices,
        )
        assert bond.face_value == 100.0
        assert len(bond.call_dates) == 3

    def test_putable_bond_creation(self):
        """Test putable bond instantiation."""
        put_dates = jnp.array([3.0, 4.0])
        put_prices = jnp.array([98.0, 99.0])
        bond = CallablePutableBond(
            T=5.0,
            face_value=100.0,
            coupon_rate=0.05,
            put_dates=put_dates,
            put_prices=put_prices,
        )
        assert len(bond.put_dates) == 2

    def test_bond_payoff_path(self):
        """Test bond value calculation."""
        bond = CallablePutableBond(
            T=5.0, face_value=100.0, coupon_rate=0.05, payment_freq=2
        )
        path = jnp.linspace(0.03, 0.04, 20)
        payoff = bond.payoff_path(path)
        assert payoff > 0


class TestCMSSpreadRangeAccrual:
    """Test CMS spread range accrual."""

    def test_cms_range_accrual_creation(self):
        """Test CMS range accrual instantiation."""
        note = CMSSpreadRangeAccrual(
            T=5.0,
            notional=1_000_000,
            base_rate=0.03,
            spread_lower=0.01,
            spread_upper=0.03,
            cms_tenor_1=10.0,
            cms_tenor_2=2.0,
        )
        assert note.spread_lower == 0.01
        assert note.spread_upper == 0.03

    def test_cms_range_accrual_full_accrual(self):
        """Test full accrual when always in range."""
        note = CMSSpreadRangeAccrual(
            T=1.0,
            notional=1_000_000,
            base_rate=0.03,
            spread_lower=0.01,
            spread_upper=0.03,
        )
        # CMS rates with spread always in range
        cms_10y = jnp.ones(252) * 0.05
        cms_2y = jnp.ones(252) * 0.03
        path = jnp.array([cms_10y, cms_2y])
        payoff = note.payoff_path(path)

        # Full accrual: 1M * 3% * 1.0 = 30,000
        assert jnp.allclose(payoff, 30_000, rtol=0.01)

    def test_cms_range_accrual_no_accrual(self):
        """Test no accrual when always out of range."""
        note = CMSSpreadRangeAccrual(
            T=1.0,
            notional=1_000_000,
            base_rate=0.03,
            spread_lower=0.01,
            spread_upper=0.03,
        )
        # CMS rates with spread always out of range
        cms_10y = jnp.ones(252) * 0.06
        cms_2y = jnp.ones(252) * 0.02
        path = jnp.array([cms_10y, cms_2y])
        payoff = note.payoff_path(path)

        # No accrual
        assert jnp.allclose(payoff, 0.0, atol=100)


class TestConstantMaturitySwap:
    """Test constant maturity swap."""

    def test_cms_creation(self):
        """Test CMS instantiation."""
        cms = ConstantMaturitySwap(
            T=5.0,
            notional=1_000_000,
            cms_tenor=10.0,
            fixed_rate=0.03,
            is_fixed_vs_cms=True,
        )
        assert cms.cms_tenor == 10.0
        assert cms.fixed_rate == 0.03

    def test_cms_payoff(self):
        """Test CMS payoff calculation."""
        cms = ConstantMaturitySwap(
            T=5.0,
            notional=1_000_000,
            cms_tenor=10.0,
            fixed_rate=0.03,
            payment_freq=2,
        )
        cms_rate = jnp.array(0.04)
        payoff = cms.payoff_terminal(cms_rate)

        # Receive CMS, pay fixed: (0.04 - 0.03) * 5 * 2 * 0.5 = 0.05
        expected = 1_000_000 * 0.01 * 5
        assert jnp.allclose(payoff, expected)

    def test_convexity_adjustment(self):
        """Test CMS convexity adjustment calculation."""
        cms = ConstantMaturitySwap(
            T=5.0, notional=1_000_000, cms_tenor=10.0, fixed_rate=0.03
        )
        adjustment = cms.convexity_adjustment(
            forward_rate=0.04, volatility=0.2, time_to_expiry=1.0
        )
        assert adjustment >= 0


class TestYieldCurveOption:
    """Test yield curve options."""

    def test_steepener_option(self):
        """Test curve steepener option."""
        option = YieldCurveOption(
            T=1.0,
            notional=1_000_000,
            option_type='steepener',
            K=0.01,
            tenors=jnp.array([2.0, 10.0]),
        )
        # Steepening curve: 10Y - 2Y = 0.05 - 0.03 = 0.02 > 0.01
        rates = jnp.array([0.03, 0.05])
        payoff = option.payoff_terminal(rates)

        # Payoff = max(0.02 - 0.01, 0) * 1M = 10,000
        assert jnp.allclose(payoff, 10_000)

    def test_flattener_option(self):
        """Test curve flattener option."""
        option = YieldCurveOption(
            T=1.0,
            notional=1_000_000,
            option_type='flattener',
            K=0.02,
            tenors=jnp.array([2.0, 10.0]),
        )
        # Flattening curve: spread = 0.01 < 0.02
        rates = jnp.array([0.03, 0.04])
        payoff = option.payoff_terminal(rates)

        # Payoff = max(0.02 - 0.01, 0) * 1M = 10,000
        assert jnp.allclose(payoff, 10_000)

    def test_butterfly_option(self):
        """Test butterfly option on curve curvature."""
        option = YieldCurveOption(
            T=1.0,
            notional=1_000_000,
            option_type='butterfly',
            K=0.0,
            tenors=jnp.array([2.0, 5.0, 10.0]),
        )
        # Positive curvature: 2*5Y - 2Y - 10Y
        rates = jnp.array([0.03, 0.04, 0.05])
        payoff = option.payoff_terminal(rates)

        # Curvature = 2*0.04 - 0.03 - 0.05 = 0
        assert payoff >= 0


class TestRangeAccrualSwap:
    """Test range accrual swap."""

    def test_range_accrual_creation(self):
        """Test range accrual swap instantiation."""
        swap = RangeAccrualSwap(
            T=5.0,
            notional=1_000_000,
            fixed_rate=0.03,
            range_lower=0.02,
            range_upper=0.05,
        )
        assert swap.range_lower == 0.02
        assert swap.range_upper == 0.05

    def test_range_accrual_full_accrual(self):
        """Test swap with full accrual."""
        swap = RangeAccrualSwap(
            T=1.0,
            notional=1_000_000,
            fixed_rate=0.03,
            range_lower=0.02,
            range_upper=0.06,
        )
        # Rates always in range
        path = jnp.ones(252) * 0.04
        payoff = swap.payoff_path(path)

        # Floating - Fixed = 0.04 - 0.03 = 0.01
        expected = 1_000_000 * 0.01
        assert jnp.allclose(payoff, expected)


class TestTargetRedemptionNote:
    """Test target redemption note (TARN)."""

    def test_tarn_creation(self):
        """Test TARN instantiation."""
        tarn = TargetRedemptionNote(
            T=5.0,
            notional=1_000_000,
            target_coupon=100_000,
            coupon_rate=0.05,
            payment_freq=4,
        )
        assert tarn.target_coupon == 100_000
        assert tarn.coupon_rate == 0.05

    def test_tarn_early_redemption(self):
        """Test TARN with early redemption."""
        tarn = TargetRedemptionNote(
            T=5.0,
            notional=100.0,
            target_coupon=10.0,
            coupon_rate=0.05,
            payment_freq=4,
        )
        # High rates to trigger early redemption
        path = jnp.ones(20) * 0.06
        payoff = tarn.payoff_path(path)

        # Should include principal
        assert payoff >= 100.0

    def test_tarn_no_early_redemption(self):
        """Test TARN without early redemption."""
        tarn = TargetRedemptionNote(
            T=5.0,
            notional=100.0,
            target_coupon=50.0,
            coupon_rate=0.01,
            payment_freq=2,
        )
        # Low rates - won't hit target
        path = jnp.ones(20) * 0.01
        payoff = tarn.payoff_path(path)

        # Should include principal at maturity
        assert payoff >= 100.0
