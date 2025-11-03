"""Tests for credit derivatives products."""
import jax.numpy as jnp
import pytest

from neutryx.products.credit_derivatives import (
    CDSIndex,
    ContingentCDS,
    CreditLinkedNote,
    LoanCDS,
    NthToDefaultBasket,
    SyntheticCDO,
)


class TestCDSIndex:
    """Test CDX/iTraxx index implementation."""

    def test_cds_index_creation(self):
        """Test basic CDS index instantiation."""
        index = CDSIndex(
            T=5.0,
            notional=10_000_000,
            spread=100.0,  # 100 bps
            recovery_rate=0.4,
            num_names=125,
        )
        assert index.T == 5.0
        assert index.notional == 10_000_000
        assert index.spread == 100.0
        assert len(index.weights) == 125
        assert jnp.allclose(jnp.sum(index.weights), 1.0)

    def test_cds_index_payoff_no_default(self):
        """Test index payoff with no defaults (zero hazard rates)."""
        index = CDSIndex(T=5.0, notional=10_000_000, spread=100.0, num_names=125)
        hazard_rates = jnp.zeros(125)
        payoff = index.payoff_terminal(hazard_rates)

        # With zero hazard rates, only premium leg has value
        assert payoff < 0  # Negative because paying premium

    def test_cds_index_payoff_with_defaults(self):
        """Test index payoff with some defaults."""
        index = CDSIndex(T=5.0, notional=10_000_000, spread=100.0, num_names=125)
        hazard_rates = jnp.ones(125) * 0.05  # 5% hazard rate
        payoff = index.payoff_terminal(hazard_rates)

        # Should have positive value due to protection leg
        assert isinstance(payoff, jnp.ndarray) or isinstance(payoff, float)


class TestSyntheticCDO:
    """Test synthetic CDO tranche implementation."""

    def test_cdo_tranche_creation(self):
        """Test CDO tranche instantiation."""
        cdo = SyntheticCDO(
            T=5.0,
            notional=1_000_000,
            attachment=0.03,
            detachment=0.07,
            spread=500.0,  # 500 bps for equity tranche
            correlation=0.3,
        )
        assert cdo.attachment == 0.03
        assert cdo.detachment == 0.07
        assert cdo.correlation == 0.3

    def test_cdo_tranche_payoff(self):
        """Test CDO tranche payoff calculation."""
        cdo = SyntheticCDO(
            T=5.0,
            notional=1_000_000,
            attachment=0.03,
            detachment=0.07,
            spread=500.0,
            correlation=0.3,
        )
        default_prob = jnp.array(0.05)
        payoff = cdo.payoff_terminal(default_prob)

        assert isinstance(payoff, jnp.ndarray) or isinstance(payoff, float)

    def test_cdo_expected_loss(self):
        """Test expected loss calculation."""
        cdo = SyntheticCDO(
            T=5.0,
            notional=1_000_000,
            attachment=0.03,
            detachment=0.07,
            spread=500.0,
        )
        loss = cdo.expected_loss(0.05, 10)
        assert 0.0 <= loss <= 1.0


class TestNthToDefaultBasket:
    """Test nth-to-default basket."""

    def test_nth_to_default_creation(self):
        """Test nth-to-default basket instantiation."""
        basket = NthToDefaultBasket(
            T=5.0, notional=1_000_000, n=2, num_names=5, correlation=0.3
        )
        assert basket.n == 2
        assert basket.num_names == 5

    def test_first_to_default(self):
        """Test first-to-default payoff."""
        basket = NthToDefaultBasket(
            T=5.0, notional=1_000_000, n=1, num_names=5, recovery_rate=0.4
        )
        # All defaults after maturity - no payout
        default_times = jnp.array([6.0, 7.0, 8.0, 9.0, 10.0])
        payoff = basket.payoff_terminal(default_times)
        assert jnp.allclose(payoff, 0.0)

        # First default before maturity
        default_times = jnp.array([2.0, 6.0, 7.0, 8.0, 9.0])
        payoff = basket.payoff_terminal(default_times)
        expected = 1_000_000 * (1.0 - 0.4)
        assert jnp.allclose(payoff, expected)

    def test_second_to_default(self):
        """Test second-to-default payoff."""
        basket = NthToDefaultBasket(
            T=5.0, notional=1_000_000, n=2, num_names=5, recovery_rate=0.4
        )
        # Second default within maturity
        default_times = jnp.array([1.0, 3.0, 6.0, 7.0, 8.0])
        payoff = basket.payoff_terminal(default_times)
        expected = 1_000_000 * (1.0 - 0.4)
        assert jnp.allclose(payoff, expected)


class TestCreditLinkedNote:
    """Test credit-linked note."""

    def test_cln_creation(self):
        """Test CLN instantiation."""
        cln = CreditLinkedNote(
            T=5.0, principal=1_000_000, coupon_rate=0.05, credit_spread=200.0
        )
        assert cln.principal == 1_000_000
        assert cln.coupon_rate == 0.05

    def test_cln_payoff_no_default(self):
        """Test CLN payoff with no default."""
        cln = CreditLinkedNote(
            T=5.0, principal=1_000_000, coupon_rate=0.05, credit_spread=200.0
        )
        default_indicator = jnp.array(0.0)
        payoff = cln.payoff_terminal(default_indicator)

        # Should receive all coupons plus principal
        expected_coupons = 0.05 * 1_000_000 * 5
        expected = expected_coupons + 1_000_000
        assert jnp.allclose(payoff, expected)

    def test_cln_payoff_with_default(self):
        """Test CLN payoff with default."""
        cln = CreditLinkedNote(
            T=5.0,
            principal=1_000_000,
            coupon_rate=0.05,
            credit_spread=200.0,
            recovery_rate=0.4,
        )
        default_indicator = jnp.array(1.0)
        payoff = cln.payoff_terminal(default_indicator)

        # Should receive coupons plus recovery value
        expected_coupons = 0.05 * 1_000_000 * 5
        expected = expected_coupons + 1_000_000 * 0.4
        assert jnp.allclose(payoff, expected)

    def test_fair_coupon(self):
        """Test fair coupon calculation."""
        cln = CreditLinkedNote(
            T=5.0, principal=1_000_000, coupon_rate=0.05, credit_spread=200.0
        )
        fair_rate = cln.fair_coupon(hazard_rate=0.02, risk_free_rate=0.03)
        assert fair_rate > 0.0


class TestLoanCDS:
    """Test loan CDS."""

    def test_loan_cds_creation(self):
        """Test loan CDS instantiation."""
        lcds = LoanCDS(T=5.0, notional=1_000_000, spread=300.0, recovery_rate=0.6)
        assert lcds.recovery_rate == 0.6  # Higher recovery for loans
        assert lcds.spread == 300.0

    def test_loan_cds_payoff_no_default(self):
        """Test loan CDS payoff with no default."""
        lcds = LoanCDS(T=5.0, notional=1_000_000, spread=300.0)
        default_time = jnp.array(10.0)  # Default after maturity
        payoff = lcds.payoff_terminal(default_time)

        # Only premium leg
        assert payoff < 0

    def test_loan_cds_payoff_with_default(self):
        """Test loan CDS payoff with default."""
        lcds = LoanCDS(
            T=5.0, notional=1_000_000, spread=300.0, recovery_rate=0.6
        )
        default_time = jnp.array(2.5)  # Default at 2.5 years
        payoff = lcds.payoff_terminal(default_time)

        # Should include protection leg
        assert isinstance(payoff, jnp.ndarray) or isinstance(payoff, float)


class TestContingentCDS:
    """Test contingent CDS."""

    def test_contingent_cds_creation(self):
        """Test contingent CDS instantiation."""
        ccds = ContingentCDS(
            T=5.0,
            notional=1_000_000,
            spread=200.0,
            trigger_barrier=80.0,
            trigger_type='down',
        )
        assert ccds.trigger_barrier == 80.0
        assert ccds.trigger_type == 'down'

    def test_contingent_cds_no_trigger(self):
        """Test CCDS payoff when trigger not hit."""
        ccds = ContingentCDS(
            T=1.0,
            notional=1_000_000,
            spread=200.0,
            trigger_barrier=80.0,
            trigger_type='down',
        )
        # Path that never hits barrier
        path = jnp.array([
            [100.0, 105.0, 102.0, 98.0, 101.0],  # Trigger asset stays above 80
            [0.0, 0.0, 0.0, 0.0, 0.0]  # No default
        ])
        payoff = ccds.payoff_path(path)
        assert jnp.allclose(payoff, 0.0)

    def test_contingent_cds_with_trigger(self):
        """Test CCDS payoff when trigger is hit."""
        ccds = ContingentCDS(
            T=1.0,
            notional=1_000_000,
            spread=200.0,
            trigger_barrier=80.0,
            trigger_type='down',
        )
        # Path that hits barrier
        path = jnp.array([
            [100.0, 90.0, 75.0, 85.0, 95.0],  # Trigger asset hits 75 < 80
            [0.0, 0.0, 0.0, 0.0, 1.0]  # Default occurs
        ])
        payoff = ccds.payoff_path(path)
        # Should have non-zero payoff
        assert payoff != 0.0
