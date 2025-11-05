"""Tests for credit derivatives products."""
import jax.numpy as jnp
import pytest

from neutryx.products.credit_derivatives import (
    CDSIndex,
    CDSOption,
    CollateralizedLoanObligation,
    ContingentCDS,
    CreditDefaultSwap,
    CreditLinkedNote,
    FirstToDefaultBasket,
    LoanCDS,
    NthToDefaultBasket,
    RecoveryLock,
    RecoverySwap,
    SyntheticCDO,
    TotalReturnSwap,
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


class TestCreditDefaultSwap:
    """Test single-name Credit Default Swap."""

    def test_cds_creation(self):
        """Test CDS instantiation."""
        cds = CreditDefaultSwap(
            T=5.0,
            notional=10_000_000,
            spread=100.0,  # 100 bps
            recovery_rate=0.4,
            coupon_freq=4,
        )
        assert cds.T == 5.0
        assert cds.notional == 10_000_000
        assert cds.spread == 100.0
        assert cds.recovery_rate == 0.4
        assert cds.coupon_freq == 4

    def test_survival_probability(self):
        """Test survival probability calculation."""
        cds = CreditDefaultSwap(T=5.0, notional=10_000_000, spread=100.0)
        hazard_rate = 0.02

        # Survival probability at T=0 should be 1.0
        surv_0 = cds.survival_probability(hazard_rate, 0.0)
        assert jnp.allclose(surv_0, 1.0)

        # Survival probability decreases with time
        surv_5 = cds.survival_probability(hazard_rate, 5.0)
        assert 0.0 < surv_5 < 1.0
        assert surv_5 < surv_0

    def test_default_probability(self):
        """Test default probability calculation."""
        cds = CreditDefaultSwap(T=5.0, notional=10_000_000, spread=100.0)
        hazard_rate = 0.02

        # Default probability at T=0 should be 0
        def_0 = cds.default_probability(hazard_rate, 0.0)
        assert jnp.allclose(def_0, 0.0)

        # Default probability increases with time
        def_5 = cds.default_probability(hazard_rate, 5.0)
        assert 0.0 < def_5 < 1.0

        # Default prob + survival prob = 1
        surv_5 = cds.survival_probability(hazard_rate, 5.0)
        assert jnp.allclose(def_5 + surv_5, 1.0)

    def test_premium_leg_pv(self):
        """Test premium leg present value calculation."""
        cds = CreditDefaultSwap(T=5.0, notional=10_000_000, spread=100.0)
        hazard_rate = 0.02
        discount_rate = 0.03

        premium_pv = cds.premium_leg_pv(hazard_rate, discount_rate)

        # Premium leg should be positive
        assert premium_pv > 0
        # Should be less than notional
        assert premium_pv < cds.notional

    def test_protection_leg_pv(self):
        """Test protection leg present value calculation."""
        cds = CreditDefaultSwap(T=5.0, notional=10_000_000, spread=100.0)
        hazard_rate = 0.02
        discount_rate = 0.03

        protection_pv = cds.protection_leg_pv(hazard_rate, discount_rate)

        # Protection leg should be positive
        assert protection_pv > 0
        # Should be related to LGD
        max_pv = cds.notional * (1.0 - cds.recovery_rate)
        assert protection_pv <= max_pv

    def test_cds_payoff(self):
        """Test CDS payoff calculation."""
        cds = CreditDefaultSwap(T=5.0, notional=10_000_000, spread=100.0)
        hazard_rate = jnp.array(0.02)

        payoff = cds.payoff_terminal(hazard_rate)

        # Payoff should be a number
        assert isinstance(payoff, jnp.ndarray) or isinstance(payoff, float)

    def test_fair_spread(self):
        """Test fair spread calculation."""
        cds = CreditDefaultSwap(T=5.0, notional=10_000_000, spread=100.0)
        hazard_rate = 0.02
        discount_rate = 0.03

        fair_spread = cds.fair_spread(hazard_rate, discount_rate)

        # Fair spread should be positive
        assert fair_spread > 0
        # Should be reasonable (between 0 and 1000 bps)
        assert 0 < fair_spread < 1000

    def test_credit_dv01(self):
        """Test credit DV01 calculation."""
        cds = CreditDefaultSwap(T=5.0, notional=10_000_000, spread=100.0)
        hazard_rate = 0.02
        discount_rate = 0.03

        dv01 = cds.credit_dv01(hazard_rate, discount_rate)

        # DV01 should be positive
        assert dv01 > 0
        # Should be less than notional * T
        assert dv01 < cds.notional * cds.T


class TestTotalReturnSwap:
    """Test Total Return Swap."""

    def test_trs_creation(self):
        """Test TRS instantiation."""
        trs = TotalReturnSwap(
            T=1.0,
            notional=10_000_000,
            asset_coupon=0.05,
            funding_spread=150.0,  # 150 bps
            initial_asset_price=100.0,
        )
        assert trs.T == 1.0
        assert trs.notional == 10_000_000
        assert trs.asset_coupon == 0.05
        assert trs.funding_spread == 150.0
        assert trs.initial_asset_price == 100.0

    def test_total_return_leg(self):
        """Test total return leg calculation."""
        trs = TotalReturnSwap(
            T=1.0,
            notional=10_000_000,
            asset_coupon=0.05,
            funding_spread=150.0,
            initial_asset_price=100.0,
        )

        final_price = 105.0
        hazard_rate = 0.02
        libor_rate = 0.03

        total_return = trs.total_return_leg(final_price, hazard_rate, libor_rate)

        # Should be positive (coupons + price appreciation)
        assert total_return > 0

    def test_funding_leg(self):
        """Test funding leg calculation."""
        trs = TotalReturnSwap(
            T=1.0,
            notional=10_000_000,
            asset_coupon=0.05,
            funding_spread=150.0,
        )

        libor_rate = 0.03
        funding = trs.funding_leg(libor_rate)

        # Funding should be positive
        assert funding > 0

    def test_trs_payoff(self):
        """Test TRS payoff calculation."""
        trs = TotalReturnSwap(
            T=1.0,
            notional=10_000_000,
            asset_coupon=0.05,
            funding_spread=150.0,
            initial_asset_price=100.0,
        )

        state = jnp.array([105.0, 0.02, 0.03])  # final_price, hazard_rate, libor_rate
        payoff = trs.payoff_terminal(state)

        # Payoff should be a number
        assert isinstance(payoff, jnp.ndarray) or isinstance(payoff, float)

    def test_trs_price_appreciation(self):
        """Test TRS with price appreciation."""
        trs = TotalReturnSwap(
            T=1.0,
            notional=10_000_000,
            asset_coupon=0.05,
            funding_spread=150.0,
            initial_asset_price=100.0,
        )

        # Asset appreciates
        state_up = jnp.array([110.0, 0.01, 0.03])
        payoff_up = trs.payoff_terminal(state_up)

        # Asset depreciates
        state_down = jnp.array([95.0, 0.01, 0.03])
        payoff_down = trs.payoff_terminal(state_down)

        # Higher asset price should give higher payoff
        assert payoff_up > payoff_down

    def test_breakeven_spread(self):
        """Test breakeven spread calculation."""
        trs = TotalReturnSwap(
            T=1.0,
            notional=10_000_000,
            asset_coupon=0.05,
            funding_spread=150.0,
        )

        breakeven = trs.breakeven_spread(105.0, 0.02, 0.03)

        # Breakeven spread should be positive
        assert breakeven > 0


class TestCollateralizedLoanObligation:
    """Test Collateralized Loan Obligation."""

    def test_clo_creation(self):
        """Test CLO instantiation."""
        clo = CollateralizedLoanObligation(
            T=7.0,
            notional=50_000_000,
            attachment=0.10,
            detachment=0.20,
            spread=250.0,  # 250 bps
            recovery_rate=0.65,
            correlation=0.25,
        )
        assert clo.T == 7.0
        assert clo.notional == 50_000_000
        assert clo.attachment == 0.10
        assert clo.detachment == 0.20
        assert clo.recovery_rate == 0.65  # Higher recovery for loans

    def test_tranche_loss(self):
        """Test tranche loss calculation."""
        clo = CollateralizedLoanObligation(
            T=7.0,
            notional=50_000_000,
            attachment=0.10,
            detachment=0.20,
            spread=250.0,
        )

        # No portfolio loss
        loss_0 = clo.tranche_loss(0.0)
        assert jnp.allclose(loss_0, 0.0)

        # Portfolio loss below attachment
        loss_low = clo.tranche_loss(0.05)
        assert jnp.allclose(loss_low, 0.0)

        # Portfolio loss within tranche
        loss_mid = clo.tranche_loss(0.15)
        assert 0.0 < loss_mid < 1.0

        # Portfolio loss above detachment
        loss_high = clo.tranche_loss(0.30)
        assert jnp.allclose(loss_high, 1.0)

    def test_expected_tranche_loss_lhp(self):
        """Test expected tranche loss using LHP."""
        clo = CollateralizedLoanObligation(
            T=7.0,
            notional=50_000_000,
            attachment=0.10,
            detachment=0.20,
            spread=250.0,
            correlation=0.25,
        )

        default_prob = 0.05
        expected_loss = clo.expected_tranche_loss_lhp(default_prob)

        # Expected loss should be between 0 and 1
        assert 0.0 <= expected_loss <= 1.0

    def test_interest_waterfall(self):
        """Test interest waterfall calculation."""
        clo = CollateralizedLoanObligation(
            T=7.0,
            notional=50_000_000,
            attachment=0.10,
            detachment=0.20,
            spread=250.0,
            loan_coupon=400.0,
        )

        libor_rate = 0.03
        default_prob = 0.05

        interest = clo.interest_waterfall(libor_rate, default_prob)

        # Interest should be positive
        assert interest > 0

    def test_clo_payoff(self):
        """Test CLO tranche payoff calculation."""
        clo = CollateralizedLoanObligation(
            T=7.0,
            notional=50_000_000,
            attachment=0.10,
            detachment=0.20,
            spread=250.0,
        )

        params = jnp.array([0.05, 0.03])  # default_prob, libor_rate
        payoff = clo.payoff_terminal(params)

        # Payoff should be positive (includes interest + principal)
        assert payoff > 0
        # Total value can exceed notional due to interest payments
        # but should be reasonable (< notional * 2 for 7 year maturity)
        assert payoff < clo.notional * 2.0

    def test_senior_vs_equity_tranche(self):
        """Test that senior tranches have lower spreads and losses."""
        # Senior tranche
        senior = CollateralizedLoanObligation(
            T=7.0,
            notional=50_000_000,
            attachment=0.20,
            detachment=1.00,
            spread=100.0,  # Lower spread for senior
            correlation=0.25,
        )

        # Equity tranche
        equity = CollateralizedLoanObligation(
            T=7.0,
            notional=50_000_000,
            attachment=0.00,
            detachment=0.10,
            spread=800.0,  # Higher spread for equity
            correlation=0.25,
        )

        default_prob = 0.05

        # Equity tranche should have higher expected loss
        senior_loss = senior.expected_tranche_loss_lhp(default_prob)
        equity_loss = equity.expected_tranche_loss_lhp(default_prob)

        assert equity_loss >= senior_loss

    def test_expected_return(self):
        """Test expected return calculation."""
        clo = CollateralizedLoanObligation(
            T=7.0,
            notional=50_000_000,
            attachment=0.00,
            detachment=0.10,
            spread=800.0,
        )

        expected_return = clo.expected_return(0.05, 0.03)

        # Return should be reasonable
        assert isinstance(expected_return, jnp.ndarray) or isinstance(expected_return, float)


class TestCDSOption:
    """Test CDS options (payer and receiver swaptions)."""

    def test_cds_option_creation(self):
        """Test CDS option instantiation."""
        cds_option = CDSOption(
            T=1.0,
            cds_maturity=5.0,
            strike_spread=100.0,  # 100 bps
            notional=10_000_000,
            option_type='payer',
            volatility=0.50,
        )
        assert cds_option.T == 1.0
        assert cds_option.cds_maturity == 5.0
        assert cds_option.strike_spread == 100.0
        assert cds_option.option_type == 'payer'
        assert cds_option.volatility == 0.50

    def test_payer_option_payoff(self):
        """Test payer CDS option payoff (right to buy protection)."""
        payer_option = CDSOption(
            T=1.0,
            cds_maturity=5.0,
            strike_spread=100.0,
            notional=10_000_000,
            option_type='payer',
            volatility=0.50,
        )

        # Forward spread = 150 bps (higher than strike)
        # Should have positive value
        params = jnp.array([150.0, 0.97, 0.98])  # forward_spread, discount, survival
        payoff = payer_option.payoff_terminal(params)

        # Payer option should have positive value when forward > strike
        assert payoff > 0

    def test_receiver_option_payoff(self):
        """Test receiver CDS option payoff (right to sell protection)."""
        receiver_option = CDSOption(
            T=1.0,
            cds_maturity=5.0,
            strike_spread=100.0,
            notional=10_000_000,
            option_type='receiver',
            volatility=0.50,
        )

        # Forward spread = 50 bps (lower than strike)
        # Should have positive value
        params = jnp.array([50.0, 0.97, 0.98])  # forward_spread, discount, survival
        payoff = receiver_option.payoff_terminal(params)

        # Receiver option should have positive value when forward < strike
        assert payoff > 0

    def test_at_the_money_option(self):
        """Test ATM CDS option."""
        atm_option = CDSOption(
            T=1.0,
            cds_maturity=5.0,
            strike_spread=100.0,
            notional=10_000_000,
            option_type='payer',
            volatility=0.50,
        )

        # Forward spread equals strike
        params = jnp.array([100.0, 0.97, 0.98])
        payoff = atm_option.payoff_terminal(params)

        # ATM option should still have time value
        assert payoff > 0

    def test_knockout_feature(self):
        """Test knockout feature when default occurs before expiry."""
        knockout_option = CDSOption(
            T=1.0,
            cds_maturity=5.0,
            strike_spread=100.0,
            notional=10_000_000,
            option_type='payer',
            is_knockout=True,
            volatility=0.50,
        )

        # Low survival probability (high chance of default before expiry)
        params = jnp.array([150.0, 0.97, 0.50])  # survival_prob = 0.5
        payoff_knockout = knockout_option.payoff_terminal(params)

        # Non-knockout version for comparison
        no_knockout_option = CDSOption(
            T=1.0,
            cds_maturity=5.0,
            strike_spread=100.0,
            notional=10_000_000,
            option_type='payer',
            is_knockout=False,
            volatility=0.50,
        )
        payoff_no_knockout = no_knockout_option.payoff_terminal(params)

        # Knockout option should have lower value
        assert payoff_knockout < payoff_no_knockout

    def test_volatility_impact(self):
        """Test that higher volatility increases option value."""
        low_vol_option = CDSOption(
            T=1.0,
            cds_maturity=5.0,
            strike_spread=100.0,
            notional=10_000_000,
            option_type='payer',
            volatility=0.30,
        )

        high_vol_option = CDSOption(
            T=1.0,
            cds_maturity=5.0,
            strike_spread=100.0,
            notional=10_000_000,
            option_type='payer',
            volatility=0.70,
        )

        params = jnp.array([100.0, 0.97, 0.98])

        payoff_low_vol = low_vol_option.payoff_terminal(params)
        payoff_high_vol = high_vol_option.payoff_terminal(params)

        # Higher volatility should give higher option value
        assert payoff_high_vol > payoff_low_vol


class TestRecoveryLock:
    """Test recovery lock contracts."""

    def test_recovery_lock_creation(self):
        """Test recovery lock instantiation."""
        recovery_lock = RecoveryLock(
            T=5.0,
            notional=10_000_000,
            locked_recovery=0.40,
            reference_entity="ACME Corp",
        )
        assert recovery_lock.T == 5.0
        assert recovery_lock.notional == 10_000_000
        assert recovery_lock.locked_recovery == 0.40
        assert recovery_lock.reference_entity == "ACME Corp"

    def test_recovery_lock_no_default(self):
        """Test recovery lock payoff when no default occurs."""
        recovery_lock = RecoveryLock(
            T=5.0,
            notional=10_000_000,
            locked_recovery=0.40,
        )

        # No default occurred
        params = jnp.array([0.0, 0.30])  # default_indicator=0, realized_recovery=0.30
        payoff = recovery_lock.payoff_terminal(params)

        # Should pay zero (no default, so recovery lock not triggered)
        assert jnp.allclose(payoff, 0.0)

    def test_recovery_lock_positive_payoff(self):
        """Test recovery lock payoff when realized recovery < locked recovery."""
        recovery_lock = RecoveryLock(
            T=5.0,
            notional=10_000_000,
            locked_recovery=0.40,
        )

        # Default occurred, realized recovery = 30%
        params = jnp.array([1.0, 0.30])  # default_indicator=1, realized_recovery=0.30
        payoff = recovery_lock.payoff_terminal(params)

        # Payoff = notional * (locked - realized) = 10M * (0.40 - 0.30) = 1M
        expected = 10_000_000 * (0.40 - 0.30)
        assert jnp.allclose(payoff, expected)

    def test_recovery_lock_negative_payoff(self):
        """Test recovery lock payoff when realized recovery > locked recovery."""
        recovery_lock = RecoveryLock(
            T=5.0,
            notional=10_000_000,
            locked_recovery=0.40,
        )

        # Default occurred, realized recovery = 50%
        params = jnp.array([1.0, 0.50])  # default_indicator=1, realized_recovery=0.50
        payoff = recovery_lock.payoff_terminal(params)

        # Payoff = notional * (locked - realized) = 10M * (0.40 - 0.50) = -1M
        expected = 10_000_000 * (0.40 - 0.50)
        assert jnp.allclose(payoff, expected)


class TestRecoverySwap:
    """Test recovery swap contracts."""

    def test_recovery_swap_creation(self):
        """Test recovery swap instantiation."""
        recovery_swap = RecoverySwap(
            T=5.0,
            notional=10_000_000,
            fixed_recovery=0.40,
            payment_freq=4,
            reference_entities=["Entity1", "Entity2", "Entity3"],
        )
        assert recovery_swap.T == 5.0
        assert recovery_swap.notional == 10_000_000
        assert recovery_swap.fixed_recovery == 0.40
        assert recovery_swap.payment_freq == 4

    def test_recovery_swap_no_defaults(self):
        """Test recovery swap payoff when no defaults occur."""
        recovery_swap = RecoverySwap(
            T=5.0,
            notional=10_000_000,
            fixed_recovery=0.40,
            payment_freq=4,
        )

        # No defaults
        params = jnp.array([0.0, 0.40, 0.03])  # num_defaults=0, avg_recovery=0.40, discount=0.03
        payoff = recovery_swap.payoff_terminal(params)

        # No defaults means no payments on either leg
        assert jnp.allclose(payoff, 0.0)

    def test_recovery_swap_positive_value(self):
        """Test recovery swap with realized recovery > fixed recovery."""
        recovery_swap = RecoverySwap(
            T=5.0,
            notional=10_000_000,
            fixed_recovery=0.40,
            payment_freq=4,
        )

        # 2 defaults, average recovery = 50%
        params = jnp.array([2.0, 0.50, 0.03])
        payoff = recovery_swap.payoff_terminal(params)

        # Receive floating (0.50) - pay fixed (0.40) = positive
        assert payoff > 0

    def test_recovery_swap_negative_value(self):
        """Test recovery swap with realized recovery < fixed recovery."""
        recovery_swap = RecoverySwap(
            T=5.0,
            notional=10_000_000,
            fixed_recovery=0.40,
            payment_freq=4,
        )

        # 2 defaults, average recovery = 30%
        params = jnp.array([2.0, 0.30, 0.03])
        payoff = recovery_swap.payoff_terminal(params)

        # Receive floating (0.30) - pay fixed (0.40) = negative
        assert payoff < 0

    def test_recovery_swap_multiple_defaults(self):
        """Test recovery swap value scales with number of defaults."""
        recovery_swap = RecoverySwap(
            T=5.0,
            notional=10_000_000,
            fixed_recovery=0.40,
            payment_freq=4,
        )

        # 1 default
        params_1 = jnp.array([1.0, 0.50, 0.03])
        payoff_1 = recovery_swap.payoff_terminal(params_1)

        # 3 defaults
        params_3 = jnp.array([3.0, 0.50, 0.03])
        payoff_3 = recovery_swap.payoff_terminal(params_3)

        # More defaults should scale the payoff
        assert abs(payoff_3) > abs(payoff_1)


class TestFirstToDefaultBasket:
    """Test first-to-default basket (convenience wrapper)."""

    def test_first_to_default_creation(self):
        """Test FTD basket instantiation."""
        ftd = FirstToDefaultBasket(
            T=5.0,
            notional=10_000_000,
            num_names=5,
            recovery_rate=0.40,
            correlation=0.30,
        )
        assert ftd.T == 5.0
        assert ftd.notional == 10_000_000
        assert ftd.n == 1  # First default
        assert ftd.num_names == 5
        assert ftd.recovery_rate == 0.40
        assert ftd.correlation == 0.30

    def test_ftd_no_defaults_before_maturity(self):
        """Test FTD payoff when no defaults occur before maturity."""
        ftd = FirstToDefaultBasket(
            T=5.0,
            notional=10_000_000,
            num_names=5,
            recovery_rate=0.40,
        )

        # All defaults after maturity
        default_times = jnp.array([6.0, 7.0, 8.0, 9.0, 10.0])
        payoff = ftd.payoff_terminal(default_times)

        # Should pay zero
        assert jnp.allclose(payoff, 0.0)

    def test_ftd_first_default_before_maturity(self):
        """Test FTD payoff when first default occurs before maturity."""
        ftd = FirstToDefaultBasket(
            T=5.0,
            notional=10_000_000,
            num_names=5,
            recovery_rate=0.40,
        )

        # First default at 2 years
        default_times = jnp.array([2.0, 6.0, 7.0, 8.0, 9.0])
        payoff = ftd.payoff_terminal(default_times)

        # Should pay (1 - recovery_rate) * notional
        expected = 10_000_000 * (1.0 - 0.40)
        assert jnp.allclose(payoff, expected)

    def test_ftd_multiple_defaults(self):
        """Test FTD pays on first default even if multiple defaults occur."""
        ftd = FirstToDefaultBasket(
            T=5.0,
            notional=10_000_000,
            num_names=5,
            recovery_rate=0.40,
        )

        # Multiple defaults before maturity
        default_times = jnp.array([1.0, 2.0, 3.0, 6.0, 7.0])
        payoff = ftd.payoff_terminal(default_times)

        # Should still pay based on first default only
        expected = 10_000_000 * (1.0 - 0.40)
        assert jnp.allclose(payoff, expected)

    def test_ftd_equivalence_to_nth_with_n1(self):
        """Test that FTD is equivalent to NthToDefaultBasket with n=1."""
        ftd = FirstToDefaultBasket(
            T=5.0,
            notional=10_000_000,
            num_names=5,
            recovery_rate=0.40,
            correlation=0.30,
        )

        nth = NthToDefaultBasket(
            T=5.0,
            notional=10_000_000,
            n=1,
            num_names=5,
            recovery_rate=0.40,
            correlation=0.30,
        )

        # Test with various default scenarios
        default_times_1 = jnp.array([2.0, 6.0, 7.0, 8.0, 9.0])
        default_times_2 = jnp.array([6.0, 7.0, 8.0, 9.0, 10.0])

        assert jnp.allclose(
            ftd.payoff_terminal(default_times_1),
            nth.payoff_terminal(default_times_1)
        )
        assert jnp.allclose(
            ftd.payoff_terminal(default_times_2),
            nth.payoff_terminal(default_times_2)
        )
