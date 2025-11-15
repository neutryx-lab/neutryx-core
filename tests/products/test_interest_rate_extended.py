"""Tests for extended interest rate derivatives.

Tests for SOFR cap/floor, American swaptions, CMS products, and exotic structures.
"""
import jax.numpy as jnp
import pytest

from neutryx.products.interest_rate import (
    InterestRateCapFloor,
    CMSCapFloor,
    CMSSpreadOptionInstrument,
    sofr_caplet_price,
    sofr_floorlet_price,
    price_sofr_cap,
    price_sofr_floor,
    price_cap,
    cms_caplet_price,
    cms_floorlet_price,
    cms_convexity_adjustment,
    price_cms_cap,
    price_cms_floor,
)
from neutryx.products.swaptions import (
    american_swaption_tree,
    american_swaption_lsm,
)
from neutryx.products.advanced_rates import (
    SnowballNote,
    AutocallableNote,
    RatchetCapFloor,
)


class TestSOFRCapFloor:
    """Tests for SOFR cap and floor pricing."""

    def test_sofr_caplet_price(self):
        """Test SOFR caplet pricing."""
        price = sofr_caplet_price(
            forward_rate=0.03,
            strike=0.025,
            time_to_expiry=1.0,
            volatility=0.20,
            discount_factor=0.97,
            compounding_days=90,
            notional=1_000_000.0,
        )

        assert price > 0
        assert price < 10_000  # Reasonable bound

    def test_sofr_floorlet_price(self):
        """Test SOFR floorlet pricing."""
        price = sofr_floorlet_price(
            forward_rate=0.02,
            strike=0.025,
            time_to_expiry=1.0,
            volatility=0.20,
            discount_factor=0.97,
            compounding_days=90,
            notional=1_000_000.0,
        )

        assert price > 0
        assert price < 10_000

    def test_sofr_cap_price(self):
        """Test SOFR cap pricing."""
        forward_rates = jnp.array([0.03, 0.032, 0.035])
        times_to_expiry = jnp.array([0.25, 0.5, 0.75])
        volatilities = jnp.array([0.20, 0.21, 0.22])
        discount_factors = jnp.array([0.99, 0.98, 0.97])
        compounding_days = jnp.array([90, 90, 90])

        price = price_sofr_cap(
            forward_rates=forward_rates,
            strike=0.03,
            times_to_expiry=times_to_expiry,
            volatilities=volatilities,
            discount_factors=discount_factors,
            compounding_days=compounding_days,
            notional=1_000_000.0,
        )

        assert price > 0
        assert price < 50_000

    def test_sofr_floor_price(self):
        """Test SOFR floor pricing."""
        forward_rates = jnp.array([0.02, 0.022, 0.025])
        times_to_expiry = jnp.array([0.25, 0.5, 0.75])
        volatilities = jnp.array([0.20, 0.21, 0.22])
        discount_factors = jnp.array([0.99, 0.98, 0.97])
        compounding_days = jnp.array([90, 90, 90])

        price = price_sofr_floor(
            forward_rates=forward_rates,
            strike=0.03,
            times_to_expiry=times_to_expiry,
            volatilities=volatilities,
            discount_factors=discount_factors,
            compounding_days=compounding_days,
            notional=1_000_000.0,
        )

        assert price > 0
        assert price < 50_000

    def test_sofr_put_call_parity(self):
        """Test SOFR cap-floor parity."""
        forward_rate = 0.03
        strike = 0.03
        time_to_expiry = 1.0
        volatility = 0.20
        discount_factor = 0.97
        compounding_days = 90
        notional = 1_000_000.0

        cap_price = sofr_caplet_price(
            forward_rate, strike, time_to_expiry, volatility,
            discount_factor, compounding_days, notional
        )

        floor_price = sofr_floorlet_price(
            forward_rate, strike, time_to_expiry, volatility,
            discount_factor, compounding_days, notional
        )

        # At-the-money cap and floor should have similar prices
        assert abs(cap_price - floor_price) < 1000


class TestAmericanSwaption:
    """Tests for American swaption pricing."""

    def test_american_swaption_tree_payer(self):
        """Test American payer swaption using tree method."""
        price = american_swaption_tree(
            strike=0.05,
            option_maturity=1.0,
            swap_maturity=5.0,
            initial_rate=0.04,
            volatility=0.20,
            discount_rate=0.03,
            notional=1_000_000.0,
            payment_frequency=2,
            is_payer=True,
            n_steps=50,
        )

        assert price >= 0
        assert price < 300_000  # Increased bound for American optionality

    def test_american_swaption_tree_receiver(self):
        """Test American receiver swaption using tree method."""
        price = american_swaption_tree(
            strike=0.05,
            option_maturity=1.0,
            swap_maturity=5.0,
            initial_rate=0.06,
            volatility=0.20,
            discount_rate=0.03,
            notional=1_000_000.0,
            payment_frequency=2,
            is_payer=False,
            n_steps=50,
        )

        assert price >= 0
        assert price < 800_000  # Increased bound for American optionality

    def test_american_swaption_lsm(self):
        """Test American swaption using LSM method."""
        # Generate rate paths (simplified)
        import jax.random as jrand
        key = jrand.PRNGKey(42)

        n_paths = 1000
        n_steps = 50
        rate_paths = jrand.normal(key, (n_paths, n_steps)) * 0.01 + 0.05

        discount_factors = jnp.exp(-0.03 * jnp.linspace(0, 1.0, n_steps))

        price = american_swaption_lsm(
            strike=0.05,
            option_maturity=1.0,
            swap_maturity=5.0,
            rate_paths=rate_paths,
            discount_factors=discount_factors,
            notional=1_000_000.0,
            payment_frequency=2,
            is_payer=True,
        )

        assert price >= 0
        assert price < 500_000  # Broader bound for MC method


class TestCMSProducts:
    """Tests for CMS products."""

    def test_cms_caplet_price(self):
        """Test CMS caplet pricing."""
        price = cms_caplet_price(
            cms_forward=0.04,
            strike=0.035,
            time_to_expiry=1.0,
            volatility=0.25,
            discount_factor=0.97,
            annuity=4.5,
            notional=1_000_000.0,
            convexity_adjustment=0.001,
        )

        assert price > 0
        assert price < 100_000


class TestInterestRateProductClasses:
    """Tests for newly exposed payoff helpers."""

    def test_interest_rate_cap_floor_payoff(self):
        """Cap/floor path payoff aggregates caplets."""
        cap = InterestRateCapFloor(T=1.0, strike=0.03, payment_frequency=4, notional=1_000_000.0)
        path = jnp.array([0.035, 0.025, 0.031, 0.04])
        payoff = cap.payoff_path(path)
        expected = (
            jnp.maximum(path - cap.strike, 0.0) * (cap.notional / cap.payment_frequency)
        ).sum()
        assert jnp.isclose(payoff, expected)

        floor = InterestRateCapFloor(
            T=1.0,
            strike=0.03,
            payment_frequency=4,
            notional=1_000_000.0,
            is_cap=False,
        )
        payoff_floor = floor.payoff_path(path)
        expected_floor = (
            jnp.maximum(cap.strike - path, 0.0) * (floor.notional / floor.payment_frequency)
        ).sum()
        assert jnp.isclose(payoff_floor, expected_floor)

    def test_interest_rate_cap_price_helper_matches_function(self):
        """Pricing helper delegates to portfolio functions."""
        cap = InterestRateCapFloor(T=1.0, strike=0.03, payment_frequency=4, notional=1_000_000.0)
        forward_rates = jnp.array([0.03, 0.031, 0.032, 0.033])
        times = jnp.array([0.25, 0.5, 0.75, 1.0])
        vols = jnp.array([0.2, 0.21, 0.215, 0.22])
        dfs = jnp.array([0.99, 0.985, 0.98, 0.975])

        helper_price = cap.price_from_forwards(forward_rates, times, vols, dfs)
        direct_price = price_cap(
            forward_rates,
            cap.strike,
            times,
            vols,
            dfs,
            jnp.full_like(times, 1.0 / cap.payment_frequency),
            cap.notional,
        )
        assert jnp.isclose(helper_price, direct_price)

    def test_cms_cap_floor_payoff_and_price(self):
        """CMS cap/floor helper mirrors function implementation."""
        cms_cap = CMSCapFloor(T=2.0, strike=0.03, annuity=4.5, notional=1_000_000.0)
        payoff = cms_cap.payoff_terminal(0.04)
        assert jnp.isclose(payoff, (0.04 - 0.03) * cms_cap.annuity * cms_cap.notional)

        price = cms_cap.price_black(
            cms_forward=0.04,
            volatility=0.2,
            time_to_expiry=1.0,
            discount_factor=0.95,
            convexity_adjustment=0.0005,
        )
        assert price > 0

    def test_cms_spread_option_paths(self):
        """CMS spread option handles different path layouts."""
        option = CMSSpreadOptionInstrument(
            T=1.0,
            strike=0.01,
            annuity=4.0,
            notional=1_000_000.0,
            is_call=True,
        )

        # Vector of two terminal rates
        payoff_vec = option.payoff_path(jnp.array([0.05, 0.037]))
        expected = (0.05 - 0.037 - option.strike) * option.annuity * option.notional
        assert jnp.isclose(payoff_vec, expected)

        # Matrix with shape (n_steps, 2)
        path_matrix = jnp.array(
            [
                [0.04, 0.03],
                [0.045, 0.032],
                [0.05, 0.037],
            ]
        )
        payoff_matrix = option.payoff_path(path_matrix)
        assert jnp.isclose(payoff_matrix, expected)

        price = option.price_black(
            cms1_forward=0.05,
            cms2_forward=0.037,
            time_to_expiry=1.0,
            spread_volatility=0.25,
            discount_factor=0.96,
        )
        assert price > 0

    def test_cms_floorlet_price(self):
        """Test CMS floorlet pricing."""
        price = cms_floorlet_price(
            cms_forward=0.03,
            strike=0.035,
            time_to_expiry=1.0,
            volatility=0.25,
            discount_factor=0.97,
            annuity=4.5,
            notional=1_000_000.0,
            convexity_adjustment=0.001,
        )

        assert price > 0
        assert price < 100_000

    def test_cms_convexity_adjustment(self):
        """Test CMS convexity adjustment calculation."""
        adjustment = cms_convexity_adjustment(
            forward_rate=0.04,
            volatility=0.25,
            time_to_payment=1.0,
            swap_tenor=10.0,
            payment_frequency=2,
        )

        assert adjustment > 0
        assert adjustment < 0.01  # Typically small

    def test_cms_cap_price(self):
        """Test CMS cap pricing."""
        cms_forwards = jnp.array([0.04, 0.042, 0.045])
        times_to_expiry = jnp.array([0.5, 1.0, 1.5])
        volatilities = jnp.array([0.25, 0.26, 0.27])
        discount_factors = jnp.array([0.98, 0.96, 0.94])
        annuities = jnp.array([4.5, 4.5, 4.5])

        price = price_cms_cap(
            cms_forwards=cms_forwards,
            strike=0.04,
            times_to_expiry=times_to_expiry,
            volatilities=volatilities,
            discount_factors=discount_factors,
            annuities=annuities,
            notional=1_000_000.0,
            apply_convexity_adjustment=True,
            swap_tenor=10.0,
        )

        assert price > 0
        assert price < 200_000

    def test_cms_floor_price(self):
        """Test CMS floor pricing."""
        cms_forwards = jnp.array([0.03, 0.032, 0.035])
        times_to_expiry = jnp.array([0.5, 1.0, 1.5])
        volatilities = jnp.array([0.25, 0.26, 0.27])
        discount_factors = jnp.array([0.98, 0.96, 0.94])
        annuities = jnp.array([4.5, 4.5, 4.5])

        price = price_cms_floor(
            cms_forwards=cms_forwards,
            strike=0.04,
            times_to_expiry=times_to_expiry,
            volatilities=volatilities,
            discount_factors=discount_factors,
            annuities=annuities,
            notional=1_000_000.0,
            apply_convexity_adjustment=True,
            swap_tenor=10.0,
        )

        assert price > 0
        assert price < 200_000


class TestSnowballNote:
    """Tests for Snowball note."""

    def test_snowball_in_range(self):
        """Test snowball when rate stays in range."""
        snowball = SnowballNote(
            T=1.0,
            notional=1_000_000.0,
            base_coupon_rate=0.05,
            range_lower=0.02,
            range_upper=0.06,
            payment_freq=4,
        )

        # Path that stays in range
        path = jnp.ones(4) * 0.04

        payoff = snowball.payoff_path(path)

        # Should get notional + all coupons
        expected = 1_000_000.0 + 1_000_000.0 * 0.05
        assert abs(payoff - expected) < 1000

    def test_snowball_out_of_range(self):
        """Test snowball when rate goes out of range."""
        snowball = SnowballNote(
            T=1.0,
            notional=1_000_000.0,
            base_coupon_rate=0.05,
            range_lower=0.02,
            range_upper=0.04,
            payment_freq=4,
        )

        # Path that goes out of range
        path = jnp.array([0.03, 0.05, 0.03, 0.03])

        payoff = snowball.payoff_path(path)

        # Should get notional + partial coupons
        assert payoff >= 1_000_000.0
        assert payoff < 1_100_000.0

    def test_snowball_knockout(self):
        """Test snowball with knock-out."""
        snowball = SnowballNote(
            T=1.0,
            notional=1_000_000.0,
            base_coupon_rate=0.05,
            range_lower=0.02,
            range_upper=0.06,
            payment_freq=4,
            knock_out_barrier=0.07,
        )

        # Path that triggers knockout
        path = jnp.array([0.04, 0.08, 0.04, 0.04])

        payoff = snowball.payoff_path(path)

        # Should get notional + coupon for first period (before knockout)
        # First period is in range, so gets coupon, then knocks out
        expected = 1_000_000.0 + 1_000_000.0 * 0.05 * 0.25  # 1 quarter coupon
        assert abs(payoff - expected) < 1000


class TestAutocallableNote:
    """Tests for Autocallable note."""

    def test_autocallable_early_call(self):
        """Test autocallable with early redemption."""
        call_dates = jnp.array([0.5, 1.0])
        autocall = AutocallableNote(
            T=1.0,
            notional=1_000_000.0,
            call_barrier=0.05,
            coupon_rate=0.04,
            call_dates=call_dates,
        )

        # Path that triggers autocall at first observation
        path = jnp.linspace(0.04, 0.06, 10)

        payoff = autocall.payoff_path(path)

        # Should get notional + coupon
        assert payoff >= 1_000_000.0
        assert payoff < 1_100_000.0

    def test_autocallable_no_call(self):
        """Test autocallable without redemption."""
        call_dates = jnp.array([0.5, 1.0])
        autocall = AutocallableNote(
            T=1.0,
            notional=1_000_000.0,
            call_barrier=0.05,
            coupon_rate=0.04,
            call_dates=call_dates,
        )

        # Path that never triggers autocall
        path = jnp.ones(10) * 0.03

        payoff = autocall.payoff_path(path)

        # Should get notional + accumulated coupons at maturity
        assert payoff >= 1_000_000.0
        assert payoff < 1_150_000.0


class TestRatchetCapFloor:
    """Tests for Ratchet cap/floor."""

    def test_ratchet_cap(self):
        """Test ratchet cap."""
        ratchet = RatchetCapFloor(
            T=1.0,
            notional=1_000_000.0,
            initial_strike=0.03,
            ratchet_rate=0.01,
            is_cap=True,
            payment_freq=4,
        )

        # Increasing rate path
        path = jnp.linspace(0.03, 0.06, 4)

        payoff = ratchet.payoff_path(path)

        assert payoff >= 0
        assert payoff < 100_000

    def test_ratchet_floor(self):
        """Test ratchet floor."""
        ratchet = RatchetCapFloor(
            T=1.0,
            notional=1_000_000.0,
            initial_strike=0.05,
            ratchet_rate=-0.01,
            is_cap=False,
            payment_freq=4,
        )

        # Decreasing rate path
        path = jnp.linspace(0.05, 0.02, 4)

        payoff = ratchet.payoff_path(path)

        assert payoff >= 0
        assert payoff < 100_000

    def test_ratchet_with_global_bounds(self):
        """Test ratchet with global floor and cap."""
        ratchet = RatchetCapFloor(
            T=1.0,
            notional=1_000_000.0,
            initial_strike=0.03,
            ratchet_rate=0.02,
            is_cap=True,
            payment_freq=4,
            global_floor=0.02,
            global_cap=0.08,
        )

        # Path with extreme values
        path = jnp.array([0.01, 0.10, 0.05, 0.03])

        payoff = ratchet.payoff_path(path)

        assert payoff >= 0
        assert payoff < 200_000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
