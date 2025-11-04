"""Tests for extended inflation-linked products.

Tests for inflation swaptions, LPI bonds, and inflation-linked swaps.
"""
import jax.numpy as jnp
import pytest

from neutryx.products.inflation import (
    inflation_swap_pv01,
    asset_swap_spread_inflation,
    inflation_swaption_price,
    limited_price_indexation_bond,
    inflation_risk_premium,
    InflationLinkedSwap,
    InflationFloorCertificate,
)


class TestInflationSwapVariants:
    """Tests for inflation swap variants."""

    def test_inflation_swap_pv01(self):
        """Test PV01 calculation for inflation swap."""
        pv01 = inflation_swap_pv01(
            notional=1_000_000.0,
            maturity=5.0,
            payment_frequency=1,
        )

        assert pv01 > 0
        assert pv01 < 5000  # Reasonable bound

    def test_asset_swap_spread_inflation(self):
        """Test asset swap spread for inflation bond."""
        spread = asset_swap_spread_inflation(
            bond_price=102.0,
            par_value=100.0,
            real_coupon_rate=0.02,
            swap_rate=0.03,
            maturity=5.0,
        )

        # Spread should be reasonable
        assert abs(spread) < 0.05

    def test_asset_swap_spread_par(self):
        """Test asset swap spread when bond is at par."""
        spread = asset_swap_spread_inflation(
            bond_price=100.0,
            par_value=100.0,
            real_coupon_rate=0.03,
            swap_rate=0.03,
            maturity=5.0,
        )

        # Spread should be near zero when at par
        assert abs(spread) < 0.01


class TestInflationSwaption:
    """Tests for inflation swaption pricing."""

    def test_inflation_swaption_payer(self):
        """Test inflation payer swaption."""
        price = inflation_swaption_price(
            strike_inflation=0.025,
            forward_inflation=0.03,
            time_to_expiry=1.0,
            volatility=0.01,
            discount_factor=0.97,
            notional=1_000_000.0,
            is_payer=True,
        )

        assert price > 0
        assert price < 50_000

    def test_inflation_swaption_receiver(self):
        """Test inflation receiver swaption."""
        price = inflation_swaption_price(
            strike_inflation=0.03,
            forward_inflation=0.025,
            time_to_expiry=1.0,
            volatility=0.01,
            discount_factor=0.97,
            notional=1_000_000.0,
            is_payer=False,
        )

        assert price > 0
        assert price < 50_000

    def test_inflation_swaption_atm(self):
        """Test at-the-money inflation swaption."""
        strike = 0.025

        payer_price = inflation_swaption_price(
            strike_inflation=strike,
            forward_inflation=strike,
            time_to_expiry=1.0,
            volatility=0.01,
            discount_factor=0.97,
            notional=1_000_000.0,
            is_payer=True,
        )

        receiver_price = inflation_swaption_price(
            strike_inflation=strike,
            forward_inflation=strike,
            time_to_expiry=1.0,
            volatility=0.01,
            discount_factor=0.97,
            notional=1_000_000.0,
            is_payer=False,
        )

        # ATM payer and receiver should have similar prices
        assert abs(payer_price - receiver_price) < 1000


class TestLimitedPriceIndexationBond:
    """Tests for LPI bonds."""

    def test_lpi_bond_no_cap_or_floor(self):
        """Test LPI bond when inflation is between cap and floor."""
        price = limited_price_indexation_bond(
            face_value=100.0,
            real_coupon_rate=0.02,
            real_yield=0.01,
            maturity=5.0,
            index_ratio=1.15,  # 15% inflation
            inflation_cap=0.20,  # 20% cap
            inflation_floor=0.00,  # 0% floor
            frequency=2,
        )

        assert price > 100.0  # Should be above par
        assert price < 150.0

    def test_lpi_bond_capped(self):
        """Test LPI bond when inflation exceeds cap."""
        # High inflation
        price_capped = limited_price_indexation_bond(
            face_value=100.0,
            real_coupon_rate=0.02,
            real_yield=0.01,
            maturity=5.0,
            index_ratio=1.30,  # 30% inflation
            inflation_cap=0.05,  # 5% cap
            inflation_floor=0.00,
            frequency=2,
        )

        # Lower inflation
        price_uncapped = limited_price_indexation_bond(
            face_value=100.0,
            real_coupon_rate=0.02,
            real_yield=0.01,
            maturity=5.0,
            index_ratio=1.05,  # 5% inflation
            inflation_cap=0.05,
            inflation_floor=0.00,
            frequency=2,
        )

        # Capped bond should have similar price to uncapped at cap level
        assert abs(price_capped - price_uncapped) < 5.0

    def test_lpi_bond_floored(self):
        """Test LPI bond when inflation is below floor."""
        # Deflation
        price_floored = limited_price_indexation_bond(
            face_value=100.0,
            real_coupon_rate=0.02,
            real_yield=0.01,
            maturity=5.0,
            index_ratio=0.95,  # -5% inflation (deflation)
            inflation_cap=0.05,
            inflation_floor=0.00,  # 0% floor
            frequency=2,
        )

        # Zero inflation
        price_zero = limited_price_indexation_bond(
            face_value=100.0,
            real_coupon_rate=0.02,
            real_yield=0.01,
            maturity=5.0,
            index_ratio=1.00,
            inflation_cap=0.05,
            inflation_floor=0.00,
            frequency=2,
        )

        # Floored bond should have similar price to zero inflation
        assert abs(price_floored - price_zero) < 5.0


class TestInflationRiskPremium:
    """Tests for inflation risk premium."""

    def test_inflation_risk_premium_positive(self):
        """Test positive inflation risk premium."""
        premium = inflation_risk_premium(
            nominal_yield=0.05,
            real_yield=0.02,
            expected_inflation=0.025,
        )

        # Premium should be positive (nominal > real + inflation)
        assert premium > 0
        assert premium < 0.02

    def test_inflation_risk_premium_zero(self):
        """Test zero inflation risk premium."""
        premium = inflation_risk_premium(
            nominal_yield=0.0502,
            real_yield=0.02,
            expected_inflation=0.03,
        )

        # Premium should be near zero
        assert abs(premium) < 0.001

    def test_inflation_risk_premium_consistency(self):
        """Test Fisher equation consistency."""
        nominal = 0.05
        real = 0.02
        inflation = 0.025

        premium = inflation_risk_premium(nominal, real, inflation)

        # Verify Fisher equation: (1+nom) = (1+real)*(1+inf)*(1+premium)
        lhs = 1.0 + nominal
        rhs = (1.0 + real) * (1.0 + inflation) * (1.0 + premium)

        assert abs(lhs - rhs) < 1e-6


class TestInflationLinkedSwap:
    """Tests for inflation-linked swap."""

    def test_inflation_linked_swap_pricing(self):
        """Test inflation-linked swap pricing."""
        swap = InflationLinkedSwap(
            notional=1_000_000.0,
            maturity=5.0,
            fixed_real_rate=0.01,
            payment_frequency=1,
        )

        # Create CPI path (5% annual inflation)
        n_periods = 5
        cpi_path = jnp.array([100.0 * (1.05 ** i) for i in range(n_periods + 1)])

        # Create discount factors
        discount_factors = jnp.exp(-0.03 * jnp.arange(1, n_periods + 1))

        value = swap.price(cpi_path, discount_factors)

        # Swap value should be reasonable
        assert abs(value) < 500_000

    def test_inflation_linked_swap_zero_inflation(self):
        """Test swap with zero inflation."""
        swap = InflationLinkedSwap(
            notional=1_000_000.0,
            maturity=5.0,
            fixed_real_rate=0.02,
            payment_frequency=1,
        )

        # Zero inflation (constant CPI)
        n_periods = 5
        cpi_path = jnp.ones(n_periods + 1) * 100.0

        discount_factors = jnp.exp(-0.03 * jnp.arange(1, n_periods + 1))

        value = swap.price(cpi_path, discount_factors)

        # With zero inflation, fixed leg should dominate
        # Value should be negative (paying fixed, receiving nothing)
        assert value < 0

    def test_inflation_linked_swap_high_inflation(self):
        """Test swap with high inflation."""
        swap = InflationLinkedSwap(
            notional=1_000_000.0,
            maturity=5.0,
            fixed_real_rate=0.01,
            payment_frequency=1,
        )

        # High inflation (10% annual)
        n_periods = 5
        cpi_path = jnp.array([100.0 * (1.10 ** i) for i in range(n_periods + 1)])

        discount_factors = jnp.exp(-0.03 * jnp.arange(1, n_periods + 1))

        value = swap.price(cpi_path, discount_factors)

        # With high inflation, floating leg should dominate
        # Value should be positive (receiving high inflation)
        assert value > 0


class TestInflationFloorCertificate:
    """Tests for inflation floor certificate."""

    def test_inflation_floor_certificate_above_floor(self):
        """Test certificate when inflation is above floor."""
        cert = InflationFloorCertificate(
            notional=100_000.0,
            maturity=5.0,
            participation_rate=1.0,
            guaranteed_floor=0.0,
        )

        # 15% realized inflation
        payoff = cert.payoff(realized_inflation=0.15)

        expected = 100_000.0 * (1.0 + 0.15)
        assert abs(payoff - expected) < 1.0

    def test_inflation_floor_certificate_below_floor(self):
        """Test certificate when inflation is below floor."""
        cert = InflationFloorCertificate(
            notional=100_000.0,
            maturity=5.0,
            participation_rate=1.0,
            guaranteed_floor=0.05,  # 5% guaranteed
        )

        # Only 2% realized inflation
        payoff = cert.payoff(realized_inflation=0.02)

        # Should get guaranteed floor
        expected = 100_000.0 * (1.0 + 0.05)
        assert abs(payoff - expected) < 1.0

    def test_inflation_floor_certificate_deflation(self):
        """Test certificate with deflation."""
        cert = InflationFloorCertificate(
            notional=100_000.0,
            maturity=5.0,
            participation_rate=1.0,
            guaranteed_floor=0.0,  # 0% guaranteed
        )

        # -5% realized inflation (deflation)
        payoff = cert.payoff(realized_inflation=-0.05)

        # Should get floor (no loss)
        expected = 100_000.0 * (1.0 + 0.0)
        assert abs(payoff - expected) < 1.0

    def test_inflation_floor_certificate_partial_participation(self):
        """Test certificate with partial participation."""
        cert = InflationFloorCertificate(
            notional=100_000.0,
            maturity=5.0,
            participation_rate=0.75,  # 75% participation
            guaranteed_floor=0.02,  # 2% guaranteed
        )

        # 10% realized inflation
        payoff = cert.payoff(realized_inflation=0.10)

        # Should get 75% of 10% = 7.5%
        expected = 100_000.0 * (1.0 + 0.075)
        assert abs(payoff - expected) < 1.0


class TestInflationProductsIntegration:
    """Integration tests for inflation products."""

    def test_inflation_swaption_vs_swap(self):
        """Test that swaption value is related to swap value."""
        # An in-the-money inflation swaption should have positive value
        itm_price = inflation_swaption_price(
            strike_inflation=0.02,
            forward_inflation=0.04,
            time_to_expiry=1.0,
            volatility=0.01,
            discount_factor=0.97,
            notional=1_000_000.0,
            is_payer=True,
        )

        # An out-of-the-money swaption should have lower value
        otm_price = inflation_swaption_price(
            strike_inflation=0.05,
            forward_inflation=0.04,
            time_to_expiry=1.0,
            volatility=0.01,
            discount_factor=0.97,
            notional=1_000_000.0,
            is_payer=True,
        )

        assert itm_price > otm_price

    def test_lpi_bond_vs_regular_inflation_bond(self):
        """Test LPI bond vs regular inflation bond."""
        # LPI bond with very high cap
        lpi_price = limited_price_indexation_bond(
            face_value=100.0,
            real_coupon_rate=0.02,
            real_yield=0.01,
            maturity=5.0,
            index_ratio=1.15,
            inflation_cap=0.50,  # Very high cap
            inflation_floor=0.00,
            frequency=2,
        )

        # Regular inflation bond (via import)
        from neutryx.products.inflation import inflation_linked_bond_price

        regular_price = inflation_linked_bond_price(
            face_value=100.0,
            real_coupon_rate=0.02,
            real_yield=0.01,
            maturity=5.0,
            index_ratio=1.15,
            frequency=2,
        )

        # With very high cap, LPI should be similar to regular
        assert abs(lpi_price - regular_price) < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
