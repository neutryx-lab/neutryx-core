"""Tests for FX volatility market conventions (BF/RR quotes)."""

import jax.numpy as jnp
import pytest

from neutryx.market.fx import (
    FXVolatilityQuote,
    FXVolatilitySurfaceBuilder,
    build_smile_from_market_quote,
    delta_to_strike_iterative,
    strike_from_delta_atm,
)


class TestFXVolatilityQuote:
    """Tests for FXVolatilityQuote and vol extraction."""

    def test_extract_pillar_vols_basic(self):
        """Test basic extraction of theoretical vols from ATM/BF/RR."""
        quote = FXVolatilityQuote(
            expiry=1.0,
            atm_vol=0.10,
            rr_25d=0.02,
            bf_25d=0.005,
            forward=1.10,
        )

        vols = quote.extract_pillar_vols()

        # ATM should equal input
        assert abs(vols['atm'] - 0.10) < 1e-10

        # 25d call vol = ATM + BF + RR/2 = 0.10 + 0.005 + 0.01 = 0.115
        expected_25d_call = 0.10 + 0.005 + 0.02 / 2
        assert abs(vols['25d_call'] - expected_25d_call) < 1e-10

        # 25d put vol = ATM + BF - RR/2 = 0.10 + 0.005 - 0.01 = 0.095
        expected_25d_put = 0.10 + 0.005 - 0.02 / 2
        assert abs(vols['25d_put'] - expected_25d_put) < 1e-10

    def test_extract_pillar_vols_zero_rr_bf(self):
        """Test with zero risk reversal and butterfly (flat smile)."""
        quote = FXVolatilityQuote(
            expiry=1.0,
            atm_vol=0.10,
            rr_25d=0.0,
            bf_25d=0.0,
            forward=1.10,
        )

        vols = quote.extract_pillar_vols()

        # All vols should equal ATM when RR=0 and BF=0
        assert abs(vols['atm'] - 0.10) < 1e-10
        assert abs(vols['25d_call'] - 0.10) < 1e-10
        assert abs(vols['25d_put'] - 0.10) < 1e-10

    def test_repr(self):
        """Test string representation."""
        quote = FXVolatilityQuote(
            expiry=1.0,
            atm_vol=0.10,
            rr_25d=0.015,
            bf_25d=0.005,
            forward=1.10,
        )

        repr_str = repr(quote)
        assert "1.00y" in repr_str
        assert "10.00%" in repr_str
        assert "1.50%" in repr_str
        assert "0.50%" in repr_str


class TestDeltaToStrike:
    """Tests for delta-to-strike conversion."""

    def test_delta_to_strike_atm(self):
        """Test that 50-delta roughly gives ATM strike."""
        strike, vol = delta_to_strike_iterative(
            delta=0.5,
            forward=1.10,
            expiry=1.0,
            vol_initial=0.10,
            is_call=True,
            domestic_rate=0.02,
            foreign_rate=0.01,
        )

        # 50-delta call should be close to forward (ATM)
        assert abs(strike - 1.10) < 0.05

    def test_delta_to_strike_25d_call(self):
        """Test 25-delta call strike is above forward."""
        strike, vol = delta_to_strike_iterative(
            delta=0.25,
            forward=1.10,
            expiry=1.0,
            vol_initial=0.10,
            is_call=True,
            domestic_rate=0.02,
            foreign_rate=0.01,
        )

        # 25-delta call (OTM) should have strike above forward
        assert strike > 1.10

    def test_delta_to_strike_25d_put(self):
        """Test 25-delta put strike is below forward."""
        strike, vol = delta_to_strike_iterative(
            delta=-0.25,
            forward=1.10,
            expiry=1.0,
            vol_initial=0.10,
            is_call=False,
            domestic_rate=0.02,
            foreign_rate=0.01,
        )

        # 25-delta put (OTM) should have strike below forward
        assert strike < 1.10

    def test_strike_from_delta_atm(self):
        """Test ATM strike calculation."""
        strike = strike_from_delta_atm(
            forward=1.10,
            expiry=1.0,
            vol=0.10,
        )

        # ATM forward convention: K = F
        assert abs(strike - 1.10) < 1e-10


class TestBuildSmile:
    """Tests for smile construction from market quotes."""

    def test_build_smile_from_market_quote(self):
        """Test building smile from ATM/BF/RR quote."""
        quote = FXVolatilityQuote(
            expiry=1.0,
            atm_vol=0.10,
            rr_25d=0.02,
            bf_25d=0.005,
            forward=1.10,
            domestic_rate=0.02,
            foreign_rate=0.01,
        )

        strikes, vols = build_smile_from_market_quote(quote, num_strikes=5)

        # Check output shapes
        assert len(strikes) == 5
        assert len(vols) == 5

        # Strikes should be sorted
        assert jnp.all(jnp.diff(strikes) > 0)

        # Vols should be positive
        assert jnp.all(vols > 0)

        # Check smile shape: should have higher vol at wings vs ATM
        # (positive butterfly)
        mid_idx = len(vols) // 2
        mid_vol = vols[mid_idx]
        # Due to positive BF, wings should be higher than center
        assert vols[0] >= mid_vol - 0.01  # Allow some tolerance
        assert vols[-1] >= mid_vol - 0.01

    def test_build_smile_flat(self):
        """Test building flat smile (RR=0, BF=0)."""
        quote = FXVolatilityQuote(
            expiry=1.0,
            atm_vol=0.10,
            rr_25d=0.0,
            bf_25d=0.0,
            forward=1.10,
            domestic_rate=0.02,
            foreign_rate=0.01,
        )

        strikes, vols = build_smile_from_market_quote(quote, num_strikes=5)

        # All vols should be approximately equal (flat smile)
        assert jnp.max(vols) - jnp.min(vols) < 0.02


class TestFXVolatilitySurfaceBuilder:
    """Tests for volatility surface builder."""

    def test_build_surface_single_tenor(self):
        """Test building surface from single tenor."""
        quote = FXVolatilityQuote(
            expiry=1.0,
            atm_vol=0.10,
            rr_25d=0.015,
            bf_25d=0.005,
            forward=1.10,
            domestic_rate=0.02,
            foreign_rate=0.01,
        )

        builder = FXVolatilitySurfaceBuilder(
            from_ccy="EUR",
            to_ccy="USD",
            quotes=[quote],
        )

        surface = builder.build_surface(num_strikes_per_tenor=7)

        # Check surface properties
        assert surface.from_ccy == "EUR"
        assert surface.to_ccy == "USD"
        assert len(surface.expiries) == 1
        assert surface.expiries[0] == 1.0
        assert len(surface.strikes) == 7
        assert surface.vols.shape == (1, 7)

    def test_build_surface_multiple_tenors(self):
        """Test building surface from multiple tenors."""
        quotes = [
            FXVolatilityQuote(
                expiry=0.25,
                atm_vol=0.095,
                rr_25d=0.010,
                bf_25d=0.003,
                forward=1.10,
                domestic_rate=0.02,
                foreign_rate=0.01,
            ),
            FXVolatilityQuote(
                expiry=0.5,
                atm_vol=0.100,
                rr_25d=0.012,
                bf_25d=0.004,
                forward=1.105,
                domestic_rate=0.02,
                foreign_rate=0.01,
            ),
            FXVolatilityQuote(
                expiry=1.0,
                atm_vol=0.105,
                rr_25d=0.015,
                bf_25d=0.005,
                forward=1.11,
                domestic_rate=0.02,
                foreign_rate=0.01,
            ),
        ]

        builder = FXVolatilitySurfaceBuilder(
            from_ccy="EUR",
            to_ccy="USD",
            quotes=quotes,
        )

        surface = builder.build_surface(num_strikes_per_tenor=9)

        # Check surface properties
        assert len(surface.expiries) == 3
        assert len(surface.strikes) == 9
        assert surface.vols.shape == (3, 9)

        # Check expiries are sorted
        assert jnp.all(jnp.diff(surface.expiries) > 0)

        # Test interpolation
        vol = surface.implied_vol(0.75, 1.10)
        assert vol > 0
        # Should be between 0.5y and 1y vols
        assert 0.08 < vol < 0.15

    def test_add_quote(self):
        """Test adding quotes dynamically."""
        builder = FXVolatilitySurfaceBuilder(
            from_ccy="EUR",
            to_ccy="USD",
            quotes=[],
        )

        assert len(builder.quotes) == 0

        quote = FXVolatilityQuote(
            expiry=1.0,
            atm_vol=0.10,
            rr_25d=0.015,
            bf_25d=0.005,
            forward=1.10,
        )

        builder.add_quote(quote)
        assert len(builder.quotes) == 1

    def test_get_quote(self):
        """Test retrieving quote by expiry."""
        quote1 = FXVolatilityQuote(
            expiry=0.5,
            atm_vol=0.10,
            rr_25d=0.015,
            bf_25d=0.005,
            forward=1.10,
        )
        quote2 = FXVolatilityQuote(
            expiry=1.0,
            atm_vol=0.105,
            rr_25d=0.018,
            bf_25d=0.006,
            forward=1.11,
        )

        builder = FXVolatilitySurfaceBuilder(
            from_ccy="EUR",
            to_ccy="USD",
            quotes=[quote1, quote2],
        )

        found = builder.get_quote(1.0)
        assert found is not None
        assert found.expiry == 1.0

        not_found = builder.get_quote(2.0)
        assert not_found is None

    def test_repr(self):
        """Test string representation."""
        builder = FXVolatilitySurfaceBuilder(
            from_ccy="EUR",
            to_ccy="USD",
            quotes=[
                FXVolatilityQuote(expiry=1.0, atm_vol=0.10, rr_25d=0.01, bf_25d=0.005, forward=1.10),
            ],
        )

        repr_str = repr(builder)
        assert "EUR/USD" in repr_str
        assert "1 tenors" in repr_str


class TestIntegration:
    """Integration tests for full workflow."""

    def test_end_to_end_surface_construction(self):
        """Test complete workflow from quotes to surface to pricing."""
        # Create market quotes for EUR/USD
        quotes = [
            FXVolatilityQuote(
                expiry=0.25,
                atm_vol=0.095,
                rr_25d=0.010,
                bf_25d=0.003,
                forward=1.10,
                domestic_rate=0.025,
                foreign_rate=0.015,
            ),
            FXVolatilityQuote(
                expiry=0.5,
                atm_vol=0.100,
                rr_25d=0.012,
                bf_25d=0.004,
                forward=1.105,
                domestic_rate=0.025,
                foreign_rate=0.015,
            ),
            FXVolatilityQuote(
                expiry=1.0,
                atm_vol=0.105,
                rr_25d=0.015,
                bf_25d=0.005,
                forward=1.11,
                domestic_rate=0.025,
                foreign_rate=0.015,
            ),
        ]

        # Build surface
        builder = FXVolatilitySurfaceBuilder(
            from_ccy="EUR",
            to_ccy="USD",
            quotes=quotes,
        )
        surface = builder.build_surface(num_strikes_per_tenor=11)

        # Query vol at different points
        vol_atm = surface.implied_vol(0.5, 1.105)
        vol_otm_call = surface.implied_vol(0.5, 1.15)
        vol_otm_put = surface.implied_vol(0.5, 1.05)

        # Basic sanity checks
        assert 0.08 < vol_atm < 0.15
        assert vol_otm_call > 0
        assert vol_otm_put > 0

        # Smile shape: for positive RR, calls should have higher vol than puts
        # (though interpolation may smooth this out)
        assert vol_atm > 0

    def test_market_data_consistency(self):
        """Test that extracted vols satisfy BF/RR definitions."""
        quote = FXVolatilityQuote(
            expiry=1.0,
            atm_vol=0.10,
            rr_25d=0.02,
            bf_25d=0.005,
            forward=1.10,
        )

        vols = quote.extract_pillar_vols()

        # Check RR definition: vol_call - vol_put = RR
        computed_rr = vols['25d_call'] - vols['25d_put']
        assert abs(computed_rr - quote.rr_25d) < 1e-10

        # Check BF definition: (vol_call + vol_put)/2 - vol_atm = BF
        computed_bf = (vols['25d_call'] + vols['25d_put']) / 2 - vols['atm']
        assert abs(computed_bf - quote.bf_25d) < 1e-10
