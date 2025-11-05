"""Tests for interest rate caps, floors, and collars."""
from __future__ import annotations

import jax.numpy as jnp
import pytest

from neutryx.products.linear_rates.caps_floors import (
    Cap,
    CapFloorType,
    Collar,
    Floor,
    InterestRateCapFloorCollar,
    black_caplet,
    black_floorlet,
)


class TestInterestRateCapFloorCollar:
    """Test cases for caps, floors, and collars."""

    def test_cap_pricing_basic(self):
        """Test basic cap pricing."""
        cap = InterestRateCapFloorCollar(
            T=5.0,
            notional=1_000_000,
            strike=0.03,
            cap_floor_type=CapFloorType.CAP,
            payment_frequency=4,
            volatility=0.20,
            forward_rates=jnp.full(20, 0.035),  # Rates slightly above strike
        )

        pv = cap.payoff_terminal(0.0)
        assert pv > 0, "Cap should have positive value when forwards > strike"

    def test_floor_pricing_basic(self):
        """Test basic floor pricing."""
        floor = InterestRateCapFloorCollar(
            T=5.0,
            notional=1_000_000,
            strike=0.04,
            cap_floor_type=CapFloorType.FLOOR,
            payment_frequency=4,
            volatility=0.20,
            forward_rates=jnp.full(20, 0.035),  # Rates below strike
        )

        pv = floor.payoff_terminal(0.0)
        assert pv > 0, "Floor should have positive value when forwards < strike"

    def test_collar_pricing(self):
        """Test collar pricing (long cap, short floor)."""
        collar = InterestRateCapFloorCollar(
            T=5.0,
            notional=1_000_000,
            strike=0.04,  # Cap strike
            cap_floor_type=CapFloorType.COLLAR,
            collar_floor_strike=0.02,  # Floor strike
            payment_frequency=4,
            volatility=0.20,
            forward_rates=jnp.full(20, 0.03),
        )

        pv = collar.payoff_terminal(0.0)
        assert isinstance(pv, (float, jnp.ndarray)), "Collar should return numeric value"

    def test_cap_zero_volatility(self):
        """Test cap pricing with zero volatility (intrinsic value only)."""
        cap = InterestRateCapFloorCollar(
            T=5.0,
            notional=1_000_000,
            strike=0.03,
            cap_floor_type=CapFloorType.CAP,
            payment_frequency=4,
            volatility=0.0,  # Zero vol
            forward_rates=jnp.full(20, 0.04),
        )

        pv = cap.payoff_terminal(0.0)
        assert pv > 0, "Cap should have intrinsic value even with zero volatility"

    def test_cap_out_of_money(self):
        """Test cap that is out of the money."""
        cap = InterestRateCapFloorCollar(
            T=5.0,
            notional=1_000_000,
            strike=0.05,  # High strike
            cap_floor_type=CapFloorType.CAP,
            payment_frequency=4,
            volatility=0.20,
            forward_rates=jnp.full(20, 0.03),  # Low forwards
        )

        pv = cap.payoff_terminal(0.0)
        # Out of money cap should have low time value
        assert pv > 0, "Cap should have positive time value"

    def test_floor_out_of_money(self):
        """Test floor that is out of the money."""
        floor = InterestRateCapFloorCollar(
            T=5.0,
            notional=1_000_000,
            strike=0.02,  # Low strike
            cap_floor_type=CapFloorType.FLOOR,
            payment_frequency=4,
            volatility=0.20,
            forward_rates=jnp.full(20, 0.04),  # High forwards
        )

        pv = floor.payoff_terminal(0.0)
        assert pv > 0, "Floor should have positive time value"

    def test_vega_sensitivity(self):
        """Test vega calculation for cap."""
        cap = InterestRateCapFloorCollar(
            T=5.0,
            notional=1_000_000,
            strike=0.03,
            cap_floor_type=CapFloorType.CAP,
            payment_frequency=4,
            volatility=0.20,
            forward_rates=jnp.full(20, 0.03),
        )

        vega = cap.vega()
        assert vega > 0, "Vega should be positive for cap"

    def test_cap_increasing_volatility(self):
        """Test that cap value increases with volatility."""
        base_cap = InterestRateCapFloorCollar(
            T=5.0,
            notional=1_000_000,
            strike=0.03,
            cap_floor_type=CapFloorType.CAP,
            payment_frequency=4,
            volatility=0.15,
            forward_rates=jnp.full(20, 0.03),
        )

        high_vol_cap = InterestRateCapFloorCollar(
            T=5.0,
            notional=1_000_000,
            strike=0.03,
            cap_floor_type=CapFloorType.CAP,
            payment_frequency=4,
            volatility=0.30,
            forward_rates=jnp.full(20, 0.03),
        )

        pv_base = base_cap.payoff_terminal(0.0)
        pv_high_vol = high_vol_cap.payoff_terminal(0.0)

        assert pv_high_vol > pv_base, "Cap value should increase with volatility"


class TestCapletFloorlet:
    """Test cases for individual caplet/floorlet pricing."""

    def test_black_caplet_atm(self):
        """Test Black caplet pricing at-the-money."""
        caplet = black_caplet(
            notional=1_000_000,
            strike=0.03,
            forward_rate=0.03,  # ATM
            volatility=0.20,
            time_to_reset=1.0,
            period_length=0.25,
            discount_factor=0.97,
        )

        assert caplet > 0, "ATM caplet should have positive time value"

    def test_black_caplet_itm(self):
        """Test Black caplet in-the-money."""
        caplet = black_caplet(
            notional=1_000_000,
            strike=0.03,
            forward_rate=0.04,  # ITM
            volatility=0.20,
            time_to_reset=1.0,
            period_length=0.25,
            discount_factor=0.97,
        )

        assert caplet > 0, "ITM caplet should have positive value"

    def test_black_floorlet_atm(self):
        """Test Black floorlet pricing at-the-money."""
        floorlet = black_floorlet(
            notional=1_000_000,
            strike=0.03,
            forward_rate=0.03,  # ATM
            volatility=0.20,
            time_to_reset=1.0,
            period_length=0.25,
            discount_factor=0.97,
        )

        assert floorlet > 0, "ATM floorlet should have positive time value"

    def test_black_floorlet_itm(self):
        """Test Black floorlet in-the-money."""
        floorlet = black_floorlet(
            notional=1_000_000,
            strike=0.04,
            forward_rate=0.03,  # ITM
            volatility=0.20,
            time_to_reset=1.0,
            period_length=0.25,
            discount_factor=0.97,
        )

        assert floorlet > 0, "ITM floorlet should have positive value"

    def test_put_call_parity(self):
        """Test put-call parity for caplets and floorlets."""
        params = {
            "notional": 1_000_000,
            "strike": 0.03,
            "forward_rate": 0.035,
            "volatility": 0.20,
            "time_to_reset": 1.0,
            "period_length": 0.25,
            "discount_factor": 0.97,
        }

        caplet = black_caplet(**params)
        floorlet = black_floorlet(**params)

        # Put-call parity: Caplet - Floorlet = PV(Forward - Strike)
        intrinsic = (
            params["notional"]
            * (params["forward_rate"] - params["strike"])
            * params["period_length"]
            * params["discount_factor"]
        )

        assert (
            abs((caplet - floorlet) - intrinsic) < 1.0
        ), "Put-call parity should hold"

    def test_caplet_zero_volatility(self):
        """Test caplet with zero volatility gives intrinsic value."""
        caplet = black_caplet(
            notional=1_000_000,
            strike=0.03,
            forward_rate=0.04,
            volatility=0.0,
            time_to_reset=1.0,
            period_length=0.25,
            discount_factor=0.97,
        )

        # Intrinsic value: 1M × (0.04 - 0.03) × 0.25 × 0.97 = 2,425
        intrinsic = 1_000_000 * (0.04 - 0.03) * 0.25 * 0.97

        assert abs(caplet - intrinsic) < 1.0, "Zero vol caplet should equal intrinsic"


class TestConvenienceClasses:
    """Test convenience classes Cap, Floor, Collar."""

    def test_cap_class(self):
        """Test Cap convenience class."""
        cap = Cap(
            T=5.0,
            notional=1_000_000,
            strike=0.03,
            payment_frequency=4,
            volatility=0.20,
        )

        assert cap.cap_floor_type == CapFloorType.CAP
        pv = cap.payoff_terminal(0.0)
        assert pv > 0, "Cap should have positive value"

    def test_floor_class(self):
        """Test Floor convenience class."""
        floor = Floor(
            T=5.0,
            notional=1_000_000,
            strike=0.03,
            payment_frequency=4,
            volatility=0.20,
        )

        assert floor.cap_floor_type == CapFloorType.FLOOR
        pv = floor.payoff_terminal(0.0)
        assert pv > 0, "Floor should have positive value"

    def test_collar_class(self):
        """Test Collar convenience class."""
        collar = Collar(
            T=5.0,
            notional=1_000_000,
            cap_strike=0.04,
            floor_strike=0.02,
            payment_frequency=4,
            volatility=0.20,
        )

        assert collar.cap_floor_type == CapFloorType.COLLAR
        assert collar.strike == 0.04  # Cap strike
        assert collar.collar_floor_strike == 0.02
        pv = collar.payoff_terminal(0.0)
        assert isinstance(pv, (float, jnp.ndarray)), "Collar should return numeric value"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
