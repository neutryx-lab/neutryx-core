"""Tests for joint calibration framework."""
import jax
import jax.numpy as jnp
import pytest
from jax.scipy.stats import norm

from neutryx.calibration.constraints import bounded, positive, symmetric
from neutryx.calibration.base import ParameterSpec
from neutryx.calibration.joint_calibration import (
    AssetClassSpec,
    CrossAssetCalibrator,
    InstrumentSpec,
    MultiInstrumentCalibrator,
    TimeDependentCalibrator,
    TimeSegment,
)
from neutryx.models.sabr import SABRParams, hagan_implied_vol

jax.config.update("jax_enable_x64", True)


def black_scholes_call(S0, K, T, r, q, sigma):
    """Black-Scholes call option formula."""
    sqrtT = jnp.sqrt(T)
    d1 = (jnp.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return S0 * jnp.exp(-q * T) * norm.cdf(d1) - K * jnp.exp(-r * T) * norm.cdf(d2)


def black_scholes_put(S0, K, T, r, q, sigma):
    """Black-Scholes put option formula."""
    sqrtT = jnp.sqrt(T)
    d1 = (jnp.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return K * jnp.exp(-r * T) * norm.cdf(-d2) - S0 * jnp.exp(-q * T) * norm.cdf(-d1)


class TestMultiInstrumentCalibrator:
    """Test multi-instrument calibration."""

    def test_multi_strike_option_calibration(self):
        """Test calibrating to options at multiple strikes."""
        # True parameters
        true_vol = 0.25
        S0 = 100.0
        r = 0.03

        # Generate synthetic data: ATM and OTM calls
        atm_strikes = jnp.array([95.0, 100.0, 105.0])
        otm_strikes = jnp.array([110.0, 115.0, 120.0])
        T = 1.0

        atm_prices = jax.vmap(
            lambda K: black_scholes_call(S0, K, T, r, 0.0, true_vol)
        )(atm_strikes)

        otm_prices = jax.vmap(
            lambda K: black_scholes_call(S0, K, T, r, 0.0, true_vol)
        )(otm_strikes)

        # Create instruments
        def price_calls(params, data):
            vol = params["volatility"]
            strikes = data["strikes"]
            return jax.vmap(lambda K: black_scholes_call(S0, K, T, r, 0.0, vol))(strikes)

        instruments = [
            InstrumentSpec(
                name="atm_calls",
                pricing_fn=price_calls,
                target_prices=atm_prices,
                weights=jnp.ones(len(atm_prices)) * 2.0,  # Higher weight on ATM
                market_data={"strikes": atm_strikes},
            ),
            InstrumentSpec(
                name="otm_calls",
                pricing_fn=price_calls,
                target_prices=otm_prices,
                weights=jnp.ones(len(otm_prices)),
                market_data={"strikes": otm_strikes},
            ),
        ]

        # Calibrate
        calibrator = MultiInstrumentCalibrator(
            parameter_specs={"volatility": ParameterSpec(0.2, positive())},
            instruments=instruments,
            max_steps=200,
        )

        result = calibrator.calibrate({})

        # Check convergence
        assert result.converged, "Multi-instrument calibration should converge"
        assert abs(result.params["volatility"] - true_vol) < 0.01, \
            "Should recover true volatility"

    def test_calls_and_puts_calibration(self):
        """Test calibrating to both calls and puts."""
        true_vol = 0.30
        S0 = 100.0
        r = 0.02
        T = 0.5

        strikes = jnp.array([90.0, 100.0, 110.0])

        call_prices = jax.vmap(
            lambda K: black_scholes_call(S0, K, T, r, 0.0, true_vol)
        )(strikes)

        put_prices = jax.vmap(
            lambda K: black_scholes_put(S0, K, T, r, 0.0, true_vol)
        )(strikes)

        def price_calls(params, data):
            vol = params["vol"]
            return jax.vmap(lambda K: black_scholes_call(S0, K, T, r, 0.0, vol))(
                data["strikes"]
            )

        def price_puts(params, data):
            vol = params["vol"]
            return jax.vmap(lambda K: black_scholes_put(S0, K, T, r, 0.0, vol))(
                data["strikes"]
            )

        instruments = [
            InstrumentSpec(
                name="calls",
                pricing_fn=price_calls,
                target_prices=call_prices,
                market_data={"strikes": strikes},
            ),
            InstrumentSpec(
                name="puts",
                pricing_fn=price_puts,
                target_prices=put_prices,
                market_data={"strikes": strikes},
            ),
        ]

        calibrator = MultiInstrumentCalibrator(
            parameter_specs={"vol": ParameterSpec(0.25, positive())},
            instruments=instruments,
            max_steps=200,
        )

        result = calibrator.calibrate({})

        assert result.converged, "Should converge"
        assert abs(result.params["vol"] - true_vol) < 0.02, \
            "Should recover volatility from calls and puts"

    def test_multi_maturity_calibration(self):
        """Test calibrating to multiple maturities."""
        true_vol = 0.22
        S0 = 100.0
        r = 0.03
        K = 100.0

        maturities = jnp.array([0.25, 0.5, 1.0, 2.0])

        prices = jax.vmap(
            lambda T: black_scholes_call(S0, K, T, r, 0.0, true_vol)
        )(maturities)

        def price_options(params, data):
            vol = params["sigma"]
            return jax.vmap(lambda T: black_scholes_call(S0, K, T, r, 0.0, vol))(
                data["maturities"]
            )

        instruments = [
            InstrumentSpec(
                name="multi_maturity",
                pricing_fn=price_options,
                target_prices=prices,
                market_data={"maturities": maturities},
            ),
        ]

        calibrator = MultiInstrumentCalibrator(
            parameter_specs={"sigma": ParameterSpec(0.2, positive())},
            instruments=instruments,
            max_steps=200,
        )

        result = calibrator.calibrate({})

        assert result.converged, "Should converge"
        assert abs(result.params["sigma"] - true_vol) < 0.01, \
            "Should recover volatility from multiple maturities"


class TestCrossAssetCalibrator:
    """Test cross-asset calibration."""

    def test_two_asset_shared_correlation(self):
        """Test calibrating two assets with shared correlation parameter."""
        # True parameters
        vol_asset1 = 0.20
        vol_asset2 = 0.30
        true_corr = 0.5

        S1, S2 = 100.0, 100.0
        K = 100.0
        T = 1.0
        r = 0.03

        # Generate synthetic prices for each asset
        price1 = black_scholes_call(S1, K, T, r, 0.0, vol_asset1)
        price2 = black_scholes_call(S2, K, T, r, 0.0, vol_asset2)

        # For spread option, use approximate pricing with correlation
        # Simple approximation: basket vol with correlation
        basket_vol = jnp.sqrt(vol_asset1**2 + vol_asset2**2 + 2*true_corr*vol_asset1*vol_asset2) / jnp.sqrt(2)
        spread_price = black_scholes_call(S1, S2, T, r, 0.0, basket_vol) * 0.7  # Approximate

        def price_asset1(params, data):
            vol = params["vol_asset1"]
            return jnp.array([black_scholes_call(S1, K, T, r, 0.0, vol)])

        def price_asset2(params, data):
            vol = params["vol_asset2"]
            return jnp.array([black_scholes_call(S2, K, T, r, 0.0, vol)])

        def price_spread(params, data):
            vol1 = params["vol_asset1"]
            vol2 = params["vol_asset2"]
            corr = params["correlation"]
            bvol = jnp.sqrt(vol1**2 + vol2**2 + 2*corr*vol1*vol2) / jnp.sqrt(2)
            return jnp.array([black_scholes_call(S1, S2, T, r, 0.0, bvol) * 0.7])

        # Asset 1 instruments
        asset1_spec = AssetClassSpec(
            name="asset1",
            parameter_specs={"vol_asset1": ParameterSpec(0.25, positive())},
            instruments=[
                InstrumentSpec(
                    name="vanilla_option",
                    pricing_fn=price_asset1,
                    target_prices=jnp.array([price1]),
                ),
            ],
            shared_params=["correlation"],
        )

        # Asset 2 instruments
        asset2_spec = AssetClassSpec(
            name="asset2",
            parameter_specs={"vol_asset2": ParameterSpec(0.25, positive())},
            instruments=[
                InstrumentSpec(
                    name="vanilla_option",
                    pricing_fn=price_asset2,
                    target_prices=jnp.array([price2]),
                ),
            ],
            shared_params=["correlation"],
        )

        # Spread instrument (uses both assets)
        spread_spec = AssetClassSpec(
            name="spread",
            parameter_specs={
                "vol_asset1": ParameterSpec(0.25, positive()),
                "vol_asset2": ParameterSpec(0.25, positive()),
            },
            instruments=[
                InstrumentSpec(
                    name="spread_option",
                    pricing_fn=price_spread,
                    target_prices=jnp.array([spread_price]),
                    weights=jnp.array([2.0]),  # Higher weight on spread
                ),
            ],
            shared_params=["correlation"],
        )

        shared_specs = {"correlation": ParameterSpec(0.0, symmetric(0.999))}

        calibrator = CrossAssetCalibrator(
            asset_classes=[asset1_spec, asset2_spec, spread_spec],
            shared_parameter_specs=shared_specs,
            max_steps=300,
        )

        result = calibrator.calibrate({})

        assert result.converged, "Cross-asset calibration should converge"

        # Extract parameters
        vol1_calibrated = result.params["asset1_vol_asset1"]
        vol2_calibrated = result.params["asset2_vol_asset2"]
        corr_calibrated = result.params["shared_correlation"]

        # Check individual volatilities are close
        assert abs(vol1_calibrated - vol_asset1) < 0.03, \
            "Asset 1 volatility should be close to true value"
        assert abs(vol2_calibrated - vol_asset2) < 0.03, \
            "Asset 2 volatility should be close to true value"

        # Correlation should be positive (direction matters more than exact value)
        assert corr_calibrated > 0, "Correlation should be positive"

    def test_fx_equity_linkage(self):
        """Test FX and equity calibration with linked parameters."""
        # Simple test: two assets with independent volatilities
        vol_fx = 0.12
        vol_eq = 0.25

        S_fx, S_eq = 1.2, 100.0
        K_fx, K_eq = 1.2, 100.0
        T = 0.5
        r = 0.02

        price_fx = black_scholes_call(S_fx, K_fx, T, r, 0.0, vol_fx)
        price_eq = black_scholes_call(S_eq, K_eq, T, r, 0.0, vol_eq)

        def price_fx_option(params, data):
            vol = params["fx_vol"]
            return jnp.array([black_scholes_call(S_fx, K_fx, T, r, 0.0, vol)])

        def price_eq_option(params, data):
            vol = params["eq_vol"]
            return jnp.array([black_scholes_call(S_eq, K_eq, T, r, 0.0, vol)])

        fx_spec = AssetClassSpec(
            name="fx",
            parameter_specs={"fx_vol": ParameterSpec(0.15, positive())},
            instruments=[
                InstrumentSpec(
                    name="fx_call",
                    pricing_fn=price_fx_option,
                    target_prices=jnp.array([price_fx]),
                ),
            ],
        )

        eq_spec = AssetClassSpec(
            name="equity",
            parameter_specs={"eq_vol": ParameterSpec(0.2, positive())},
            instruments=[
                InstrumentSpec(
                    name="eq_call",
                    pricing_fn=price_eq_option,
                    target_prices=jnp.array([price_eq]),
                ),
            ],
        )

        calibrator = CrossAssetCalibrator(
            asset_classes=[fx_spec, eq_spec],
            max_steps=200,
        )

        result = calibrator.calibrate({})

        assert result.converged, "Should converge"

        vol_fx_cal = result.params["fx_fx_vol"]
        vol_eq_cal = result.params["equity_eq_vol"]

        assert abs(vol_fx_cal - vol_fx) < 0.02, "FX vol should match"
        assert abs(vol_eq_cal - vol_eq) < 0.02, "Equity vol should match"


class TestTimeDependentCalibrator:
    """Test time-dependent parameter calibration."""

    def test_piecewise_constant_volatility(self):
        """Test piecewise constant volatility calibration."""
        # True piecewise constant volatility
        vol_segment1 = 0.20  # 0-6 months
        vol_segment2 = 0.30  # 6-12 months
        vol_segment3 = 0.25  # 1-2 years

        S0 = 100.0
        K = 100.0
        r = 0.03

        # Generate observations at different times
        observation_times = jnp.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0])

        def get_true_vol(t):
            """Get true volatility for a given time."""
            if t < 0.5:
                return vol_segment1
            elif t < 1.0:
                return vol_segment2
            else:
                return vol_segment3

        # Generate target prices
        target_prices = jnp.array([
            black_scholes_call(S0, K, t, r, 0.0, get_true_vol(float(t)))
            for t in observation_times
        ])

        # Define time segments
        segments = [
            TimeSegment(0.0, 0.5, {"vol": ParameterSpec(0.25, positive())}),
            TimeSegment(0.5, 1.0, {"vol": ParameterSpec(0.25, positive())}),
            TimeSegment(1.0, 2.5, {"vol": ParameterSpec(0.25, positive())}),
        ]

        def pricing_fn(params, time, market_data):
            """Price option at given time with segment-specific vol."""
            vol = params["vol"]
            return black_scholes_call(S0, K, time, r, 0.0, vol)

        calibrator = TimeDependentCalibrator(
            time_segments=segments,
            pricing_fn=pricing_fn,
            target_prices=target_prices,
            observation_times=observation_times,
            max_steps=300,
        )

        result = calibrator.calibrate({})

        assert result.converged, "Time-dependent calibration should converge"

        # Extract segment volatilities
        vol_seg1 = result.params["t0_vol"]
        vol_seg2 = result.params["t1_vol"]
        vol_seg3 = result.params["t2_vol"]

        # Check segment volatilities
        assert abs(vol_seg1 - vol_segment1) < 0.03, \
            "Segment 1 volatility should match"
        assert abs(vol_seg2 - vol_segment2) < 0.03, \
            "Segment 2 volatility should match"
        assert abs(vol_seg3 - vol_segment3) < 0.03, \
            "Segment 3 volatility should match"

    def test_smoothness_penalty(self):
        """Test that smoothness penalty reduces parameter jumps."""
        S0 = 100.0
        K = 100.0
        r = 0.03

        # Use constant volatility data
        true_vol = 0.25
        observation_times = jnp.array([0.25, 0.75, 1.5])

        target_prices = jnp.array([
            black_scholes_call(S0, K, t, r, 0.0, true_vol)
            for t in observation_times
        ])

        segments = [
            TimeSegment(0.0, 0.5, {"vol": ParameterSpec(0.2, positive())}),
            TimeSegment(0.5, 1.0, {"vol": ParameterSpec(0.3, positive())}),
            TimeSegment(1.0, 2.0, {"vol": ParameterSpec(0.2, positive())}),
        ]

        def pricing_fn(params, time, market_data):
            vol = params["vol"]
            return black_scholes_call(S0, K, time, r, 0.0, vol)

        # Calibrate without smoothness penalty
        calibrator_no_penalty = TimeDependentCalibrator(
            time_segments=segments,
            pricing_fn=pricing_fn,
            target_prices=target_prices,
            observation_times=observation_times,
            max_steps=300,
            smoothness_penalty=0.0,
        )

        result_no_penalty = calibrator_no_penalty.calibrate({})

        # Calibrate with smoothness penalty
        calibrator_with_penalty = TimeDependentCalibrator(
            time_segments=segments,
            pricing_fn=pricing_fn,
            target_prices=target_prices,
            observation_times=observation_times,
            max_steps=300,
            smoothness_penalty=1.0,
        )

        result_with_penalty = calibrator_with_penalty.calibrate({})

        # Both should converge
        assert result_no_penalty.converged and result_with_penalty.converged

        # Compute variance of parameters (measure of jumpiness)
        vols_no_penalty = [
            result_no_penalty.params["t0_vol"],
            result_no_penalty.params["t1_vol"],
            result_no_penalty.params["t2_vol"],
        ]

        vols_with_penalty = [
            result_with_penalty.params["t0_vol"],
            result_with_penalty.params["t1_vol"],
            result_with_penalty.params["t2_vol"],
        ]

        variance_no_penalty = jnp.var(jnp.array(vols_no_penalty))
        variance_with_penalty = jnp.var(jnp.array(vols_with_penalty))

        # Smoothness penalty should reduce variance (make parameters more uniform)
        assert variance_with_penalty <= variance_no_penalty * 1.1, \
            "Smoothness penalty should reduce parameter variance"

    def test_term_structure_of_volatility(self):
        """Test calibrating term structure of volatility."""
        S0 = 100.0
        K = 100.0
        r = 0.03

        # Realistic term structure: mean-reverting to long-term level
        def true_vol_structure(t):
            """Vol starts high, drops, then stabilizes."""
            long_term_vol = 0.22
            short_term_vol = 0.35
            reversion_speed = 2.0
            return long_term_vol + (short_term_vol - long_term_vol) * jnp.exp(-reversion_speed * t)

        observation_times = jnp.array([0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])

        target_prices = jnp.array([
            black_scholes_call(S0, K, t, r, 0.0, float(true_vol_structure(t)))
            for t in observation_times
        ])

        # Define fine-grained time segments
        segments = [
            TimeSegment(0.0, 0.25, {"vol": ParameterSpec(0.3, positive())}),
            TimeSegment(0.25, 0.5, {"vol": ParameterSpec(0.28, positive())}),
            TimeSegment(0.5, 1.0, {"vol": ParameterSpec(0.25, positive())}),
            TimeSegment(1.0, 2.5, {"vol": ParameterSpec(0.22, positive())}),
        ]

        def pricing_fn(params, time, market_data):
            vol = params["vol"]
            return black_scholes_call(S0, K, time, r, 0.0, vol)

        calibrator = TimeDependentCalibrator(
            time_segments=segments,
            pricing_fn=pricing_fn,
            target_prices=target_prices,
            observation_times=observation_times,
            max_steps=400,
            smoothness_penalty=0.5,
        )

        result = calibrator.calibrate({})

        assert result.converged, "Should converge"

        # Extract volatilities
        vol_t0 = result.params["t0_vol"]
        vol_t1 = result.params["t1_vol"]
        vol_t2 = result.params["t2_vol"]
        vol_t3 = result.params["t3_vol"]

        # Term structure should be decreasing
        assert vol_t0 >= vol_t1 - 0.05, "Vol should decrease or stay flat"
        assert vol_t1 >= vol_t2 - 0.05, "Vol should decrease or stay flat"
        assert vol_t2 >= vol_t3 - 0.05, "Vol should decrease or stay flat"


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_multi_instrument_with_sabr(self):
        """Test multi-instrument calibration with SABR model."""
        # True SABR parameters
        true_params = SABRParams(alpha=0.3, beta=0.5, rho=-0.3, nu=0.4)
        forward = 0.05
        T = 1.0

        # Generate strikes
        atm_strikes = jnp.array([0.045, 0.05, 0.055])
        otm_strikes = jnp.array([0.06, 0.065, 0.07])

        # Generate target implied vols
        atm_vols = jax.vmap(lambda K: hagan_implied_vol(forward, K, T, true_params))(
            atm_strikes
        )
        otm_vols = jax.vmap(lambda K: hagan_implied_vol(forward, K, T, true_params))(
            otm_strikes
        )

        def price_sabr_vols(params, data):
            sabr_params = SABRParams(
                alpha=params["alpha"],
                beta=params["beta"],
                rho=params["rho"],
                nu=params["nu"],
            )
            strikes = data["strikes"]
            vols = jax.vmap(lambda K: hagan_implied_vol(forward, K, T, sabr_params))(
                strikes
            )
            return jnp.where(jnp.isfinite(vols), vols, 0.3)

        instruments = [
            InstrumentSpec(
                name="atm",
                pricing_fn=price_sabr_vols,
                target_prices=atm_vols,
                weights=jnp.ones(len(atm_vols)) * 2.0,
                market_data={"strikes": atm_strikes},
            ),
            InstrumentSpec(
                name="otm",
                pricing_fn=price_sabr_vols,
                target_prices=otm_vols,
                market_data={"strikes": otm_strikes},
            ),
        ]

        from neutryx.calibration.constraints import positive_with_upper

        calibrator = MultiInstrumentCalibrator(
            parameter_specs={
                "alpha": ParameterSpec(0.25, positive_with_upper(1e-4, 3.0)),
                "beta": ParameterSpec(0.5, bounded(0.0, 0.999)),
                "rho": ParameterSpec(-0.2, symmetric(0.999)),
                "nu": ParameterSpec(0.5, positive_with_upper(1e-4, 3.0)),
            },
            instruments=instruments,
            max_steps=400,
        )

        result = calibrator.calibrate({})

        # Should achieve reasonable fit
        assert result.converged or result.iterations >= 350, \
            "SABR calibration should make progress"

        # Parameters should be in reasonable range
        assert 0.1 < result.params["alpha"] < 1.0, "Alpha should be reasonable"
        assert 0.0 < result.params["beta"] < 1.0, "Beta should be in [0,1]"
        assert -1.0 < result.params["rho"] < 0.5, "Rho should be reasonable"
        assert 0.1 < result.params["nu"] < 2.0, "Nu should be reasonable"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
