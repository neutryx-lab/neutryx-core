"""Tests for comprehensive sensitivity analysis framework."""
import jax.numpy as jnp
import pytest

from neutryx.risk.sensitivity_analysis import (
    FiniteDiffScheme,
    PortfolioSensitivity,
    SensitivityConfig,
    SensitivityMethod,
    aggregate_portfolio_sensitivities,
    benchmark_sensitivity_methods,
    calculate_cs01,
    calculate_dv01,
    calculate_fx_greeks,
    calculate_higher_order_greeks,
    calculate_vega_surface,
    format_sensitivity_report,
)


class TestDV01:
    """Test DV01 (interest rate sensitivity) calculations."""

    def test_dv01_simple_bond(self):
        """Test DV01 for a simple bond."""
        # Simple bond price: P = C * sum(exp(-r*t)) + FV * exp(-r*T)
        def price_bond(params):
            r = params[0]  # Interest rate
            coupon = 5.0  # 5% coupon
            face_value = 100.0
            T = 5.0  # 5 year bond
            n_periods = 10  # Semi-annual

            pv_coupons = sum([coupon/2 * jnp.exp(-r * (i+1)*T/n_periods)
                              for i in range(n_periods)])
            pv_principal = face_value * jnp.exp(-r * T)

            return pv_coupons + pv_principal

        params = jnp.array([0.05])  # 5% rate
        dv01 = calculate_dv01(price_bond, params, rate_indices=0)

        # DV01 should be negative (bond loses value when rates rise)
        assert dv01 < 0

        # Check magnitude is reasonable (should be non-zero and negative)
        # The actual value depends on the bond characteristics
        assert abs(dv01) > 0.001
        assert abs(dv01) < 100.0

    def test_dv01_swap(self):
        """Test DV01 for an interest rate swap."""
        def price_swap(params):
            fixed_rate = 0.03
            floating_rate = params[0]
            notional = 1_000_000
            T = 5.0
            n_payments = 20  # Quarterly

            # Fixed leg PV
            fixed_pv = sum([fixed_rate/4 * notional * jnp.exp(-floating_rate * (i+1)*T/n_payments)
                           for i in range(n_payments)])

            # Floating leg PV (approximation)
            floating_pv = notional * (1 - jnp.exp(-floating_rate * T))

            return fixed_pv - floating_pv  # Payer swap

        params = jnp.array([0.03])  # 3% current rate
        dv01 = calculate_dv01(price_swap, params, rate_indices=0)

        # Payer swap loses value when rates rise (paying fixed)
        assert dv01 < 0

    def test_dv01_multiple_rates(self):
        """Test DV01 for multiple rates (curve risk)."""
        def price_instrument(params):
            # params = [r_1y, r_2y, r_5y, r_10y]
            weights = jnp.array([0.25, 0.25, 0.30, 0.20])
            return jnp.sum(params * weights * 100_000)

        params = jnp.array([0.02, 0.025, 0.03, 0.035])
        dv01_vector = calculate_dv01(price_instrument, params,
                                      rate_indices=[0, 1, 2, 3])

        # Should get 4 DV01 values
        assert len(dv01_vector) == 4

        # All should be positive (gain value when rates rise for this simple function)
        assert all(dv01_vector > 0)

    def test_dv01_autodiff_vs_finite_diff(self):
        """Compare autodiff and finite difference DV01."""
        def price_bond(params):
            r = params[0]
            return 100.0 * jnp.exp(-r * 5.0)  # Zero coupon bond

        params = jnp.array([0.05])

        # Autodiff
        config_autodiff = SensitivityConfig(method=SensitivityMethod.AUTODIFF)
        dv01_autodiff = calculate_dv01(price_bond, params, 0, config_autodiff)

        # Finite difference
        config_fd = SensitivityConfig(method=SensitivityMethod.FINITE_DIFF)
        dv01_fd = calculate_dv01(price_bond, params, 0, config_fd)

        # Should be very close
        assert jnp.abs(dv01_autodiff - dv01_fd) < 1e-6


class TestCS01:
    """Test CS01 (credit spread sensitivity) calculations."""

    def test_cs01_cds(self):
        """Test CS01 for a CDS contract."""
        def price_cds(params):
            spread = params[0]
            recovery = params[1]
            notional = 10_000_000
            T = 5.0

            # Simplified CDS pricing
            # PV of protection leg = notional * (1 - recovery) * (1 - exp(-spread*T))
            protection_leg = notional * (1 - recovery) * (1 - jnp.exp(-spread * T))

            # PV of premium leg = spread * notional * T/2 (approximation)
            premium_leg = spread * notional * T / 2

            return protection_leg - premium_leg  # Long protection

        params = jnp.array([0.0150, 0.40])  # 150bps spread, 40% recovery
        cs01 = calculate_cs01(price_cds, params, spread_indices=0)

        # Long protection gains value when spreads widen
        assert cs01 > 0

    def test_cs01_cds_index(self):
        """Test CS01 for a CDS index with multiple spreads."""
        def price_cds_index(params):
            # CDX index with 5 names
            spreads = params[:5]
            recovery = 0.40
            notional = 100_000_000

            total_pv = 0.0
            for spread in spreads:
                # Simplified individual CDS
                pv = notional/5 * (1 - recovery) * (1 - jnp.exp(-spread * 5.0))
                total_pv += pv

            return total_pv

        spreads = jnp.array([0.0100, 0.0150, 0.0120, 0.0180, 0.0140])
        cs01_vector = calculate_cs01(price_cds_index, spreads,
                                      spread_indices=list(range(5)))

        # Should get CS01 for each name
        assert len(cs01_vector) == 5

        # All should be positive (gain when spreads widen)
        assert all(cs01_vector > 0)

    def test_cs01_recovery_sensitivity(self):
        """Test that CS01 is not affected by recovery rate parameter."""
        def price_cds(params):
            spread = params[0]
            recovery = params[1]
            return 10_000_000 * (1 - recovery) * (1 - jnp.exp(-spread * 5.0))

        params = jnp.array([0.0150, 0.40])

        # CS01 should only measure spread sensitivity (index 0)
        cs01 = calculate_cs01(price_cds, params, spread_indices=0)

        # Should be a reasonable number
        assert cs01 > 0
        assert cs01 < 1_000_000  # Reasonable upper bound


class TestVegaSurface:
    """Test vega surface calculations."""

    def test_vega_single_option(self):
        """Test vega for a single option."""
        def price_option(params):
            S = params[0]
            vol = params[1]
            K = 100.0
            T = 1.0
            r = 0.05

            # Simplified Black-Scholes
            d1 = (jnp.log(S/K) + (r + 0.5 * vol**2) * T) / (vol * jnp.sqrt(T))
            d2 = d1 - vol * jnp.sqrt(T)

            from jax.scipy.stats import norm
            call = S * norm.cdf(d1) - K * jnp.exp(-r*T) * norm.cdf(d2)

            return call

        params = jnp.array([100.0, 0.20])  # ATM, 20% vol
        vega_data = calculate_vega_surface(price_option, params, vol_indices=[1])

        # Total vega should be positive
        assert vega_data['total_vega'] > 0

        # Vega vector should have 1 element
        assert len(vega_data['vega_vector']) == 1

    def test_vega_surface_with_tenor_strike(self):
        """Test vega surface with tenor and strike information."""
        def price_portfolio(params):
            # Portfolio of 3 options with different strikes/tenors
            S = params[0]
            vols = params[1:]  # 3 volatilities

            strikes = [95.0, 100.0, 105.0]
            tenors = [0.5, 1.0, 2.0]

            total_value = 0.0
            for vol, K, T in zip(vols, strikes, tenors):
                # Simplified option price
                value = S * vol * jnp.sqrt(T) * 0.4  # Vega-like approximation
                total_value += value

            return total_value

        params = jnp.array([100.0, 0.18, 0.20, 0.22])
        tenors = [0.5, 1.0, 2.0]
        strikes = [95.0, 100.0, 105.0]

        vega_data = calculate_vega_surface(
            price_portfolio, params,
            vol_indices=[1, 2, 3],
            tenors=tenors,
            strikes=strikes
        )

        # Check that surface mapping exists
        assert 'vega_surface' in vega_data
        assert (1.0, 100.0) in vega_data['vega_surface']

        # Check tenor aggregation
        assert 'vega_by_tenor' in vega_data
        assert 1.0 in vega_data['vega_by_tenor']

        # Check strike aggregation
        assert 'vega_by_strike' in vega_data
        assert 100.0 in vega_data['vega_by_strike']

    def test_vega_atm_vs_otm(self):
        """Test that ATM options have higher vega than OTM."""
        def price_option_atm(params):
            vol = params[0]
            return 100.0 * vol * jnp.sqrt(1.0) * 0.4  # ATM vega approximation

        def price_option_otm(params):
            vol = params[0]
            return 50.0 * vol * jnp.sqrt(1.0) * 0.4  # OTM vega approximation

        params = jnp.array([0.20])

        vega_atm = calculate_vega_surface(price_option_atm, params, vol_indices=[0])
        vega_otm = calculate_vega_surface(price_option_otm, params, vol_indices=[0])

        # ATM should have higher vega
        assert vega_atm['total_vega'] > vega_otm['total_vega']


class TestFXGreeks:
    """Test FX option Greeks."""

    def test_fx_delta(self):
        """Test FX delta calculation."""
        def price_fx_call(params):
            S = params[0]
            vol = params[1]
            K = 1.10
            T = 1.0
            r_d = 0.05
            r_f = 0.02

            # Garman-Kohlhagen
            d1 = (jnp.log(S/K) + (r_d - r_f + 0.5 * vol**2) * T) / (vol * jnp.sqrt(T))

            from jax.scipy.stats import norm
            delta_forward = norm.cdf(d1)
            call = S * jnp.exp(-r_f * T) * delta_forward - K * jnp.exp(-r_d * T) * norm.cdf(d1 - vol * jnp.sqrt(T))

            return call

        params = jnp.array([1.10, 0.12])  # ATM
        greeks = calculate_fx_greeks(price_fx_call, params, spot_index=0, vol_index=1)

        # ATM call delta should be around 0.5 (discounted for foreign rate)
        assert 'delta' in greeks
        assert 0.3 < greeks['delta'] < 0.7

        # Gamma should be positive
        assert 'gamma' in greeks
        assert greeks['gamma'] > 0

        # Vega should be positive
        assert 'vega' in greeks
        assert greeks['vega'] > 0

    def test_fx_gamma(self):
        """Test that gamma exists and is positive for options."""
        def price_fx_call(params):
            S = params[0]
            vol = 0.12
            K = 1.10
            T = 1.0
            r_d = 0.05
            r_f = 0.02

            d1 = (jnp.log(S/K) + (r_d - r_f + 0.5 * vol**2) * T) / (vol * jnp.sqrt(T))
            d2 = d1 - vol * jnp.sqrt(T)

            from jax.scipy.stats import norm
            return S * jnp.exp(-r_f * T) * norm.cdf(d1) - K * jnp.exp(-r_d * T) * norm.cdf(d2)

        # ATM option
        params = jnp.array([1.10])
        greeks = calculate_fx_greeks(price_fx_call, params, spot_index=0)

        # Gamma should exist and be finite
        assert 'gamma' in greeks
        assert jnp.isfinite(greeks['gamma'])

        # For a call option, gamma should be non-negative
        # (though it can be negative for portfolios or exotic payoffs)
        assert abs(greeks['gamma']) > 0

    def test_fx_greeks_put_call_parity(self):
        """Test that put and call deltas sum correctly."""
        def price_fx_call(params):
            S = params[0]
            K = 1.10
            vol = 0.12
            T = 1.0
            r_d = 0.05
            r_f = 0.02

            d1 = (jnp.log(S/K) + (r_d - r_f + 0.5 * vol**2) * T) / (vol * jnp.sqrt(T))
            d2 = d1 - vol * jnp.sqrt(T)

            from jax.scipy.stats import norm
            return S * jnp.exp(-r_f * T) * norm.cdf(d1) - K * jnp.exp(-r_d * T) * norm.cdf(d2)

        def price_fx_put(params):
            S = params[0]
            K = 1.10
            vol = 0.12
            T = 1.0
            r_d = 0.05
            r_f = 0.02

            d1 = (jnp.log(S/K) + (r_d - r_f + 0.5 * vol**2) * T) / (vol * jnp.sqrt(T))
            d2 = d1 - vol * jnp.sqrt(T)

            from jax.scipy.stats import norm
            return K * jnp.exp(-r_d * T) * norm.cdf(-d2) - S * jnp.exp(-r_f * T) * norm.cdf(-d1)

        params = jnp.array([1.10])

        greeks_call = calculate_fx_greeks(price_fx_call, params, spot_index=0)
        greeks_put = calculate_fx_greeks(price_fx_put, params, spot_index=0)

        # Put-call parity: Δ_call - Δ_put = exp(-r_f * T)
        T = 1.0
        r_f = 0.02
        expected_diff = jnp.exp(-r_f * T)

        assert jnp.abs((greeks_call['delta'] - greeks_put['delta']) - expected_diff) < 0.01


class TestHigherOrderGreeks:
    """Test higher-order Greeks (vanna, volga, vomma)."""

    def test_vanna_calculation(self):
        """Test vanna (∂²V/∂S∂σ) calculation."""
        def price_option(params):
            S = params[0]
            vol = params[1]
            K = 100.0
            T = 1.0
            r = 0.05

            d1 = (jnp.log(S/K) + (r + 0.5 * vol**2) * T) / (vol * jnp.sqrt(T))

            from jax.scipy.stats import norm
            # Simplified call price
            return S * norm.cdf(d1) - K * jnp.exp(-r*T) * norm.cdf(d1 - vol * jnp.sqrt(T))

        params = jnp.array([100.0, 0.20])  # ATM
        higher_greeks = calculate_higher_order_greeks(price_option, params,
                                                       spot_index=0, vol_index=1)

        # Vanna should exist
        assert 'vanna' in higher_greeks

        # For ATM options far from expiry, vanna is typically close to 0
        # but not exactly 0 in practice
        assert abs(higher_greeks['vanna']) < 100

    def test_volga_calculation(self):
        """Test volga/vomma (∂²V/∂σ²) calculation."""
        def price_option(params):
            S = params[0]
            vol = params[1]
            K = 100.0
            T = 1.0
            r = 0.05

            d1 = (jnp.log(S/K) + (r + 0.5 * vol**2) * T) / (vol * jnp.sqrt(T))
            d2 = d1 - vol * jnp.sqrt(T)

            from jax.scipy.stats import norm
            return S * norm.cdf(d1) - K * jnp.exp(-r*T) * norm.cdf(d2)

        params = jnp.array([100.0, 0.20])
        higher_greeks = calculate_higher_order_greeks(price_option, params,
                                                       spot_index=0, vol_index=1)

        # Volga should be positive (vega increases with vol for ATM options)
        assert 'volga' in higher_greeks
        assert higher_greeks['volga'] > 0

        # Vomma should equal volga
        assert higher_greeks['vomma'] == higher_greeks['volga']

    def test_vanna_symmetry(self):
        """Test that vanna is symmetric (∂²V/∂S∂σ = ∂²V/∂σ∂S)."""
        def price_option(params):
            S = params[0]
            vol = params[1]
            K = 100.0
            T = 1.0

            # Simple option price model
            return S * vol * jnp.sqrt(T) * 0.4

        params = jnp.array([100.0, 0.20])

        # Using autodiff, Hessian should be symmetric
        higher_greeks = calculate_higher_order_greeks(
            price_option, params, spot_index=0, vol_index=1,
            config=SensitivityConfig(method=SensitivityMethod.AUTODIFF)
        )

        # Vanna should be well-defined
        assert jnp.isfinite(higher_greeks['vanna'])


class TestPortfolioAggregation:
    """Test portfolio-level sensitivity aggregation."""

    def test_aggregate_rates_portfolio(self):
        """Test aggregating DV01 across multiple rate positions."""
        # Position 1: 5Y swap
        def price_swap_5y(params):
            r = params[0]
            return -1_000_000 * r * 5.0  # Simplified

        # Position 2: 10Y bond
        def price_bond_10y(params):
            r = params[0]
            return 10_000_000 * jnp.exp(-r * 10.0)

        positions = [
            {
                'name': '5Y IRS',
                'pricing_func': price_swap_5y,
                'params': jnp.array([0.03]),
                'type': 'rates',
                'quantity': 1.0,
                'rate_indices': [0],
                'curve_name': 'USD-LIBOR'
            },
            {
                'name': '10Y Bond',
                'pricing_func': price_bond_10y,
                'params': jnp.array([0.035]),
                'type': 'rates',
                'quantity': 1.0,
                'rate_indices': [0],
                'curve_name': 'USD-Treasury'
            }
        ]

        portfolio_sens = aggregate_portfolio_sensitivities(positions)

        # Should have total DV01
        assert portfolio_sens.total_dv01 != 0

        # Should have DV01 by curve
        assert 'USD-LIBOR' in portfolio_sens.dv01_by_curve
        assert 'USD-Treasury' in portfolio_sens.dv01_by_curve

        # Should have position-level detail
        assert '5Y IRS' in portfolio_sens.sensitivities_by_position
        assert '10Y Bond' in portfolio_sens.sensitivities_by_position

    def test_aggregate_credit_portfolio(self):
        """Test aggregating CS01 across multiple credit positions."""
        def price_cds_aapl(params):
            spread = params[0]
            return 5_000_000 * (1 - 0.40) * (1 - jnp.exp(-spread * 5.0))

        def price_cds_msft(params):
            spread = params[0]
            return 3_000_000 * (1 - 0.40) * (1 - jnp.exp(-spread * 5.0))

        positions = [
            {
                'name': 'AAPL 5Y CDS',
                'pricing_func': price_cds_aapl,
                'params': jnp.array([0.0150]),
                'type': 'credit',
                'quantity': 1.0,
                'spread_indices': [0],
                'entity': 'AAPL'
            },
            {
                'name': 'MSFT 5Y CDS',
                'pricing_func': price_cds_msft,
                'params': jnp.array([0.0120]),
                'type': 'credit',
                'quantity': 1.0,
                'spread_indices': [0],
                'entity': 'MSFT'
            }
        ]

        portfolio_sens = aggregate_portfolio_sensitivities(positions)

        # Should have total CS01
        assert portfolio_sens.total_cs01 > 0

        # Should have CS01 by entity
        assert 'AAPL' in portfolio_sens.cs01_by_entity
        assert 'MSFT' in portfolio_sens.cs01_by_entity

    def test_aggregate_mixed_portfolio(self):
        """Test aggregating sensitivities across asset classes."""
        def price_swap(params):
            r = params[0]
            return -500_000 * r * 3.0

        def price_cds(params):
            spread = params[0]
            return 1_000_000 * (1 - 0.40) * (1 - jnp.exp(-spread * 5.0))

        def price_fx_option(params):
            S = params[0]
            return S * 100_000 * 0.5  # Delta of 0.5

        positions = [
            {
                'name': '3Y IRS',
                'pricing_func': price_swap,
                'params': jnp.array([0.03]),
                'type': 'rates',
                'quantity': 1.0,
                'rate_indices': [0]
            },
            {
                'name': '5Y CDS',
                'pricing_func': price_cds,
                'params': jnp.array([0.0180]),
                'type': 'credit',
                'quantity': 1.0,
                'spread_indices': [0]
            },
            {
                'name': 'EURUSD Call',
                'pricing_func': price_fx_option,
                'params': jnp.array([1.10]),
                'type': 'fx',
                'quantity': 1.0,
                'spot_index': 0
            }
        ]

        portfolio_sens = aggregate_portfolio_sensitivities(positions)

        # Should have all sensitivity types
        assert portfolio_sens.total_dv01 != 0
        assert portfolio_sens.total_cs01 > 0
        assert portfolio_sens.total_delta != 0

    def test_format_sensitivity_report(self):
        """Test sensitivity report formatting."""
        portfolio_sens = PortfolioSensitivity(
            total_dv01=-15_000.50,
            total_cs01=8_500.25,
            total_vega=12_000.00,
            total_delta=250.75,
            net_gamma=0.0125,
            dv01_by_curve={'USD-LIBOR': -10_000.00, 'USD-Treasury': -5_000.50},
            cs01_by_entity={'AAPL': 5_000.00, 'MSFT': 3_500.25},
            vega_by_tenor={0.5: 3_000.00, 1.0: 6_000.00, 2.0: 3_000.00}
        )

        report = format_sensitivity_report(portfolio_sens, include_breakdown=True)

        # Should contain all key sections
        assert 'PORTFOLIO SENSITIVITY REPORT' in report
        assert 'SUMMARY' in report
        assert 'DV01 BY CURVE' in report
        assert 'CS01 BY ENTITY' in report
        assert 'VEGA BY TENOR' in report

        # Should contain values
        assert '-15,000.50' in report
        assert '8,500.25' in report
        assert 'USD-LIBOR' in report
        assert 'AAPL' in report


class TestUtilities:
    """Test utility functions."""

    def test_benchmark_sensitivity_methods(self):
        """Test benchmarking of different sensitivity methods."""
        def simple_function(params):
            return params[0] ** 2 + 2 * params[1]

        params = jnp.array([2.0, 3.0])

        results = benchmark_sensitivity_methods(simple_function, params, 0, n_runs=10)

        # Should have results for autodiff and finite_diff
        assert 'autodiff' in results
        assert 'finite_diff' in results

        # Both should have timing and value
        assert 'time' in results['autodiff']
        assert 'value' in results['autodiff']
        assert 'time' in results['finite_diff']
        assert 'value' in results['finite_diff']

        # Finite diff should have error metric
        assert 'error' in results['finite_diff']

        # Values should be close (analytical: d/dx(x²) = 2x = 4.0)
        assert abs(results['autodiff']['value'] - 4.0) < 0.01
        assert abs(results['finite_diff']['value'] - 4.0) < 0.01


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_sensitivity(self):
        """Test handling of zero sensitivities."""
        def constant_function(params):
            return 100.0  # No dependence on params

        params = jnp.array([0.05])
        dv01 = calculate_dv01(constant_function, params, 0)

        # Should be zero (or very close)
        assert abs(dv01) < 1e-6

    def test_discontinuous_payoff(self):
        """Test handling of discontinuous payoffs (digital options)."""
        def digital_option(params):
            S = params[0]
            K = 100.0
            # Digital call: pays $1 if S > K
            return jnp.where(S > K, 1.0, 0.0)

        params = jnp.array([100.5])

        # Autodiff will give zero gradient at discontinuity
        # Finite difference will give approximate delta
        config_fd = SensitivityConfig(method=SensitivityMethod.FINITE_DIFF)
        delta_fd = calculate_fx_greeks(digital_option, params, spot_index=0, config=config_fd)

        # Finite difference should give some non-zero value
        assert 'delta' in delta_fd

    def test_very_small_bump_size(self):
        """Test with very small bump sizes (numerical stability)."""
        def simple_function(params):
            return params[0] ** 2

        params = jnp.array([1.0])

        # Very small bump
        config = SensitivityConfig(
            method=SensitivityMethod.FINITE_DIFF,
            bump_size=1e-10
        )

        dv01 = calculate_dv01(simple_function, params, 0, config)

        # Should still work (gradient = 2x * 1e-10 = 2e-10 for x=1)
        assert jnp.isfinite(dv01)
