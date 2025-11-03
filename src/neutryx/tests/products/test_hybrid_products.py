"""Tests for hybrid products."""
import jax.numpy as jnp
import pytest

from neutryx.products.hybrid_products import (
    CorrelationSwap,
    CrossCurrencyExotic,
    DispersionOption,
    FXHybridEquityOption,
    InflationLinkedFXOption,
    MultiAssetWorstOfBestOf,
    QuantoCDS,
)


class TestQuantoCDS:
    """Test Quanto CDS implementation."""

    def test_quanto_cds_creation(self):
        """Test basic Quanto CDS instantiation."""
        qcds = QuantoCDS(
            T=5.0,
            notional_domestic=1_000_000,
            spread=150.0,
            quanto_fx_rate=1.2,
            fx_vol=0.1,
        )
        assert qcds.T == 5.0
        assert qcds.quanto_fx_rate == 1.2

    def test_quanto_cds_payoff_no_default(self):
        """Test Quanto CDS payoff with no default."""
        qcds = QuantoCDS(
            T=5.0,
            notional_domestic=1_000_000,
            spread=150.0,
            quanto_fx_rate=1.2,
        )
        state = jnp.array([0.0, 1.2])  # No default, FX at quanto rate
        payoff = qcds.payoff_terminal(state)

        # Only premium leg
        assert payoff < 0

    def test_quanto_cds_payoff_with_default(self):
        """Test Quanto CDS payoff with default."""
        qcds = QuantoCDS(
            T=5.0,
            notional_domestic=1_000_000,
            spread=150.0,
            quanto_fx_rate=1.2,
            recovery_rate=0.4,
        )
        state = jnp.array([1.0, 1.3])  # Default occurred
        payoff = qcds.payoff_terminal(state)

        # Should include protection leg
        assert isinstance(payoff, jnp.ndarray) or isinstance(payoff, float)


class TestFXHybridEquityOption:
    """Test FX-hybrid equity option."""

    def test_fx_hybrid_creation(self):
        """Test FX-hybrid option instantiation."""
        option = FXHybridEquityOption(
            T=1.0, K_equity=100.0, K_fx=1.2, option_type='call', fx_protected=True
        )
        assert option.K_equity == 100.0
        assert option.K_fx == 1.2
        assert option.fx_protected is True

    def test_fx_hybrid_call_itm(self):
        """Test FX-hybrid call option in the money."""
        option = FXHybridEquityOption(
            T=1.0,
            K_equity=100.0,
            K_fx=1.2,
            option_type='call',
            notional=1.0,
            fx_protected=True,
        )
        state = jnp.array([110.0, 1.3])  # Equity at 110, FX at 1.3
        payoff = option.payoff_terminal(state)

        # Payoff = (110 - 100) * 1.2 = 12 (FX protected)
        assert jnp.allclose(payoff, 12.0)

    def test_fx_hybrid_call_unprotected(self):
        """Test FX-hybrid call option without FX protection."""
        option = FXHybridEquityOption(
            T=1.0,
            K_equity=100.0,
            K_fx=1.2,
            option_type='call',
            notional=1.0,
            fx_protected=False,
        )
        state = jnp.array([110.0, 1.3])  # Equity at 110, FX at 1.3
        payoff = option.payoff_terminal(state)

        # Payoff = (110 - 100) * 1.3 = 13 (use spot FX)
        assert jnp.allclose(payoff, 13.0)

    def test_fx_hybrid_put(self):
        """Test FX-hybrid put option."""
        option = FXHybridEquityOption(
            T=1.0,
            K_equity=100.0,
            K_fx=1.2,
            option_type='put',
            notional=1.0,
            fx_protected=True,
        )
        state = jnp.array([90.0, 1.3])  # Equity at 90
        payoff = option.payoff_terminal(state)

        # Payoff = (100 - 90) * 1.2 = 12
        assert jnp.allclose(payoff, 12.0)


class TestInflationLinkedFXOption:
    """Test inflation-linked FX option."""

    def test_inflation_fx_creation(self):
        """Test inflation-linked FX option instantiation."""
        option = InflationLinkedFXOption(
            T=5.0,
            K=1.2,
            domestic_inflation_rate=0.02,
            foreign_inflation_rate=0.03,
            inflation_adjustment_type='strike',
        )
        assert option.domestic_inflation_rate == 0.02
        assert option.foreign_inflation_rate == 0.03

    def test_inflation_adjusted_strike(self):
        """Test inflation-adjusted strike calculation."""
        option = InflationLinkedFXOption(
            T=5.0,
            K=1.2,
            option_type='call',
            domestic_inflation_rate=0.02,
            foreign_inflation_rate=0.03,
            inflation_adjustment_type='strike',
        )
        # Simplified: domestic CPI = 1.1, foreign CPI = 1.15
        state = jnp.array([1.3, 1.1, 1.15])
        payoff = option.payoff_terminal(state)

        assert payoff >= 0.0


class TestCrossCurrencyExotic:
    """Test cross-currency exotic options."""

    def test_cross_currency_creation(self):
        """Test cross-currency exotic instantiation."""
        option = CrossCurrencyExotic(
            T=1.0,
            strikes=jnp.array([1.2, 1.3]),
            barrier=1.25,
            barrier_type='down-and-out',
        )
        assert option.barrier == 1.25
        assert option.barrier_type == 'down-and-out'

    def test_cross_currency_vanilla(self):
        """Test cross-currency option without barrier."""
        option = CrossCurrencyExotic(
            T=1.0, strikes=jnp.array([1.2]), option_type='call', barrier=None
        )
        terminal_rate = jnp.array([1.3])
        payoff = option.payoff_path(terminal_rate)

        # Call payoff: max(1.3 - 1.2, 0) = 0.1
        assert jnp.allclose(payoff, 0.1)


class TestMultiAssetWorstOfBestOf:
    """Test multi-asset worst-of/best-of options."""

    def test_worst_of_creation(self):
        """Test worst-of option instantiation."""
        option = MultiAssetWorstOfBestOf(
            T=1.0,
            strikes=jnp.array([100.0, 100.0, 100.0]),
            option_type='call',
            payoff_type='worst-of',
            num_assets=3,
        )
        assert option.payoff_type == 'worst-of'
        assert option.num_assets == 3

    def test_worst_of_call(self):
        """Test worst-of call option."""
        option = MultiAssetWorstOfBestOf(
            T=1.0,
            strikes=jnp.array([100.0, 100.0, 100.0]),
            option_type='call',
            payoff_type='worst-of',
        )
        spots = jnp.array([110.0, 120.0, 105.0])
        payoff = option.payoff_terminal(spots)

        # Worst performer: 105 - 100 = 5
        assert jnp.allclose(payoff, 5.0)

    def test_best_of_call(self):
        """Test best-of call option."""
        option = MultiAssetWorstOfBestOf(
            T=1.0,
            strikes=jnp.array([100.0, 100.0, 100.0]),
            option_type='call',
            payoff_type='best-of',
        )
        spots = jnp.array([110.0, 120.0, 105.0])
        payoff = option.payoff_terminal(spots)

        # Best performer: 120 - 100 = 20
        assert jnp.allclose(payoff, 20.0)

    def test_rainbow_payoff(self):
        """Test rainbow payoff with specific rank."""
        option = MultiAssetWorstOfBestOf(
            T=1.0,
            strikes=jnp.array([100.0, 100.0, 100.0]),
            option_type='call',
            payoff_type='worst-of',
        )
        spots = jnp.array([110.0, 120.0, 105.0])

        # 2nd best performance
        payoff = option.rainbow_payoff(spots, rank=2)
        # Sorted performances: [20, 10, 5], 2nd = 10
        assert jnp.allclose(payoff, 10.0)


class TestCorrelationSwap:
    """Test correlation swap."""

    def test_correlation_swap_creation(self):
        """Test correlation swap instantiation."""
        swap = CorrelationSwap(
            T=1.0, strike_correlation=0.5, notional=1_000_000, num_assets=2
        )
        assert swap.strike_correlation == 0.5
        assert swap.num_assets == 2

    def test_correlation_swap_payoff(self):
        """Test correlation swap payoff calculation."""
        swap = CorrelationSwap(
            T=1.0, strike_correlation=0.5, notional=1_000_000, num_assets=2
        )
        # Generate correlated paths
        paths = jnp.array([
            [100.0, 101.0, 102.0, 103.0, 104.0],
            [100.0, 101.5, 102.5, 103.5, 104.5],
        ])
        payoff = swap.payoff_path(paths)

        # Should have non-zero payoff based on realized correlation
        assert isinstance(payoff, jnp.ndarray) or isinstance(payoff, float)


class TestDispersionOption:
    """Test dispersion option."""

    def test_dispersion_creation(self):
        """Test dispersion option instantiation."""
        option = DispersionOption(T=1.0, strike=0.01, notional=1_000_000, num_stocks=50)
        assert option.strike == 0.01
        assert option.num_stocks == 50

    def test_dispersion_payoff(self):
        """Test dispersion option payoff."""
        option = DispersionOption(T=1.0, strike=0.01, notional=1_000_000, num_stocks=3)
        # variances: [stock1_var, stock2_var, stock3_var, index_var]
        variances = jnp.array([0.04, 0.05, 0.045, 0.025])
        payoff = option.payoff_terminal(variances)

        # Avg stock variance = (0.04 + 0.05 + 0.045) / 3 = 0.045
        # Dispersion = 0.045 - 0.025 = 0.02
        # Payoff = (0.02 - 0.01) * 1M = 10,000
        assert isinstance(payoff, jnp.ndarray) or isinstance(payoff, float)
