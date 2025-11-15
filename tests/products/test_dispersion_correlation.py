"""Comprehensive tests for dispersion and correlation trading products."""
from __future__ import annotations

import jax.numpy as jnp
import pytest

from neutryx.products.dispersion_correlation import (
    IndexVarianceSwap,
    SingleNameVarianceSwap,
    CorrelationSwap,
    BasicDispersionStrategy,
    ImpliedCorrelationStrategy,
    RealizedCorrelationDispersion,
)


# ============================================================================
# Index Variance Swap Tests
# ============================================================================


class TestIndexVarianceSwap:
    """Test cases for index variance swaps."""

    def test_index_var_swap_payoff(self):
        """Test basic index variance swap payoff calculation."""
        var_swap = IndexVarianceSwap(
            T=0.25,
            strike_variance=0.04,  # 20% vol
            notional_per_point=50_000.0,
            annualization_factor=252.0
        )

        # Create a simple path with known volatility
        # Use 64 observations (roughly 3 months of daily data)
        path = jnp.array([100.0] * 64)
        # Add some returns
        for i in range(1, 64):
            path = path.at[i].set(path[i-1] * jnp.exp(0.01 * ((-1) ** i)))

        payoff = var_swap.payoff_path(path)
        # Payoff should be notional * (realized_var - strike_var)
        assert isinstance(float(payoff), float), "Payoff should be a number"

    def test_index_var_swap_zero_vol(self):
        """Test variance swap with zero volatility path."""
        var_swap = IndexVarianceSwap(
            T=0.25,
            strike_variance=0.04,
            notional_per_point=50_000.0
        )

        # Constant path (zero volatility)
        path = jnp.ones(100) * 100.0

        payoff = var_swap.payoff_path(path)
        # Realized var is 0, strike is 0.04
        # Payoff = 50,000 * (0 - 0.04) = -2,000
        expected = 50_000.0 * (0.0 - 0.04)
        assert abs(payoff - expected) < 1e-4, "Zero vol path should give negative payoff"

    def test_index_var_swap_with_cap(self):
        """Test variance swap with cap on realized variance."""
        var_swap = IndexVarianceSwap(
            T=0.25,
            strike_variance=0.04,
            notional_per_point=50_000.0,
            cap=0.06  # Cap at 24.5% vol
        )

        # Create a high volatility path
        path = jnp.linspace(100, 120, 100)  # Strong trend
        payoff = var_swap.payoff_path(path)

        # Even if realized variance > 0.06, it should be capped
        # Max payoff = 50,000 * (0.06 - 0.04) = 1,000
        assert payoff <= 50_000.0 * (0.06 - 0.04) + 1e-2, "Cap should limit payoff"

    def test_vega_notional_calculation(self):
        """Test vega notional calculation."""
        var_swap = IndexVarianceSwap(
            T=0.25,
            strike_variance=0.04,  # 20% vol
            notional_per_point=50_000.0
        )

        vega_not = var_swap.vega_notional()
        # Vega notional = variance_notional / (2 * sqrt(K_var))
        expected = 50_000.0 / (2.0 * jnp.sqrt(0.04))
        assert abs(vega_not - expected) < 1e-4, "Vega notional calculation incorrect"


# ============================================================================
# Single-Name Variance Swap Tests
# ============================================================================


class TestSingleNameVarianceSwap:
    """Test cases for single-name variance swaps."""

    def test_single_name_var_swap_payoff(self):
        """Test single-name variance swap payoff."""
        var_swap = SingleNameVarianceSwap(
            T=0.5,
            strike_variance=0.06,  # 24.5% vol
            notional_per_point=10_000.0,
            ticker="AAPL"
        )

        # Simulate stock path
        path = jnp.array([150.0] * 100)
        for i in range(1, 100):
            path = path.at[i].set(path[i-1] * jnp.exp(0.02 * jnp.sin(i/10)))

        payoff = var_swap.payoff_path(path)
        assert isinstance(float(payoff), float), "Payoff should be a number"

    def test_single_name_ticker_attribute(self):
        """Test that ticker attribute is stored correctly."""
        var_swap = SingleNameVarianceSwap(
            T=0.5,
            strike_variance=0.06,
            notional_per_point=10_000.0,
            ticker="TSLA"
        )

        assert var_swap.ticker == "TSLA", "Ticker should be stored correctly"

    def test_single_name_with_floor(self):
        """Test variance swap with floor."""
        var_swap = SingleNameVarianceSwap(
            T=0.5,
            strike_variance=0.06,
            notional_per_point=10_000.0,
            floor=0.04  # Floor at 20% vol
        )

        # Very low volatility path
        path = jnp.ones(100) * 100.0

        payoff = var_swap.payoff_path(path)
        # Realized var close to 0, but floored at 0.04
        # Min payoff = 10,000 * (0.04 - 0.06) = -200
        expected_min = 10_000.0 * (0.04 - 0.06)
        assert payoff >= expected_min - 1e-2, "Floor should limit downside"


# ============================================================================
# Correlation Swap Tests
# ============================================================================


class TestCorrelationSwap:
    """Test cases for correlation swaps."""

    def test_correlation_swap_specific_pair(self):
        """Test correlation swap on specific pair."""
        corr_swap = CorrelationSwap(
            T=1.0,
            strike_correlation=0.50,
            notional_per_point=100_000.0,
            num_assets=2,
            correlation_type='specific',
            asset_1_idx=0,
            asset_2_idx=1
        )

        # Create two correlated paths
        n_steps = 100
        path1 = jnp.array([100.0] * n_steps)
        path2 = jnp.array([100.0] * n_steps)

        # Highly correlated moves
        for i in range(1, n_steps):
            move = 0.01 * ((-1) ** i)
            path1 = path1.at[i].set(path1[i-1] * jnp.exp(move))
            path2 = path2.at[i].set(path2[i-1] * jnp.exp(move * 0.9))  # 90% correlation

        paths = jnp.array([path1, path2])
        payoff = corr_swap.payoff_path(paths)

        # Realized correlation should be high (close to 0.9)
        # P&L ≈ 100,000 * (0.9 - 0.50) * 100 = 4,000,000
        assert payoff > 0, "Correlated paths should give positive payoff"

    def test_correlation_swap_average(self):
        """Test average pairwise correlation swap."""
        corr_swap = CorrelationSwap(
            T=1.0,
            strike_correlation=0.30,
            notional_per_point=100_000.0,
            num_assets=3,
            correlation_type='average'
        )

        # Create three paths with varying correlation
        n_steps = 100
        path1 = jnp.linspace(100, 110, n_steps)
        path2 = jnp.linspace(100, 105, n_steps)
        path3 = jnp.linspace(110, 100, n_steps)  # Negative trend

        paths = jnp.array([path1, path2, path3])
        payoff = corr_swap.payoff_path(paths)

        # Average correlation will be calculated
        assert isinstance(float(payoff), float), "Payoff should be a number"

    def test_correlation_swap_single_asset(self):
        """Test correlation swap with single asset returns zero."""
        corr_swap = CorrelationSwap(
            T=1.0,
            strike_correlation=0.50,
            notional_per_point=100_000.0
        )

        # Single path
        path = jnp.linspace(100, 110, 100)
        payoff = corr_swap.payoff_path(path)

        assert payoff == 0.0, "Single asset should have no correlation payoff"

    def test_correlation_swap_perfect_correlation(self):
        """Test correlation swap with perfectly correlated assets."""
        corr_swap = CorrelationSwap(
            T=1.0,
            strike_correlation=0.50,
            notional_per_point=100_000.0,
            num_assets=2,
            correlation_type='specific'
        )

        # Two paths with perfect correlation (identical returns, not identical prices)
        n_steps = 100
        path1 = jnp.array([100.0] * n_steps)
        for i in range(1, n_steps):
            path1 = path1.at[i].set(path1[i-1] * jnp.exp(0.01 * ((-1) ** i)))

        # Path2 has identical returns to path1
        path2 = jnp.array([100.0] * n_steps)
        for i in range(1, n_steps):
            path2 = path2.at[i].set(path2[i-1] * jnp.exp(0.01 * ((-1) ** i)))

        paths = jnp.array([path1, path2])
        payoff = corr_swap.payoff_path(paths)

        # Realized correlation ≈ 1.0
        # P&L ≈ 100,000 * (1.0 - 0.50) * 100 = 5,000,000
        assert payoff > 4_000_000, "Perfect correlation should give large positive payoff"


# ============================================================================
# Basic Dispersion Strategy Tests
# ============================================================================


class TestBasicDispersionStrategy:
    """Test cases for basic dispersion trading."""

    def test_basic_dispersion_payoff(self):
        """Test basic dispersion strategy payoff."""
        dispersion = BasicDispersionStrategy(
            T=0.25,
            index_strike_var=0.04,
            stock_strike_var=jnp.array([0.06] * 5),
            index_notional=250_000.0,
            stock_notionals=jnp.array([50_000.0] * 5),
            num_stocks=5
        )

        # Create paths: 5 stocks + 1 index
        n_steps = 64
        stock_paths = []
        for i in range(5):
            path = jnp.array([100.0] * n_steps)
            for j in range(1, n_steps):
                path = path.at[j].set(path[j-1] * jnp.exp(0.015 * ((-1) ** j)))
            stock_paths.append(path)

        # Index has lower vol (due to diversification)
        index_path = jnp.array([100.0] * n_steps)
        for j in range(1, n_steps):
            index_path = index_path.at[j].set(index_path[j-1] * jnp.exp(0.008 * ((-1) ** j)))

        paths = jnp.array(stock_paths + [index_path])
        payoff = dispersion.payoff_path(paths)

        # Dispersion should profit when stocks have higher vol than index
        assert isinstance(float(payoff), float), "Payoff should be a number"

    def test_dispersion_equal_weights(self):
        """Test dispersion with equal weights."""
        dispersion = BasicDispersionStrategy(
            T=0.25,
            index_strike_var=0.04,
            stock_strike_var=jnp.array([0.06, 0.05, 0.07, 0.055, 0.065]),
            index_notional=500_000.0,
            stock_notionals=jnp.array([100_000.0] * 5),
            num_stocks=5
        )

        # Equal weights should be set by default
        assert jnp.allclose(
            dispersion.index_weights, jnp.array([0.2, 0.2, 0.2, 0.2, 0.2])
        ), "Equal weights should be default"

    def test_dispersion_single_dimensional_path(self):
        """Test dispersion with 1D path returns zero."""
        dispersion = BasicDispersionStrategy(
            T=0.25,
            index_strike_var=0.04,
            stock_strike_var=jnp.array([0.06] * 5),
            index_notional=250_000.0,
            stock_notionals=jnp.array([50_000.0] * 5)
        )

        # 1D path
        path = jnp.linspace(100, 110, 100)
        payoff = dispersion.payoff_path(path)

        assert payoff == 0.0, "1D path should return zero payoff"


# ============================================================================
# Implied Correlation Strategy Tests
# ============================================================================


class TestImpliedCorrelationStrategy:
    """Test cases for implied correlation strategies."""

    def test_implied_corr_strategy_payoff(self):
        """Test implied correlation strategy payoff."""
        strat = ImpliedCorrelationStrategy(
            T=0.5,
            implied_correlation=0.55,
            index_var_strike=0.04,
            avg_stock_var_strike=0.06,
            notional_per_corr_point=200_000.0,
            num_stocks=10
        )

        # Create 10 stock paths + 1 index
        n_steps = 100
        stock_paths = []
        for i in range(10):
            path = jnp.array([100.0] * n_steps)
            for j in range(1, n_steps):
                path = path.at[j].set(path[j-1] * jnp.exp(0.01 * jnp.sin((i+j)/5)))
            stock_paths.append(path)

        # Index with moderate volatility
        index_path = jnp.array([100.0] * n_steps)
        for j in range(1, n_steps):
            index_path = index_path.at[j].set(index_path[j-1] * jnp.exp(0.008 * ((-1) ** j)))

        paths = jnp.array(stock_paths + [index_path])
        payoff = strat.payoff_path(paths)

        # P&L depends on realized vs implied correlation
        assert isinstance(float(payoff), float), "Payoff should be a number"

    def test_implied_corr_single_dimensional(self):
        """Test implied correlation with 1D path."""
        strat = ImpliedCorrelationStrategy(
            T=0.5,
            implied_correlation=0.55,
            index_var_strike=0.04,
            avg_stock_var_strike=0.06,
            notional_per_corr_point=200_000.0
        )

        path = jnp.linspace(100, 110, 100)
        payoff = strat.payoff_path(path)

        assert payoff == 0.0, "1D path should return zero"


# ============================================================================
# Realized Correlation Dispersion Tests
# ============================================================================


class TestRealizedCorrelationDispersion:
    """Test cases for realized correlation dispersion."""

    def test_realized_corr_dispersion_payoff(self):
        """Test realized correlation dispersion payoff."""
        strat = RealizedCorrelationDispersion(
            T=1.0,
            target_correlation=0.50,
            index_notional=1_000_000.0,
            stock_notionals=jnp.array([100_000.0] * 10),
            correlation_notional=500_000.0,
            num_stocks=10
        )

        # Create 10 stock paths + 1 index
        n_steps = 100
        stock_paths = []
        for i in range(10):
            path = jnp.array([100.0] * n_steps)
            for j in range(1, n_steps):
                # Add some correlation structure
                base_move = 0.01 * ((-1) ** j)
                idio_move = 0.005 * jnp.sin((i+j)/3)
                path = path.at[j].set(path[j-1] * jnp.exp(base_move + idio_move))
            stock_paths.append(path)

        # Index
        index_path = jnp.array([100.0] * n_steps)
        for j in range(1, n_steps):
            index_path = index_path.at[j].set(index_path[j-1] * jnp.exp(0.01 * ((-1) ** j)))

        paths = jnp.array(stock_paths + [index_path])
        payoff = strat.payoff_path(paths)

        # Payoff based on realized vs target correlation
        assert isinstance(float(payoff), float), "Payoff should be a number"

    def test_realized_corr_high_correlation(self):
        """Test realized correlation with high actual correlation."""
        strat = RealizedCorrelationDispersion(
            T=1.0,
            target_correlation=0.30,
            index_notional=1_000_000.0,
            stock_notionals=jnp.array([100_000.0] * 5),
            correlation_notional=500_000.0,
            num_stocks=5
        )

        # Create highly correlated paths
        n_steps = 100
        stock_paths = []
        base = jnp.array([100.0] * n_steps)
        for j in range(1, n_steps):
            base = base.at[j].set(base[j-1] * jnp.exp(0.01 * ((-1) ** j)))

        for i in range(5):
            # Add tiny idiosyncratic moves
            path = base * (1.0 + 0.001 * i)
            stock_paths.append(path)

        # Index
        index_path = base.copy()

        paths = jnp.array(stock_paths + [index_path])
        payoff = strat.payoff_path(paths)

        # High realized correlation (>0.9), target is 0.30
        # Should be large positive payoff
        assert payoff > 0, "High correlation should give positive payoff vs low target"


# ============================================================================
# Integration Tests
# ============================================================================


class TestDispersionCorrelationIntegration:
    """Integration tests for dispersion and correlation products."""

    def test_combined_strategy(self):
        """Test combining variance swaps and correlation swaps."""
        # Index variance swap
        index_var = IndexVarianceSwap(
            T=0.25,
            strike_variance=0.04,
            notional_per_point=50_000.0
        )

        # Stock variance swaps
        stock_var = SingleNameVarianceSwap(
            T=0.25,
            strike_variance=0.06,
            notional_per_point=10_000.0,
            ticker="STOCK1"
        )

        # Correlation swap
        corr_swap = CorrelationSwap(
            T=0.25,
            strike_correlation=0.50,
            notional_per_point=100_000.0,
            num_assets=2
        )

        # All products should be instantiated correctly
        assert index_var.T == 0.25
        assert stock_var.T == 0.25
        assert corr_swap.T == 0.25

    def test_vega_notional_consistency(self):
        """Test vega notional consistency across products."""
        index_var = IndexVarianceSwap(
            T=0.25,
            strike_variance=0.04,
            notional_per_point=50_000.0
        )

        stock_var = SingleNameVarianceSwap(
            T=0.25,
            strike_variance=0.04,  # Same strike
            notional_per_point=50_000.0  # Same notional
        )

        # Vega notionals should be equal
        index_vega = index_var.vega_notional()
        stock_vega = stock_var.vega_notional()

        assert abs(index_vega - stock_vega) < 1e-6, "Vega notionals should match"


# ============================================================================
# Edge Cases and Validation
# ============================================================================


class TestEdgeCases:
    """Test edge cases and validation."""

    def test_negative_notional(self):
        """Test that negative notionals flip the sign correctly."""
        var_swap = IndexVarianceSwap(
            T=0.25,
            strike_variance=0.04,
            notional_per_point=-50_000.0  # Short variance
        )

        path = jnp.ones(100) * 100.0
        payoff = var_swap.payoff_path(path)

        # With zero realized variance and negative notional
        # Payoff = -50,000 * (0 - 0.04) = +2,000 (profit from shorting)
        expected = -50_000.0 * (0.0 - 0.04)
        assert abs(payoff - expected) < 1e-4, "Negative notional should work"

    def test_zero_strike_variance(self):
        """Test variance swap with zero strike."""
        var_swap = IndexVarianceSwap(
            T=0.25,
            strike_variance=0.0,
            notional_per_point=50_000.0
        )

        path = jnp.array([100.0] * 10)
        for i in range(1, 10):
            path = path.at[i].set(path[i-1] * 1.01)

        payoff = var_swap.payoff_path(path)
        # Any positive realized variance should give positive payoff
        assert payoff > 0, "Zero strike with positive realized var should profit"

    def test_very_short_path(self):
        """Test with very short paths."""
        var_swap = IndexVarianceSwap(
            T=0.25,
            strike_variance=0.04,
            notional_per_point=50_000.0
        )

        # Just 2 points (1 return)
        path = jnp.array([100.0, 101.0])
        payoff = var_swap.payoff_path(path)

        # Should not crash
        assert isinstance(float(payoff), float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
