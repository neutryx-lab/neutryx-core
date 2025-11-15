"""Tests for performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neutryx.research.performance import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_calmar_ratio,
    calculate_information_ratio,
    calculate_var,
    calculate_cvar,
    calculate_comprehensive_metrics,
)


@pytest.fixture
def sample_returns():
    """Create sample return series."""
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=252, freq="D")
    returns = pd.Series(
        np.random.normal(0.001, 0.02, 252),
        index=dates,
    )
    return returns


@pytest.fixture
def losing_returns():
    """Create return series with losses."""
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=252, freq="D")
    returns = pd.Series(
        np.random.normal(-0.001, 0.02, 252),
        index=dates,
    )
    return returns


class TestSharpeRatio:
    """Test Sharpe ratio calculation."""

    def test_positive_returns(self, sample_returns):
        """Test Sharpe with positive returns."""
        sharpe = calculate_sharpe_ratio(sample_returns)
        assert isinstance(sharpe, float)
        assert sharpe > 0  # Positive expected return should give positive Sharpe

    def test_with_risk_free_rate(self, sample_returns):
        """Test Sharpe with risk-free rate."""
        sharpe_rf = calculate_sharpe_ratio(sample_returns, risk_free_rate=0.02)
        sharpe_no_rf = calculate_sharpe_ratio(sample_returns, risk_free_rate=0.0)

        # With risk-free rate should be lower
        assert sharpe_rf < sharpe_no_rf

    def test_zero_volatility(self):
        """Test Sharpe with near-zero volatility."""
        returns = pd.Series([0.001] * 100)
        sharpe = calculate_sharpe_ratio(returns)
        # With constant returns, std might be near-zero, giving very large Sharpe
        # or the function returns 0 if it detects zero std
        assert sharpe == 0.0 or np.isinf(sharpe) or sharpe > 1000

    def test_empty_returns(self):
        """Test Sharpe with empty returns."""
        returns = pd.Series([])
        sharpe = calculate_sharpe_ratio(returns)
        assert sharpe == 0.0


class TestSortinoRatio:
    """Test Sortino ratio calculation."""

    def test_sortino_calculation(self, sample_returns):
        """Test Sortino ratio."""
        sortino = calculate_sortino_ratio(sample_returns)
        assert isinstance(sortino, float)

    def test_sortino_higher_than_sharpe(self, sample_returns):
        """Sortino should typically be higher than Sharpe."""
        sharpe = calculate_sharpe_ratio(sample_returns)
        sortino = calculate_sortino_ratio(sample_returns)

        # Sortino penalizes only downside, so typically higher
        assert sortino >= sharpe * 0.8  # Allow some tolerance


class TestMaxDrawdown:
    """Test maximum drawdown calculation."""

    def test_max_drawdown(self, sample_returns):
        """Test max drawdown calculation."""
        max_dd, duration, recovery = calculate_max_drawdown(sample_returns)

        assert isinstance(max_dd, float)
        assert max_dd <= 0.0  # Drawdown is negative
        assert isinstance(duration, int)
        assert duration >= 0

    def test_no_drawdown(self):
        """Test with no drawdowns (only positive returns)."""
        returns = pd.Series([0.01] * 100)
        max_dd, duration, recovery = calculate_max_drawdown(returns)

        assert max_dd == 0.0
        assert duration == 0

    def test_single_drawdown(self):
        """Test with single drawdown."""
        returns = pd.Series([0.01] * 50 + [-0.02] * 10 + [0.01] * 50)
        max_dd, duration, recovery = calculate_max_drawdown(returns)

        assert max_dd < 0.0
        assert duration >= 10  # At least the drawdown period


class TestCalmarRatio:
    """Test Calmar ratio calculation."""

    def test_calmar_calculation(self, sample_returns):
        """Test Calmar ratio."""
        calmar = calculate_calmar_ratio(sample_returns)
        assert isinstance(calmar, float)

    def test_zero_drawdown(self):
        """Test Calmar with zero drawdown."""
        returns = pd.Series([0.01] * 100)
        calmar = calculate_calmar_ratio(returns)
        assert calmar == 0.0  # Zero drawdown gives undefined Calmar


class TestInformationRatio:
    """Test information ratio calculation."""

    def test_information_ratio(self, sample_returns):
        """Test information ratio vs benchmark."""
        # Create benchmark returns (slightly lower)
        benchmark = sample_returns * 0.9

        ir = calculate_information_ratio(sample_returns, benchmark)

        assert isinstance(ir, float)
        assert ir > 0  # Should outperform benchmark

    def test_equal_returns(self, sample_returns):
        """Test IR when strategy matches benchmark."""
        ir = calculate_information_ratio(sample_returns, sample_returns)
        assert abs(ir) < 0.01  # Should be close to zero


class TestVaRandCVaR:
    """Test Value at Risk and Conditional VaR."""

    def test_var_calculation(self, sample_returns):
        """Test VaR calculation."""
        var_95 = calculate_var(sample_returns, confidence_level=0.95)

        assert isinstance(var_95, float)
        assert var_95 >= 0  # VaR is reported as positive loss

    def test_cvar_higher_than_var(self, sample_returns):
        """CVaR should be higher than VaR."""
        var_95 = calculate_var(sample_returns, 0.95)
        cvar_95 = calculate_cvar(sample_returns, 0.95)

        assert cvar_95 >= var_95

    def test_var_confidence_levels(self, sample_returns):
        """VaR should increase with confidence level."""
        var_90 = calculate_var(sample_returns, 0.90)
        var_95 = calculate_var(sample_returns, 0.95)
        var_99 = calculate_var(sample_returns, 0.99)

        assert var_99 >= var_95 >= var_90


class TestComprehensiveMetrics:
    """Test comprehensive metrics calculation."""

    def test_comprehensive_metrics(self, sample_returns):
        """Test calculating all metrics together."""
        metrics = calculate_comprehensive_metrics(sample_returns)

        # Check all fields exist
        assert hasattr(metrics, "total_return")
        assert hasattr(metrics, "annualized_return")
        assert hasattr(metrics, "sharpe_ratio")
        assert hasattr(metrics, "sortino_ratio")
        assert hasattr(metrics, "max_drawdown")
        assert hasattr(metrics, "var_95")
        assert hasattr(metrics, "cvar_95")

        # Check all are floats
        assert isinstance(metrics.total_return, float)
        assert isinstance(metrics.sharpe_ratio, float)
        assert isinstance(metrics.max_drawdown, float)

    def test_metrics_with_benchmark(self, sample_returns):
        """Test metrics with benchmark."""
        benchmark = sample_returns * 0.9

        metrics = calculate_comprehensive_metrics(
            sample_returns,
            benchmark_returns=benchmark,
        )

        assert metrics.information_ratio != 0.0

    def test_metrics_to_dict(self, sample_returns):
        """Test converting metrics to dictionary."""
        metrics = calculate_comprehensive_metrics(sample_returns)
        metrics_dict = metrics.to_dict()

        assert isinstance(metrics_dict, dict)
        assert "Sharpe Ratio" in metrics_dict
        assert "Max Drawdown" in metrics_dict


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
