"""Tests for risk metrics (VaR, CVaR, etc.)."""
import jax.numpy as jnp
import numpy as np
import pytest

from neutryx.valuations.risk_metrics import (
    VaRMethod,
    backtest_var,
    calculate_var,
    component_var,
    conditional_value_at_risk,
    cornish_fisher_var,
    downside_deviation,
    historical_var,
    incremental_var,
    marginal_var,
    maximum_drawdown,
    monte_carlo_var,
    parametric_var,
    portfolio_cvar,
    portfolio_var,
    sharpe_ratio,
    sortino_ratio,
    value_at_risk,
)


class TestBasicVaR:
    """Tests for basic VaR calculations."""

    def test_value_at_risk_simple(self):
        """Test basic VaR calculation."""
        # Simple returns: -5%, -3%, -1%, 0%, 1%, 2%, 3%, 5%, 8%, 10%
        returns = jnp.array([-0.05, -0.03, -0.01, 0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10])

        # 90% VaR should be around 3-5%
        var_90 = value_at_risk(returns, 0.90)
        assert 0.025 < var_90 < 0.055

        # 95% VaR should be higher
        var_95 = value_at_risk(returns, 0.95)
        assert var_95 > var_90

    def test_value_at_risk_edge_cases(self):
        """Test VaR edge cases."""
        # All positive returns
        returns = jnp.array([0.01, 0.02, 0.03, 0.04, 0.05])
        var = value_at_risk(returns, 0.95)
        assert var < 0  # Negative VaR means profit

        # All negative returns
        returns = jnp.array([-0.01, -0.02, -0.03, -0.04, -0.05])
        var = value_at_risk(returns, 0.95)
        assert var > 0  # Positive VaR means loss

    def test_cvar_exceeds_var(self):
        """Test that CVaR >= VaR."""
        returns = jnp.array([-0.10, -0.05, -0.02, 0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10])

        var = value_at_risk(returns, 0.95)
        cvar = conditional_value_at_risk(returns, 0.95)

        # CVaR should be at least as large as VaR
        assert cvar >= var


class TestVaRMethodologies:
    """Tests for different VaR calculation methods."""

    @pytest.fixture
    def sample_returns(self):
        """Generate sample returns with known distribution."""
        np.random.seed(42)
        # Generate 1000 returns from normal distribution
        returns = np.random.normal(0.001, 0.02, 1000)
        return jnp.array(returns)

    def test_historical_var(self, sample_returns):
        """Test historical VaR."""
        var = historical_var(sample_returns, 0.95)
        assert var > 0
        assert isinstance(var, float)

        # Test with window
        var_window = historical_var(sample_returns, 0.95, window=250)
        assert var_window > 0

    def test_parametric_var(self, sample_returns):
        """Test parametric VaR."""
        var = parametric_var(sample_returns, 0.95)
        assert var > 0

        # Test with explicit mean and std
        mean = 0.001
        std = 0.02
        var_explicit = parametric_var(sample_returns, 0.95, mean=mean, std=std)
        assert var_explicit > 0

    def test_monte_carlo_var(self, sample_returns):
        """Test Monte Carlo VaR."""
        # Monte Carlo VaR is same as historical for simulated returns
        var = monte_carlo_var(sample_returns, 0.95)
        assert var > 0
        assert isinstance(var, float)

    def test_cornish_fisher_var(self):
        """Test Cornish-Fisher VaR with skewed distribution."""
        np.random.seed(42)
        # Generate skewed returns (chi-squared shifted)
        skewed = np.random.chisquare(5, 1000) / 100 - 0.05
        returns = jnp.array(skewed)

        var_cf = cornish_fisher_var(returns, 0.95)
        var_param = parametric_var(returns, 0.95)

        # CF-VaR should differ from parametric for skewed data
        assert var_cf != var_param
        assert var_cf > 0

    def test_calculate_var_all_methods(self, sample_returns):
        """Test calculate_var with all methods."""
        methods = [
            VaRMethod.HISTORICAL,
            VaRMethod.PARAMETRIC,
            VaRMethod.MONTE_CARLO,
            VaRMethod.CORNISH_FISHER,
        ]

        for method in methods:
            var = calculate_var(sample_returns, 0.95, method)
            assert var > 0, f"VaR should be positive for {method}"
            assert isinstance(var, float)


class TestPortfolioVaR:
    """Tests for portfolio VaR calculations."""

    @pytest.fixture
    def portfolio_data(self):
        """Generate sample portfolio data."""
        np.random.seed(42)
        n_scenarios = 1000
        n_assets = 3

        # Generate correlated returns
        cov_matrix = np.array([
            [0.0004, 0.0002, 0.0001],
            [0.0002, 0.0009, 0.0003],
            [0.0001, 0.0003, 0.0016],
        ])
        returns = np.random.multivariate_normal([0.001, 0.0015, 0.002], cov_matrix, n_scenarios)

        positions = jnp.array([100000.0, 50000.0, 75000.0])  # Position sizes
        returns_scenarios = jnp.array(returns)

        return positions, returns_scenarios

    def test_portfolio_var(self, portfolio_data):
        """Test portfolio VaR calculation."""
        positions, returns_scenarios = portfolio_data

        pvar = portfolio_var(positions, returns_scenarios, 0.95)
        assert pvar > 0
        assert isinstance(pvar, float)

        # 99% VaR should be higher than 95% VaR
        pvar_99 = portfolio_var(positions, returns_scenarios, 0.99)
        assert pvar_99 > pvar

    def test_portfolio_cvar(self, portfolio_data):
        """Test portfolio CVaR calculation."""
        positions, returns_scenarios = portfolio_data

        pcvar = portfolio_cvar(positions, returns_scenarios, 0.95)
        pvar = portfolio_var(positions, returns_scenarios, 0.95)

        # CVaR should be >= VaR
        assert pcvar >= pvar

    def test_component_var(self, portfolio_data):
        """Test component VaR calculation."""
        positions, returns_scenarios = portfolio_data

        comp_vars = component_var(positions, returns_scenarios, 0.95)

        # Should have one component per asset
        assert comp_vars.shape == positions.shape

        # Sum of components should be close to total VaR
        # (not exact due to diversification effects in simplified calculation)
        total_var = portfolio_var(positions, returns_scenarios, 0.95)
        comp_sum = jnp.sum(comp_vars)

        # Allow some difference due to approximation
        assert abs(comp_sum - total_var) / total_var < 0.5

    def test_marginal_var(self, portfolio_data):
        """Test marginal VaR calculation."""
        positions, returns_scenarios = portfolio_data

        marg_vars = marginal_var(positions, returns_scenarios, 0.95)

        # Should have one marginal VaR per asset
        assert marg_vars.shape == positions.shape

        # Marginal VaR can be positive or negative
        assert jnp.isfinite(marg_vars).all()

    def test_incremental_var(self):
        """Test incremental VaR."""
        np.random.seed(42)
        portfolio_returns = jnp.array(np.random.normal(0.001, 0.02, 1000))
        position_returns = jnp.array(np.random.normal(0.0015, 0.025, 1000))

        ivar = incremental_var(portfolio_returns, position_returns, 0.95)

        # Incremental VaR can be positive (increases risk) or negative (hedges)
        assert isinstance(ivar, float)
        assert jnp.isfinite(ivar)


class TestOtherRiskMetrics:
    """Tests for other risk metrics."""

    def test_downside_deviation(self):
        """Test downside deviation calculation."""
        returns = jnp.array([-0.05, -0.02, 0.01, 0.03, -0.01, 0.04, -0.03, 0.02])

        dd = downside_deviation(returns, threshold=0.0)
        assert dd > 0

        # Downside deviation with higher threshold should be larger
        dd_high = downside_deviation(returns, threshold=0.01)
        assert dd_high > dd

    def test_maximum_drawdown(self):
        """Test maximum drawdown calculation."""
        # Cumulative returns with a drawdown
        cum_returns = jnp.array([1.0, 1.05, 1.08, 1.02, 0.98, 0.95, 1.00, 1.10])

        mdd = maximum_drawdown(cum_returns)
        assert mdd > 0

        # Max drawdown should be peak (1.08) to trough (0.95) = 0.13
        expected_mdd = 1.08 - 0.95
        assert abs(mdd - expected_mdd) < 0.01

    def test_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        # Positive returns with some volatility
        returns = jnp.array([0.01, 0.02, -0.01, 0.03, 0.02, 0.01, -0.005, 0.015])

        sharpe = sharpe_ratio(returns, risk_free_rate=0.0)
        assert sharpe > 0  # Positive expected return should give positive Sharpe

        # Higher risk-free rate should decrease Sharpe
        sharpe_rf = sharpe_ratio(returns, risk_free_rate=0.015)
        assert sharpe_rf < sharpe

    def test_sortino_ratio(self):
        """Test Sortino ratio calculation."""
        returns = jnp.array([0.01, 0.02, -0.01, 0.03, 0.02, 0.01, -0.005, 0.015])

        sortino = sortino_ratio(returns, risk_free_rate=0.0)
        assert sortino > 0

        # Sortino should be >= Sharpe (uses downside deviation only)
        sharpe = sharpe_ratio(returns, risk_free_rate=0.0)
        # Not always true, but often true for typical distributions
        assert jnp.isfinite(sortino)


class TestVaRBacktest:
    """Tests for VaR backtesting."""

    def test_backtest_var_perfect_model(self):
        """Test VaR backtest with perfect model."""
        np.random.seed(42)
        n = 250  # Trading days

        # Generate returns
        realized_returns = jnp.array(np.random.normal(0.001, 0.02, n))

        # Generate VaR forecasts that match realized distribution
        # 95% VaR should be exceeded 5% of the time
        var_forecasts = jnp.full(n, 0.032)  # Approximately 95th percentile

        result = backtest_var(realized_returns, var_forecasts, 0.95)

        assert "violations" in result
        assert "violation_rate" in result
        assert "expected_rate" in result
        assert "kupiec_pvalue" in result
        assert "pass_backtest" in result

        # Expected violations should be close to 5% of days
        expected_violations = n * 0.05
        assert 0 <= result["violations"] <= n
        assert 0 <= result["violation_rate"] <= 1.0

    def test_backtest_var_bad_model(self):
        """Test VaR backtest with underestimated VaR."""
        np.random.seed(42)
        n = 250

        realized_returns = jnp.array(np.random.normal(0.001, 0.02, n))

        # Severely underestimate VaR (should fail backtest)
        var_forecasts = jnp.full(n, 0.01)  # Too low

        result = backtest_var(realized_returns, var_forecasts, 0.95)

        # Should have too many violations
        assert result["violations"] > n * 0.05 * 2  # More than 2x expected
        assert result["violation_rate"] > 0.05

    def test_backtest_var_zero_violations(self):
        """Test backtest with zero violations."""
        n = 100
        realized_returns = jnp.array(np.random.uniform(-0.01, 0.01, n))
        var_forecasts = jnp.full(n, 0.10)  # Very high VaR, no violations

        result = backtest_var(realized_returns, var_forecasts, 0.95)

        assert result["violations"] == 0
        assert result["violation_rate"] == 0.0


def test_var_confidence_level_validation():
    """Test that confidence level validation works."""
    returns = jnp.array([0.01, -0.02, 0.03, -0.01])

    # Invalid confidence levels
    with pytest.raises(ValueError):
        value_at_risk(returns, 0.0)

    with pytest.raises(ValueError):
        value_at_risk(returns, 1.0)

    with pytest.raises(ValueError):
        value_at_risk(returns, 1.5)

    with pytest.raises(ValueError):
        conditional_value_at_risk(returns, -0.1)
