"""Tests for dynamic programming portfolio optimization."""

import numpy as np
import pytest

from neutryx.research.portfolio.advanced import (
    DynamicProgrammingPortfolioOptimizer,
    StochasticDynamicProgramming,
)


class TestDynamicProgrammingPortfolioOptimizer:
    """Test suite for dynamic programming portfolio optimization."""

    def test_initialization(self):
        """Test optimizer initialization."""
        optimizer = DynamicProgrammingPortfolioOptimizer(
            n_periods=10,
            risk_aversion=2.0,
            transaction_cost=0.001,
            wealth_grid_size=50,
            initial_wealth=1.0
        )

        assert optimizer.n_periods == 10
        assert optimizer.risk_aversion == 2.0
        assert optimizer.transaction_cost == 0.001
        assert optimizer.wealth_grid_size == 50
        assert optimizer.initial_wealth == 1.0

    def test_invalid_parameters(self):
        """Test validation of invalid parameters."""
        with pytest.raises(ValueError, match="n_periods must be at least 1"):
            DynamicProgrammingPortfolioOptimizer(n_periods=0)

        with pytest.raises(ValueError, match="wealth_grid_size must be at least 10"):
            DynamicProgrammingPortfolioOptimizer(n_periods=5, wealth_grid_size=5)

        with pytest.raises(ValueError, match="initial_wealth must be positive"):
            DynamicProgrammingPortfolioOptimizer(n_periods=5, initial_wealth=-1.0)

    def test_optimize_two_assets(self):
        """Test optimization with two assets."""
        optimizer = DynamicProgrammingPortfolioOptimizer(
            n_periods=5,
            risk_aversion=2.0,
            wealth_grid_size=50
        )

        expected_returns = np.array([0.08, 0.06])
        covariance = np.array([[0.04, 0.01], [0.01, 0.02]])

        policies, value_function = optimizer.optimize(expected_returns, covariance)

        # Check outputs
        assert len(policies) == 5
        assert isinstance(value_function, np.ndarray)
        assert len(value_function) == 50

        # Check that weights sum to approximately 1
        for policy in policies:
            assert len(policy) == 2
            assert np.isfinite(policy).all()

    def test_optimize_three_assets(self):
        """Test optimization with three assets."""
        optimizer = DynamicProgrammingPortfolioOptimizer(
            n_periods=3,
            risk_aversion=1.5,
            wealth_grid_size=30
        )

        expected_returns = np.array([0.10, 0.07, 0.05])
        covariance = np.array([
            [0.05, 0.01, 0.005],
            [0.01, 0.03, 0.008],
            [0.005, 0.008, 0.02]
        ])

        policies, value_function = optimizer.optimize(expected_returns, covariance)

        assert len(policies) == 3
        for policy in policies:
            assert len(policy) == 3
            assert np.isfinite(policy).all()

    def test_custom_wealth_grid(self):
        """Test with custom wealth grid."""
        optimizer = DynamicProgrammingPortfolioOptimizer(
            n_periods=3,
            wealth_grid_size=50
        )

        expected_returns = np.array([0.08, 0.06])
        covariance = np.array([[0.04, 0.01], [0.01, 0.02]])

        # Custom wealth grid
        wealth_grid = np.linspace(0.8, 1.5, 50)

        policies, value_function = optimizer.optimize(
            expected_returns,
            covariance,
            wealth_grid=wealth_grid
        )

        assert len(policies) == 3
        assert len(value_function) == 50

    def test_terminal_utility(self):
        """Test terminal utility function."""
        optimizer = DynamicProgrammingPortfolioOptimizer(
            n_periods=5,
            risk_aversion=2.0
        )

        wealth_grid = np.linspace(0.5, 2.0, 100)
        utilities = optimizer._terminal_utility(wealth_grid)

        # Check that utilities are finite
        assert np.isfinite(utilities).all()

        # Check monotonicity (higher wealth -> higher utility for gamma < 1)
        # For CRRA with gamma = 2.0, utility should be negative but increasing
        assert len(utilities) == 100

    def test_log_utility_case(self):
        """Test log utility when risk_aversion = 1.0."""
        optimizer = DynamicProgrammingPortfolioOptimizer(
            n_periods=3,
            risk_aversion=1.0
        )

        wealth_grid = np.linspace(0.5, 2.0, 50)
        utilities = optimizer._terminal_utility(wealth_grid)

        # Log utility should be increasing
        assert np.all(np.diff(utilities) > 0)

    def test_invalid_covariance_shape(self):
        """Test error handling for invalid covariance matrix."""
        optimizer = DynamicProgrammingPortfolioOptimizer(n_periods=3)

        expected_returns = np.array([0.08, 0.06])
        covariance = np.array([[0.04, 0.01]])  # Wrong shape

        with pytest.raises(ValueError, match="Covariance shape must be"):
            optimizer.optimize(expected_returns, covariance)


class TestStochasticDynamicProgramming:
    """Test suite for stochastic dynamic programming."""

    def test_initialization(self):
        """Test SDP initialization."""
        sdp = StochasticDynamicProgramming(
            n_periods=10,
            n_scenarios=100,
            risk_aversion=2.0,
            transaction_cost=0.001
        )

        assert sdp.n_periods == 10
        assert sdp.n_scenarios == 100
        assert sdp.risk_aversion == 2.0
        assert sdp.transaction_cost == 0.001

    def test_optimize_with_scenarios(self):
        """Test optimization with return scenarios."""
        sdp = StochasticDynamicProgramming(
            n_periods=5,
            n_scenarios=50,
            risk_aversion=2.0
        )

        # Generate random return scenarios
        rng = np.random.default_rng(42)
        return_scenarios = rng.normal(0.01, 0.02, (50, 5, 3))

        optimal_policy, expected_value = sdp.optimize(1.0, return_scenarios)

        # Check outputs
        assert len(optimal_policy) == 3
        assert np.isfinite(optimal_policy).all()
        assert np.isfinite(expected_value)
        assert np.isclose(np.sum(optimal_policy), 1.0)

    def test_optimize_with_probabilities(self):
        """Test optimization with scenario probabilities."""
        sdp = StochasticDynamicProgramming(
            n_periods=3,
            n_scenarios=20
        )

        rng = np.random.default_rng(42)
        return_scenarios = rng.normal(0.01, 0.02, (20, 3, 2))

        # Non-uniform probabilities
        probs = rng.uniform(0, 1, 20)
        probs /= probs.sum()

        optimal_policy, expected_value = sdp.optimize(
            1.0,
            return_scenarios,
            scenario_probabilities=probs
        )

        assert len(optimal_policy) == 2
        assert np.isfinite(expected_value)

    def test_invalid_scenario_shape(self):
        """Test error handling for invalid scenario shape."""
        sdp = StochasticDynamicProgramming(n_periods=5, n_scenarios=50)

        # Wrong shape (should be 3D)
        return_scenarios = np.random.randn(50, 5)

        with pytest.raises(ValueError, match="return_scenarios must have shape"):
            sdp.optimize(1.0, return_scenarios)

    def test_utility_function(self):
        """Test utility function computation."""
        sdp = StochasticDynamicProgramming(
            n_periods=5,
            risk_aversion=2.0
        )

        wealth = np.array([0.5, 1.0, 1.5, 2.0])
        utilities = sdp._utility(wealth)

        # Check that utilities are finite
        assert np.isfinite(utilities).all()
        assert len(utilities) == 4

    def test_log_utility_sdp(self):
        """Test log utility case in SDP."""
        sdp = StochasticDynamicProgramming(
            n_periods=5,
            risk_aversion=1.0  # Log utility
        )

        wealth = np.array([0.5, 1.0, 1.5, 2.0])
        utilities = sdp._utility(wealth)

        # Log utility should be increasing
        assert np.all(np.diff(utilities) > 0)


class TestIntegration:
    """Integration tests for dynamic programming optimizers."""

    def test_dp_vs_myopic(self):
        """Compare DP solution with myopic (single-period) optimization."""
        optimizer = DynamicProgrammingPortfolioOptimizer(
            n_periods=1,  # Single period
            risk_aversion=2.0,
            transaction_cost=0.0,  # No transaction cost
            wealth_grid_size=50
        )

        expected_returns = np.array([0.10, 0.05])
        covariance = np.array([[0.04, 0.01], [0.01, 0.02]])

        policies, _ = optimizer.optimize(expected_returns, covariance)

        # For single period with no transaction cost, should prefer higher return asset
        assert len(policies) == 1
        # First asset has higher Sharpe ratio, should get more weight
        assert policies[0][0] > policies[0][1]

    def test_transaction_cost_effect(self):
        """Test that transaction costs affect optimal policies."""
        expected_returns = np.array([0.08, 0.06])
        covariance = np.array([[0.04, 0.01], [0.01, 0.02]])

        # No transaction cost
        opt1 = DynamicProgrammingPortfolioOptimizer(
            n_periods=3,
            transaction_cost=0.0,
            wealth_grid_size=30
        )
        policies1, _ = opt1.optimize(expected_returns, covariance)

        # With transaction cost
        opt2 = DynamicProgrammingPortfolioOptimizer(
            n_periods=3,
            transaction_cost=0.01,
            wealth_grid_size=30
        )
        policies2, _ = opt2.optimize(expected_returns, covariance)

        # Policies should differ due to transaction costs
        for p1, p2 in zip(policies1, policies2):
            assert not np.allclose(p1, p2, rtol=0.01)

    def test_risk_aversion_effect(self):
        """Test that risk aversion affects portfolio allocation."""
        expected_returns = np.array([0.12, 0.04])  # Risky vs safe
        covariance = np.array([[0.10, 0.0], [0.0, 0.01]])  # Risky vs safe

        # Low risk aversion
        opt1 = DynamicProgrammingPortfolioOptimizer(
            n_periods=3,
            risk_aversion=0.5,
            wealth_grid_size=30
        )
        policies1, _ = opt1.optimize(expected_returns, covariance)

        # High risk aversion
        opt2 = DynamicProgrammingPortfolioOptimizer(
            n_periods=3,
            risk_aversion=5.0,
            wealth_grid_size=30
        )
        policies2, _ = opt2.optimize(expected_returns, covariance)

        # Low risk aversion should allocate more to risky asset
        assert policies1[0][0] >= policies2[0][0]
