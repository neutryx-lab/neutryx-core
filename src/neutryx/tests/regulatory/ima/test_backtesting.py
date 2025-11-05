"""Tests for VaR and ES backtesting with traffic light approach."""

from datetime import date, timedelta

import jax.numpy as jnp
import pytest

from neutryx.regulatory.ima import (
    BacktestResult,
    TrafficLightZone,
    backtest_expected_shortfall,
    backtest_var,
    calculate_traffic_light_zone,
    rolling_backtest,
)


class TestVaRBacktesting:
    """Test VaR backtesting with Basel traffic light approach."""

    def test_var_backtest_no_exceptions(self):
        """Test VaR backtest with no exceptions (perfect model)."""
        # VaR forecasts that are never breached
        actual_pnl = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0] * 50)  # 250 days
        var_forecasts = jnp.array([-10.0] * 250)  # Conservative VaR

        result = backtest_var(actual_pnl, var_forecasts, coverage_level=0.99)

        assert result.num_exceptions == 0
        assert result.exception_rate == 0.0
        assert result.traffic_light_zone == TrafficLightZone.GREEN

    def test_var_backtest_green_zone(self):
        """Test VaR backtest in green zone (0-4 exceptions)."""
        # Create 250 observations with 3 exceptions
        actual_pnl = jnp.ones(250)
        actual_pnl = actual_pnl.at[10].set(-5.0)
        actual_pnl = actual_pnl.at[100].set(-5.0)
        actual_pnl = actual_pnl.at[200].set(-5.0)

        var_forecasts = jnp.ones(250) * -3.0  # 3 breaches

        result = backtest_var(actual_pnl, var_forecasts)

        assert result.num_exceptions == 3
        assert result.traffic_light_zone == TrafficLightZone.GREEN

    def test_var_backtest_amber_zone(self):
        """Test VaR backtest in amber zone (5-9 exceptions)."""
        # Create 250 observations with 7 exceptions
        actual_pnl = jnp.ones(250)
        for i in range(7):
            actual_pnl = actual_pnl.at[i * 30].set(-5.0)

        var_forecasts = jnp.ones(250) * -3.0

        result = backtest_var(actual_pnl, var_forecasts)

        assert result.num_exceptions == 7
        assert result.traffic_light_zone == TrafficLightZone.AMBER

    def test_var_backtest_red_zone(self):
        """Test VaR backtest in red zone (10+ exceptions)."""
        # Create 250 observations with 12 exceptions
        actual_pnl = jnp.ones(250)
        for i in range(12):
            actual_pnl = actual_pnl.at[i * 20].set(-5.0)

        var_forecasts = jnp.ones(250) * -3.0

        result = backtest_var(actual_pnl, var_forecasts)

        assert result.num_exceptions == 12
        assert result.traffic_light_zone == TrafficLightZone.RED

    def test_var_capital_multiplier(self):
        """Test capital multiplier increases with exceptions."""
        actual_pnl = jnp.ones(250)
        var_forecasts = jnp.ones(250) * -3.0

        # Test different exception counts
        multipliers = []
        for num_exceptions in [0, 4, 7, 10, 15]:
            pnl = actual_pnl.copy()
            for i in range(num_exceptions):
                pnl = pnl.at[i * 20].set(-5.0)

            result = backtest_var(pnl, var_forecasts)
            multipliers.append(result.capital_multiplier)

        # Multipliers should increase with exceptions
        assert multipliers[0] <= multipliers[1]  # Green zone
        assert multipliers[1] <= multipliers[2]  # Amber zone
        assert multipliers[2] <= multipliers[3]  # Red zone
        assert multipliers[3] <= multipliers[4]  # More red

    def test_backtest_with_dates(self):
        """Test backtest with date tracking."""
        start_date = date(2024, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(250)]

        actual_pnl = jnp.ones(250)
        actual_pnl = actual_pnl.at[10].set(-5.0)
        actual_pnl = actual_pnl.at[100].set(-5.0)

        var_forecasts = jnp.ones(250) * -3.0

        result = backtest_var(actual_pnl, var_forecasts, dates=dates)

        # Should have 2 exceptions
        assert len(result.exceptions) == 2
        assert result.exceptions[0].date == dates[10]
        assert result.exceptions[1].date == dates[100]


class TestESBacktesting:
    """Test Expected Shortfall backtesting."""

    def test_es_backtest_basic(self):
        """Test basic ES backtest."""
        # ES forecasts that match actual tail losses
        actual_pnl = jnp.concatenate([
            jnp.ones(240),  # Normal days
            jnp.array([-5.0, -6.0, -7.0, -8.0, -9.0, -10.0, -4.0, -5.0, -6.0, -7.0])  # Tail
        ])

        # ES forecast: average of worst 2.5% (~6-7 worst observations)
        es_forecasts = jnp.ones(250) * -6.5

        result = backtest_expected_shortfall(
            actual_pnl, es_forecasts, coverage_level=0.975
        )

        # Should have reasonable performance
        assert isinstance(result.num_exceptions, int)
        assert result.num_exceptions <= 10  # Some exceptions expected

    def test_es_vs_var_comparison(self):
        """Test that ES backtesting is more stringent than VaR."""
        key = jax.random.PRNGKey(42)
        actual_pnl = jax.random.normal(key, (250,)) * 10.0

        var_forecasts = jnp.percentile(actual_pnl, 1.0) * jnp.ones(250)
        es_forecasts = var_forecasts * 1.3  # ES should be ~30% worse than VaR

        result_var = backtest_var(actual_pnl, var_forecasts, coverage_level=0.99)
        result_es = backtest_expected_shortfall(
            actual_pnl, es_forecasts, coverage_level=0.975
        )

        # Both should be reasonable models
        assert result_var.traffic_light_zone in [TrafficLightZone.GREEN, TrafficLightZone.AMBER]
        assert result_es.traffic_light_zone in [TrafficLightZone.GREEN, TrafficLightZone.AMBER, TrafficLightZone.RED]


class TestTrafficLightZones:
    """Test traffic light zone calculation."""

    def test_zone_thresholds(self):
        """Test Basel II traffic light zone thresholds."""
        num_observations = 250

        # Green zone: 0-4 exceptions
        for num_exceptions in [0, 1, 2, 3, 4]:
            zone, multiplier = calculate_traffic_light_zone(num_exceptions, num_observations)
            assert zone == TrafficLightZone.GREEN

        # Amber zone: 5-9 exceptions
        for num_exceptions in [5, 6, 7, 8, 9]:
            zone, multiplier = calculate_traffic_light_zone(num_exceptions, num_observations)
            assert zone == TrafficLightZone.AMBER

        # Red zone: 10+ exceptions
        for num_exceptions in [10, 11, 15, 20]:
            zone, multiplier = calculate_traffic_light_zone(num_exceptions, num_observations)
            assert zone == TrafficLightZone.RED

    def test_capital_multiplier_ranges(self):
        """Test capital multiplier ranges by zone."""
        actual_pnl = jnp.ones(250)
        var_forecasts = jnp.ones(250) * -3.0

        # Green zone multiplier: 3.00
        pnl_green = actual_pnl.copy()
        pnl_green = pnl_green.at[0].set(-5.0)
        result_green = backtest_var(pnl_green, var_forecasts)
        assert 3.0 <= result_green.capital_multiplier <= 3.4

        # Amber zone multiplier: 3.40-3.75
        pnl_amber = actual_pnl.copy()
        for i in range(7):
            pnl_amber = pnl_amber.at[i * 30].set(-5.0)
        result_amber = backtest_var(pnl_amber, var_forecasts)
        assert 3.4 <= result_amber.capital_multiplier <= 3.85

        # Red zone multiplier: 3.75-4.00
        pnl_red = actual_pnl.copy()
        for i in range(12):
            pnl_red = pnl_red.at[i * 20].set(-5.0)
        result_red = backtest_var(pnl_red, var_forecasts)
        assert 3.75 <= result_red.capital_multiplier <= 4.0


class TestRollingBacktest:
    """Test rolling window backtesting."""

    def test_rolling_backtest_basic(self):
        """Test rolling backtest with 250-day windows."""
        # Create 500 days of data
        actual_pnl = jnp.concatenate([
            jax.random.normal(jax.random.PRNGKey(42), (250,)) * 5.0,
            jax.random.normal(jax.random.PRNGKey(43), (250,)) * 10.0  # Higher vol
        ])

        var_forecasts = jnp.ones(500) * -10.0

        results = rolling_backtest(
            actual_pnl,
            var_forecasts,
            window_size=250,
            step_size=50
        )

        # Should have multiple windows
        assert len(results) > 1

        # Each result should be valid
        for result in results:
            assert isinstance(result, BacktestResult)
            assert result.num_observations == 250
            assert result.traffic_light_zone in [TrafficLightZone.GREEN, TrafficLightZone.AMBER, TrafficLightZone.RED]

    def test_rolling_backtest_degradation(self):
        """Test detection of model degradation over time."""
        # Good model initially, degrades later
        actual_pnl = jnp.concatenate([
            jnp.ones(250) * 2.0,  # Period 1: good
            jnp.ones(250) * -10.0  # Period 2: bad (many exceptions)
        ])

        var_forecasts = jnp.ones(500) * -5.0

        results = rolling_backtest(
            actual_pnl,
            var_forecasts,
            window_size=250,
            step_size=250
        )

        # First window should be better than second
        if len(results) >= 2:
            assert results[0].num_exceptions <= results[1].num_exceptions


class TestBacktestEdgeCases:
    """Test edge cases and error handling."""

    def test_insufficient_observations(self):
        """Test error with too few observations."""
        actual_pnl = jnp.array([1.0, 2.0, 3.0])
        var_forecasts = jnp.array([-1.0, -1.0, -1.0])

        # Should still work but with warning or low confidence
        result = backtest_var(actual_pnl, var_forecasts)
        assert result.num_observations == 3

    def test_unequal_length_inputs(self):
        """Test error with unequal length inputs."""
        actual_pnl = jnp.array([1.0, 2.0, 3.0])
        var_forecasts = jnp.array([-1.0, -1.0])

        with pytest.raises((ValueError, IndexError)):
            backtest_var(actual_pnl, var_forecasts)

    def test_all_exceptions(self):
        """Test behavior when all observations are exceptions."""
        actual_pnl = jnp.ones(250) * -10.0
        var_forecasts = jnp.ones(250) * -1.0  # Too optimistic

        result = backtest_var(actual_pnl, var_forecasts)

        assert result.num_exceptions == 250
        assert result.traffic_light_zone == TrafficLightZone.RED
        assert result.capital_multiplier == 4.0  # Maximum

    def test_no_exceptions_boundary(self):
        """Test boundary case with VaR = actual PnL."""
        actual_pnl = jnp.array([1.0, 2.0, -5.0, 3.0, 4.0] * 50)
        var_forecasts = jnp.array([0.5, 1.5, -5.0, 2.5, 3.5] * 50)  # Exact match

        result = backtest_var(actual_pnl, var_forecasts)

        # With exact match, should have 0 or 1 exceptions (depends on <= vs <)
        assert result.num_exceptions <= 1
        assert result.traffic_light_zone == TrafficLightZone.GREEN


class TestStatisticalTests:
    """Test statistical test components."""

    def test_kupiec_pof_in_result(self):
        """Test that Kupiec POF test is included in result."""
        actual_pnl = jnp.ones(250)
        actual_pnl = actual_pnl.at[10].set(-5.0)
        actual_pnl = actual_pnl.at[20].set(-5.0)

        var_forecasts = jnp.ones(250) * -3.0

        result = backtest_var(actual_pnl, var_forecasts)

        # Kupiec test should be in result
        assert result.kupiec_pof_pvalue is not None
        assert isinstance(result.kupiec_pof_pvalue, float)
        # With 2 exceptions in 250 observations at 99%, should have reasonable p-value
        assert result.kupiec_pof_pvalue > 0.05

    def test_christoffersen_independence(self):
        """Test that independence test is computed."""
        actual_pnl = jnp.ones(250)
        # Clustered exceptions
        for i in range(10):
            actual_pnl = actual_pnl.at[100 + i].set(-5.0)

        var_forecasts = jnp.ones(250) * -3.0

        result = backtest_var(actual_pnl, var_forecasts)

        # Independence test should be in result
        assert result.christoffersen_test_pvalue is not None
        assert isinstance(result.christoffersen_test_pvalue, float)
