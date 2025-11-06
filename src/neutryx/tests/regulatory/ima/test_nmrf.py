"""Tests for Non-Modellable Risk Factors (NMRF) identification and treatment."""

from datetime import date, timedelta

import jax
import jax.numpy as jnp
import pytest

from neutryx.regulatory.ima import (
    ModellabilityStatus,
    ModellabilityTester,
    NMRFCapitalCalculator,
    StressScenarioCalibrator,
    calculate_nmrf_capital_total,
    identify_nmrfs,
)


class TestModellabilityTester:
    """Test modellability testing for risk factors."""

    def test_modellable_risk_factor(self):
        """Test risk factor that passes modellability test."""
        # 24+ observations over 12+ months with no large gaps
        start_date = date(2024, 1, 1)
        # Use 26 observations over ~13 months to ensure >= 365 days
        observation_dates = [start_date + timedelta(days=i * 15) for i in range(26)]

        tester = ModellabilityTester()
        result = tester.test_risk_factor("USD.JPY", observation_dates)

        assert result.status == ModellabilityStatus.MODELLABLE
        assert result.is_modellable
        assert result.observation_count == 26
        assert len(result.gaps_exceeding_threshold) == 0

    def test_insufficient_observations(self):
        """Test risk factor with insufficient observations."""
        # Only 20 observations (need 24)
        start_date = date(2024, 1, 1)
        observation_dates = [start_date + timedelta(days=i * 15) for i in range(20)]

        tester = ModellabilityTester(min_observations=24)
        result = tester.test_risk_factor("EXOTIC.EQUITY", observation_dates)

        assert result.status == ModellabilityStatus.NON_MODELLABLE
        assert not result.is_modellable
        assert result.observation_count == 20
        assert "Insufficient observations" in result.reason

    def test_excessive_gaps(self):
        """Test risk factor with excessive gaps between observations."""
        start_date = date(2024, 1, 1)
        # Gap of 60 days exceeds 30-day threshold
        observation_dates = [
            start_date,
            start_date + timedelta(days=60),  # Large gap
            start_date + timedelta(days=90),
        ] + [start_date + timedelta(days=90 + i * 10) for i in range(22)]

        tester = ModellabilityTester()
        result = tester.test_risk_factor("ILLIQUID.BOND", observation_dates)

        assert result.status == ModellabilityStatus.NON_MODELLABLE
        assert not result.is_modellable
        assert len(result.gaps_exceeding_threshold) > 0
        assert result.max_gap_days == 60
        assert "Excessive gaps" in result.reason

    def test_insufficient_period(self):
        """Test risk factor with insufficient observation period."""
        # 30 observations but only over 6 months
        start_date = date(2024, 1, 1)
        observation_dates = [start_date + timedelta(days=i * 5) for i in range(30)]

        tester = ModellabilityTester(observation_period_days=365)
        result = tester.test_risk_factor("SHORT.HISTORY", observation_dates)

        # Observations over ~150 days, need 365
        assert result.status == ModellabilityStatus.NON_MODELLABLE
        assert "Insufficient observation period" in result.reason

    def test_no_observations(self):
        """Test risk factor with no observations."""
        tester = ModellabilityTester()
        result = tester.test_risk_factor("NO.DATA", [])

        assert result.status == ModellabilityStatus.INSUFFICIENT_DATA
        assert result.observation_count == 0
        assert "No observations" in result.reason

    def test_real_price_requirement(self):
        """Test requirement for real (non-proxy) prices."""
        start_date = date(2024, 1, 1)
        observation_dates = [start_date + timedelta(days=i * 15) for i in range(25)]

        tester = ModellabilityTester(require_real_prices=True)

        # Only 20 real prices (rest are proxies)
        result = tester.test_risk_factor(
            "PROXY.EQUITY", observation_dates, real_price_observations=20
        )

        assert result.status == ModellabilityStatus.NON_MODELLABLE
        assert "Insufficient real prices" in result.reason

    def test_gap_detection(self):
        """Test accurate detection of observation gaps."""
        start_date = date(2024, 1, 1)
        observation_dates = [
            start_date,
            start_date + timedelta(days=5),
            start_date + timedelta(days=40),  # Gap of 35 days
            start_date + timedelta(days=50),
        ]

        tester = ModellabilityTester(max_gap_days=30)
        result = tester.test_risk_factor("GAPPY.RF", observation_dates)

        # Should detect one gap exceeding threshold
        assert len(result.gaps_exceeding_threshold) == 1
        assert result.gaps_exceeding_threshold[0].days == 35
        assert result.max_gap_days == 35

    def test_custom_thresholds(self):
        """Test modellability tester with custom thresholds."""
        start_date = date(2024, 1, 1)
        observation_dates = [start_date + timedelta(days=i * 20) for i in range(15)]

        # Relaxed thresholds
        tester_relaxed = ModellabilityTester(
            min_observations=15,
            observation_period_days=200,
            max_gap_days=25
        )

        result = tester_relaxed.test_risk_factor("CUSTOM.RF", observation_dates)
        assert result.status == ModellabilityStatus.MODELLABLE


class TestStressScenarioCalibrator:
    """Test stress scenario calibration for NMRFs."""

    def test_identify_stress_period(self):
        """Test identification of most severe stress period."""
        # Create returns with clear stress period
        returns = jnp.concatenate([
            jax.random.normal(jax.random.PRNGKey(1), (200,)) * 2.0,  # Normal
            jax.random.normal(jax.random.PRNGKey(2), (200,)) * 10.0 - 5.0,  # Stress
            jax.random.normal(jax.random.PRNGKey(3), (200,)) * 2.0,  # Normal
        ])

        start_date = date(2023, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(len(returns))]

        calibrator = StressScenarioCalibrator(stress_period_days=200)
        stress_period = calibrator.identify_stress_period(returns, dates)

        # Stress period should be around middle section
        assert stress_period.duration_days <= 200
        # Severity should be high
        assert stress_period.severity_score > 0
        # Should have stressed ES (positive value for loss)
        assert stress_period.stressed_es > 0

    def test_stress_period_severity_metrics(self):
        """Test that stress period captures high volatility and losses."""
        # Create distinct periods
        low_vol = jnp.ones(200) * 0.5
        high_vol = jax.random.normal(jax.random.PRNGKey(42), (200,)) * 50.0 - 10.0

        returns = jnp.concatenate([low_vol, high_vol])
        dates = [date(2023, 1, 1) + timedelta(days=i) for i in range(len(returns))]

        calibrator = StressScenarioCalibrator(stress_period_days=200)
        stress_period = calibrator.identify_stress_period(returns, dates)

        # Should identify high volatility period
        stressed_vol = float(jnp.std(stress_period.stressed_returns))
        assert stressed_vol > 5.0

    def test_calibrate_stress_scenarios(self):
        """Test calibration of stress scenarios via bootstrap."""
        # Create stress period
        key = jax.random.PRNGKey(42)
        stressed_returns = jax.random.normal(key, (250,)) * 15.0 - 5.0

        from neutryx.regulatory.ima.nmrf import StressPeriod

        stress_period = StressPeriod(
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            stressed_returns=stressed_returns,
            stressed_var=-20.0,
            stressed_es=-25.0,
            severity_score=100.0
        )

        calibrator = StressScenarioCalibrator()
        scenarios = calibrator.calibrate_stress_scenarios(stress_period, num_scenarios=500)

        # Should generate scenarios
        assert len(scenarios) == 500
        # Scenarios should have similar statistics to stress period
        assert abs(float(jnp.mean(scenarios)) - float(jnp.mean(stressed_returns))) < 2.0
        assert abs(float(jnp.std(scenarios)) - float(jnp.std(stressed_returns))) < 2.0

    def test_insufficient_data_for_stress_period(self):
        """Test error when insufficient data for stress period."""
        returns = jax.random.normal(jax.random.PRNGKey(42), (100,))
        dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(100)]

        calibrator = StressScenarioCalibrator(stress_period_days=365)

        with pytest.raises(ValueError, match="Insufficient data"):
            calibrator.identify_stress_period(returns, dates)


class TestNMRFCapitalCalculator:
    """Test NMRF capital calculation."""

    def test_basic_capital_calculation(self):
        """Test basic NMRF capital calculation."""
        from neutryx.regulatory.ima.nmrf import StressPeriod

        # Create stress scenarios
        key = jax.random.PRNGKey(42)
        stress_scenarios = jax.random.normal(key, (1000,)) * 20.0 - 10.0

        stress_period = StressPeriod(
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            stressed_returns=stress_scenarios[:250],
            stressed_var=-30.0,
            stressed_es=-40.0,
            severity_score=150.0
        )

        calculator = NMRFCapitalCalculator(
            confidence_level=0.975,
            liquidity_horizon_days=20
        )

        capital = calculator.calculate_capital(
            risk_factor="ILLIQUID.EQUITY",
            stress_period=stress_period,
            stress_scenarios=stress_scenarios,
            position_size=1000000.0  # $1M position
        )

        # Capital should be positive
        assert capital.capital_charge > 0
        # Scaling factor should account for liquidity horizon
        assert capital.scaling_factor == pytest.approx(jnp.sqrt(20.0 / 10.0))
        # Check diagnostics
        assert capital.diagnostics['num_scenarios'] == 1000
        assert capital.diagnostics['position_size'] == 1000000.0

    def test_liquidity_horizon_scaling(self):
        """Test that capital scales with liquidity horizon."""
        from neutryx.regulatory.ima.nmrf import StressPeriod

        stress_scenarios = jax.random.normal(jax.random.PRNGKey(42), (1000,)) * 10.0

        stress_period = StressPeriod(
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            stressed_returns=stress_scenarios[:250],
            stressed_var=-15.0,
            stressed_es=-20.0,
            severity_score=100.0
        )

        # Test different liquidity horizons
        horizons = [10, 20, 40, 120]
        capitals = []

        for horizon in horizons:
            calculator = NMRFCapitalCalculator(liquidity_horizon_days=horizon)
            capital = calculator.calculate_capital(
                "RF", stress_period, stress_scenarios, position_size=1.0
            )
            capitals.append(capital.capital_charge)

        # Capital should increase with liquidity horizon (sqrt relationship)
        assert capitals[1] > capitals[0]  # 20d > 10d
        assert capitals[2] > capitals[1]  # 40d > 20d
        assert capitals[3] > capitals[2]  # 120d > 40d

        # Check sqrt relationship: capital(40d) ≈ capital(10d) * sqrt(4)
        assert capitals[2] / capitals[0] == pytest.approx(2.0, rel=0.1)

    def test_position_size_scaling(self):
        """Test that capital scales linearly with position size."""
        from neutryx.regulatory.ima.nmrf import StressPeriod

        stress_scenarios = jax.random.normal(jax.random.PRNGKey(42), (500,)) * 10.0

        stress_period = StressPeriod(
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            stressed_returns=stress_scenarios[:250],
            stressed_var=-15.0,
            stressed_es=-20.0,
            severity_score=100.0
        )

        calculator = NMRFCapitalCalculator()

        # Calculate capital for different position sizes
        capital_1m = calculator.calculate_capital(
            "RF", stress_period, stress_scenarios, position_size=1000000.0
        )
        capital_2m = calculator.calculate_capital(
            "RF", stress_period, stress_scenarios, position_size=2000000.0
        )

        # Should scale linearly
        assert capital_2m.capital_charge == pytest.approx(2.0 * capital_1m.capital_charge)


class TestIdentifyNMRFs:
    """Test NMRF identification across multiple risk factors."""

    def test_identify_mixed_portfolio(self):
        """Test identifying NMRFs in mixed portfolio."""
        start_date = date(2024, 1, 1)

        risk_factors = {
            # Modellable: good data (>= 365 days, >= 24 observations)
            "USD.EUR": [start_date + timedelta(days=i * 15) for i in range(26)],
            "US.10Y": [start_date + timedelta(days=i * 14) for i in range(27)],
            # Non-modellable: insufficient data
            "EXOTIC.OPTION": [start_date + timedelta(days=i * 20) for i in range(10)],
            # Non-modellable: excessive gaps
            "ILLIQUID.BOND": [
                start_date,
                start_date + timedelta(days=50),
                start_date + timedelta(days=100),
            ] + [start_date + timedelta(days=100 + i * 10) for i in range(20)],
        }

        modellable, non_modellable, results = identify_nmrfs(
            risk_factors,
            min_observations=24,
            observation_period_days=365,
            max_gap_days=30
        )

        # Should identify modellable and non-modellable RFs
        assert "USD.EUR" in modellable
        assert "US.10Y" in modellable
        assert "EXOTIC.OPTION" in non_modellable
        assert "ILLIQUID.BOND" in non_modellable

        # Check results
        assert results["USD.EUR"].is_modellable
        assert not results["EXOTIC.OPTION"].is_modellable

    def test_all_modellable(self):
        """Test case where all risk factors are modellable."""
        start_date = date(2024, 1, 1)

        risk_factors = {
            f"RF{i}": [start_date + timedelta(days=j * 15) for j in range(26)]
            for i in range(5)
        }

        modellable, non_modellable, results = identify_nmrfs(risk_factors)

        assert len(modellable) == 5
        assert len(non_modellable) == 0

    def test_all_non_modellable(self):
        """Test case where all risk factors are non-modellable."""
        start_date = date(2024, 1, 1)

        risk_factors = {
            f"RF{i}": [start_date + timedelta(days=j * 50) for j in range(5)]
            for i in range(3)
        }

        modellable, non_modellable, results = identify_nmrfs(risk_factors)

        assert len(modellable) == 0
        assert len(non_modellable) == 3


class TestNMRFCapitalAggregation:
    """Test aggregation of NMRF capital across multiple risk factors."""

    def test_capital_aggregation_no_diversification(self):
        """Test capital aggregation with no diversification benefit."""
        from neutryx.regulatory.ima.nmrf import NMRFCapital, StressPeriod

        # Create dummy capitals
        capitals = []
        for i in range(3):
            stress_period = StressPeriod(
                start_date=date(2023, 1, 1),
                end_date=date(2023, 12, 31),
                stressed_returns=jnp.zeros(250),
                stressed_var=-10.0,
                stressed_es=-15.0,
                severity_score=100.0
            )

            capitals.append(NMRFCapital(
                risk_factor=f"RF{i}",
                stress_period=stress_period,
                stressed_es=-15.0,
                capital_charge=100000.0 * (i + 1),
                liquidity_horizon_days=10,
                scaling_factor=1.0,
                diagnostics={}
            ))

        # Perfect correlation (no diversification)
        result = calculate_nmrf_capital_total(capitals, correlation_factor=1.0)

        # Total should be sum of individual capitals
        expected_total = sum(cap.capital_charge for cap in capitals)
        assert result['total_capital'] == expected_total
        assert result['num_nmrfs'] == 3
        assert result['correlation_factor'] == 1.0

    def test_capital_aggregation_with_diversification(self):
        """Test capital aggregation with diversification benefit."""
        from neutryx.regulatory.ima.nmrf import NMRFCapital, StressPeriod

        stress_period = StressPeriod(
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            stressed_returns=jnp.zeros(250),
            stressed_var=-10.0,
            stressed_es=-15.0,
            severity_score=100.0
        )

        capitals = [
            NMRFCapital(
                risk_factor="RF1",
                stress_period=stress_period,
                stressed_es=-15.0,
                capital_charge=100000.0,
                liquidity_horizon_days=10,
                scaling_factor=1.0,
                diagnostics={}
            ),
            NMRFCapital(
                risk_factor="RF2",
                stress_period=stress_period,
                stressed_es=-20.0,
                capital_charge=150000.0,
                liquidity_horizon_days=10,
                scaling_factor=1.0,
                diagnostics={}
            ),
        ]

        # With diversification (0.7 correlation)
        result = calculate_nmrf_capital_total(capitals, correlation_factor=0.7)

        undiversified = sum(cap.capital_charge for cap in capitals)
        diversified = result['total_capital']

        # Diversified capital should be less than undiversified
        assert diversified < undiversified
        assert diversified == pytest.approx(undiversified * 0.7)

    def test_empty_capital_list(self):
        """Test aggregation with no NMRF capitals."""
        result = calculate_nmrf_capital_total([])

        assert result['total_capital'] == 0.0
        assert result['num_nmrfs'] == 0
