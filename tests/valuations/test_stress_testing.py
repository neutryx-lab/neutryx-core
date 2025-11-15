"""Tests for enhanced stress testing framework."""
import jax.numpy as jnp
import pytest

from neutryx.valuations.stress_testing import (
    AdvancedReverseStressTest,
    ConcentrationRiskMetrics,
    HistoricalScenarioLibrary,
    HypotheticalScenarioBuilder,
)
from neutryx.valuations.stress_test import StressScenario


class TestHistoricalScenarioLibrary:
    """Test historical scenario library."""

    def test_library_initialization(self):
        """Test library contains all scenarios."""
        library = HistoricalScenarioLibrary()

        # Should have original scenarios plus new ones
        assert len(library.scenarios) >= 20, "Should have at least 20 scenarios"

        # Check some key scenarios exist
        assert "financial_crisis_2008" in library.scenarios
        assert "covid_crash_2020" in library.scenarios
        assert "asian_crisis_1997" in library.scenarios
        assert "brexit_2016" in library.scenarios

    def test_get_scenario(self):
        """Test retrieving specific scenario."""
        library = HistoricalScenarioLibrary()

        scenario = library.get_scenario("financial_crisis_2008")

        assert scenario.name == "Financial Crisis 2008"
        assert "equity" in scenario.shocks
        assert scenario.shocks["equity"] < 0, "Equity should be negative in crisis"

    def test_get_unknown_scenario_raises(self):
        """Test that unknown scenario raises error."""
        library = HistoricalScenarioLibrary()

        with pytest.raises(ValueError, match="Unknown scenario"):
            library.get_scenario("nonexistent_scenario")

    def test_list_scenarios(self):
        """Test listing all scenario names."""
        library = HistoricalScenarioLibrary()

        names = library.list_scenarios()

        assert isinstance(names, list)
        assert len(names) > 0
        assert "financial_crisis_2008" in names

    def test_filter_by_crisis_type_equity(self):
        """Test filtering scenarios by equity crisis."""
        library = HistoricalScenarioLibrary()

        equity_crises = library.filter_by_crisis_type("equity")

        assert len(equity_crises) > 0
        # Should include crashes and bubbles
        scenario_names = [s.name.lower() for s in equity_crises]
        assert any("crash" in name or "bubble" in name for name in scenario_names)

    def test_filter_by_crisis_type_credit(self):
        """Test filtering scenarios by credit crisis."""
        library = HistoricalScenarioLibrary()

        credit_crises = library.filter_by_crisis_type("credit")

        assert len(credit_crises) > 0
        scenario_names = [s.name.lower() for s in credit_crises]
        assert any("credit" in name or "default" in name for name in scenario_names)

    def test_filter_by_crisis_type_fx(self):
        """Test filtering scenarios by FX crisis."""
        library = HistoricalScenarioLibrary()

        fx_crises = library.filter_by_crisis_type("fx")

        assert len(fx_crises) > 0
        # Should include currency crises
        scenario_names = [s.name.lower() for s in fx_crises]
        assert any("currency" in name or "franc" in name or "devaluation" in name
                   for name in scenario_names)


class TestHypotheticalScenarioBuilder:
    """Test hypothetical scenario builder."""

    def test_basic_scenario_build(self):
        """Test building a basic scenario."""
        scenario = (
            HypotheticalScenarioBuilder("Test Scenario")
            .add_shock("equity", -0.20)
            .add_shock("rates", 0.01)
            .build()
        )

        assert scenario.name == "Test Scenario"
        assert "equity" in scenario.shocks
        assert scenario.shocks["equity"] == -0.20
        assert scenario.shocks["rates"] == 0.01

    def test_parallel_shift(self):
        """Test parallel shift."""
        scenario = (
            HypotheticalScenarioBuilder("Parallel Shift")
            .parallel_shift("rates", 0.02)
            .build()
        )

        assert "rates" in scenario.shocks
        assert scenario.shocks["rates"] == 0.02

    def test_twist_scenario(self):
        """Test curve twist."""
        scenario = (
            HypotheticalScenarioBuilder("Steepening")
            .twist("rates_2y", "rates_10y", -0.01, 0.02)
            .build()
        )

        assert "rates_2y" in scenario.shocks
        assert "rates_10y" in scenario.shocks
        assert scenario.shocks["rates_2y"] == -0.01
        assert scenario.shocks["rates_10y"] == 0.02

    def test_correlation_shock(self):
        """Test correlation shock."""
        scenario = (
            HypotheticalScenarioBuilder("Correlation Break")
            .correlation_shock("equity", "bonds", 0.5)
            .build()
        )

        assert "corr_equity_bonds" in scenario.shocks
        assert scenario.shocks["corr_equity_bonds"] == 0.5

    def test_method_chaining(self):
        """Test that all methods support chaining."""
        scenario = (
            HypotheticalScenarioBuilder("Complex Scenario")
            .add_shock("equity", -0.15)
            .parallel_shift("rates", 0.015)
            .twist("rates_short", "rates_long", 0.01, -0.005)
            .correlation_shock("equity", "credit", -0.3)
            .build()
        )

        assert len(scenario.shocks) == 5, "Should have 5 shocks"

    def test_auto_description(self):
        """Test that description is auto-generated if not provided."""
        scenario = (
            HypotheticalScenarioBuilder("Test")
            .add_shock("factor1", 0.1)
            .add_shock("factor2", 0.2)
            .build()
        )

        assert "2 shocks" in scenario.description


class TestConcentrationRiskMetrics:
    """Test concentration risk metrics."""

    def test_herfindahl_perfectly_diversified(self):
        """Test HHI for perfectly diversified portfolio."""
        # 10 equal exposures
        exposures = jnp.ones(10)

        hhi = ConcentrationRiskMetrics.herfindahl_index(exposures)

        # HHI should be 1/N = 0.1 for perfect diversification
        assert abs(hhi - 0.1) < 0.001, "HHI should be 0.1 for equal exposures"

    def test_herfindahl_fully_concentrated(self):
        """Test HHI for fully concentrated portfolio."""
        # One exposure dominates
        exposures = jnp.array([100.0, 1.0, 1.0, 1.0])

        hhi = ConcentrationRiskMetrics.herfindahl_index(exposures)

        # HHI should be close to 1
        assert hhi > 0.9, "HHI should be high for concentrated portfolio"

    def test_herfindahl_handles_negative(self):
        """Test that HHI handles negative exposures."""
        exposures = jnp.array([10.0, -5.0, 3.0, -2.0])

        hhi = ConcentrationRiskMetrics.herfindahl_index(exposures)

        assert 0.0 <= hhi <= 1.0, "HHI should be in [0, 1]"

    def test_gini_coefficient_equal(self):
        """Test Gini for equal distribution."""
        exposures = jnp.ones(10)

        gini = ConcentrationRiskMetrics.gini_coefficient(exposures)

        # Should be close to 0 for perfect equality
        assert abs(gini) < 0.05, "Gini should be near 0 for equal distribution"

    def test_gini_coefficient_unequal(self):
        """Test Gini for unequal distribution."""
        # Very unequal distribution
        exposures = jnp.array([100.0, 10.0, 5.0, 1.0, 1.0])

        gini = ConcentrationRiskMetrics.gini_coefficient(exposures)

        # Should be significantly positive
        assert gini > 0.3, "Gini should be high for unequal distribution"

    def test_top_n_concentration(self):
        """Test top-N concentration."""
        exposures = jnp.array([50.0, 30.0, 10.0, 5.0, 3.0, 1.0, 1.0])

        # Top 2 should be 80% of total (50+30)/100
        top_2 = ConcentrationRiskMetrics.top_n_concentration(exposures, n=2)

        assert abs(top_2 - 0.80) < 0.01, "Top 2 should be 80%"

        # Top 5 should be 98%
        top_5 = ConcentrationRiskMetrics.top_n_concentration(exposures, n=5)

        assert abs(top_5 - 0.98) < 0.01, "Top 5 should be 98%"

    def test_entropy_uniform(self):
        """Test entropy for uniform distribution."""
        # Uniform distribution should have maximum entropy
        exposures = jnp.ones(8)  # 8 equal exposures

        entropy = ConcentrationRiskMetrics.entropy(exposures)

        # Entropy of uniform distribution with N items is log(N)
        expected_entropy = jnp.log(8.0)

        assert abs(entropy - expected_entropy) < 0.01, \
            "Entropy should be log(N) for uniform distribution"

    def test_entropy_concentrated(self):
        """Test entropy for concentrated distribution."""
        # One dominant exposure
        exposures = jnp.array([100.0, 1.0, 1.0, 1.0])

        entropy = ConcentrationRiskMetrics.entropy(exposures)

        # Entropy should be low for concentrated distribution
        max_entropy = jnp.log(4.0)  # Maximum for 4 items

        assert entropy < max_entropy * 0.5, \
            "Entropy should be low for concentrated distribution"

    def test_compute_all_metrics(self):
        """Test computing all metrics at once."""
        exposures = jnp.array([40.0, 30.0, 20.0, 10.0])

        metrics = ConcentrationRiskMetrics.compute_all_metrics(exposures, top_n=2)

        assert "herfindahl_index" in metrics
        assert "gini_coefficient" in metrics
        assert "top_2_concentration" in metrics
        assert "entropy" in metrics

        # All metrics should be numeric
        for value in metrics.values():
            assert isinstance(value, float)


class TestAdvancedReverseStressTest:
    """Test advanced reverse stress testing."""

    def test_find_minimal_shock_single_factor(self):
        """Test finding minimal shock for single factor."""
        # Simple linear valuation function
        def valuation_fn(equity=100.0, rates=0.03):
            return equity * 1000 + rates * 10000

        base_params = {"equity": 100.0, "rates": 0.03}

        reverse_test = AdvancedReverseStressTest(
            valuation_fn=valuation_fn,
            base_params=base_params,
            risk_factors=["equity", "rates"],
        )

        # Want to lose 10,000 (10% equity drop should do it)
        target_loss = -10000.0

        shock = reverse_test.find_minimal_shock(
            target_loss=target_loss,
            factor="equity",
            max_steps=500,
        )

        # Should find approximately -10% shock
        assert abs(shock - (-0.10)) < 0.01, \
            "Should find ~10% equity shock for 10k loss"

    def test_find_realistic_scenario(self):
        """Test finding realistic multi-factor scenario."""
        # Portfolio value depends on equity and rates
        def valuation_fn(equity=100.0, rates=0.03, volatility=0.20):
            return equity * 1000 - rates * 5000 + volatility * 2000

        base_params = {"equity": 100.0, "rates": 0.03, "volatility": 0.20}

        reverse_test = AdvancedReverseStressTest(
            valuation_fn=valuation_fn,
            base_params=base_params,
            risk_factors=["equity", "rates", "volatility"],
        )

        # Target loss
        target_loss = -15000.0

        result = reverse_test.find_realistic_scenario(
            target_loss=target_loss,
            max_steps=500,
        )

        assert result.target_loss == target_loss
        assert len(result.required_shocks) > 0
        assert result.implied_scenario.name == "Reverse Stress Test Result"

        # Verify the scenario actually produces approximately the target loss
        stressed_params = base_params.copy()
        for factor, shock in result.required_shocks.items():
            if factor in stressed_params:
                stressed_params[factor] = base_params[factor] * (1 + shock)

        stressed_value = valuation_fn(**stressed_params)
        actual_loss = stressed_value - reverse_test.base_value

        # Should be close to target loss
        assert abs(actual_loss - target_loss) < 1000, \
            "Implied scenario should produce target loss"

    def test_plausibility_scoring(self):
        """Test plausibility scoring with historical scenarios."""
        def valuation_fn(equity=100.0, volatility=0.20):
            return equity * 1000 + volatility * 500

        base_params = {"equity": 100.0, "volatility": 0.20}

        # Create some historical scenarios
        historical = [
            StressScenario(
                name="Hist 1",
                description="",
                shocks={"equity": -0.30, "volatility": 1.0}
            ),
            StressScenario(
                name="Hist 2",
                description="",
                shocks={"equity": -0.40, "volatility": 2.0}
            ),
        ]

        reverse_test = AdvancedReverseStressTest(
            valuation_fn=valuation_fn,
            base_params=base_params,
            risk_factors=["equity", "volatility"],
        )

        result = reverse_test.find_realistic_scenario(
            target_loss=-25000.0,
            historical_scenarios=historical,
            max_steps=500,
        )

        assert result.plausibility_score is not None
        assert 0.0 <= result.plausibility_score <= 1.0, \
            "Plausibility score should be in [0, 1]"


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_scenario_library_with_stress_testing(self):
        """Test using library scenarios for stress testing."""
        library = HistoricalScenarioLibrary()

        # Get a crisis scenario
        crisis = library.get_scenario("financial_crisis_2008")

        # Simple portfolio valuation
        def portfolio_value(equity=100.0, credit_spread=0.02):
            return equity * 10000 - credit_spread * 100000

        base_params = {"equity": 100.0, "credit_spread": 0.02}

        base_value = portfolio_value(**base_params)

        # Apply crisis shocks
        stressed_params = base_params.copy()
        if "equity" in crisis.shocks:
            stressed_params["equity"] *= (1 + crisis.shocks["equity"])
        if "credit_spread" in crisis.shocks:
            stressed_params["credit_spread"] += crisis.shocks["credit_spread"]

        stressed_value = portfolio_value(**stressed_params)
        loss = stressed_value - base_value

        # 2008 crisis should cause significant loss
        assert loss < -400000, "2008 crisis should cause significant portfolio loss"

    def test_hypothetical_scenario_with_concentration(self):
        """Test building hypothetical scenario and measuring concentration."""
        # Build a scenario with multiple shocks
        scenario = (
            HypotheticalScenarioBuilder("Multi-Factor Stress")
            .add_shock("equity_us", -0.25)
            .add_shock("equity_eu", -0.20)
            .add_shock("equity_asia", -0.30)
            .add_shock("rates_usd", 0.02)
            .build()
        )

        # Portfolio exposures across regions
        exposures = jnp.array([500000, 300000, 200000])  # US, EU, Asia

        # Check concentration
        metrics = ConcentrationRiskMetrics.compute_all_metrics(exposures, top_n=2)

        # Portfolio is somewhat concentrated in US
        assert metrics["herfindahl_index"] > 0.33, \
            "HHI should show concentration"
        assert metrics["top_2_concentration"] > 0.75, \
            "Top 2 regions dominate"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
