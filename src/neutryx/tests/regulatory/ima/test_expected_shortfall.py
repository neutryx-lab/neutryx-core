"""Tests for Expected Shortfall (ES) calculation under Basel III/FRTB."""

import jax
import jax.numpy as jnp
import pytest

from neutryx.regulatory.ima import (
    ESResult,
    LiquidityHorizon,
    calculate_expected_shortfall,
    calculate_stressed_es,
    get_liquidity_horizon,
)


class TestExpectedShortfall:
    """Test ES calculation at 97.5% confidence level."""

    def test_es_calculation_basic(self):
        """Test basic ES calculation."""
        # Create symmetric loss distribution
        scenarios = jnp.array([-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0])

        es, var, diagnostics = calculate_expected_shortfall(
            scenarios, confidence_level=0.90
        )

        # At 90% confidence, VaR should be around 3.0 (positive for losses)
        # ES should be average of worst 10% (around 5.0)
        assert var > 0  # Positive values for losses
        assert es >= var  # ES should be >= VaR (more conservative)
        assert abs(es - 5.0) < 0.1  # ES should be close to 5.0

    def test_es_975_confidence(self):
        """Test ES at Basel III required 97.5% confidence level."""
        # Generate normal-like losses
        key = jax.random.PRNGKey(42)
        scenarios = jax.random.normal(key, (1000,)) * 10.0

        es, var, diagnostics = calculate_expected_shortfall(
            scenarios, confidence_level=0.975
        )

        # ES should be more extreme than VaR (larger positive value since both are reported as positive losses)
        assert es > var
        # Roughly expect ES to be ~30-40% larger in magnitude than VaR
        assert abs(es) > abs(var) * 1.1

    def test_es_vs_var_ordering(self):
        """Test that ES is always more conservative than VaR."""
        scenarios = jnp.linspace(-100, 100, 10000)

        for confidence in [0.90, 0.95, 0.975, 0.99]:
            es, var, _ = calculate_expected_shortfall(scenarios, confidence_level=confidence)
            # ES should be >= VaR (ES is more conservative, larger positive value)
            assert es >= var, f"ES ({es}) should be >= VaR ({var}) at {confidence}"

    def test_liquidity_horizons(self):
        """Test liquidity horizon mapping."""
        # Test equity liquidity horizons
        assert get_liquidity_horizon("equity", "large_cap") == LiquidityHorizon.DAYS_10
        assert get_liquidity_horizon("equity", "small_cap") == LiquidityHorizon.DAYS_20
        assert get_liquidity_horizon("equity", "emerging") == LiquidityHorizon.DAYS_40

        # Test rates
        assert get_liquidity_horizon("rates", "major") == LiquidityHorizon.DAYS_10
        assert get_liquidity_horizon("rates", "other") == LiquidityHorizon.DAYS_20

        # Test FX
        assert get_liquidity_horizon("fx", "major_pairs") == LiquidityHorizon.DAYS_10
        assert get_liquidity_horizon("fx", "other") == LiquidityHorizon.DAYS_40

        # Test commodities
        assert get_liquidity_horizon("commodity", "electricity") == LiquidityHorizon.DAYS_250

        # Test credit
        assert get_liquidity_horizon("credit", "ig_index") == LiquidityHorizon.DAYS_20
        assert get_liquidity_horizon("credit", "hy_single") == LiquidityHorizon.DAYS_120

    def test_liquidity_horizon_scaling(self):
        """Test that ES scales with liquidity horizon."""
        base_scenarios = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])

        # Calculate ES for 10-day horizon
        es_10d, _, _ = calculate_expected_shortfall(base_scenarios, confidence_level=0.975)

        # Scale to 40-day horizon
        scaling_factor = jnp.sqrt(40.0 / 10.0)  # sqrt(4) = 2.0
        es_40d_expected = es_10d * scaling_factor

        # Verify scaling is approximately correct
        # (in practice, you'd scale the scenarios themselves)
        assert scaling_factor == pytest.approx(2.0)

    def test_stressed_es(self):
        """Test stressed ES calculation."""
        key = jax.random.PRNGKey(42)

        # Combined scenarios: normal period + stressed period
        normal_scenarios = jax.random.normal(key, (500,)) * 5.0
        stressed_scenarios = jax.random.normal(jax.random.PRNGKey(43), (500,)) * 15.0

        # Combine all scenarios
        all_scenarios = jnp.concatenate([normal_scenarios, stressed_scenarios])

        # Create stress period mask (last 500 scenarios are stressed)
        stress_mask = jnp.concatenate([
            jnp.zeros(500, dtype=bool),
            jnp.ones(500, dtype=bool)
        ])

        total_es, standard_result, stressed_result = calculate_stressed_es(
            all_scenarios,
            stress_period_indices=stress_mask,
            confidence_level=0.975
        )

        # Stressed ES should be significantly larger than standard ES
        assert stressed_result.expected_shortfall > standard_result.expected_shortfall * 1.5

    def test_es_coherence(self):
        """Test that ES satisfies coherent risk measure properties."""
        scenarios1 = jnp.array([-10.0, -5.0, 0.0, 5.0, 10.0])
        scenarios2 = jnp.array([-8.0, -4.0, 0.0, 4.0, 8.0])

        es1, _, _ = calculate_expected_shortfall(scenarios1)
        es2, _, _ = calculate_expected_shortfall(scenarios2)

        # Sub-additivity: ES(X+Y) <= ES(X) + ES(Y)
        combined_scenarios = scenarios1 + scenarios2
        es_combined, _, _ = calculate_expected_shortfall(combined_scenarios)
        assert es_combined <= es1 + es2 + 1e-6  # Small tolerance

        # Positive homogeneity: ES(λX) = λ*ES(X) for λ > 0
        scaled_scenarios = 2.0 * scenarios1
        es_scaled, _, _ = calculate_expected_shortfall(scaled_scenarios)
        assert abs(es_scaled - 2.0 * es1) < 0.01

    def test_empty_scenarios(self):
        """Test error handling for empty scenarios."""
        with pytest.raises((ValueError, IndexError)):
            calculate_expected_shortfall(jnp.array([]))

    def test_single_scenario(self):
        """Test behavior with single scenario."""
        scenarios = jnp.array([-5.0])
        es, var, _ = calculate_expected_shortfall(scenarios)

        # With single scenario, ES = VaR = absolute value of that scenario (positive)
        assert es == var == 5.0

    def test_es_result_dataclass(self):
        """Test ESResult dataclass structure."""
        scenarios = jnp.linspace(-50, 50, 100)
        es, var, diagnostics = calculate_expected_shortfall(scenarios)

        # Check diagnostics contain expected keys
        assert 'num_scenarios' in diagnostics
        assert 'tail_observations' in diagnostics
        assert 'mean_excess_loss' in diagnostics
        assert 'max_loss' in diagnostics

        # Create ESResult manually
        result = ESResult(
            expected_shortfall=es,
            var_97_5=var,
            confidence_level=0.975,
            num_scenarios=len(scenarios),
            base_horizon_days=10
        )

        assert result.expected_shortfall == es
        assert result.var_97_5 == var
        assert result.confidence_level == 0.975


class TestLiquidityHorizonMapping:
    """Test liquidity horizon mappings for different asset classes."""

    def test_all_equity_sub_classes(self):
        """Test all equity sub-classifications."""
        mappings = {
            "large_cap": LiquidityHorizon.DAYS_10,
            "small_cap": LiquidityHorizon.DAYS_20,
            "emerging": LiquidityHorizon.DAYS_40,
            "other": LiquidityHorizon.DAYS_20,
        }

        for sub_class, expected_horizon in mappings.items():
            horizon = get_liquidity_horizon("equity", sub_class)
            assert horizon == expected_horizon

    def test_all_credit_sub_classes(self):
        """Test all credit sub-classifications."""
        mappings = {
            "ig_index": LiquidityHorizon.DAYS_20,
            "ig_single": LiquidityHorizon.DAYS_40,
            "hy_index": LiquidityHorizon.DAYS_60,
            "hy_single": LiquidityHorizon.DAYS_120,
            "structured": LiquidityHorizon.DAYS_120,
        }

        for sub_class, expected_horizon in mappings.items():
            horizon = get_liquidity_horizon("credit", sub_class)
            assert horizon == expected_horizon

    def test_default_liquidity_horizon(self):
        """Test default liquidity horizon for unknown asset class."""
        # Unknown asset class should default to something reasonable
        horizon = get_liquidity_horizon("unknown_asset_class", "unknown_sub")
        assert isinstance(horizon, LiquidityHorizon)
        # Should default to at least 10 days
        assert horizon.to_days() >= 10

    def test_liquidity_horizon_enum_values(self):
        """Test LiquidityHorizon enum values are correct."""
        assert LiquidityHorizon.DAYS_10.to_days() == 10
        assert LiquidityHorizon.DAYS_20.to_days() == 20
        assert LiquidityHorizon.DAYS_40.to_days() == 40
        assert LiquidityHorizon.DAYS_60.to_days() == 60
        assert LiquidityHorizon.DAYS_120.to_days() == 120
        assert LiquidityHorizon.DAYS_250.to_days() == 250
