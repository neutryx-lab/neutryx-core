"""Tests for P&L Attribution (PLA) testing under Basel III/FRTB."""

import jax
import jax.numpy as jnp
import pytest

from neutryx.regulatory.ima import (
    PLAMetrics,
    PLATestResult,
    calculate_pla_metrics,
    diagnose_pla_failures,
)


class TestPLAMetrics:
    """Test P&L Attribution metrics calculation."""

    def test_perfect_attribution(self):
        """Test PLA with perfect attribution (HPL = RTPL)."""
        # Perfect match between hypothetical and risk-theoretical P&L
        hypothetical_pnl = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        risk_theoretical_pnl = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

        metrics = calculate_pla_metrics(hypothetical_pnl, risk_theoretical_pnl)

        # Perfect correlation
        assert metrics.spearman_correlation == pytest.approx(1.0)
        assert metrics.kolmogorov_smirnov_statistic == pytest.approx(0.0)

        # Should be in green zone
        assert metrics.test_result == PLATestResult.GREEN
        assert metrics.passes_test

    def test_good_attribution(self):
        """Test PLA with good (but not perfect) attribution."""
        # Strong correlation but with some noise
        key = jax.random.PRNGKey(42)
        hypothetical_pnl = jax.random.normal(key, (250,)) * 10.0
        # RTPL closely tracks HPL with small noise
        noise = jax.random.normal(jax.random.PRNGKey(43), (250,)) * 1.0
        risk_theoretical_pnl = hypothetical_pnl + noise

        metrics = calculate_pla_metrics(hypothetical_pnl, risk_theoretical_pnl)

        # Should have high correlation (>0.85)
        assert metrics.spearman_correlation > 0.85
        # Should be in green zone
        assert metrics.test_result == PLATestResult.GREEN
        assert metrics.passes_test

    def test_poor_attribution(self):
        """Test PLA with poor attribution."""
        # Weak correlation
        key = jax.random.PRNGKey(42)
        hypothetical_pnl = jax.random.normal(key, (100,))
        risk_theoretical_pnl = jax.random.normal(jax.random.PRNGKey(43), (100,))

        metrics = calculate_pla_metrics(hypothetical_pnl, risk_theoretical_pnl)

        # Should have low correlation
        assert metrics.spearman_correlation < 0.85
        # Should fail attribution test
        assert not metrics.passes_attribution
        assert metrics.zone in [PLATestResult.AMBER, PLATestResult.RED]

    def test_spearman_threshold(self):
        """Test Spearman correlation threshold."""
        # Test boundary case at 0.85 threshold
        hypothetical_pnl = jnp.arange(100, dtype=float)

        # Create RTPL with correlation just above threshold
        risk_theoretical_pnl = hypothetical_pnl + jax.random.normal(
            jax.random.PRNGKey(42), (100,)
        ) * 5.0

        metrics = calculate_pla_metrics(
            hypothetical_pnl,
            risk_theoretical_pnl,
            spearman_threshold=0.85
        )

        # Check threshold is applied correctly
        if metrics.spearman_correlation >= 0.85:
            assert metrics.passes_spearman
        else:
            assert not metrics.passes_spearman

    def test_ks_threshold(self):
        """Test Kolmogorov-Smirnov threshold."""
        # Identical distributions should have KS = 0
        pnl = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

        metrics = calculate_pla_metrics(pnl, pnl, ks_threshold=0.09)

        assert metrics.kolmogorov_smirnov_statistic < 0.09
        assert metrics.passes_ks

    def test_zone_classification(self):
        """Test zone classification logic."""
        base_pnl = jnp.arange(100, dtype=float)

        # Green zone: both tests pass
        green_rtpl = base_pnl + jax.random.normal(jax.random.PRNGKey(1), (100,)) * 2.0
        metrics_green = calculate_pla_metrics(base_pnl, green_rtpl)
        if metrics_green.passes_attribution:
            assert metrics_green.zone == PLATestResult.GREEN

        # Red zone: both tests fail
        red_rtpl = jax.random.normal(jax.random.PRNGKey(2), (100,)) * 50.0
        metrics_red = calculate_pla_metrics(base_pnl, red_rtpl)
        if not metrics_red.passes_spearman and not metrics_red.passes_ks:
            assert metrics_red.zone == PLATestResult.RED

    def test_pla_with_negative_correlation(self):
        """Test PLA with negative correlation."""
        hypothetical_pnl = jnp.arange(50, dtype=float)
        # Perfectly negative correlation
        risk_theoretical_pnl = -hypothetical_pnl

        metrics = calculate_pla_metrics(hypothetical_pnl, risk_theoretical_pnl)

        # Spearman correlation should be -1
        assert metrics.spearman_correlation == pytest.approx(-1.0)
        # Should fail attribution (negative correlation is bad)
        assert not metrics.passes_attribution

    def test_diagnose_pla_failures(self):
        """Test PLA failure diagnostic function."""
        key = jax.random.PRNGKey(42)
        hypothetical_pnl = jax.random.normal(key, (100,))
        risk_theoretical_pnl = jax.random.normal(jax.random.PRNGKey(43), (100,))

        diagnostics = diagnose_pla_failures(hypothetical_pnl, risk_theoretical_pnl)

        # Check diagnostic keys exist
        assert 'mean_difference' in diagnostics
        assert 'std_difference' in diagnostics
        assert 'mean_unexplained_pnl' in diagnostics
        assert 'max_unexplained_pnl' in diagnostics
        assert 'rmse' in diagnostics

        # Check values are reasonable
        assert isinstance(diagnostics['mean_difference'], float)
        assert isinstance(diagnostics['rmse'], float)
        assert diagnostics['rmse'] >= 0

    def test_unequal_length_inputs(self):
        """Test error handling for unequal length inputs."""
        hypothetical_pnl = jnp.array([1.0, 2.0, 3.0])
        risk_theoretical_pnl = jnp.array([1.0, 2.0])

        with pytest.raises((ValueError, IndexError)):
            calculate_pla_metrics(hypothetical_pnl, risk_theoretical_pnl)

    def test_insufficient_observations(self):
        """Test behavior with very few observations."""
        # Less than typical minimum for statistical tests
        hypothetical_pnl = jnp.array([1.0, 2.0])
        risk_theoretical_pnl = jnp.array([1.0, 2.0])

        # Should still compute metrics (though not statistically meaningful)
        metrics = calculate_pla_metrics(hypothetical_pnl, risk_theoretical_pnl)
        assert isinstance(metrics, PLAMetrics)

    def test_constant_pnl(self):
        """Test behavior with constant P&L."""
        # Constant P&L (no variation)
        hypothetical_pnl = jnp.ones(100)
        risk_theoretical_pnl = jnp.ones(100)

        # Should handle gracefully (Spearman undefined, but distributions match)
        metrics = calculate_pla_metrics(hypothetical_pnl, risk_theoretical_pnl)
        # KS test should show perfect match
        assert metrics.kolmogorov_smirnov_statistic == pytest.approx(0.0)


class TestPLATestResults:
    """Test PLA zone classifications."""

    def test_zone_enum_values(self):
        """Test PLATestResult enum values."""
        assert PLATestResult.GREEN.value == "green"
        assert PLATestResult.AMBER.value == "amber"
        assert PLATestResult.RED.value == "red"

    def test_zone_from_metrics(self):
        """Test zone determination from metrics."""
        # Green zone
        pnl = jnp.arange(100, dtype=float)
        metrics_green = calculate_pla_metrics(
            pnl,
            pnl + jax.random.normal(jax.random.PRNGKey(1), (100,)) * 1.0
        )

        # Red zone
        metrics_red = calculate_pla_metrics(
            pnl,
            jax.random.normal(jax.random.PRNGKey(2), (100,)) * 100.0
        )

        # At least one should be green and one should be red/amber
        zones = {metrics_green.zone, metrics_red.zone}
        assert len(zones) > 1  # Different zones

    def test_zone_consequences(self):
        """Test that zone has appropriate regulatory consequences."""
        pnl_good = jnp.arange(100, dtype=float)
        pnl_good_rtpl = pnl_good + jax.random.normal(jax.random.PRNGKey(1), (100,)) * 2.0

        pnl_bad = jnp.arange(100, dtype=float)
        pnl_bad_rtpl = jax.random.normal(jax.random.PRNGKey(2), (100,)) * 100.0

        metrics_good = calculate_pla_metrics(pnl_good, pnl_good_rtpl)
        metrics_bad = calculate_pla_metrics(pnl_bad, pnl_bad_rtpl)

        # Good attribution should pass
        if metrics_good.zone == PLATestResult.GREEN:
            assert metrics_good.passes_attribution

        # Bad attribution should fail
        if metrics_bad.zone == PLATestResult.RED:
            assert not metrics_bad.passes_attribution


class TestPLADiagnostics:
    """Test PLA diagnostic functions."""

    def test_diagnose_systematic_bias(self):
        """Test diagnosis of systematic bias in attribution."""
        hypothetical_pnl = jnp.arange(100, dtype=float)
        # Systematic bias: RTPL consistently over-estimates
        risk_theoretical_pnl = hypothetical_pnl + 5.0

        diagnostics = diagnose_pla_failures(hypothetical_pnl, risk_theoretical_pnl)

        # Should detect positive mean difference (HPL - RTPL = negative when RTPL > HPL)
        assert diagnostics['systematic_bias']['mean_difference'] < 0

    def test_diagnose_volatility_mismatch(self):
        """Test diagnosis of volatility mismatch."""
        key = jax.random.PRNGKey(42)
        hypothetical_pnl = jax.random.normal(key, (100,)) * 10.0
        # RTPL has much higher volatility
        risk_theoretical_pnl = hypothetical_pnl + jax.random.normal(
            jax.random.PRNGKey(43), (100,)
        ) * 50.0

        diagnostics = diagnose_pla_failures(hypothetical_pnl, risk_theoretical_pnl)

        # Should detect large RMSE
        assert diagnostics['rmse'] > 10.0
        # Should detect std difference
        assert abs(diagnostics['std_difference']) > 5.0

    def test_diagnose_large_outliers(self):
        """Test diagnosis of large unexplained outliers."""
        hypothetical_pnl = jnp.ones(100) * 1.0
        risk_theoretical_pnl = jnp.ones(100) * 1.0
        # Add a large outlier
        risk_theoretical_pnl = risk_theoretical_pnl.at[50].set(100.0)

        diagnostics = diagnose_pla_failures(hypothetical_pnl, risk_theoretical_pnl)

        # Should detect large outlier in largest_discrepancies
        assert len(diagnostics['largest_discrepancies']) > 0
        assert abs(diagnostics['largest_discrepancies'][0]['difference']) > 50.0
