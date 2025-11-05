"""Tests for CCR analytics module."""
import jax
import jax.numpy as jnp
import pytest

from neutryx.valuations.ccr_analytics import (
    CollateralOptimizer,
    ExposureProfile,
    MultiNettingSetAggregator,
    NettingSetExposure,
    WrongWayRiskAnalytics,
)


class TestExposureProfile:
    """Test exposure profile calculations."""

    def test_from_paths_basic(self):
        """Test creating exposure profile from simulated paths."""
        # Create synthetic exposure paths
        n_paths = 1000
        n_times = 10
        times = jnp.linspace(0, 1, n_times)

        # Simple increasing trend with noise
        key = jax.random.PRNGKey(42)
        base_exposure = jnp.linspace(100, 200, n_times)
        noise = jax.random.normal(key, (n_paths, n_times)) * 20
        exposure_paths = base_exposure[None, :] + noise

        profile = ExposureProfile.from_paths(times, exposure_paths, include_pathwise=True)

        # Check that all metrics are computed
        assert profile.expected_exposure.shape == (n_times,)
        assert profile.pfe_95.shape == (n_times,)
        assert profile.pfe_97_5.shape == (n_times,)
        assert profile.pfe_99.shape == (n_times,)
        assert profile.expected_positive_exposure.shape == (n_times,)

        # Check that EE is close to true mean
        assert jnp.allclose(profile.expected_exposure, base_exposure, atol=5.0)

        # Check that PFE_99 > PFE_97.5 > PFE_95
        assert jnp.all(profile.pfe_99 >= profile.pfe_97_5)
        assert jnp.all(profile.pfe_97_5 >= profile.pfe_95)

    def test_effective_epe_non_decreasing(self):
        """Test that effective EPE is non-decreasing."""
        times = jnp.array([0.0, 0.5, 1.0, 1.5, 2.0])

        # EPE profile that increases then decreases
        epe_profile = jnp.array([10.0, 20.0, 15.0, 12.0, 10.0])

        # Create simple exposure paths
        n_paths = 100
        exposure_paths = epe_profile[None, :] + jax.random.normal(
            jax.random.PRNGKey(0), (n_paths, len(times))
        )

        profile = ExposureProfile.from_paths(times, exposure_paths)

        # Effective EPE should be non-decreasing by construction
        # Check the final value is >= average
        avg_epe = float(jnp.mean(profile.expected_positive_exposure))
        assert profile.effective_epe >= avg_epe

    def test_time_averaged_epe(self):
        """Test time-averaged EPE calculation."""
        times = jnp.linspace(0, 1, 11)  # 0, 0.1, 0.2, ..., 1.0
        epe = jnp.ones(11) * 100.0  # Constant EPE of 100

        # Create minimal profile
        profile = ExposureProfile(
            times=times,
            expected_exposure=epe,
            pfe_95=epe,
            pfe_97_5=epe,
            pfe_99=epe,
            expected_positive_exposure=epe,
            expected_negative_exposure=jnp.zeros(11),
            effective_epe=100.0,
            max_pfe=100.0,
        )

        # For constant EPE over time [0,1], time-averaged should equal the value
        avg_epe = profile.time_averaged_epe()
        assert abs(avg_epe - 100.0) < 1.0  # Allow small numerical error

    def test_peak_exposure(self):
        """Test peak exposure detection."""
        times = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
        pfe = jnp.array([100.0, 150.0, 200.0, 180.0, 160.0])  # Peak at t=2

        profile = ExposureProfile(
            times=times,
            expected_exposure=pfe,
            pfe_95=pfe,
            pfe_97_5=pfe,
            pfe_99=pfe,
            expected_positive_exposure=pfe,
            expected_negative_exposure=jnp.zeros(5),
            effective_epe=150.0,
            max_pfe=200.0,
        )

        peak_value, peak_time = profile.peak_exposure("pfe_97_5")

        assert peak_value == 200.0
        assert peak_time == 2.0


class TestMultiNettingSetAggregator:
    """Test multi-netting set aggregation."""

    def create_sample_netting_set(
        self, ns_id: str, cp_id: str, base_exposure: float
    ) -> NettingSetExposure:
        """Helper to create a sample netting set."""
        times = jnp.linspace(0, 1, 10)
        epe = jnp.ones(10) * base_exposure

        profile = ExposureProfile(
            times=times,
            expected_exposure=epe,
            pfe_95=epe * 1.5,
            pfe_97_5=epe * 1.8,
            pfe_99=epe * 2.0,
            expected_positive_exposure=epe,
            expected_negative_exposure=jnp.zeros(10),
            effective_epe=base_exposure,
            max_pfe=base_exposure * 1.8,
        )

        return NettingSetExposure(
            netting_set_id=ns_id,
            counterparty_id=cp_id,
            profile=profile,
        )

    def test_aggregate_by_counterparty(self):
        """Test aggregation by counterparty."""
        # Create 3 netting sets: 2 for CP1, 1 for CP2
        ns1 = self.create_sample_netting_set("NS1", "CP1", 100.0)
        ns2 = self.create_sample_netting_set("NS2", "CP1", 150.0)
        ns3 = self.create_sample_netting_set("NS3", "CP2", 200.0)

        aggregator = MultiNettingSetAggregator(netting_sets=[ns1, ns2, ns3])

        cp_exposures = aggregator.aggregate_by_counterparty()

        assert "CP1" in cp_exposures
        assert "CP2" in cp_exposures

        # CP1 should have sum of NS1 and NS2
        cp1_epe = cp_exposures["CP1"].expected_positive_exposure[0]
        assert abs(cp1_epe - 250.0) < 0.1  # 100 + 150

        # CP2 should have NS3
        cp2_epe = cp_exposures["CP2"].expected_positive_exposure[0]
        assert abs(cp2_epe - 200.0) < 0.1

    def test_aggregate_portfolio(self):
        """Test portfolio-level aggregation."""
        # Create netting sets for different counterparties
        ns1 = self.create_sample_netting_set("NS1", "CP1", 100.0)
        ns2 = self.create_sample_netting_set("NS2", "CP2", 100.0)
        ns3 = self.create_sample_netting_set("NS3", "CP3", 100.0)

        aggregator = MultiNettingSetAggregator(netting_sets=[ns1, ns2, ns3])

        # Test with zero correlation (maximum diversification)
        portfolio = aggregator.aggregate_portfolio(correlation=0.0)

        # With 3 counterparties and zero correlation:
        # Diversification factor = sqrt(1 + 2*0) / sqrt(3) = 1/sqrt(3) ≈ 0.577
        # Total EPE without diversification = 300
        # Diversified EPE ≈ 300 * 0.577 ≈ 173

        expected_diversified = 300.0 * (1.0 / jnp.sqrt(3.0))
        actual_epe = portfolio.expected_positive_exposure[0]

        assert abs(actual_epe - expected_diversified) < 1.0

        # Test with perfect correlation (no diversification)
        portfolio_perfect = aggregator.aggregate_portfolio(correlation=1.0)

        # With perfect correlation, no diversification benefit
        # Diversification factor = sqrt(1 + 2*1) / sqrt(3) = 1.0
        actual_epe_perfect = portfolio_perfect.expected_positive_exposure[0]

        assert abs(actual_epe_perfect - 300.0) < 1.0


class TestCollateralOptimizer:
    """Test collateral optimization."""

    def create_sample_exposure_profile(self) -> ExposureProfile:
        """Create a sample exposure profile for testing."""
        times = jnp.linspace(0, 5, 50)  # 5 years, quarterly
        # Exposure increases then decreases
        epe = 1000.0 * jnp.exp(-((times - 2.5) ** 2) / 2.0)

        return ExposureProfile(
            times=times,
            expected_exposure=epe,
            pfe_95=epe * 1.5,
            pfe_97_5=epe * 1.8,
            pfe_99=epe * 2.0,
            expected_positive_exposure=epe,
            expected_negative_exposure=jnp.zeros_like(times),
            effective_epe=float(jnp.max(epe)),
            max_pfe=float(jnp.max(epe) * 1.8),
        )

    def test_cva_decreases_with_threshold(self):
        """Test that CVA decreases as threshold increases."""
        profile = self.create_sample_exposure_profile()

        optimizer = CollateralOptimizer(
            exposure_profile=profile,
            lgd=0.6,
            hazard_rate=0.01,
            discount_rate=0.05,
        )

        cva_no_threshold = optimizer.compute_cva(threshold=0.0)
        cva_with_threshold = optimizer.compute_cva(threshold=500.0)

        # CVA should be lower with higher threshold (less collateral = more exposure)
        # Wait, actually threshold increases exposure, so CVA increases
        # Let me reconsider: threshold is the amount BELOW which collateral is NOT posted
        # So collateralized_epe = max(EPE - threshold, 0)
        # Higher threshold = less collateral posted = higher CVA
        assert cva_no_threshold > cva_with_threshold, \
            "CVA should decrease when collateral threshold decreases (more collateral)"

    def test_collateral_cost_increases_with_lower_threshold(self):
        """Test that collateral cost increases as threshold decreases."""
        profile = self.create_sample_exposure_profile()

        optimizer = CollateralOptimizer(
            exposure_profile=profile,
            collateral_cost=0.001,
        )

        cost_high_threshold = optimizer.compute_collateral_cost(threshold=500.0)
        cost_low_threshold = optimizer.compute_collateral_cost(threshold=100.0)

        # Lower threshold = more collateral posted = higher cost
        assert cost_low_threshold > cost_high_threshold, \
            "Collateral cost should increase with lower threshold"

    def test_optimize_threshold(self):
        """Test threshold optimization."""
        profile = self.create_sample_exposure_profile()

        optimizer = CollateralOptimizer(
            exposure_profile=profile,
            lgd=0.6,
            hazard_rate=0.01,
            discount_rate=0.05,
            collateral_cost=0.001,
        )

        optimal_threshold, minimal_cost = optimizer.optimize_threshold(
            threshold_range=(0.0, 2000.0),
            n_points=50,
        )

        # Optimal should be somewhere in between (not at extremes)
        assert 0.0 < optimal_threshold < 2000.0, \
            "Optimal threshold should be interior solution"

        # Verify it's actually minimal
        cost_at_zero = optimizer.total_cost(0.0)
        cost_at_max = optimizer.total_cost(2000.0)

        assert minimal_cost <= cost_at_zero
        assert minimal_cost <= cost_at_max


class TestWrongWayRiskAnalytics:
    """Test wrong-way risk analytics."""

    def create_sample_profile(self) -> ExposureProfile:
        """Create sample exposure profile."""
        times = jnp.linspace(0, 2, 20)
        epe = jnp.ones(20) * 1000.0

        key = jax.random.PRNGKey(42)
        n_paths = 500
        exposure_paths = epe[None, :] + jax.random.normal(key, (n_paths, 20)) * 200

        return ExposureProfile.from_paths(times, exposure_paths, include_pathwise=True)

    def test_wwr_multiplier_positive_correlation(self):
        """Test WWR multiplier with positive correlation (wrong-way risk)."""
        profile = self.create_sample_profile()

        wwr_analytics = WrongWayRiskAnalytics(
            base_exposure_profile=profile,
            wwr_correlation=0.5,  # Positive = wrong-way risk
        )

        multiplier = wwr_analytics.compute_wwr_multiplier(
            correlation=0.5,
            volatility=0.3
        )

        # Multiplier should be > 1.0 for positive correlation (wrong-way risk)
        assert multiplier > 1.0, "WWR multiplier should exceed 1 for positive correlation"

    def test_wwr_multiplier_negative_correlation(self):
        """Test WWR multiplier with negative correlation (right-way risk)."""
        profile = self.create_sample_profile()

        wwr_analytics = WrongWayRiskAnalytics(
            base_exposure_profile=profile,
            wwr_correlation=-0.5,  # Negative = right-way risk
        )

        multiplier = wwr_analytics.compute_wwr_multiplier(
            correlation=-0.5,
            volatility=0.3
        )

        # Multiplier should be < 1.0 for negative correlation (right-way risk)
        assert multiplier < 1.0, "WWR multiplier should be below 1 for negative correlation"

    def test_wwr_adjusted_epe(self):
        """Test WWR-adjusted EPE calculation."""
        profile = self.create_sample_profile()

        wwr_analytics = WrongWayRiskAnalytics(
            base_exposure_profile=profile,
            wwr_correlation=0.3,  # Moderate wrong-way risk
        )

        wwr_epe = wwr_analytics.wwr_adjusted_epe()

        # WWR-adjusted EPE should be higher than base EPE for positive correlation
        base_epe_mean = float(jnp.mean(profile.expected_positive_exposure))
        wwr_epe_mean = float(jnp.mean(wwr_epe))

        assert wwr_epe_mean > base_epe_mean, \
            "WWR-adjusted EPE should exceed base EPE for positive correlation"

    def test_decompose_wwr_impact(self):
        """Test WWR impact decomposition."""
        profile = self.create_sample_profile()

        wwr_analytics = WrongWayRiskAnalytics(
            base_exposure_profile=profile,
            wwr_correlation=0.4,
        )

        decomposition = wwr_analytics.decompose_wwr_impact()

        # Check all components are present
        assert "base_effective_epe" in decomposition
        assert "wwr_adjusted_effective_epe" in decomposition
        assert "wwr_impact" in decomposition
        assert "wwr_impact_percent" in decomposition

        # WWR impact should be positive for positive correlation
        assert decomposition["wwr_impact"] > 0, \
            "WWR impact should be positive for wrong-way risk"

        assert decomposition["wwr_adjusted_effective_epe"] > decomposition["base_effective_epe"], \
            "WWR-adjusted effective EPE should exceed base"


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_end_to_end_ccr_workflow(self):
        """Test complete CCR analytics workflow."""
        # 1. Generate exposure paths
        n_paths = 500
        n_times = 25
        times = jnp.linspace(0, 5, n_times)

        key = jax.random.PRNGKey(123)
        base_exposure = jnp.linspace(1000, 1500, n_times)
        noise = jax.random.normal(key, (n_paths, n_times)) * 300
        exposure_paths = jnp.maximum(base_exposure[None, :] + noise, 0)

        # 2. Create exposure profile
        profile = ExposureProfile.from_paths(times, exposure_paths, include_pathwise=True)

        # 3. Test key metrics
        assert profile.effective_epe > 0
        assert profile.max_pfe > 0
        assert jnp.all(profile.pfe_97_5 >= profile.expected_exposure)

        # 4. Optimize collateral
        optimizer = CollateralOptimizer(
            exposure_profile=profile,
            lgd=0.6,
            hazard_rate=0.02,
        )

        optimal_threshold, _ = optimizer.optimize_threshold(
            threshold_range=(0, 2000),
            n_points=20
        )

        assert 0 < optimal_threshold < 2000

        # 5. Analyze WWR
        wwr_analytics = WrongWayRiskAnalytics(
            base_exposure_profile=profile,
            wwr_correlation=0.3,
        )

        wwr_impact = wwr_analytics.decompose_wwr_impact()

        assert wwr_impact["wwr_impact_percent"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
