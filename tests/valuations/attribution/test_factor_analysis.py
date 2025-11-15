"""Tests for Factor Analysis framework.

Tests cover:
- PCA for dimension reduction
- Factor risk models (Barra-style)
- Style attribution
- Factor timing signals
- Factor allocation optimization
"""
import pytest
import jax.numpy as jnp
import jax

from neutryx.valuations.attribution.factor_analysis import (
    AssetRiskDecomposition,
    FactorAllocation,
    FactorAllocationOptimizer,
    FactorExposure,
    FactorReturn,
    FactorRiskModel,
    FactorRiskModelEstimator,
    FactorTimingSignal,
    FactorTimingStrategy,
    IndustryFactor,
    PCAResult,
    PCTransform,
    PrincipalComponentAnalysis,
    StyleAttribution,
    StyleAttributionAnalyzer,
    StyleFactor,
)


# ==============================================================================
# PCA Tests
# ==============================================================================


class TestPrincipalComponentAnalysis:
    """Test PCA implementation."""

    def test_pca_basic(self):
        """Test basic PCA on synthetic data."""
        # Generate correlated data
        key = jax.random.PRNGKey(42)
        n_samples = 100
        n_features = 5

        # Create data with known structure
        data = jax.random.normal(key, (n_samples, n_features))
        # Add correlation structure
        data = data @ jnp.array(
            [[1.0, 0.5, 0.3, 0.1, 0.0], [0.5, 1.0, 0.4, 0.2, 0.1], [0.3, 0.4, 1.0, 0.3, 0.2], [0.1, 0.2, 0.3, 1.0, 0.5], [0.0, 0.1, 0.2, 0.5, 1.0]]
        )

        pca = PrincipalComponentAnalysis(n_components=3)
        result = pca.fit(data)

        assert isinstance(result, PCAResult)
        assert result.n_components == 3
        assert result.n_features == n_features
        assert result.principal_components.shape == (n_features, 3)
        assert result.explained_variance.shape == (3,)
        assert jnp.all(result.explained_variance >= 0)

    def test_pca_explained_variance(self):
        """Test that explained variance is computed correctly."""
        key = jax.random.PRNGKey(123)
        data = jax.random.normal(key, (50, 4))

        pca = PrincipalComponentAnalysis(n_components=4)
        result = pca.fit(data)

        # Explained variance ratio should sum to ~1.0 for all components
        total_explained = jnp.sum(result.explained_variance_ratio)
        assert 0.95 < total_explained <= 1.01  # Allow small numerical error

        # Cumulative variance should be monotonically increasing
        assert jnp.all(jnp.diff(result.cumulative_variance_ratio) >= 0)

    def test_pca_transform(self):
        """Test PCA transformation."""
        key = jax.random.PRNGKey(456)
        data = jax.random.normal(key, (30, 6))

        pca = PrincipalComponentAnalysis(n_components=3)
        result = pca.fit(data)
        transform = pca.transform(data, result)

        assert isinstance(transform, PCTransform)
        assert transform.transformed_data.shape == (30, 3)
        assert transform.reconstruction.shape == (30, 6)
        assert transform.reconstruction_error >= 0

    def test_pca_variance_threshold(self):
        """Test PCA with variance threshold."""
        key = jax.random.PRNGKey(789)
        data = jax.random.normal(key, (50, 10))

        pca = PrincipalComponentAnalysis(variance_threshold=0.90)
        result = pca.fit(data)

        # Should retain components explaining 90% of variance
        assert result.cumulative_variance_ratio[-1] >= 0.90
        assert result.n_components <= 10


# ==============================================================================
# Factor Risk Model Tests
# ==============================================================================


class TestFactorRiskModelEstimator:
    """Test factor risk model estimation."""

    def test_estimate_factor_model(self):
        """Test factor model estimation."""
        key = jax.random.PRNGKey(42)

        # Simulate data
        n_periods = 252
        n_assets = 20
        n_factors = 3

        # Factor returns
        factor_returns_true = jax.random.normal(key, (n_periods, n_factors)) * 0.01

        # Factor exposures (betas)
        exposures = jax.random.normal(jax.random.PRNGKey(43), (n_assets, n_factors))
        exposures = exposures / jnp.linalg.norm(exposures, axis=1, keepdims=True)  # Normalize

        # Generate asset returns
        returns = (exposures @ factor_returns_true.T).T
        # Add specific returns (noise)
        specific_returns = jax.random.normal(jax.random.PRNGKey(44), (n_periods, n_assets)) * 0.005
        returns = returns + specific_returns

        # Estimate model
        estimator = FactorRiskModelEstimator(estimation_window=252)
        asset_ids = [f"ASSET_{i}" for i in range(n_assets)]
        factor_names = ["FACTOR_1", "FACTOR_2", "FACTOR_3"]

        risk_model = estimator.estimate_factor_model(
            returns=returns,
            exposures=exposures,
            asset_ids=asset_ids,
            factor_names=factor_names,
            estimation_date="2024-01-01",
        )

        assert isinstance(risk_model, FactorRiskModel)
        assert risk_model.factor_covariance.shape == (n_factors, n_factors)
        assert len(risk_model.specific_variances) == n_assets
        assert risk_model.factor_names == factor_names

    def test_decompose_asset_risk(self):
        """Test asset risk decomposition."""
        key = jax.random.PRNGKey(123)

        # Create simple risk model
        n_factors = 3
        factor_covariance = jnp.eye(n_factors) * 0.0001  # Daily variance
        specific_variances = {"ASSET_1": 0.0002}
        factor_names = ["VALUE", "MOMENTUM", "SIZE"]

        risk_model = FactorRiskModel(
            factor_covariance=factor_covariance,
            specific_variances=specific_variances,
            factor_names=factor_names,
            estimation_date="2024-01-01",
            estimation_window=252,
        )

        # Asset exposures
        exposures = {"VALUE": 1.0, "MOMENTUM": 0.5, "SIZE": -0.3}

        decomp = FactorRiskModelEstimator().decompose_asset_risk(
            asset_id="ASSET_1", exposures=exposures, risk_model=risk_model
        )

        assert isinstance(decomp, AssetRiskDecomposition)
        assert decomp.total_risk > 0
        assert decomp.factor_risk > 0
        assert decomp.specific_risk > 0
        # Total variance = factor variance + specific variance
        total_var = (decomp.total_risk / jnp.sqrt(252)) ** 2
        factor_var = (decomp.factor_risk / jnp.sqrt(252)) ** 2
        specific_var = (decomp.specific_risk / jnp.sqrt(252)) ** 2
        assert jnp.abs(total_var - (factor_var + specific_var)) < 1e-6

    def test_factor_contributions(self):
        """Test factor risk contributions."""
        # Simple 2-factor model
        factor_covariance = jnp.array([[0.0001, 0.0], [0.0, 0.0001]])
        specific_variances = {"ASSET_1": 0.0001}
        factor_names = ["FACTOR_A", "FACTOR_B"]

        risk_model = FactorRiskModel(
            factor_covariance=factor_covariance,
            specific_variances=specific_variances,
            factor_names=factor_names,
            estimation_date="2024-01-01",
            estimation_window=252,
        )

        # Equal exposures
        exposures = {"FACTOR_A": 1.0, "FACTOR_B": 1.0}

        decomp = FactorRiskModelEstimator().decompose_asset_risk(
            asset_id="ASSET_1", exposures=exposures, risk_model=risk_model
        )

        # Both factors should contribute equally
        contrib_a = decomp.factor_contributions["FACTOR_A"]
        contrib_b = decomp.factor_contributions["FACTOR_B"]
        assert jnp.abs(contrib_a - contrib_b) < 1e-6


# ==============================================================================
# Style Attribution Tests
# ==============================================================================


class TestStyleAttributionAnalyzer:
    """Test style attribution."""

    def test_attribute_performance(self):
        """Test performance attribution by style factors."""
        analyzer = StyleAttributionAnalyzer()

        # Portfolio return
        portfolio_return = 0.15  # 15%

        # Portfolio exposures
        portfolio_exposures = {
            StyleFactor.VALUE: 0.8,
            StyleFactor.MOMENTUM: 0.3,
            StyleFactor.SIZE: -0.2,
            StyleFactor.QUALITY: 0.5,
        }

        # Factor returns
        factor_returns = {
            StyleFactor.VALUE: 0.10,  # 10% return to value
            StyleFactor.MOMENTUM: 0.05,
            StyleFactor.SIZE: -0.02,
            StyleFactor.QUALITY: 0.08,
        }

        attribution = analyzer.attribute_performance(
            portfolio_return=portfolio_return,
            portfolio_exposures=portfolio_exposures,
            factor_returns=factor_returns,
            period_start="2024-01-01",
            period_end="2024-12-31",
        )

        assert isinstance(attribution, StyleAttribution)
        assert attribution.total_return == portfolio_return

        # Check factor contributions
        value_contribution = attribution.factor_returns[StyleFactor.VALUE]
        assert jnp.abs(value_contribution - 0.8 * 0.10) < 1e-6  # exposure × return

        # Total = factor contributions + specific
        total_factor = sum(attribution.factor_returns.values())
        assert jnp.abs(attribution.total_return - (total_factor + attribution.specific_return)) < 1e-6

    def test_alpha_calculation(self):
        """Test alpha (specific return) calculation."""
        analyzer = StyleAttributionAnalyzer()

        # Simple case: no factor exposures
        attribution = analyzer.attribute_performance(
            portfolio_return=0.12,
            portfolio_exposures={},
            factor_returns={},
            period_start="2024-01-01",
            period_end="2024-12-31",
        )

        # All return should be alpha
        assert jnp.abs(attribution.specific_return - 0.12) < 1e-6
        assert sum(attribution.factor_returns.values()) == 0.0


# ==============================================================================
# Factor Timing Tests
# ==============================================================================


class TestFactorTimingStrategy:
    """Test factor timing signals."""

    def test_generate_timing_signal(self):
        """Test timing signal generation."""
        strategy = FactorTimingStrategy(lookback_window=63, momentum_window=126)

        # Simulate factor returns with positive momentum
        key = jax.random.PRNGKey(42)
        factor_returns_history = jax.random.normal(key, (200,)) * 0.01 + 0.0005  # Positive drift

        market_regime_indicators = {"vix": 18.0, "credit_spread": 1.5}

        signal = strategy.generate_timing_signal(
            factor=StyleFactor.MOMENTUM,
            factor_returns_history=factor_returns_history,
            market_regime_indicators=market_regime_indicators,
            date="2024-01-01",
        )

        assert isinstance(signal, FactorTimingSignal)
        assert signal.factor == StyleFactor.MOMENTUM
        assert -3.0 <= signal.signal_value <= 3.0
        assert 0.0 <= signal.confidence <= 1.0
        assert signal.regime in ["risk_on", "risk_off", "neutral"]

    def test_regime_classification(self):
        """Test market regime classification."""
        strategy = FactorTimingStrategy()
        key = jax.random.PRNGKey(123)
        factor_returns = jax.random.normal(key, (200,)) * 0.01

        # Risk-on regime (low VIX)
        signal_risk_on = strategy.generate_timing_signal(
            factor=StyleFactor.VALUE,
            factor_returns_history=factor_returns,
            market_regime_indicators={"vix": 12.0},
            date="2024-01-01",
        )
        assert signal_risk_on.regime == "risk_on"

        # Risk-off regime (high VIX)
        signal_risk_off = strategy.generate_timing_signal(
            factor=StyleFactor.VALUE,
            factor_returns_history=factor_returns,
            market_regime_indicators={"vix": 30.0},
            date="2024-01-01",
        )
        assert signal_risk_off.regime == "risk_off"

        # Neutral regime
        signal_neutral = strategy.generate_timing_signal(
            factor=StyleFactor.VALUE,
            factor_returns_history=factor_returns,
            market_regime_indicators={"vix": 20.0},
            date="2024-01-01",
        )
        assert signal_neutral.regime == "neutral"


# ==============================================================================
# Factor Allocation Tests
# ==============================================================================


class TestFactorAllocationOptimizer:
    """Test factor allocation optimization."""

    def test_optimize_mean_variance(self):
        """Test mean-variance factor allocation."""
        optimizer = FactorAllocationOptimizer(risk_aversion=2.5)

        # Expected returns
        expected_returns = {
            StyleFactor.VALUE: 0.08,
            StyleFactor.MOMENTUM: 0.06,
            StyleFactor.QUALITY: 0.05,
        }

        # Factor covariance (3x3)
        covariance_matrix = jnp.array(
            [[0.04, 0.01, 0.005], [0.01, 0.03, 0.008], [0.005, 0.008, 0.025]]
        )

        factor_order = [StyleFactor.VALUE, StyleFactor.MOMENTUM, StyleFactor.QUALITY]

        allocation = optimizer.optimize_mean_variance(
            expected_returns=expected_returns,
            covariance_matrix=covariance_matrix,
            factor_order=factor_order,
            date="2024-01-01",
        )

        assert isinstance(allocation, FactorAllocation)
        assert allocation.optimization_method == "mean_variance"

        # Weights should sum to 1
        total_weight = sum(allocation.factor_weights.values())
        assert jnp.abs(total_weight - 1.0) < 1e-6

        # Should have non-zero expected return and volatility
        assert allocation.expected_return != 0.0
        assert allocation.expected_volatility > 0.0

    def test_optimize_risk_parity(self):
        """Test risk parity allocation."""
        optimizer = FactorAllocationOptimizer()

        # Factor covariance (3x3)
        covariance_matrix = jnp.array(
            [[0.04, 0.01, 0.005], [0.01, 0.03, 0.008], [0.005, 0.008, 0.025]]
        )

        factor_order = [StyleFactor.VALUE, StyleFactor.MOMENTUM, StyleFactor.QUALITY]

        allocation = optimizer.optimize_risk_parity(
            covariance_matrix=covariance_matrix, factor_order=factor_order, date="2024-01-01"
        )

        assert isinstance(allocation, FactorAllocation)
        assert allocation.optimization_method == "risk_parity"

        # Weights should sum to 1
        total_weight = sum(allocation.factor_weights.values())
        assert jnp.abs(total_weight - 1.0) < 1e-6

        # Higher volatility factors should have lower weights
        factor_vols = jnp.sqrt(jnp.diag(covariance_matrix))
        weights = jnp.array(
            [allocation.factor_weights[f] for f in factor_order]
        )

        # Verify inverse relationship (approximately)
        expected_weights = 1.0 / factor_vols
        expected_weights = expected_weights / jnp.sum(expected_weights)
        assert jnp.allclose(weights, expected_weights, rtol=0.01)

    def test_weight_constraints(self):
        """Test weight constraints in optimization."""
        optimizer = FactorAllocationOptimizer(risk_aversion=2.5)

        expected_returns = {
            StyleFactor.VALUE: 0.10,
            StyleFactor.MOMENTUM: 0.08,
        }

        covariance_matrix = jnp.array([[0.04, 0.01], [0.01, 0.03]])

        factor_order = [StyleFactor.VALUE, StyleFactor.MOMENTUM]

        # Apply max weight constraint
        allocation = optimizer.optimize_mean_variance(
            expected_returns=expected_returns,
            covariance_matrix=covariance_matrix,
            factor_order=factor_order,
            date="2024-01-01",
            constraints={"max_weight": 0.6},
        )

        # All weights should respect constraint
        for weight in allocation.factor_weights.values():
            assert abs(weight) <= 0.6


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestFactorAnalysisIntegration:
    """Test integrated factor analysis workflows."""

    def test_full_factor_workflow(self):
        """Test complete factor analysis workflow."""
        key = jax.random.PRNGKey(42)

        # 1. Generate synthetic returns data
        n_periods = 252
        n_assets = 30
        n_factors = 4

        factor_returns = jax.random.normal(key, (n_periods, n_factors)) * 0.01
        exposures = jax.random.normal(jax.random.PRNGKey(43), (n_assets, n_factors))
        exposures = exposures / jnp.linalg.norm(exposures, axis=1, keepdims=True)

        returns = (exposures @ factor_returns.T).T
        specific = jax.random.normal(jax.random.PRNGKey(44), (n_periods, n_assets)) * 0.005
        returns = returns + specific

        # 2. Estimate factor model
        estimator = FactorRiskModelEstimator()
        asset_ids = [f"ASSET_{i}" for i in range(n_assets)]
        factor_names = ["VALUE", "MOMENTUM", "SIZE", "QUALITY"]

        risk_model = estimator.estimate_factor_model(
            returns=returns,
            exposures=exposures,
            asset_ids=asset_ids,
            factor_names=factor_names,
            estimation_date="2024-12-31",
        )

        # 3. Decompose asset risk
        asset_exposures = {factor_names[i]: float(exposures[0, i]) for i in range(n_factors)}
        decomp = estimator.decompose_asset_risk(
            asset_id=asset_ids[0], exposures=asset_exposures, risk_model=risk_model
        )

        # Verify risk decomposition
        assert decomp.total_risk > 0
        assert decomp.factor_risk >= 0
        assert decomp.specific_risk >= 0

        # 4. Factor timing
        timing_strategy = FactorTimingStrategy()
        signal = timing_strategy.generate_timing_signal(
            factor=StyleFactor.VALUE,
            factor_returns_history=returns[:, 0],
            market_regime_indicators={"vix": 18.0},
            date="2024-12-31",
        )

        assert isinstance(signal, FactorTimingSignal)

        # 5. Factor allocation
        expected_returns_dict = {StyleFactor(f.lower()): 0.08 for f in factor_names if f.lower() in [f.value for f in StyleFactor]}
        # Use subset of factors that exist in StyleFactor enum
        valid_factors = [StyleFactor.VALUE, StyleFactor.MOMENTUM, StyleFactor.SIZE, StyleFactor.QUALITY]
        expected_returns_dict = {f: 0.08 for f in valid_factors}

        optimizer = FactorAllocationOptimizer()
        allocation = optimizer.optimize_risk_parity(
            covariance_matrix=risk_model.factor_covariance,
            factor_order=valid_factors,
            date="2024-12-31",
        )

        assert isinstance(allocation, FactorAllocation)
        assert sum(allocation.factor_weights.values()) > 0.99  # ~1.0

    def test_pca_factor_model_integration(self):
        """Test PCA for factor dimension reduction."""
        key = jax.random.PRNGKey(789)

        # Generate high-dimensional returns with factor structure
        n_periods = 100
        n_assets = 50
        n_true_factors = 5

        # Create structured returns (not purely random)
        true_factors = jax.random.normal(key, (n_periods, n_true_factors)) * 0.015
        loadings = jax.random.normal(jax.random.PRNGKey(790), (n_assets, n_true_factors))
        returns = (loadings @ true_factors.T).T
        # Add small idiosyncratic noise
        returns = returns + jax.random.normal(jax.random.PRNGKey(791), (n_periods, n_assets)) * 0.003

        # Apply PCA to reduce dimensions
        pca = PrincipalComponentAnalysis(n_components=5)
        pca_result = pca.fit(returns)

        # Should capture meaningful structure (with factor structure, should capture >50%)
        assert pca_result.n_components == 5
        assert pca_result.cumulative_variance_ratio[-1] > 0.5  # At least 50% variance

        # Can use principal components as factors
        principal_factors = pca_result.principal_components

        # Verify orthogonality of principal components
        assert jnp.allclose(
            principal_factors.T @ principal_factors, jnp.eye(5), atol=1e-5
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
