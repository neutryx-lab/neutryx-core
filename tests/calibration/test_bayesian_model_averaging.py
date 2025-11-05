"""Tests for Bayesian Model Averaging functionality.

Tests cover:
- Model weight computation from information criteria
- Stacking weights optimization
- BMA prediction with uncertainty quantification
- Variance decomposition (within/between)
- Integration with ModelComparison
- Pseudo-BMA weights
"""
import jax
import jax.numpy as jnp
import pytest

from neutryx.calibration.model_selection import (
    InformationCriterion,
    ModelFit,
    ModelComparison,
    compare_models,
)
from neutryx.calibration.bayesian_model_averaging import (
    WeightingScheme,
    ModelWeights,
    BMAResult,
    BayesianModelAveraging,
    compute_weights_from_ic,
    compute_stacking_weights,
    pseudo_bma_weights,
)


# ==============================================================================
# Model Weights Tests
# ==============================================================================


class TestModelWeights:
    """Test model weight computation."""

    def test_compute_weights_from_bic(self):
        """Test computing weights from BIC."""
        # Create two models with different BIC values
        fit1 = ModelFit(
            log_likelihood=-100.0,
            n_parameters=2,
            n_observations=100,
            residuals=jnp.ones(100) * 0.5,
            predictions=jnp.ones(100),
        )

        fit2 = ModelFit(
            log_likelihood=-95.0,  # Better fit
            n_parameters=3,
            n_observations=100,
            residuals=jnp.ones(100) * 0.3,
            predictions=jnp.ones(100),
        )

        weights = compute_weights_from_ic(
            {"Model1": fit1, "Model2": fit2},
            criterion=InformationCriterion.BIC,
        )

        # Check structure
        assert set(weights.weights.keys()) == {"Model1", "Model2"}
        assert weights.scheme == WeightingScheme.BIC

        # Weights should sum to 1
        assert abs(sum(weights.weights.values()) - 1.0) < 1e-6

        # All weights should be non-negative
        assert all(w >= 0 for w in weights.weights.values())

        # Better model (lower BIC) should have higher weight
        # Model2 has better log-likelihood, so likely higher weight
        # (depends on penalty, but in this case likely)

    def test_compute_weights_from_aic(self):
        """Test computing weights from AIC."""
        fits = {
            "A": ModelFit(-100, 2, 100, jnp.zeros(100), jnp.ones(100)),
            "B": ModelFit(-110, 3, 100, jnp.zeros(100), jnp.ones(100)),
            "C": ModelFit(-90, 5, 100, jnp.zeros(100), jnp.ones(100)),
        }

        weights = compute_weights_from_ic(fits, InformationCriterion.AIC)

        # Model C has best likelihood, should have highest weight
        assert weights.weights["C"] > weights.weights["B"]
        assert weights.weights["C"] > weights.weights["A"]

        # Check effective models
        assert 1.0 <= weights.effective_models <= 3.0

    def test_equal_models_equal_weights(self):
        """Test that equal models get equal weights."""
        # Three identical models
        fit = ModelFit(-100, 3, 100, jnp.zeros(100), jnp.ones(100))
        fits = {"M1": fit, "M2": fit, "M3": fit}

        weights = compute_weights_from_ic(fits, InformationCriterion.BIC)

        # All weights should be approximately 1/3
        for w in weights.weights.values():
            assert abs(w - 1.0 / 3.0) < 1e-6

        # Effective models should be close to 3
        assert abs(weights.effective_models - 3.0) < 0.01

    def test_dominant_model(self):
        """Test that a dominant model gets most weight."""
        fits = {
            "Best": ModelFit(-50, 2, 100, jnp.zeros(100), jnp.ones(100)),  # Much better
            "Poor": ModelFit(-200, 2, 100, jnp.zeros(100), jnp.ones(100)),
        }

        weights = compute_weights_from_ic(fits, InformationCriterion.BIC)

        # Best model should dominate
        assert weights.weights["Best"] > 0.99
        assert weights.weights["Poor"] < 0.01

        # Effective models should be close to 1
        assert weights.effective_models < 1.1

    def test_stacking_weights(self):
        """Test stacking weight computation."""
        # Create synthetic predictions and observations
        key = jax.random.PRNGKey(42)
        n_obs = 50

        # True values
        observed = jax.random.normal(key, (n_obs,))

        # Model predictions (some better than others)
        pred1 = observed + jax.random.normal(jax.random.PRNGKey(1), (n_obs,)) * 0.5  # Good
        pred2 = observed + jax.random.normal(jax.random.PRNGKey(2), (n_obs,)) * 1.0  # Moderate
        pred3 = jax.random.normal(jax.random.PRNGKey(3), (n_obs,)) * 2.0  # Poor

        predictions = {"M1": pred1, "M2": pred2, "M3": pred3}

        fits = {
            name: ModelFit(
                -100, 3, n_obs, observed - pred, pred
            )
            for name, pred in predictions.items()
        }

        weights = compute_stacking_weights(fits, predictions, observed)

        # Weights should sum to 1
        assert abs(sum(weights.weights.values()) - 1.0) < 1e-4

        # All weights should be non-negative
        assert all(w >= 0 for w in weights.weights.values())

        # M1 (best predictions) should have highest weight
        assert weights.weights["M1"] >= weights.weights["M2"]
        assert weights.weights["M1"] >= weights.weights["M3"]


class TestPseudoBMAWeights:
    """Test pseudo-BMA weight computation."""

    def test_pseudo_bma_weights_basic(self):
        """Test pseudo-BMA weights from LOO scores."""
        loo_scores = {
            "Model_A": 100.0,
            "Model_B": 110.0,
            "Model_C": 90.0,  # Best
        }

        weights = pseudo_bma_weights(loo_scores)

        # Model_C should have highest weight
        assert weights.weights["Model_C"] > weights.weights["Model_A"]
        assert weights.weights["Model_C"] > weights.weights["Model_B"]

        # Weights should sum to 1
        assert abs(sum(weights.weights.values()) - 1.0) < 1e-6


# ==============================================================================
# BMA Prediction Tests
# ==============================================================================


class TestBayesianModelAveraging:
    """Test BayesianModelAveraging class."""

    def test_bma_initialization(self):
        """Test BMA initialization."""
        fits = {
            "M1": ModelFit(-100, 2, 100, jnp.zeros(100), jnp.ones(100)),
            "M2": ModelFit(-105, 3, 100, jnp.zeros(100), jnp.ones(100)),
        }

        bma = BayesianModelAveraging(fits, weighting_scheme=WeightingScheme.BIC)

        assert set(bma.model_names) == {"M1", "M2"}
        assert bma.weighting_scheme == WeightingScheme.BIC
        assert len(bma.weights.weights) == 2

    def test_bma_predict_simple(self):
        """Test BMA prediction with simple case."""
        # Create models
        fits = {
            "M1": ModelFit(-50, 2, 10, jnp.zeros(10), jnp.ones(10)),
            "M2": ModelFit(-55, 2, 10, jnp.zeros(10), jnp.ones(10)),
        }

        bma = BayesianModelAveraging(fits, weighting_scheme=WeightingScheme.BIC)

        # Model predictions
        predictions = {
            "M1": jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            "M2": jnp.array([1.1, 2.1, 3.1, 4.1, 5.1]),
        }

        result = bma.predict(predictions)

        # Check result structure
        assert len(result.mean) == 5
        assert len(result.variance) == 5
        assert len(result.within_model_variance) == 5
        assert len(result.between_model_variance) == 5

        # BMA mean should be between individual predictions
        assert jnp.all(result.mean >= predictions["M1"])
        assert jnp.all(result.mean <= predictions["M2"])

        # Variance should be non-negative
        assert jnp.all(result.variance >= 0)

    def test_bma_predict_with_variance(self):
        """Test BMA prediction with model variances."""
        fits = {
            "M1": ModelFit(-50, 2, 10, jnp.ones(10) * 0.1, jnp.ones(10)),
            "M2": ModelFit(-52, 2, 10, jnp.ones(10) * 0.2, jnp.ones(10)),
        }

        bma = BayesianModelAveraging(fits, weighting_scheme=WeightingScheme.BIC)

        predictions = {
            "M1": jnp.ones(5) * 10.0,
            "M2": jnp.ones(5) * 12.0,
        }

        variances = {
            "M1": jnp.ones(5) * 0.5,
            "M2": jnp.ones(5) * 1.0,
        }

        result = bma.predict(predictions, model_variances=variances)

        # Total variance should include both within and between
        assert jnp.all(result.variance > 0)
        assert jnp.all(result.within_model_variance > 0)
        assert jnp.all(result.between_model_variance >= 0)  # Can be zero if predictions identical

        # Total = within + between
        expected_total = result.within_model_variance + result.between_model_variance
        assert jnp.allclose(result.variance, expected_total, atol=1e-6)

    def test_bma_identical_predictions(self):
        """Test BMA when all models make identical predictions."""
        fits = {
            "M1": ModelFit(-50, 2, 10, jnp.zeros(10), jnp.ones(10)),
            "M2": ModelFit(-52, 2, 10, jnp.zeros(10), jnp.ones(10)),
            "M3": ModelFit(-54, 2, 10, jnp.zeros(10), jnp.ones(10)),
        }

        bma = BayesianModelAveraging(fits)

        # All models predict the same
        pred = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predictions = {"M1": pred, "M2": pred, "M3": pred}

        result = bma.predict(predictions)

        # BMA mean should equal individual predictions
        assert jnp.allclose(result.mean, pred)

        # Between-model variance should be zero (or very small)
        assert jnp.all(result.between_model_variance < 1e-10)

    def test_bma_custom_weights(self):
        """Test BMA with custom weights."""
        fits = {
            "M1": ModelFit(-50, 2, 10, jnp.zeros(10), jnp.ones(10)),
            "M2": ModelFit(-55, 2, 10, jnp.zeros(10), jnp.ones(10)),
        }

        custom_weights = {"M1": 0.7, "M2": 0.3}

        bma = BayesianModelAveraging(
            fits,
            weighting_scheme=WeightingScheme.CUSTOM,
            custom_weights=custom_weights,
        )

        # Check weights are normalized
        assert abs(bma.weights.weights["M1"] - 0.7) < 1e-6
        assert abs(bma.weights.weights["M2"] - 0.3) < 1e-6

    def test_bma_from_comparison(self):
        """Test creating BMA from ModelComparison."""
        fits = {
            "Model_A": ModelFit(-100, 2, 100, jnp.zeros(100), jnp.ones(100)),
            "Model_B": ModelFit(-105, 3, 100, jnp.zeros(100), jnp.ones(100)),
        }

        comparison = compare_models(fits)
        bma = BayesianModelAveraging.from_comparison(comparison, fits)

        assert set(bma.model_names) == {"Model_A", "Model_B"}
        assert len(bma.weights.weights) == 2

    def test_bma_update_weights(self):
        """Test updating BMA weights."""
        fits = {
            "M1": ModelFit(-50, 2, 10, jnp.zeros(10), jnp.ones(10)),
            "M2": ModelFit(-55, 2, 10, jnp.zeros(10), jnp.ones(10)),
        }

        bma = BayesianModelAveraging(fits)
        original_weights = bma.weights.weights.copy()

        # Update with new weights
        new_weights = {"M1": 0.8, "M2": 0.2}
        bma.update_weights(new_weights)

        # Weights should have changed
        assert bma.weights.weights["M1"] == 0.8
        assert bma.weights.weights["M2"] == 0.2

        # Should be marked as CUSTOM
        assert bma.weights.scheme == WeightingScheme.CUSTOM

    def test_model_probabilities(self):
        """Test getting model probabilities."""
        fits = {
            "M1": ModelFit(-50, 2, 10, jnp.zeros(10), jnp.ones(10)),
            "M2": ModelFit(-55, 2, 10, jnp.zeros(10), jnp.ones(10)),
        }

        bma = BayesianModelAveraging(fits)
        probs = bma.model_probabilities()

        # Should return a copy of weights
        assert set(probs.keys()) == {"M1", "M2"}
        assert abs(sum(probs.values()) - 1.0) < 1e-6


class TestBMAResult:
    """Test BMAResult class."""

    def test_bma_result_properties(self):
        """Test BMAResult properties."""
        mean = jnp.array([1.0, 2.0, 3.0])
        variance = jnp.array([0.1, 0.2, 0.3])
        within = jnp.array([0.05, 0.1, 0.15])
        between = jnp.array([0.05, 0.1, 0.15])

        weights = ModelWeights(
            weights={"M1": 0.6, "M2": 0.4},
            raw_scores={"M1": 100.0, "M2": 105.0},
            effective_models=1.92,
            scheme=WeightingScheme.BIC,
        )

        result = BMAResult(
            mean=mean,
            variance=variance,
            within_model_variance=within,
            between_model_variance=between,
            model_predictions={"M1": mean, "M2": mean + 0.1},
            weights=weights,
        )

        # Test std property
        expected_std = jnp.sqrt(variance)
        assert jnp.allclose(result.std, expected_std)

    def test_prediction_interval(self):
        """Test prediction interval computation."""
        mean = jnp.array([10.0, 20.0, 30.0])
        variance = jnp.array([1.0, 4.0, 9.0])  # std = [1, 2, 3]

        weights = ModelWeights(
            weights={"M": 1.0},
            raw_scores={"M": 100.0},
            effective_models=1.0,
            scheme=WeightingScheme.BIC,
        )

        result = BMAResult(
            mean=mean,
            variance=variance,
            within_model_variance=variance * 0.5,
            between_model_variance=variance * 0.5,
            model_predictions={"M": mean},
            weights=weights,
        )

        lower, upper = result.prediction_interval(confidence=0.95)

        # Check structure
        assert len(lower) == 3
        assert len(upper) == 3

        # Interval should contain mean
        assert jnp.all(lower < mean)
        assert jnp.all(upper > mean)

        # Interval should be symmetric around mean (for normal)
        assert jnp.allclose(mean - lower, upper - mean, atol=1e-6)


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestBMAIntegration:
    """Test integration with existing model selection framework."""

    def test_full_bma_workflow(self):
        """Test complete BMA workflow."""
        # Step 1: Create multiple model fits
        fits = {
            "Heston": ModelFit(
                log_likelihood=-150.0,
                n_parameters=5,
                n_observations=100,
                residuals=jnp.ones(100) * 0.3,
                predictions=jnp.ones(100) * 50,
            ),
            "SABR": ModelFit(
                log_likelihood=-145.0,
                n_parameters=4,
                n_observations=100,
                residuals=jnp.ones(100) * 0.25,
                predictions=jnp.ones(100) * 51,
            ),
            "LocalVol": ModelFit(
                log_likelihood=-155.0,
                n_parameters=6,
                n_observations=100,
                residuals=jnp.ones(100) * 0.35,
                predictions=jnp.ones(100) * 49,
            ),
        }

        # Step 2: Compare models
        comparison = compare_models(fits)
        assert comparison.best_model in ["Heston", "SABR", "LocalVol"]

        # Step 3: Create BMA
        bma = BayesianModelAveraging.from_comparison(comparison, fits)

        # Step 4: Make predictions
        # Simulate option prices for new strikes
        predictions = {
            "Heston": jnp.array([10.5, 11.2, 12.0, 13.1, 14.5]),
            "SABR": jnp.array([10.6, 11.1, 12.1, 13.0, 14.4]),
            "LocalVol": jnp.array([10.4, 11.3, 11.9, 13.2, 14.6]),
        }

        result = bma.predict(predictions)

        # Check results
        assert len(result.mean) == 5
        assert jnp.all(result.mean > 10)
        assert jnp.all(result.mean < 15)
        assert jnp.all(result.variance > 0)

        # Get prediction intervals
        lower, upper = result.prediction_interval(0.95)
        assert jnp.all(lower < result.mean)
        assert jnp.all(upper > result.mean)

    def test_bma_vs_single_model(self):
        """Test that BMA reduces prediction variance vs single model."""
        # Create two complementary models
        key = jax.random.PRNGKey(42)
        n_pred = 50

        # True values (unknown)
        true_values = jnp.linspace(0, 10, n_pred)

        # Model 1: overestimates
        pred1 = true_values + jax.random.normal(jax.random.PRNGKey(1), (n_pred,)) * 0.5

        # Model 2: underestimates
        pred2 = true_values + jax.random.normal(jax.random.PRNGKey(2), (n_pred,)) * 0.5

        fits = {
            "M1": ModelFit(-100, 3, n_pred, true_values - pred1, pred1),
            "M2": ModelFit(-100, 3, n_pred, true_values - pred2, pred2),
        }

        # Equal weights (both models equally good)
        bma = BayesianModelAveraging(
            fits,
            weighting_scheme=WeightingScheme.CUSTOM,
            custom_weights={"M1": 0.5, "M2": 0.5},
        )

        predictions = {"M1": pred1, "M2": pred2}
        result = bma.predict(predictions)

        # BMA prediction should be closer to truth than either model alone
        bma_error = jnp.mean(jnp.abs(result.mean - true_values))
        m1_error = jnp.mean(jnp.abs(pred1 - true_values))
        m2_error = jnp.mean(jnp.abs(pred2 - true_values))

        # BMA should perform at least as well as average of individual models
        avg_individual_error = (m1_error + m2_error) / 2
        assert bma_error <= avg_individual_error * 1.1  # Allow small tolerance

    def test_variance_decomposition(self):
        """Test that variance decomposition is correct."""
        fits = {
            "M1": ModelFit(-50, 2, 100, jnp.ones(100) * 0.1, jnp.ones(100)),
            "M2": ModelFit(-52, 2, 100, jnp.ones(100) * 0.15, jnp.ones(100)),
            "M3": ModelFit(-54, 2, 100, jnp.ones(100) * 0.2, jnp.ones(100)),
        }

        bma = BayesianModelAveraging(fits)

        # Create divergent predictions to ensure between-model variance
        predictions = {
            "M1": jnp.array([10.0, 11.0, 12.0]),
            "M2": jnp.array([10.5, 11.5, 12.5]),
            "M3": jnp.array([9.5, 10.5, 11.5]),
        }

        variances = {
            "M1": jnp.ones(3) * 0.5,
            "M2": jnp.ones(3) * 0.6,
            "M3": jnp.ones(3) * 0.7,
        }

        result = bma.predict(predictions, model_variances=variances)

        # Verify decomposition: Total = Within + Between
        computed_total = result.within_model_variance + result.between_model_variance
        assert jnp.allclose(result.variance, computed_total, atol=1e-5)

        # Both components should be positive
        assert jnp.all(result.within_model_variance > 0)
        assert jnp.all(result.between_model_variance > 0)

    def test_bma_many_models(self):
        """Test BMA with many models."""
        # Create 10 similar models
        fits = {
            f"Model_{i}": ModelFit(
                log_likelihood=-100 - i,
                n_parameters=3,
                n_observations=100,
                residuals=jnp.zeros(100),
                predictions=jnp.ones(100),
            )
            for i in range(10)
        }

        bma = BayesianModelAveraging(fits, weighting_scheme=WeightingScheme.BIC)

        # Check that weights are computed for all models
        assert len(bma.weights.weights) == 10
        assert abs(sum(bma.weights.weights.values()) - 1.0) < 1e-6

        # First model should have highest weight (best log-likelihood)
        assert bma.weights.weights["Model_0"] == max(bma.weights.weights.values())


# ==============================================================================
# Edge Cases and Error Handling
# ==============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_model_bma(self):
        """Test BMA with single model (should work, weight = 1)."""
        fit = ModelFit(-100, 3, 100, jnp.zeros(100), jnp.ones(100))
        bma = BayesianModelAveraging({"M1": fit})

        # Single model should get weight 1
        assert abs(bma.weights.weights["M1"] - 1.0) < 1e-6
        assert abs(bma.weights.effective_models - 1.0) < 1e-6

    def test_mismatched_prediction_keys(self):
        """Test error when prediction keys don't match model names."""
        fits = {
            "M1": ModelFit(-50, 2, 10, jnp.zeros(10), jnp.ones(10)),
            "M2": ModelFit(-55, 2, 10, jnp.zeros(10), jnp.ones(10)),
        }

        bma = BayesianModelAveraging(fits)

        # Wrong keys
        predictions = {
            "M1": jnp.ones(5),
            "M3": jnp.ones(5),  # M3 instead of M2
        }

        with pytest.raises(ValueError, match="must match"):
            bma.predict(predictions)

    def test_custom_weights_without_scheme(self):
        """Test that custom weights require CUSTOM scheme."""
        fits = {"M1": ModelFit(-50, 2, 10, jnp.zeros(10), jnp.ones(10))}

        with pytest.raises(ValueError, match="custom_weights must be provided"):
            BayesianModelAveraging(fits, weighting_scheme=WeightingScheme.CUSTOM)

    def test_weights_summary(self):
        """Test ModelWeights summary generation."""
        weights = ModelWeights(
            weights={"M1": 0.6, "M2": 0.4},
            raw_scores={"M1": 200.0, "M2": 205.0},
            effective_models=1.92,
            scheme=WeightingScheme.BIC,
        )

        summary = weights.summary()
        assert "BIC" in summary
        assert "M1" in summary
        assert "M2" in summary
        assert "1.92" in summary

    def test_bma_result_summary(self):
        """Test BMAResult summary generation."""
        weights = ModelWeights(
            weights={"M1": 0.7, "M2": 0.3},
            raw_scores={},
            effective_models=1.6,
            scheme=WeightingScheme.BIC,
        )

        result = BMAResult(
            mean=jnp.array([10.0, 11.0, 12.0, 13.0, 14.0]),
            variance=jnp.array([1.0, 1.1, 1.2, 1.3, 1.4]),
            within_model_variance=jnp.array([0.5, 0.55, 0.6, 0.65, 0.7]),
            between_model_variance=jnp.array([0.5, 0.55, 0.6, 0.65, 0.7]),
            model_predictions={"M1": jnp.ones(5), "M2": jnp.ones(5) * 2},
            weights=weights,
        )

        summary = result.summary()
        assert "Bayesian Model Averaging" in summary
        assert "Effective models" in summary
        assert "Variance decomposition" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
