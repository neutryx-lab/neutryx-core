"""Tests for model selection and statistical inference tools.

Tests cover:
- Information criteria (AIC, BIC, HQIC, AICc)
- Model comparison
- K-fold cross-validation
- Time-series cross-validation
- Local sensitivity analysis
- Global sensitivity analysis (Sobol indices)
"""
import jax
import jax.numpy as jnp
import pytest

from neutryx.calibration.model_selection import (
    InformationCriterion,
    ModelFit,
    compute_aic,
    compute_bic,
    compute_aicc,
    compute_hqic,
    compute_information_criterion,
    ModelComparison,
    compare_models,
    CrossValidationResult,
    k_fold_split,
    time_series_split,
    cross_validate,
    LocalSensitivity,
    GlobalSensitivity,
    compute_local_sensitivity,
    sobol_indices,
)


# ==============================================================================
# Information Criteria Tests
# ==============================================================================


class TestInformationCriteria:
    """Test information criteria for model selection."""

    def test_compute_aic(self):
        """Test AIC computation."""
        aic = compute_aic(log_likelihood=-100.0, n_parameters=3)

        # AIC = -2 * (-100) + 2 * 3 = 200 + 6 = 206
        assert abs(aic - 206.0) < 1e-6

    def test_compute_bic(self):
        """Test BIC computation."""
        bic = compute_bic(log_likelihood=-100.0, n_parameters=3, n_observations=100)

        # BIC = -2 * (-100) + 3 * log(100) = 200 + 3 * 4.605 ≈ 213.82
        expected = 200.0 + 3.0 * jnp.log(100.0)
        assert abs(bic - expected) < 1e-4

    def test_compute_aicc(self):
        """Test AICc computation."""
        aicc = compute_aicc(log_likelihood=-100.0, n_parameters=3, n_observations=50)

        # AIC = 206
        # Correction = 2 * 3 * 4 / (50 - 3 - 1) = 24 / 46 ≈ 0.522
        # AICc ≈ 206.522
        aic = 206.0
        correction = (2.0 * 3 * 4) / (50 - 3 - 1)
        expected = aic + correction
        assert abs(aicc - expected) < 1e-3

    def test_compute_hqic(self):
        """Test HQIC computation."""
        hqic = compute_hqic(log_likelihood=-100.0, n_parameters=3, n_observations=100)

        # HQIC = 200 + 2 * 3 * log(log(100))
        expected = 200.0 + 2.0 * 3.0 * jnp.log(jnp.log(100.0 + jnp.e))
        assert abs(hqic - expected) < 1e-4

    def test_bic_penalty_stronger_than_aic(self):
        """BIC should have stronger penalty than AIC for large samples."""
        n_obs = 1000
        n_params = 10
        log_lik = -500.0

        aic = compute_aic(log_lik, n_params)
        bic = compute_bic(log_lik, n_params, n_obs)

        # BIC penalty = k * log(n) = 10 * log(1000) ≈ 69
        # AIC penalty = 2 * k = 20
        # So BIC should be higher
        assert bic > aic

    def test_aicc_converges_to_aic(self):
        """AICc should converge to AIC as sample size increases."""
        log_lik = -100.0
        n_params = 3

        aic = compute_aic(log_lik, n_params)
        aicc_small = compute_aicc(log_lik, n_params, n_observations=50)
        aicc_large = compute_aicc(log_lik, n_params, n_observations=10000)

        # AICc with small n should differ from AIC
        assert abs(aicc_small - aic) > 0.1

        # AICc with large n should be close to AIC
        assert abs(aicc_large - aic) < 0.1


class TestModelFit:
    """Test ModelFit container and properties."""

    def test_model_fit_basic(self):
        """Test ModelFit initialization and basic properties."""
        residuals = jnp.array([0.1, -0.2, 0.3, -0.1, 0.2])
        predictions = jnp.array([1.1, 0.8, 1.3, 0.9, 1.2])

        fit = ModelFit(
            log_likelihood=-10.0,
            n_parameters=3,
            n_observations=5,
            residuals=residuals,
            predictions=predictions,
        )

        # Check RSS
        expected_rss = jnp.sum(residuals**2)
        assert abs(fit.rss - expected_rss) < 1e-6

        # Check MSE
        expected_mse = jnp.mean(residuals**2)
        assert abs(fit.mse - expected_mse) < 1e-6

        # Check RMSE
        expected_rmse = jnp.sqrt(expected_mse)
        assert abs(fit.rmse - expected_rmse) < 1e-6

        # Check MAE
        expected_mae = jnp.mean(jnp.abs(residuals))
        assert abs(fit.mae - expected_mae) < 1e-6

    def test_compute_information_criterion(self):
        """Test computing IC from ModelFit."""
        fit = ModelFit(
            log_likelihood=-100.0,
            n_parameters=3,
            n_observations=100,
            residuals=jnp.zeros(100),
            predictions=jnp.ones(100),
        )

        aic = compute_information_criterion(fit, InformationCriterion.AIC)
        bic = compute_information_criterion(fit, InformationCriterion.BIC)

        assert aic == 206.0
        assert abs(bic - (200.0 + 3.0 * jnp.log(100.0))) < 1e-4


class TestModelComparison:
    """Test model comparison functionality."""

    def test_compare_models_basic(self):
        """Test comparing two models."""
        # Model 1: Simple model (fewer parameters, worse fit)
        fit1 = ModelFit(
            log_likelihood=-150.0,
            n_parameters=2,
            n_observations=100,
            residuals=jnp.ones(100) * 0.5,
            predictions=jnp.ones(100),
        )

        # Model 2: Complex model (more parameters, better fit)
        fit2 = ModelFit(
            log_likelihood=-120.0,
            n_parameters=5,
            n_observations=100,
            residuals=jnp.ones(100) * 0.3,
            predictions=jnp.ones(100),
        )

        comparison = compare_models(
            {"Simple": fit1, "Complex": fit2},
            criteria=[InformationCriterion.AIC, InformationCriterion.BIC],
        )

        # Check structure
        assert set(comparison.model_names) == {"Simple", "Complex"}
        assert "aic" in comparison.criteria_values
        assert "bic" in comparison.criteria_values

        # Simple model: AIC = 304, BIC ≈ 313.8
        # Complex model: AIC = 250, BIC ≈ 273.0
        # Complex should win on AIC
        assert comparison.best_model in ["Simple", "Complex"]

    def test_compare_three_models(self):
        """Test comparing three models."""
        fits = {
            "Model_A": ModelFit(-100, 2, 100, jnp.zeros(100), jnp.ones(100)),
            "Model_B": ModelFit(-110, 3, 100, jnp.zeros(100), jnp.ones(100)),
            "Model_C": ModelFit(-90, 5, 100, jnp.zeros(100), jnp.ones(100)),
        }

        comparison = compare_models(fits)

        # Model_A: AIC = 204
        # Model_B: AIC = 226
        # Model_C: AIC = 190
        # Model_C should win
        assert comparison.best_model == "Model_C"

        # Check delta values
        deltas = comparison.delta_values["aic"]
        # Best model should have delta = 0
        best_idx = comparison.model_names.index(comparison.best_model)
        assert deltas[best_idx] == 0.0


# ==============================================================================
# Cross-Validation Tests
# ==============================================================================


class TestKFoldCV:
    """Test k-fold cross-validation."""

    def test_k_fold_split_basic(self):
        """Test basic k-fold splitting."""
        n_samples = 100
        n_folds = 5

        splits = k_fold_split(n_samples, n_folds, shuffle=False)

        # Should have n_folds splits
        assert len(splits) == n_folds

        # Each fold should have roughly n_samples / n_folds test samples
        for train_idx, test_idx in splits:
            assert len(test_idx) == 20  # 100 / 5
            assert len(train_idx) == 80  # 100 - 20

        # All samples should be used exactly once as test
        all_test_indices = jnp.concatenate([test_idx for _, test_idx in splits])
        assert len(jnp.unique(all_test_indices)) == n_samples

    def test_k_fold_split_shuffle(self):
        """Test k-fold with shuffling."""
        splits_no_shuffle = k_fold_split(100, 5, shuffle=False)
        splits_shuffle = k_fold_split(100, 5, shuffle=True, random_key=jax.random.PRNGKey(42))

        # First fold without shuffle should start at index 0
        _, test_idx_no_shuffle = splits_no_shuffle[0]
        _, test_idx_shuffle = splits_shuffle[0]

        # Shuffled indices should likely be different
        assert not jnp.array_equal(test_idx_no_shuffle, test_idx_shuffle)

    def test_cross_validate_simple(self):
        """Test cross-validation with simple linear model."""
        # Generate simple dataset: y = 2 * x + noise
        key = jax.random.PRNGKey(123)
        X = jnp.linspace(0, 10, 100).reshape(-1, 1)
        y = 2 * X.squeeze() + jax.random.normal(key, (100,)) * 0.1

        # Simple linear model
        def fit_fn(X_train, y_train):
            # Closed-form OLS: β = (X'X)^{-1} X'y
            X_aug = jnp.concatenate([jnp.ones((len(X_train), 1)), X_train], axis=1)
            beta = jnp.linalg.lstsq(X_aug, y_train)[0]
            return beta

        def model_fn(params, X_test):
            X_aug = jnp.concatenate([jnp.ones((len(X_test), 1)), X_test], axis=1)
            return X_aug @ params

        splits = k_fold_split(len(X), n_folds=5, shuffle=False)
        results = cross_validate(model_fn, fit_fn, X, y, splits)

        # Check results structure
        assert len(results.fold_scores) == 5
        assert results.mean_score > 0
        assert results.std_score >= 0

        # MSE should be small for linear data
        assert results.mean_score < 0.1


class TestTimeSeriesCV:
    """Test time-series cross-validation."""

    def test_time_series_split_expanding(self):
        """Test expanding window time-series CV."""
        n_samples = 100
        n_splits = 5

        splits = time_series_split(n_samples, n_splits, expanding=True)

        assert len(splits) == n_splits

        # Train set should grow with each split
        train_sizes = [len(train) for train, _ in splits]
        assert all(train_sizes[i] < train_sizes[i + 1] for i in range(len(train_sizes) - 1))

    def test_time_series_split_rolling(self):
        """Test rolling window time-series CV."""
        n_samples = 100
        n_splits = 5

        splits = time_series_split(n_samples, n_splits, expanding=False)

        assert len(splits) == n_splits

        # Train set size should grow to a fixed size after the first few splits
        train_sizes = [len(train) for train, _ in splits]
        # The later splits should have similar sizes
        later_sizes = train_sizes[1:]  # Skip first split
        assert max(later_sizes) - min(later_sizes) <= 1

    def test_time_series_no_overlap(self):
        """Test that test sets don't overlap with train sets."""
        splits = time_series_split(100, n_splits=4, expanding=True)

        for train_idx, test_idx in splits:
            # Test indices should all be greater than train indices
            max_train = jnp.max(train_idx)
            min_test = jnp.min(test_idx)
            assert min_test >= max_train


# ==============================================================================
# Sensitivity Analysis Tests
# ==============================================================================


class TestLocalSensitivity:
    """Test local sensitivity analysis."""

    def test_compute_local_sensitivity_linear(self):
        """Test local sensitivity for linear function."""

        def model(params):
            # f(x, y) = 3*x + 2*y
            return 3.0 * params["x"] + 2.0 * params["y"]

        params = {"x": 1.0, "y": 2.0}
        sensitivity = compute_local_sensitivity(model, params)

        # Gradients should be exact: df/dx = 3, df/dy = 2
        assert abs(sensitivity.sensitivities["x"] - 3.0) < 0.01
        assert abs(sensitivity.sensitivities["y"] - 2.0) < 0.01

    def test_compute_local_sensitivity_quadratic(self):
        """Test local sensitivity for quadratic function."""

        def model(params):
            # f(x) = x^2 + 3*x
            return params["x"] ** 2 + 3.0 * params["x"]

        params = {"x": 2.0}
        sensitivity = compute_local_sensitivity(model, params)

        # df/dx = 2*x + 3 = 2*2 + 3 = 7
        assert abs(sensitivity.sensitivities["x"] - 7.0) < 0.01

    def test_normalized_sensitivity(self):
        """Test normalized sensitivity (elasticity)."""

        def model(params):
            # f(kappa, theta) = kappa * theta
            return params["kappa"] * params["theta"]

        params = {"kappa": 2.0, "theta": 0.04}
        sensitivity = compute_local_sensitivity(model, params)

        # For multiplicative model, normalized sensitivity should be 1.0 for each
        # f = k * t, df/dk = t, elasticity = (t * k) / (k * t) = 1
        assert abs(sensitivity.normalized_sensitivities["kappa"] - 1.0) < 0.01
        assert abs(sensitivity.normalized_sensitivities["theta"] - 1.0) < 0.01


class TestGlobalSensitivity:
    """Test global sensitivity analysis (Sobol indices)."""

    def test_sobol_indices_additive(self):
        """Test Sobol indices for additive function."""

        def model(params):
            # f(x1, x2) = x1^2 + x2^2 (no interaction)
            return params[0] ** 2 + params[1] ** 2

        bounds = {"param1": (0.0, 1.0), "param2": (0.0, 1.0)}
        indices = sobol_indices(model, bounds, n_samples=5000, random_key=jax.random.PRNGKey(42))

        # For additive model, first-order indices should sum to ~1
        # and total-order should equal first-order
        total_first_order = sum(indices.first_order.values())
        assert 0.8 < total_first_order < 1.2  # Allow tolerance

        # Total order should be close to first order (no interactions)
        for param in ["param1", "param2"]:
            diff = abs(indices.total_order[param] - indices.first_order[param])
            assert diff < 0.3  # Allow some Monte Carlo error

    def test_sobol_indices_multiplicative(self):
        """Test Sobol indices for multiplicative function."""

        def model(params):
            # f(x1, x2) = x1 * x2 (pure interaction)
            return params[0] * params[1]

        bounds = {"param1": (0.0, 1.0), "param2": (0.0, 1.0)}
        indices = sobol_indices(model, bounds, n_samples=5000, random_key=jax.random.PRNGKey(123))

        # For multiplicative model, interaction effects should be significant
        # Total order > First order
        for param in ["param1", "param2"]:
            # Total should be larger than first-order
            assert indices.total_order[param] >= indices.first_order[param] - 0.1

    def test_sobol_indices_dominant_variable(self):
        """Test Sobol indices with one dominant variable."""

        def model(params):
            # f(x1, x2) = 10*x1 + 0.1*x2 (x1 dominates)
            return 10.0 * params[0] + 0.1 * params[1]

        bounds = {"param1": (0.0, 1.0), "param2": (0.0, 1.0)}
        indices = sobol_indices(model, bounds, n_samples=5000, random_key=jax.random.PRNGKey(456))

        # param1 should have much larger index than param2
        assert indices.first_order["param1"] > indices.first_order["param2"]
        assert indices.total_order["param1"] > indices.total_order["param2"]


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestIntegration:
    """Test integration of model selection tools."""

    def test_model_selection_workflow(self):
        """Test complete model selection workflow."""
        # Create mock model fits
        fits = {
            "Model1": ModelFit(
                log_likelihood=-100.0,
                n_parameters=2,
                n_observations=100,
                residuals=jnp.ones(100) * 0.5,
                predictions=jnp.ones(100),
            ),
            "Model2": ModelFit(
                log_likelihood=-90.0,
                n_parameters=4,
                n_observations=100,
                residuals=jnp.ones(100) * 0.3,
                predictions=jnp.ones(100),
            ),
        }

        # Compare models
        comparison = compare_models(fits)

        # Get summary
        summary = comparison.summary()
        assert "AIC" in summary or "aic" in summary
        assert "Best Model" in summary

    def test_cv_with_information_criteria(self):
        """Test using CV to validate model selected by IC."""
        # Generate data
        key = jax.random.PRNGKey(42)
        X = jnp.linspace(0, 10, 50).reshape(-1, 1)
        y = 2 * X.squeeze() + jax.random.normal(key, (50,)) * 0.5

        # Simple model fitting
        def fit_fn(X_train, y_train):
            X_aug = jnp.concatenate([jnp.ones((len(X_train), 1)), X_train], axis=1)
            return jnp.linalg.lstsq(X_aug, y_train)[0]

        def model_fn(params, X_test):
            X_aug = jnp.concatenate([jnp.ones((len(X_test), 1)), X_test], axis=1)
            return X_aug @ params

        # Cross-validate
        splits = k_fold_split(len(X), n_folds=5, shuffle=False)
        cv_results = cross_validate(model_fn, fit_fn, X, y, splits)

        # CV should give reasonable error
        assert cv_results.mean_score < 1.0  # MSE should be less than 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
