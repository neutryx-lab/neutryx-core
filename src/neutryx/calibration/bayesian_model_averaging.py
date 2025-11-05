"""Bayesian Model Averaging for model combination and uncertainty quantification.

This module implements Bayesian Model Averaging (BMA), which combines predictions
from multiple models using posterior model probabilities as weights:

    BMA_prediction = Σ P(M_k | Data) × Prediction_k

Key features:
1. **Model Weight Computation**: Convert information criteria (BIC, AIC) to posterior probabilities
2. **Weighted Predictions**: Combine model predictions using Bayesian weights
3. **Uncertainty Quantification**: Compute prediction intervals accounting for model uncertainty
4. **Integration**: Seamless integration with existing ModelFit and ModelComparison classes

References
----------
Hoeting, J. A., et al. (1999). Bayesian model averaging: a tutorial.
Statistical Science, 14(4), 382-417.

Raftery, A. E., et al. (1997). Bayesian model selection in social research.
Sociological Methodology, 27, 111-163.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from neutryx.calibration.model_selection import (
    InformationCriterion,
    ModelComparison,
    ModelFit,
    compute_information_criterion,
)


# ==============================================================================
# Weighting Schemes
# ==============================================================================


class WeightingScheme(Enum):
    """Methods for computing model weights in BMA."""

    BIC = "bic"  # Bayesian Information Criterion (recommended)
    AIC = "aic"  # Akaike Information Criterion
    AICC = "aicc"  # Corrected AIC (small samples)
    CUSTOM = "custom"  # User-provided weights


@dataclass
class ModelWeights:
    """Container for model weights and metadata.

    Attributes
    ----------
    weights : Dict[str, float]
        Dictionary mapping model names to their posterior probabilities
    raw_scores : Dict[str, float]
        Raw information criterion values before transformation
    effective_models : float
        Effective number of models (1 / Σ w_i^2)
    scheme : WeightingScheme
        Weighting scheme used
    """

    weights: Dict[str, float]
    raw_scores: Dict[str, float]
    effective_models: float
    scheme: WeightingScheme

    def summary(self) -> str:
        """Generate summary of model weights."""
        lines = ["Model Weights (Bayesian Model Averaging)", "=" * 60]
        lines.append(f"Weighting Scheme: {self.scheme.value.upper()}")
        lines.append(f"Effective Models: {self.effective_models:.2f}")
        lines.append("\n" + f"{'Model':<20} {'Weight':<12} {'Score':<12} {'Percentage':<12}")
        lines.append("-" * 60)

        sorted_models = sorted(self.weights.items(), key=lambda x: x[1], reverse=True)
        for model_name, weight in sorted_models:
            score = self.raw_scores.get(model_name, 0.0)
            percentage = weight * 100
            lines.append(
                f"{model_name:<20} {weight:10.6f}  {score:10.4f}  {percentage:9.2f}%"
            )

        return "\n".join(lines)


def compute_weights_from_ic(
    model_fits: Dict[str, ModelFit],
    criterion: InformationCriterion = InformationCriterion.BIC,
) -> ModelWeights:
    """Compute Bayesian model weights from information criteria.

    Uses the transformation:
        w_k = exp(-0.5 × Δ_k) / Σ exp(-0.5 × Δ_j)

    where Δ_k = IC_k - min(IC) is the difference from the best model.

    Parameters
    ----------
    model_fits : Dict[str, ModelFit]
        Dictionary mapping model names to their fit statistics
    criterion : InformationCriterion
        Information criterion to use for weighting (BIC recommended)

    Returns
    -------
    ModelWeights
        Model weights and metadata

    Notes
    -----
    BIC-based weights approximate Bayes factors under certain assumptions:
        BF_kj ≈ exp(-0.5 × (BIC_k - BIC_j))

    BIC is recommended over AIC as it:
    - Better approximates Bayes factors
    - Includes proper prior odds
    - Provides consistency (selects true model as n → ∞)

    Example
    -------
    >>> fits = {
    ...     "Heston": ModelFit(log_likelihood=-100, n_parameters=5, ...),
    ...     "SABR": ModelFit(log_likelihood=-105, n_parameters=4, ...),
    ... }
    >>> weights = compute_weights_from_ic(fits, InformationCriterion.BIC)
    >>> print(weights.summary())
    """
    model_names = list(model_fits.keys())

    # Compute information criteria
    ic_values = {
        name: compute_information_criterion(fit, criterion)
        for name, fit in model_fits.items()
    }

    # Compute deltas from best model
    min_ic = min(ic_values.values())
    deltas = {name: ic_values[name] - min_ic for name in model_names}

    # Transform to weights using exp(-0.5 × Δ)
    # This approximates posterior model probabilities
    unnormalized_weights = {name: jnp.exp(-0.5 * delta) for name, delta in deltas.items()}
    total = sum(unnormalized_weights.values())
    weights = {name: float(w / total) for name, w in unnormalized_weights.items()}

    # Compute effective number of models: 1 / Σ w_i^2
    effective_n = 1.0 / sum(w**2 for w in weights.values())

    # Map criterion enum to weighting scheme
    scheme_map = {
        InformationCriterion.AIC: WeightingScheme.AIC,
        InformationCriterion.BIC: WeightingScheme.BIC,
        InformationCriterion.AICC: WeightingScheme.AICC,
    }

    return ModelWeights(
        weights=weights,
        raw_scores=ic_values,
        effective_models=effective_n,
        scheme=scheme_map.get(criterion, WeightingScheme.BIC),
    )


def compute_stacking_weights(
    model_fits: Dict[str, ModelFit],
    predictions: Dict[str, Array],
    observed: Array,
    non_negative: bool = True,
) -> ModelWeights:
    """Compute stacking weights via optimization.

    Stacking finds weights that minimize out-of-sample prediction error:
        w* = argmin Σ (y_i - Σ w_k × pred_ki)^2
        subject to: Σ w_k = 1, w_k ≥ 0

    Parameters
    ----------
    model_fits : Dict[str, ModelFit]
        Model fit statistics
    predictions : Dict[str, Array]
        Model predictions (same keys as model_fits)
    observed : Array
        Observed values
    non_negative : bool
        If True, constrain weights to be non-negative

    Returns
    -------
    ModelWeights
        Optimized stacking weights

    Notes
    -----
    Stacking is often superior to information criteria-based weights,
    especially when models are misspecified or highly correlated.

    References
    ----------
    Breiman, L. (1996). Stacked regressions. Machine Learning, 24(1), 49-64.
    Yao, Y., et al. (2018). Using stacking to average Bayesian predictive
    distributions. Bayesian Analysis, 13(3), 917-1007.

    Example
    -------
    >>> weights = compute_stacking_weights(fits, predictions, observed_values)
    """
    from scipy.optimize import minimize

    model_names = list(model_fits.keys())
    n_models = len(model_names)

    # Stack predictions into matrix
    pred_matrix = jnp.stack([predictions[name] for name in model_names], axis=1)

    def objective(w: Array) -> float:
        """MSE of weighted combination."""
        combined = pred_matrix @ w
        return float(jnp.mean((observed - combined) ** 2))

    # Constraints: weights sum to 1
    constraints = {"type": "eq", "fun": lambda w: jnp.sum(w) - 1.0}

    # Bounds: non-negative if requested
    bounds = [(0.0, 1.0) if non_negative else (None, None)] * n_models

    # Initial guess: uniform weights
    w0 = jnp.ones(n_models) / n_models

    # Optimize
    result = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=constraints)

    if not result.success:
        raise RuntimeError(f"Stacking optimization failed: {result.message}")

    weights = {name: float(w) for name, w in zip(model_names, result.x)}
    effective_n = 1.0 / sum(w**2 for w in weights.values())

    # Use MSE as "score"
    mse_scores = {
        name: float(jnp.mean((observed - predictions[name]) ** 2)) for name in model_names
    }

    return ModelWeights(
        weights=weights,
        raw_scores=mse_scores,
        effective_models=effective_n,
        scheme=WeightingScheme.CUSTOM,
    )


# ==============================================================================
# Bayesian Model Averaging
# ==============================================================================


@dataclass
class BMAResult:
    """Result from Bayesian Model Averaging prediction.

    Attributes
    ----------
    mean : Array
        BMA point prediction (weighted average)
    variance : Array
        Total variance (within-model + between-model)
    within_model_variance : Array
        Variance within individual models
    between_model_variance : Array
        Variance across models
    model_predictions : Dict[str, Array]
        Individual model predictions
    weights : ModelWeights
        Model weights used
    """

    mean: Array
    variance: Array
    within_model_variance: Array
    between_model_variance: Array
    model_predictions: Dict[str, Array]
    weights: ModelWeights

    @property
    def std(self) -> Array:
        """Standard deviation of BMA prediction."""
        return jnp.sqrt(self.variance)

    def prediction_interval(
        self, confidence: float = 0.95
    ) -> Tuple[Array, Array]:
        """Compute prediction interval.

        Parameters
        ----------
        confidence : float
            Confidence level (e.g., 0.95 for 95% interval)

        Returns
        -------
        Tuple[Array, Array]
            Lower and upper bounds of prediction interval
        """
        from scipy.stats import norm

        # Use normal approximation
        z = norm.ppf(0.5 + confidence / 2)
        lower = self.mean - z * self.std
        upper = self.mean + z * self.std
        return lower, upper

    def summary(self, n_points: int = 5) -> str:
        """Generate summary of BMA results.

        Parameters
        ----------
        n_points : int
            Number of prediction points to display
        """
        lines = ["Bayesian Model Averaging Results", "=" * 60]
        lines.append(f"Number of models: {len(self.model_predictions)}")
        lines.append(
            f"Effective models: {self.weights.effective_models:.2f}"
        )
        lines.append(f"\nFirst {n_points} predictions:")
        lines.append(
            f"{'Index':<8} {'Mean':<12} {'Std':<12} {'95% CI Lower':<15} {'95% CI Upper':<15}"
        )
        lines.append("-" * 60)

        lower, upper = self.prediction_interval(0.95)
        for i in range(min(n_points, len(self.mean))):
            lines.append(
                f"{i:<8} {self.mean[i]:10.6f}  {self.std[i]:10.6f}  "
                f"{lower[i]:13.6f}  {upper[i]:13.6f}"
            )

        lines.append(f"\nVariance decomposition:")
        avg_within = float(jnp.mean(self.within_model_variance))
        avg_between = float(jnp.mean(self.between_model_variance))
        avg_total = float(jnp.mean(self.variance))
        lines.append(f"  Within-model:  {avg_within:10.6f} ({100*avg_within/avg_total:.1f}%)")
        lines.append(f"  Between-model: {avg_between:10.6f} ({100*avg_between/avg_total:.1f}%)")
        lines.append(f"  Total:         {avg_total:10.6f}")

        return "\n".join(lines)


class BayesianModelAveraging:
    """Bayesian Model Averaging for combining multiple model predictions.

    BMA provides a coherent framework for:
    - Combining predictions from multiple models
    - Accounting for model uncertainty
    - Quantifying total prediction uncertainty
    - Making robust predictions when model selection is uncertain

    Parameters
    ----------
    model_fits : Dict[str, ModelFit]
        Dictionary mapping model names to their fit statistics
    weighting_scheme : WeightingScheme
        Method for computing model weights
    criterion : Optional[InformationCriterion]
        Information criterion for weight computation (if using IC-based weighting)

    Attributes
    ----------
    weights : ModelWeights
        Computed model weights
    model_names : List[str]
        Names of models in the ensemble

    Example
    -------
    >>> # Fit multiple models
    >>> fits = {
    ...     "Heston": heston_fit,
    ...     "SABR": sabr_fit,
    ...     "LocalVol": local_vol_fit,
    ... }
    >>>
    >>> # Create BMA instance
    >>> bma = BayesianModelAveraging(fits, weighting_scheme=WeightingScheme.BIC)
    >>> print(bma.weights.summary())
    >>>
    >>> # Make predictions
    >>> predictions = {
    ...     "Heston": heston_prices,
    ...     "SABR": sabr_prices,
    ...     "LocalVol": local_vol_prices,
    ... }
    >>> result = bma.predict(predictions)
    >>> print(result.summary())
    """

    def __init__(
        self,
        model_fits: Dict[str, ModelFit],
        weighting_scheme: WeightingScheme = WeightingScheme.BIC,
        criterion: Optional[InformationCriterion] = None,
        custom_weights: Optional[Dict[str, float]] = None,
    ):
        """Initialize BMA with model fits and weighting scheme."""
        self.model_fits = model_fits
        self.model_names = list(model_fits.keys())
        self.weighting_scheme = weighting_scheme

        # Compute weights based on scheme
        if weighting_scheme == WeightingScheme.CUSTOM:
            if custom_weights is None:
                raise ValueError("custom_weights must be provided for CUSTOM weighting scheme")
            # Validate and normalize custom weights
            total = sum(custom_weights.values())
            normalized = {k: v / total for k, v in custom_weights.items()}
            effective_n = 1.0 / sum(w**2 for w in normalized.values())
            self.weights = ModelWeights(
                weights=normalized,
                raw_scores={},
                effective_models=effective_n,
                scheme=WeightingScheme.CUSTOM,
            )
        else:
            # Use information criterion
            if criterion is None:
                # Default mapping
                criterion_map = {
                    WeightingScheme.AIC: InformationCriterion.AIC,
                    WeightingScheme.BIC: InformationCriterion.BIC,
                    WeightingScheme.AICC: InformationCriterion.AICC,
                }
                criterion = criterion_map[weighting_scheme]

            self.weights = compute_weights_from_ic(model_fits, criterion)

    @classmethod
    def from_comparison(
        cls,
        comparison: ModelComparison,
        model_fits: Dict[str, ModelFit],
        criterion: Optional[InformationCriterion] = None,
    ) -> BayesianModelAveraging:
        """Create BMA instance from ModelComparison.

        Parameters
        ----------
        comparison : ModelComparison
            Model comparison results
        model_fits : Dict[str, ModelFit]
            Model fit statistics
        criterion : Optional[InformationCriterion]
            Information criterion to use (defaults to first in comparison)

        Returns
        -------
        BayesianModelAveraging
            Initialized BMA instance

        Example
        -------
        >>> comparison = compare_models(fits)
        >>> bma = BayesianModelAveraging.from_comparison(comparison, fits)
        """
        if criterion is None:
            # Use first criterion from comparison
            criterion_name = list(comparison.criteria_values.keys())[0]
            criterion = InformationCriterion(criterion_name)

        return cls(
            model_fits=model_fits,
            weighting_scheme=WeightingScheme.BIC
            if criterion == InformationCriterion.BIC
            else WeightingScheme.AIC,
            criterion=criterion,
        )

    def predict(
        self,
        model_predictions: Dict[str, Array],
        model_variances: Optional[Dict[str, Array]] = None,
    ) -> BMAResult:
        """Generate BMA prediction from individual model predictions.

        Parameters
        ----------
        model_predictions : Dict[str, Array]
            Dictionary mapping model names to their predictions
        model_variances : Optional[Dict[str, Array]]
            Dictionary mapping model names to prediction variances
            If None, variance is estimated from prediction spread

        Returns
        -------
        BMAResult
            BMA predictions with uncertainty quantification

        Notes
        -----
        Total variance is decomposed as:
            Var(y|Data) = E[Var(y|M)] + Var[E(y|M)]
                        = within + between

        where the expectation is over models weighted by posterior probabilities.

        Example
        -------
        >>> predictions = {
        ...     "Heston": heston_prices,
        ...     "SABR": sabr_prices,
        ... }
        >>> result = bma.predict(predictions)
        >>> print(f"BMA mean: {result.mean}")
        >>> print(f"BMA std: {result.std}")
        """
        # Validate inputs
        if set(model_predictions.keys()) != set(self.model_names):
            raise ValueError("model_predictions keys must match model_names")

        # Compute weighted mean: E[y] = Σ w_k × E[y|M_k]
        weighted_mean = jnp.zeros_like(next(iter(model_predictions.values())))
        for model_name, pred in model_predictions.items():
            weight = self.weights.weights[model_name]
            weighted_mean += weight * pred

        # Compute within-model variance: E[Var(y|M_k)]
        if model_variances is not None:
            within_var = jnp.zeros_like(weighted_mean)
            for model_name, var in model_variances.items():
                weight = self.weights.weights[model_name]
                within_var += weight * var
        else:
            # Use residual variance from model fits as proxy
            within_var = jnp.zeros_like(weighted_mean)
            for model_name, fit in self.model_fits.items():
                weight = self.weights.weights[model_name]
                within_var += weight * fit.mse

        # Compute between-model variance: Var[E(y|M_k)]
        between_var = jnp.zeros_like(weighted_mean)
        for model_name, pred in model_predictions.items():
            weight = self.weights.weights[model_name]
            between_var += weight * (pred - weighted_mean) ** 2

        # Total variance
        total_var = within_var + between_var

        return BMAResult(
            mean=weighted_mean,
            variance=total_var,
            within_model_variance=within_var,
            between_model_variance=between_var,
            model_predictions=model_predictions,
            weights=self.weights,
        )

    def predict_with_uncertainty(
        self,
        prediction_fn: Callable[[str], Array],
        n_samples: int = 1000,
        random_key: Optional[jax.random.KeyArray] = None,
    ) -> BMAResult:
        """Generate BMA prediction with Monte Carlo uncertainty sampling.

        This method is useful when you have calibrated parameter distributions
        and want to sample from the full posterior predictive distribution.

        Parameters
        ----------
        prediction_fn : Callable[[str], Array]
            Function that takes model name and returns a prediction sample
        n_samples : int
            Number of Monte Carlo samples
        random_key : Optional[jax.random.KeyArray]
            Random key for reproducibility

        Returns
        -------
        BMAResult
            BMA predictions with uncertainty

        Example
        -------
        >>> def sample_prediction(model_name):
        ...     # Sample parameters from posterior
        ...     # Generate prediction
        ...     return prediction
        >>>
        >>> result = bma.predict_with_uncertainty(sample_prediction, n_samples=5000)
        """
        if random_key is None:
            random_key = jax.random.PRNGKey(0)

        # Sample model indices according to weights
        model_weights_array = jnp.array([self.weights.weights[name] for name in self.model_names])

        keys = jax.random.split(random_key, n_samples)
        samples = []

        for key in keys:
            # Sample model according to weights
            model_idx = int(jax.random.choice(key, len(self.model_names), p=model_weights_array))
            model_name = self.model_names[model_idx]

            # Generate prediction from sampled model
            pred = prediction_fn(model_name)
            samples.append(pred)

        samples_array = jnp.stack(samples)

        # Compute statistics
        mean = jnp.mean(samples_array, axis=0)
        variance = jnp.var(samples_array, axis=0)

        # For compatibility, compute approximate within/between split
        # This is approximate as we're mixing parameter and model uncertainty
        within_var = jnp.zeros_like(mean)
        for model_name, fit in self.model_fits.items():
            weight = self.weights.weights[model_name]
            within_var += weight * fit.mse

        between_var = variance - within_var
        between_var = jnp.maximum(between_var, 0.0)  # Ensure non-negative

        # Get model predictions (means)
        model_predictions = {name: prediction_fn(name) for name in self.model_names}

        return BMAResult(
            mean=mean,
            variance=variance,
            within_model_variance=within_var,
            between_model_variance=between_var,
            model_predictions=model_predictions,
            weights=self.weights,
        )

    def model_probabilities(self) -> Dict[str, float]:
        """Get posterior model probabilities.

        Returns
        -------
        Dict[str, float]
            Dictionary mapping model names to their posterior probabilities
        """
        return self.weights.weights.copy()

    def update_weights(
        self,
        new_weights: Dict[str, float],
    ) -> None:
        """Update model weights manually.

        Parameters
        ----------
        new_weights : Dict[str, float]
            New weights for models (will be normalized)

        Example
        -------
        >>> # Give more weight to a specific model based on expert knowledge
        >>> bma.update_weights({"Heston": 0.7, "SABR": 0.3})
        """
        total = sum(new_weights.values())
        normalized = {k: v / total for k, v in new_weights.items()}
        effective_n = 1.0 / sum(w**2 for w in normalized.values())

        self.weights = ModelWeights(
            weights=normalized,
            raw_scores=self.weights.raw_scores,
            effective_models=effective_n,
            scheme=WeightingScheme.CUSTOM,
        )


# ==============================================================================
# Utility Functions
# ==============================================================================


def pseudo_bma_weights(
    loo_scores: Dict[str, float],
) -> ModelWeights:
    """Compute pseudo-BMA weights from leave-one-out cross-validation.

    Uses PSIS-LOO (Pareto Smoothed Importance Sampling LOO) scores
    to approximate Bayesian model weights.

    Parameters
    ----------
    loo_scores : Dict[str, float]
        Dictionary mapping model names to their LOO-CV scores (lower is better)

    Returns
    -------
    ModelWeights
        Pseudo-BMA weights

    References
    ----------
    Yao, Y., et al. (2018). Using stacking to average Bayesian predictive
    distributions. Bayesian Analysis, 13(3), 917-1007.
    """
    # Similar to IC-based weights
    min_loo = min(loo_scores.values())
    deltas = {name: loo_scores[name] - min_loo for name in loo_scores.keys()}

    unnormalized = {name: jnp.exp(-0.5 * delta) for name, delta in deltas.items()}
    total = sum(unnormalized.values())
    weights = {name: float(w / total) for name, w in unnormalized.items()}

    effective_n = 1.0 / sum(w**2 for w in weights.values())

    return ModelWeights(
        weights=weights,
        raw_scores=loo_scores,
        effective_models=effective_n,
        scheme=WeightingScheme.CUSTOM,
    )


__all__ = [
    "WeightingScheme",
    "ModelWeights",
    "BMAResult",
    "BayesianModelAveraging",
    "compute_weights_from_ic",
    "compute_stacking_weights",
    "pseudo_bma_weights",
]
