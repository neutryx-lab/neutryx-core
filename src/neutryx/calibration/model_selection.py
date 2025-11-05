"""Model selection and statistical inference tools.

This module provides comprehensive tools for model selection, validation,
and parameter analysis:

1. **Information Criteria**: AIC, BIC, HQIC, AICc for model comparison
2. **Cross-Validation**: K-fold, time-series, leave-one-out validation
3. **Identifiability Analysis**: Profile likelihood, parameter correlations
4. **Sensitivity Analysis**: Local and global sensitivity measures

These tools are essential for:
- Comparing competing model specifications
- Assessing model overfitting risk
- Understanding parameter identification
- Quantifying parameter importance
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from jax import Array


# ==============================================================================
# Information Criteria
# ==============================================================================


class InformationCriterion(Enum):
    """Supported information criteria for model selection."""

    AIC = "aic"  # Akaike Information Criterion
    BIC = "bic"  # Bayesian Information Criterion
    HQIC = "hqic"  # Hannan-Quinn Information Criterion
    AICC = "aicc"  # Corrected AIC (small sample)


@dataclass
class ModelFit:
    """Container for model fit statistics.

    Attributes
    ----------
    log_likelihood : float
        Log-likelihood at the optimum
    n_parameters : int
        Number of model parameters
    n_observations : int
        Number of observations
    residuals : Array
        Model residuals (observed - predicted)
    predictions : Array
        Model predictions
    """

    log_likelihood: float
    n_parameters: int
    n_observations: int
    residuals: Array
    predictions: Array

    @property
    def rss(self) -> float:
        """Residual sum of squares."""
        return float(jnp.sum(self.residuals**2))

    @property
    def mse(self) -> float:
        """Mean squared error."""
        return float(jnp.mean(self.residuals**2))

    @property
    def rmse(self) -> float:
        """Root mean squared error."""
        return float(jnp.sqrt(self.mse))

    @property
    def mae(self) -> float:
        """Mean absolute error."""
        return float(jnp.mean(jnp.abs(self.residuals)))

    @property
    def r_squared(self) -> float:
        """Coefficient of determination (R²).

        Note: This assumes predictions are compared against observations.
        For calibration, this may not be meaningful if observed are prices.
        """
        observed = self.predictions - self.residuals
        ss_tot = jnp.sum((observed - jnp.mean(observed)) ** 2)
        ss_res = jnp.sum(self.residuals**2)
        return float(1.0 - (ss_res / (ss_tot + 1e-12)))


def compute_aic(log_likelihood: float, n_parameters: int) -> float:
    """Compute Akaike Information Criterion.

    AIC = -2 * log_likelihood + 2 * k

    where k is the number of parameters.

    Parameters
    ----------
    log_likelihood : float
        Log-likelihood at the optimum
    n_parameters : int
        Number of model parameters

    Returns
    -------
    float
        AIC value (lower is better)

    Notes
    -----
    AIC balances model fit against complexity. It penalizes additional
    parameters to avoid overfitting.

    References
    ----------
    Akaike, H. (1974). A new look at the statistical model identification.
    IEEE Transactions on Automatic Control, 19(6), 716-723.

    Example
    -------
    >>> aic = compute_aic(log_lik=-1000.0, n_parameters=5)
    >>> # aic = 2000 + 10 = 2010
    """
    return -2.0 * log_likelihood + 2.0 * n_parameters


def compute_bic(log_likelihood: float, n_parameters: int, n_observations: int) -> float:
    """Compute Bayesian Information Criterion.

    BIC = -2 * log_likelihood + k * log(n)

    where k is the number of parameters and n is the sample size.

    Parameters
    ----------
    log_likelihood : float
        Log-likelihood at the optimum
    n_parameters : int
        Number of model parameters
    n_observations : int
        Number of observations

    Returns
    -------
    float
        BIC value (lower is better)

    Notes
    -----
    BIC applies a stronger penalty than AIC for additional parameters,
    especially for large samples. It approximates Bayes factors and
    is consistent (selects the true model as n → ∞).

    References
    ----------
    Schwarz, G. (1978). Estimating the dimension of a model.
    Annals of Statistics, 6(2), 461-464.

    Example
    -------
    >>> bic = compute_bic(log_lik=-1000.0, n_parameters=5, n_observations=100)
    >>> # bic = 2000 + 5 * log(100) ≈ 2023.03
    """
    return -2.0 * log_likelihood + n_parameters * jnp.log(n_observations)


def compute_aicc(log_likelihood: float, n_parameters: int, n_observations: int) -> float:
    """Compute corrected AIC for small samples.

    AICc = AIC + 2k(k+1) / (n - k - 1)

    where k is the number of parameters and n is the sample size.

    Parameters
    ----------
    log_likelihood : float
        Log-likelihood at the optimum
    n_parameters : int
        Number of model parameters
    n_observations : int
        Number of observations

    Returns
    -------
    float
        AICc value (lower is better)

    Notes
    -----
    AICc corrects for small sample bias in AIC. As n → ∞, AICc → AIC.
    Recommended when n/k < 40.

    References
    ----------
    Hurvich, C. M., & Tsai, C. L. (1989). Regression and time series
    model selection in small samples. Biometrika, 76(2), 297-307.

    Example
    -------
    >>> aicc = compute_aicc(log_lik=-100.0, n_parameters=5, n_observations=50)
    """
    aic = compute_aic(log_likelihood, n_parameters)
    correction = (2.0 * n_parameters * (n_parameters + 1)) / (
        n_observations - n_parameters - 1 + 1e-12
    )
    return aic + correction


def compute_hqic(log_likelihood: float, n_parameters: int, n_observations: int) -> float:
    """Compute Hannan-Quinn Information Criterion.

    HQIC = -2 * log_likelihood + 2k * log(log(n))

    where k is the number of parameters and n is the sample size.

    Parameters
    ----------
    log_likelihood : float
        Log-likelihood at the optimum
    n_parameters : int
        Number of model parameters
    n_observations : int
        Number of observations

    Returns
    -------
    float
        HQIC value (lower is better)

    Notes
    -----
    HQIC provides a penalty between AIC and BIC. It is consistent
    and has optimal rate of convergence.

    References
    ----------
    Hannan, E. J., & Quinn, B. G. (1979). The determination of the order
    of an autoregression. Journal of the Royal Statistical Society, 41(2), 190-195.

    Example
    -------
    >>> hqic = compute_hqic(log_lik=-1000.0, n_parameters=5, n_observations=100)
    """
    return -2.0 * log_likelihood + 2.0 * n_parameters * jnp.log(jnp.log(n_observations + jnp.e))


def compute_information_criterion(
    model_fit: ModelFit,
    criterion: InformationCriterion = InformationCriterion.AIC,
) -> float:
    """Compute specified information criterion.

    Parameters
    ----------
    model_fit : ModelFit
        Model fit statistics
    criterion : InformationCriterion
        Which criterion to compute

    Returns
    -------
    float
        Information criterion value

    Example
    -------
    >>> fit = ModelFit(log_likelihood=-100, n_parameters=3, n_observations=50, ...)
    >>> aic = compute_information_criterion(fit, InformationCriterion.AIC)
    >>> bic = compute_information_criterion(fit, InformationCriterion.BIC)
    """
    if criterion == InformationCriterion.AIC:
        return compute_aic(model_fit.log_likelihood, model_fit.n_parameters)
    elif criterion == InformationCriterion.BIC:
        return compute_bic(
            model_fit.log_likelihood,
            model_fit.n_parameters,
            model_fit.n_observations,
        )
    elif criterion == InformationCriterion.AICC:
        return compute_aicc(
            model_fit.log_likelihood,
            model_fit.n_parameters,
            model_fit.n_observations,
        )
    elif criterion == InformationCriterion.HQIC:
        return compute_hqic(
            model_fit.log_likelihood,
            model_fit.n_parameters,
            model_fit.n_observations,
        )
    else:
        raise ValueError(f"Unknown criterion: {criterion}")


@dataclass
class ModelComparison:
    """Results from comparing multiple models.

    Attributes
    ----------
    model_names : List[str]
        Names of models being compared
    criteria_values : Dict[str, List[float]]
        Dictionary mapping criterion name to list of values for each model
    best_model : str
        Name of the best model (lowest criterion value)
    delta_values : Dict[str, List[float]]
        Difference from best model for each criterion
    """

    model_names: List[str]
    criteria_values: Dict[str, List[float]]
    best_model: str
    delta_values: Dict[str, List[float]]

    def summary(self) -> str:
        """Generate a summary table of model comparison."""
        lines = ["Model Comparison Summary", "=" * 50]

        for criterion_name, values in self.criteria_values.items():
            lines.append(f"\n{criterion_name.upper()}:")
            deltas = self.delta_values[criterion_name]

            for i, (name, value, delta) in enumerate(zip(self.model_names, values, deltas)):
                marker = " *" if name == self.best_model else ""
                lines.append(f"  {name}: {value:.2f} (Δ={delta:.2f}){marker}")

        lines.append(f"\nBest Model: {self.best_model}")
        return "\n".join(lines)


def compare_models(
    model_fits: Dict[str, ModelFit],
    criteria: Sequence[InformationCriterion] = (
        InformationCriterion.AIC,
        InformationCriterion.BIC,
    ),
) -> ModelComparison:
    """Compare multiple models using information criteria.

    Parameters
    ----------
    model_fits : Dict[str, ModelFit]
        Dictionary mapping model names to their fit statistics
    criteria : Sequence[InformationCriterion]
        Which criteria to use for comparison

    Returns
    -------
    ModelComparison
        Comparison results

    Example
    -------
    >>> fits = {
    ...     "Heston": ModelFit(...),
    ...     "SABR": ModelFit(...),
    ...     "LocalVol": ModelFit(...),
    ... }
    >>> comparison = compare_models(fits)
    >>> print(comparison.summary())
    """
    model_names = list(model_fits.keys())
    criteria_values: Dict[str, List[float]] = {}
    delta_values: Dict[str, List[float]] = {}

    best_model = None
    best_aic = float("inf")

    for criterion in criteria:
        values = [
            compute_information_criterion(fit, criterion) for fit in model_fits.values()
        ]
        criteria_values[criterion.value] = values

        # Compute deltas from best (minimum)
        min_value = min(values)
        deltas = [v - min_value for v in values]
        delta_values[criterion.value] = deltas

        # Track overall best model based on first criterion
        if criterion == criteria[0]:
            best_idx = values.index(min_value)
            best_model = model_names[best_idx]

    return ModelComparison(
        model_names=model_names,
        criteria_values=criteria_values,
        best_model=best_model or model_names[0],
        delta_values=delta_values,
    )


# ==============================================================================
# Cross-Validation
# ==============================================================================


@dataclass
class CrossValidationResult:
    """Results from cross-validation.

    Attributes
    ----------
    fold_scores : List[float]
        Score (e.g., MSE) for each fold
    mean_score : float
        Mean score across folds
    std_score : float
        Standard deviation of scores
    fold_predictions : List[Array]
        Predictions for each fold
    fold_indices : List[Tuple[Array, Array]]
        Train/test indices for each fold
    """

    fold_scores: List[float]
    mean_score: float
    std_score: float
    fold_predictions: List[Array]
    fold_indices: List[Tuple[Array, Array]]

    def summary(self) -> str:
        """Generate summary of CV results."""
        return (
            f"Cross-Validation Results:\n"
            f"  Mean Score: {self.mean_score:.6f}\n"
            f"  Std Score:  {self.std_score:.6f}\n"
            f"  Folds:      {len(self.fold_scores)}\n"
            f"  Min Score:  {min(self.fold_scores):.6f}\n"
            f"  Max Score:  {max(self.fold_scores):.6f}"
        )


def k_fold_split(
    n_samples: int,
    n_folds: int = 5,
    shuffle: bool = True,
    random_key: Optional[jax.random.KeyArray] = None,
) -> List[Tuple[Array, Array]]:
    """Generate k-fold cross-validation splits.

    Parameters
    ----------
    n_samples : int
        Number of samples in dataset
    n_folds : int
        Number of folds
    shuffle : bool
        Whether to shuffle indices before splitting
    random_key : Optional[jax.random.KeyArray]
        Random key for shuffling

    Returns
    -------
    List[Tuple[Array, Array]]
        List of (train_indices, test_indices) for each fold

    Example
    -------
    >>> splits = k_fold_split(n_samples=100, n_folds=5)
    >>> for train_idx, test_idx in splits:
    ...     # Train on train_idx, validate on test_idx
    ...     pass
    """
    indices = jnp.arange(n_samples)

    if shuffle:
        if random_key is None:
            random_key = jax.random.PRNGKey(0)
        indices = jax.random.permutation(random_key, indices)

    fold_sizes = jnp.full(n_folds, n_samples // n_folds, dtype=jnp.int32)
    fold_sizes = fold_sizes.at[: n_samples % n_folds].add(1)

    splits = []
    current = 0
    for fold_size in fold_sizes:
        test_start = current
        test_end = current + int(fold_size)
        test_indices = indices[test_start:test_end]
        train_indices = jnp.concatenate([indices[:test_start], indices[test_end:]])
        splits.append((train_indices, test_indices))
        current = test_end

    return splits


def time_series_split(
    n_samples: int,
    n_splits: int = 5,
    test_size: Optional[int] = None,
    expanding: bool = True,
) -> List[Tuple[Array, Array]]:
    """Generate time-series cross-validation splits.

    Parameters
    ----------
    n_samples : int
        Number of samples
    n_splits : int
        Number of splits
    test_size : Optional[int]
        Size of test set (if None, computed automatically)
    expanding : bool
        If True, use expanding window (train size grows).
        If False, use rolling window (train size fixed).

    Returns
    -------
    List[Tuple[Array, Array]]
        List of (train_indices, test_indices)

    Notes
    -----
    Time-series CV respects temporal order and does not shuffle.
    Expanding window is recommended for most financial applications.

    Example
    -------
    >>> # Expanding window: [0:10], [0:20], [0:30], ...
    >>> splits = time_series_split(100, n_splits=5, expanding=True)
    >>>
    >>> # Rolling window: [0:20], [10:30], [20:40], ...
    >>> splits = time_series_split(100, n_splits=5, expanding=False)
    """
    if test_size is None:
        test_size = n_samples // (n_splits + 1)

    splits = []

    if expanding:
        # Expanding window
        min_train = n_samples // (n_splits + 1)
        for i in range(n_splits):
            test_start = min_train + i * test_size
            test_end = min(test_start + test_size, n_samples)
            train_end = test_start
            train_indices = jnp.arange(0, train_end)
            test_indices = jnp.arange(test_start, test_end)
            splits.append((train_indices, test_indices))
    else:
        # Rolling window
        train_size = test_size * 2
        for i in range(n_splits):
            test_end = n_samples - (n_splits - i - 1) * test_size
            test_start = test_end - test_size
            train_end = test_start
            train_start = max(0, train_end - train_size)
            train_indices = jnp.arange(train_start, train_end)
            test_indices = jnp.arange(test_start, test_end)
            splits.append((train_indices, test_indices))

    return splits


def cross_validate(
    model_fn: Callable[[Any, Array], Array],
    fit_fn: Callable[[Array, Array], Any],
    X: Array,
    y: Array,
    cv_splits: List[Tuple[Array, Array]],
    score_fn: Optional[Callable[[Array, Array], float]] = None,
) -> CrossValidationResult:
    """Perform cross-validation.

    Parameters
    ----------
    model_fn : Callable[[Any, Array], Array]
        Function that takes (params, X) and returns predictions
    fit_fn : Callable[[Array, Array], Any]
        Function that takes (X_train, y_train) and returns fitted params
    X : Array
        Feature matrix
    y : Array
        Target values
    cv_splits : List[Tuple[Array, Array]]
        Cross-validation splits (train/test indices)
    score_fn : Optional[Callable[[Array, Array], float]]
        Scoring function (y_true, y_pred) -> score
        If None, uses MSE

    Returns
    -------
    CrossValidationResult
        Cross-validation results

    Example
    -------
    >>> def fit_fn(X_train, y_train):
    ...     # Fit model and return parameters
    ...     return calibrated_params
    >>>
    >>> def model_fn(params, X):
    ...     # Generate predictions
    ...     return predictions
    >>>
    >>> splits = k_fold_split(n_samples=len(X), n_folds=5)
    >>> results = cross_validate(model_fn, fit_fn, X, y, splits)
    >>> print(results.summary())
    """
    if score_fn is None:
        # Default: mean squared error
        def score_fn(y_true, y_pred):
            return float(jnp.mean((y_true - y_pred) ** 2))

    fold_scores = []
    fold_predictions = []

    for train_idx, test_idx in cv_splits:
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]

        # Fit model on training data
        params = fit_fn(X_train, y_train)

        # Predict on test data
        y_pred = model_fn(params, X_test)

        # Score
        score = score_fn(y_test, y_pred)
        fold_scores.append(score)
        fold_predictions.append(y_pred)

    mean_score = float(jnp.mean(jnp.array(fold_scores)))
    std_score = float(jnp.std(jnp.array(fold_scores)))

    return CrossValidationResult(
        fold_scores=fold_scores,
        mean_score=mean_score,
        std_score=std_score,
        fold_predictions=fold_predictions,
        fold_indices=cv_splits,
    )


# ==============================================================================
# Sensitivity Analysis
# ==============================================================================


@dataclass
class LocalSensitivity:
    """Local sensitivity analysis results.

    Attributes
    ----------
    sensitivities : Dict[str, float]
        Sensitivity (gradient) for each parameter
    normalized_sensitivities : Dict[str, float]
        Normalized sensitivities (elasticities)
    parameter_names : List[str]
        Parameter names
    """

    sensitivities: Dict[str, float]
    normalized_sensitivities: Dict[str, float]
    parameter_names: List[str]

    def summary(self) -> str:
        """Generate summary of sensitivity analysis."""
        lines = ["Local Sensitivity Analysis", "=" * 50]
        for param in self.parameter_names:
            sens = self.sensitivities[param]
            norm_sens = self.normalized_sensitivities[param]
            lines.append(f"{param:20s}: {sens:12.6f}  (normalized: {norm_sens:12.6f})")
        return "\n".join(lines)


def compute_local_sensitivity(
    model_fn: Callable[[Any], float],
    params: Dict[str, float],
    epsilon: float = 1e-5,
) -> LocalSensitivity:
    """Compute local sensitivity via finite differences.

    Parameters
    ----------
    model_fn : Callable[[Any], float]
        Model function that takes parameters dict and returns a scalar output
    params : Dict[str, float]
        Parameter values at which to compute sensitivity
    epsilon : float
        Perturbation size for finite differences

    Returns
    -------
    LocalSensitivity
        Local sensitivity results

    Notes
    -----
    Local sensitivity measures how the output changes with respect to
    each parameter at a specific parameter point.

    Example
    -------
    >>> def model(params):
    ...     return params['kappa'] * params['theta']
    >>>
    >>> params = {'kappa': 2.0, 'theta': 0.04}
    >>> sensitivity = compute_local_sensitivity(model, params)
    """
    sensitivities = {}
    normalized_sensitivities = {}
    parameter_names = list(params.keys())

    base_value = model_fn(params)

    for param_name in parameter_names:
        # Perturb parameter
        params_plus = params.copy()
        params_plus[param_name] = params[param_name] + epsilon

        # Compute gradient
        value_plus = model_fn(params_plus)
        gradient = (value_plus - base_value) / epsilon
        sensitivities[param_name] = float(gradient)

        # Normalized sensitivity (elasticity)
        if abs(base_value) > 1e-12 and abs(params[param_name]) > 1e-12:
            elasticity = gradient * params[param_name] / base_value
            normalized_sensitivities[param_name] = float(elasticity)
        else:
            normalized_sensitivities[param_name] = 0.0

    return LocalSensitivity(
        sensitivities=sensitivities,
        normalized_sensitivities=normalized_sensitivities,
        parameter_names=parameter_names,
    )


@dataclass
class GlobalSensitivity:
    """Global sensitivity analysis results (Sobol indices).

    Attributes
    ----------
    first_order : Dict[str, float]
        First-order Sobol indices (main effects)
    total_order : Dict[str, float]
        Total-order Sobol indices (main + interaction effects)
    parameter_names : List[str]
        Parameter names
    """

    first_order: Dict[str, float]
    total_order: Dict[str, float]
    parameter_names: List[str]

    def summary(self) -> str:
        """Generate summary of global sensitivity analysis."""
        lines = ["Global Sensitivity Analysis (Sobol Indices)", "=" * 60]
        lines.append(f"{'Parameter':<20} {'First Order':<15} {'Total Order':<15} {'Interaction':<15}")
        lines.append("-" * 60)

        for param in self.parameter_names:
            first = self.first_order[param]
            total = self.total_order[param]
            interaction = total - first
            lines.append(
                f"{param:<20} {first:12.6f}    {total:12.6f}    {interaction:12.6f}"
            )

        return "\n".join(lines)


def sobol_indices(
    model_fn: Callable[[Array], float],
    param_bounds: Dict[str, Tuple[float, float]],
    n_samples: int = 1000,
    random_key: Optional[jax.random.KeyArray] = None,
) -> GlobalSensitivity:
    """Compute Sobol sensitivity indices.

    Parameters
    ----------
    model_fn : Callable[[Array], float]
        Model function taking parameter array and returning scalar output
    param_bounds : Dict[str, Tuple[float, float]]
        Parameter bounds for sampling
    n_samples : int
        Number of samples for Monte Carlo integration
    random_key : Optional[jax.random.KeyArray]
        Random key for reproducibility

    Returns
    -------
    GlobalSensitivity
        Global sensitivity results

    Notes
    -----
    Sobol indices decompose output variance into contributions from
    individual parameters and their interactions. First-order indices
    measure main effects, total-order indices include interactions.

    This is a simplified implementation using Saltelli's sampling scheme.

    References
    ----------
    Saltelli, A., et al. (2010). Variance based sensitivity analysis of
    model output. Computer Physics Communications, 181(2), 259-270.

    Example
    -------
    >>> def model(params):
    ...     return params[0]**2 + params[1]**2 + params[0] * params[1]
    >>>
    >>> bounds = {'param1': (0.0, 1.0), 'param2': (0.0, 1.0)}
    >>> indices = sobol_indices(model, bounds, n_samples=10000)
    """
    if random_key is None:
        random_key = jax.random.PRNGKey(42)

    parameter_names = list(param_bounds.keys())
    n_params = len(parameter_names)

    # Generate quasi-random samples (simplified)
    keys = jax.random.split(random_key, 2 * n_params + 2)

    # Sample parameter matrices A and B
    A_samples = []
    B_samples = []

    for i, param_name in enumerate(parameter_names):
        lower, upper = param_bounds[param_name]
        A_samples.append(
            jax.random.uniform(keys[i], (n_samples,), minval=lower, maxval=upper)
        )
        B_samples.append(
            jax.random.uniform(
                keys[n_params + i], (n_samples,), minval=lower, maxval=upper
            )
        )

    A = jnp.stack(A_samples, axis=1)
    B = jnp.stack(B_samples, axis=1)

    # Evaluate model on A and B
    f_A = jax.vmap(lambda row: model_fn(row))(A)
    f_B = jax.vmap(lambda row: model_fn(row))(B)

    # Compute variance
    f_mean = jnp.mean(jnp.concatenate([f_A, f_B]))
    total_variance = jnp.var(jnp.concatenate([f_A, f_B]))

    first_order = {}
    total_order = {}

    for i, param_name in enumerate(parameter_names):
        # Create C_i: B with i-th column from A
        C_i = B.at[:, i].set(A[:, i])
        f_C_i = jax.vmap(lambda row: model_fn(row))(C_i)

        # First-order index: S_i = V_i / V
        # V_i = E[f(A) * f(C_i)] - E[f(A)]^2
        V_i = jnp.mean(f_A * f_C_i) - f_mean**2
        first_order[param_name] = float(jnp.clip(V_i / (total_variance + 1e-12), 0, 1))

        # Total-order index: ST_i = 1 - V_~i / V
        # V_~i = E[f(B) * f(C_i)] - E[f(B)]^2
        V_not_i = jnp.mean(f_B * f_C_i) - f_mean**2
        total_order[param_name] = float(
            jnp.clip(1.0 - V_not_i / (total_variance + 1e-12), 0, 1)
        )

    return GlobalSensitivity(
        first_order=first_order,
        total_order=total_order,
        parameter_names=parameter_names,
    )


__all__ = [
    # Information Criteria
    "InformationCriterion",
    "ModelFit",
    "compute_aic",
    "compute_bic",
    "compute_aicc",
    "compute_hqic",
    "compute_information_criterion",
    "ModelComparison",
    "compare_models",
    # Cross-Validation
    "CrossValidationResult",
    "k_fold_split",
    "time_series_split",
    "cross_validate",
    # Sensitivity Analysis
    "LocalSensitivity",
    "GlobalSensitivity",
    "compute_local_sensitivity",
    "sobol_indices",
]
