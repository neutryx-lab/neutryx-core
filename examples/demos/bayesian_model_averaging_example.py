"""Example: Bayesian Model Averaging for Option Pricing

This example demonstrates how to use Bayesian Model Averaging (BMA) to combine
predictions from multiple volatility models for option pricing.

Scenario:
---------
You've calibrated three different models (Heston, SABR, Local Vol) to market data
and want to:
1. Compute model weights based on BIC
2. Generate BMA option prices with uncertainty
3. Compare BMA predictions to individual models
4. Understand variance decomposition

This approach is useful when:
- Model selection is uncertain
- Different models perform well in different regimes
- You want robust predictions accounting for model uncertainty
"""
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import Dict

from neutryx.calibration.model_selection import (
    ModelFit,
    compare_models,
    InformationCriterion,
)
from neutryx.calibration.bayesian_model_averaging import (
    BayesianModelAveraging,
    WeightingScheme,
    compute_weights_from_ic,
    compute_stacking_weights,
)


# ==============================================================================
# Generate Synthetic Calibration Data
# ==============================================================================


def generate_market_data(n_options: int = 50, seed: int = 42) -> tuple:
    """Generate synthetic option market prices and features.

    Returns
    -------
    strikes : Array
        Strike prices
    market_prices : Array
        Observed market prices
    """
    key = jax.random.PRNGKey(seed)

    # Generate strikes around ATM (S0 = 100)
    strikes = jnp.linspace(80, 120, n_options)

    # True prices (unknown to models) with some structure
    moneyness = strikes / 100.0
    true_prices = 15 * jnp.exp(-0.5 * (moneyness - 1.0) ** 2 / 0.05)

    # Add market noise
    noise = jax.random.normal(key, (n_options,)) * 0.3
    market_prices = true_prices + noise

    return strikes, market_prices


# ==============================================================================
# Simulate Model Calibrations
# ==============================================================================


def simulate_heston_calibration(strikes, market_prices, key) -> tuple:
    """Simulate Heston model calibration.

    Returns
    -------
    predictions : Array
        Model predictions
    model_fit : ModelFit
        Calibration statistics
    """
    # Heston tends to fit well but can overfit
    noise = jax.random.normal(key, market_prices.shape) * 0.25
    predictions = market_prices + noise * 0.8

    residuals = market_prices - predictions
    log_likelihood = -0.5 * jnp.sum(residuals**2) - 0.5 * len(residuals) * jnp.log(
        2 * jnp.pi * 0.25**2
    )

    return predictions, ModelFit(
        log_likelihood=float(log_likelihood),
        n_parameters=5,  # v0, kappa, theta, sigma, rho
        n_observations=len(market_prices),
        residuals=residuals,
        predictions=predictions,
    )


def simulate_sabr_calibration(strikes, market_prices, key) -> tuple:
    """Simulate SABR model calibration.

    Returns
    -------
    predictions : Array
        Model predictions
    model_fit : ModelFit
        Calibration statistics
    """
    # SABR is simpler, might underfit slightly
    noise = jax.random.normal(key, market_prices.shape) * 0.30
    predictions = market_prices + noise * 0.9

    residuals = market_prices - predictions
    log_likelihood = -0.5 * jnp.sum(residuals**2) - 0.5 * len(residuals) * jnp.log(
        2 * jnp.pi * 0.30**2
    )

    return predictions, ModelFit(
        log_likelihood=float(log_likelihood),
        n_parameters=4,  # alpha, beta, rho, nu
        n_observations=len(market_prices),
        residuals=residuals,
        predictions=predictions,
    )


def simulate_localvol_calibration(strikes, market_prices, key) -> tuple:
    """Simulate Local Volatility model calibration.

    Returns
    -------
    predictions : Array
        Model predictions
    model_fit : ModelFit
        Calibration statistics
    """
    # Local vol fits very well (can overfit)
    noise = jax.random.normal(key, market_prices.shape) * 0.20
    predictions = market_prices + noise * 0.7

    residuals = market_prices - predictions
    log_likelihood = -0.5 * jnp.sum(residuals**2) - 0.5 * len(residuals) * jnp.log(
        2 * jnp.pi * 0.20**2
    )

    return predictions, ModelFit(
        log_likelihood=float(log_likelihood),
        n_parameters=8,  # Higher dimensional
        n_observations=len(market_prices),
        residuals=residuals,
        predictions=predictions,
    )


# ==============================================================================
# Example 1: Basic BMA with Information Criteria
# ==============================================================================


def example_basic_bma():
    """Example: Basic BMA using BIC-based weights."""
    print("=" * 80)
    print("Example 1: Basic Bayesian Model Averaging")
    print("=" * 80)

    # Generate data
    strikes, market_prices = generate_market_data(n_options=50)

    # Calibrate models
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 3)

    heston_pred, heston_fit = simulate_heston_calibration(strikes, market_prices, keys[0])
    sabr_pred, sabr_fit = simulate_sabr_calibration(strikes, market_prices, keys[1])
    lv_pred, lv_fit = simulate_localvol_calibration(strikes, market_prices, keys[2])

    # Store fits
    model_fits = {
        "Heston": heston_fit,
        "SABR": sabr_fit,
        "LocalVol": lv_fit,
    }

    # Step 1: Compare models using information criteria
    print("\n1. Model Comparison")
    print("-" * 80)
    comparison = compare_models(
        model_fits, criteria=[InformationCriterion.AIC, InformationCriterion.BIC]
    )
    print(comparison.summary())

    # Step 2: Create BMA with BIC-based weights
    print("\n2. Bayesian Model Averaging")
    print("-" * 80)
    bma = BayesianModelAveraging(model_fits, weighting_scheme=WeightingScheme.BIC)
    print(bma.weights.summary())

    # Step 3: Generate BMA predictions for new strikes
    print("\n3. BMA Predictions")
    print("-" * 80)
    new_strikes = jnp.array([85, 90, 95, 100, 105, 110, 115])
    print(f"Predicting prices for strikes: {new_strikes}")

    # In practice, you'd use your calibrated models to price at new strikes
    # Here we simulate predictions
    predictions = {
        "Heston": jnp.array([13.5, 12.0, 10.8, 9.5, 8.0, 6.5, 5.2]),
        "SABR": jnp.array([13.8, 12.2, 10.6, 9.3, 7.9, 6.6, 5.3]),
        "LocalVol": jnp.array([13.3, 11.9, 10.9, 9.6, 8.1, 6.4, 5.1]),
    }

    # Optional: provide prediction variances from bootstrap/MCMC
    variances = {
        "Heston": jnp.ones(7) * 0.15,
        "SABR": jnp.ones(7) * 0.20,
        "LocalVol": jnp.ones(7) * 0.10,
    }

    result = bma.predict(predictions, model_variances=variances)

    # Display results
    print(f"\n{'Strike':<10} {'BMA Price':<12} {'Std Dev':<12} {'95% CI Lower':<15} {'95% CI Upper':<15}")
    print("-" * 80)
    lower, upper = result.prediction_interval(0.95)
    for i, strike in enumerate(new_strikes):
        print(
            f"{strike:<10.1f} {result.mean[i]:10.4f}  {result.std[i]:10.4f}  "
            f"{lower[i]:13.4f}  {upper[i]:13.4f}"
        )

    # Step 4: Variance decomposition
    print("\n4. Variance Decomposition")
    print("-" * 80)
    avg_within = float(jnp.mean(result.within_model_variance))
    avg_between = float(jnp.mean(result.between_model_variance))
    avg_total = float(jnp.mean(result.variance))

    print(f"Within-model variance:  {avg_within:.6f} ({100*avg_within/avg_total:.1f}%)")
    print(f"Between-model variance: {avg_between:.6f} ({100*avg_between/avg_total:.1f}%)")
    print(f"Total variance:         {avg_total:.6f}")
    print(
        "\nInterpretation: "
        + ("High between-model variance suggests model uncertainty is important."
           if avg_between / avg_total > 0.3
           else "Low between-model variance suggests models agree.")
    )


# ==============================================================================
# Example 2: Stacking Weights
# ==============================================================================


def example_stacking_weights():
    """Example: Using stacking weights (optimized for prediction)."""
    print("\n\n" + "=" * 80)
    print("Example 2: Stacking Weights (Optimized for Prediction)")
    print("=" * 80)

    # Generate data
    strikes, market_prices = generate_market_data(n_options=50)

    # Calibrate models
    key = jax.random.PRNGKey(123)
    keys = jax.random.split(key, 3)

    heston_pred, heston_fit = simulate_heston_calibration(strikes, market_prices, keys[0])
    sabr_pred, sabr_fit = simulate_sabr_calibration(strikes, market_prices, keys[1])
    lv_pred, lv_fit = simulate_localvol_calibration(strikes, market_prices, keys[2])

    model_fits = {"Heston": heston_fit, "SABR": sabr_fit, "LocalVol": lv_fit}
    predictions = {"Heston": heston_pred, "SABR": sabr_pred, "LocalVol": lv_pred}

    # Compute BIC weights
    bic_weights = compute_weights_from_ic(model_fits, InformationCriterion.BIC)

    # Compute stacking weights (optimized to minimize out-of-sample error)
    print("\n1. Computing Stacking Weights")
    print("-" * 80)
    stacking_weights = compute_stacking_weights(model_fits, predictions, market_prices)

    # Compare
    print("\nComparison: BIC vs Stacking Weights")
    print("-" * 80)
    print(f"{'Model':<15} {'BIC Weight':<15} {'Stacking Weight':<20}")
    print("-" * 80)
    for model in ["Heston", "SABR", "LocalVol"]:
        print(
            f"{model:<15} {bic_weights.weights[model]:13.6f}  "
            f"{stacking_weights.weights[model]:18.6f}"
        )

    print("\nNote: Stacking weights are optimized for prediction accuracy,")
    print("while BIC weights approximate Bayesian posterior probabilities.")


# ==============================================================================
# Example 3: Custom Weights and Expert Knowledge
# ==============================================================================


def example_custom_weights():
    """Example: Using custom weights based on expert knowledge."""
    print("\n\n" + "=" * 80)
    print("Example 3: Custom Weights with Expert Knowledge")
    print("=" * 80)

    # Generate data
    strikes, market_prices = generate_market_data(n_options=30)

    # Simulate calibration
    key = jax.random.PRNGKey(456)
    keys = jax.random.split(key, 3)

    _, heston_fit = simulate_heston_calibration(strikes, market_prices, keys[0])
    _, sabr_fit = simulate_sabr_calibration(strikes, market_prices, keys[1])

    model_fits = {"Heston": heston_fit, "SABR": sabr_fit}

    # Scenario: Expert knows that for short-dated options,
    # SABR tends to perform better than Heston
    print("\nScenario: Short-dated options where SABR typically outperforms")
    print("-" * 80)

    # First, see what BIC suggests
    bma_bic = BayesianModelAveraging(model_fits, weighting_scheme=WeightingScheme.BIC)
    print("\nBIC-based weights:")
    for model, weight in bma_bic.weights.weights.items():
        print(f"  {model}: {weight:.3f}")

    # Override with expert knowledge
    expert_weights = {"Heston": 0.3, "SABR": 0.7}
    bma_expert = BayesianModelAveraging(
        model_fits,
        weighting_scheme=WeightingScheme.CUSTOM,
        custom_weights=expert_weights,
    )

    print("\nExpert-adjusted weights:")
    for model, weight in bma_expert.weights.weights.items():
        print(f"  {model}: {weight:.3f}")

    print("\nBenefit: Incorporates domain knowledge while still using BMA framework")


# ==============================================================================
# Example 4: Uncertainty Quantification
# ==============================================================================


def example_uncertainty_quantification():
    """Example: Visualizing prediction uncertainty."""
    print("\n\n" + "=" * 80)
    print("Example 4: Uncertainty Quantification")
    print("=" * 80)

    # Generate data
    strikes, market_prices = generate_market_data(n_options=40)

    # Calibrate models
    key = jax.random.PRNGKey(789)
    keys = jax.random.split(key, 3)

    heston_pred, heston_fit = simulate_heston_calibration(strikes, market_prices, keys[0])
    sabr_pred, sabr_fit = simulate_sabr_calibration(strikes, market_prices, keys[1])
    lv_pred, lv_fit = simulate_localvol_calibration(strikes, market_prices, keys[2])

    model_fits = {"Heston": heston_fit, "SABR": sabr_fit, "LocalVol": lv_fit}

    # Create BMA
    bma = BayesianModelAveraging(model_fits, weighting_scheme=WeightingScheme.BIC)

    # Predictions
    predictions = {"Heston": heston_pred, "SABR": sabr_pred, "LocalVol": lv_pred}
    result = bma.predict(predictions)

    # Get intervals
    lower_95, upper_95 = result.prediction_interval(0.95)
    lower_68, upper_68 = result.prediction_interval(0.68)

    print("\nUncertainty Analysis")
    print("-" * 80)
    print(f"Mean prediction std: {jnp.mean(result.std):.4f}")
    print(f"Mean 95% CI width:   {jnp.mean(upper_95 - lower_95):.4f}")
    print(f"Mean 68% CI width:   {jnp.mean(upper_68 - lower_68):.4f}")

    # Analyze which region has highest uncertainty
    uncertainty_idx = jnp.argmax(result.std)
    print(f"\nHighest uncertainty at strike {strikes[uncertainty_idx]:.1f}")
    print(f"  BMA price: {result.mean[uncertainty_idx]:.4f}")
    print(f"  Std dev:   {result.std[uncertainty_idx]:.4f}")
    print(f"  95% CI:    [{lower_95[uncertainty_idx]:.4f}, {upper_95[uncertainty_idx]:.4f}]")

    print("\nInterpretation: High uncertainty may indicate:")
    print("  - Models disagree on this region")
    print("  - Sparse calibration data in this region")
    print("  - Need for more sophisticated models")


# ==============================================================================
# Example 5: Model Probability Evolution
# ==============================================================================


def example_model_probabilities():
    """Example: Understanding posterior model probabilities."""
    print("\n\n" + "=" * 80)
    print("Example 5: Posterior Model Probabilities")
    print("=" * 80)

    # Generate data
    strikes, market_prices = generate_market_data(n_options=50)

    # Calibrate models
    key = jax.random.PRNGKey(999)
    keys = jax.random.split(key, 3)

    _, heston_fit = simulate_heston_calibration(strikes, market_prices, keys[0])
    _, sabr_fit = simulate_sabr_calibration(strikes, market_prices, keys[1])
    _, lv_fit = simulate_localvol_calibration(strikes, market_prices, keys[2])

    model_fits = {"Heston": heston_fit, "SABR": sabr_fit, "LocalVol": lv_fit}

    # Create BMA
    bma = BayesianModelAveraging(model_fits, weighting_scheme=WeightingScheme.BIC)

    print("\nPosterior Model Probabilities (via BIC)")
    print("-" * 80)
    probs = bma.model_probabilities()

    for model, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(prob * 50)
        print(f"{model:<15} {prob:6.4f} {bar}")

    print(f"\nEffective number of models: {bma.weights.effective_models:.2f}")

    if bma.weights.effective_models > 2.5:
        print("→ High model uncertainty: All models receive substantial weight")
        print("→ BMA particularly valuable here")
    elif bma.weights.effective_models < 1.5:
        print("→ Low model uncertainty: One model clearly dominates")
        print("→ Consider using the best model directly")
    else:
        print("→ Moderate model uncertainty: 2-3 models are competitive")
        print("→ BMA provides robustness")


# ==============================================================================
# Run Examples
# ==============================================================================


if __name__ == "__main__":
    # Run all examples
    example_basic_bma()
    example_stacking_weights()
    example_custom_weights()
    example_uncertainty_quantification()
    example_model_probabilities()

    print("\n\n" + "=" * 80)
    print("Examples completed!")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("  1. BMA provides robust predictions when model selection is uncertain")
    print("  2. BIC-based weights approximate Bayesian posterior probabilities")
    print("  3. Stacking weights optimize for prediction accuracy")
    print("  4. Variance decomposition reveals model vs parameter uncertainty")
    print("  5. Prediction intervals account for all sources of uncertainty")
    print("\nFor more details, see:")
    print("  - Hoeting et al. (1999): Bayesian model averaging: a tutorial")
    print("  - Raftery et al. (1997): Bayesian model selection in social research")
