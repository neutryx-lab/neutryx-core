"""Demonstration of regularization and stability techniques for model calibration.

This script showcases the use of various regularization methods to ensure
stable, well-behaved calibration of financial models:

1. Tikhonov regularization for parameter estimation
2. L1/L2 penalties for sparse/smooth solutions
3. Smoothness regularization for volatility surfaces
4. Arbitrage-free constraints

Each example demonstrates:
- Problem setup
- Regularization configuration
- Impact on calibration results
- Comparison with/without regularization
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm

from neutryx.calibration.regularization import (
    TikhonovRegularizer,
    L1Regularizer,
    ElasticNetRegularizer,
    SmoothnessRegularizer,
    ArbitrageFreeConstraints,
    create_difference_matrix,
)


def demo_tikhonov_regularization():
    """Demonstrate Tikhonov regularization for stable parameter estimation."""
    print("\n" + "=" * 80)
    print("TIKHONOV REGULARIZATION")
    print("=" * 80)

    print("\nProblem: Estimate parameters from noisy data")
    print("Without regularization: Overfitting to noise")
    print("With regularization: Stable, smooth estimates")

    # Synthetic data: parameters with noise
    np.random.seed(42)
    n_params = 20
    true_params = np.sin(np.linspace(0, 2 * np.pi, n_params))
    noise = np.random.normal(0, 0.5, n_params)
    noisy_data = true_params + noise

    print(f"\nGenerated {n_params} noisy observations")
    print(f"Noise level: σ = 0.5")

    # Without regularization: direct fit
    params_unregularized = noisy_data.copy()

    # With Tikhonov regularization
    # Minimize: ||params - data||² + λ ||D * params||²
    # where D is second difference matrix (penalizes roughness)
    D = create_difference_matrix(n_params, order=2)
    tikhonov = TikhonovRegularizer(
        lambda_reg=10.0,
        regularization_matrix=D,
    )

    def objective(params):
        data_fit = jnp.sum((params - noisy_data) ** 2)
        return data_fit

    reg_objective = tikhonov.regularized_objective(objective)

    result = minimize(reg_objective, x0=noisy_data, method='L-BFGS-B')
    params_regularized = result.x

    # Compute errors
    error_unreg = np.mean((params_unregularized - true_params) ** 2)
    error_reg = np.mean((params_regularized - true_params) ** 2)

    print(f"\nResults:")
    print(f"  MSE without regularization: {error_unreg:.4f}")
    print(f"  MSE with regularization:    {error_reg:.4f}")
    print(f"  Improvement: {(error_unreg - error_reg) / error_unreg * 100:.1f}%")

    print(f"\n✓ Tikhonov regularization reduces overfitting")
    print(f"✓ Produces smoother, more stable parameter estimates")


def demo_l1_sparsity():
    """Demonstrate L1 regularization for sparse solutions."""
    print("\n" + "=" * 80)
    print("L1 (LASSO) REGULARIZATION")
    print("=" * 80)

    print("\nProblem: Select important features from many candidates")
    print("L1 penalty drives unimportant coefficients to exactly zero")

    # Sparse linear regression
    np.random.seed(42)
    n_features = 50
    n_samples = 100

    # Only 5 features are truly important
    true_coefficients = np.zeros(n_features)
    important_features = [5, 15, 25, 35, 45]
    true_coefficients[important_features] = [2.0, -1.5, 3.0, -2.5, 1.0]

    # Generate data
    X = np.random.randn(n_samples, n_features)
    y = X @ true_coefficients + np.random.randn(n_samples) * 0.5

    print(f"\nData: {n_samples} samples, {n_features} features")
    print(f"True model: Only {len(important_features)} features are relevant")

    # L1 regularization
    l1_reg = L1Regularizer(lambda_reg=0.5)

    def mse_objective(coeffs):
        predictions = X @ coeffs
        return jnp.mean((predictions - y) ** 2)

    reg_objective = l1_reg.regularized_objective(mse_objective)

    result = minimize(
        reg_objective,
        x0=np.zeros(n_features),
        method='L-BFGS-B',
    )

    coeffs_l1 = result.x

    # Count non-zero coefficients
    n_nonzero = np.sum(np.abs(coeffs_l1) > 0.01)

    print(f"\nResults:")
    print(f"  Number of non-zero coefficients: {n_nonzero}")
    print(f"  True number of relevant features: {len(important_features)}")

    # Show which features were selected
    selected = np.where(np.abs(coeffs_l1) > 0.01)[0]
    print(f"  Selected features: {selected.tolist()}")
    print(f"  True relevant features: {important_features}")

    # Compute overlap
    overlap = len(set(selected) & set(important_features))
    print(f"  Correct selections: {overlap}/{len(important_features)}")

    print(f"\n✓ L1 regularization induces sparsity")
    print(f"✓ Automatically selects relevant features")


def demo_elastic_net():
    """Demonstrate Elastic Net combining L1 and L2."""
    print("\n" + "=" * 80)
    print("ELASTIC NET (L1 + L2)")
    print("=" * 80)

    print("\nElastic Net combines benefits of L1 and L2:")
    print("  • L1: Feature selection (sparsity)")
    print("  • L2: Stability and grouping of correlated features")

    # Problem with correlated features
    np.random.seed(42)
    n_samples = 100
    n_features = 20

    # Create correlated feature groups
    X_base = np.random.randn(n_samples, 5)
    X = np.hstack([
        X_base[:, 0:1] + np.random.randn(n_samples, 3) * 0.1,  # Group 1
        X_base[:, 1:2] + np.random.randn(n_samples, 3) * 0.1,  # Group 2
        X_base[:, 2:3] + np.random.randn(n_samples, 3) * 0.1,  # Group 3
        np.random.randn(n_samples, 11)  # Noise features
    ])

    # True model uses first group
    true_coeffs = np.zeros(n_features)
    true_coeffs[0:3] = [2.0, 1.8, 2.2]  # Correlated features all relevant

    y = X @ true_coeffs + np.random.randn(n_samples) * 0.5

    print(f"\nData: {n_features} features with 3 correlated groups")
    print(f"True model: Features 0-2 (correlated group 1) are relevant")

    # Compare L1, L2, and Elastic Net
    def mse(coeffs):
        return jnp.mean((X @ coeffs - y) ** 2)

    # L1 only
    l1 = L1Regularizer(lambda_reg=0.3)
    result_l1 = minimize(l1.regularized_objective(mse), x0=np.zeros(n_features))
    coeffs_l1 = result_l1.x

    # L2 only
    l2 = TikhonovRegularizer(lambda_reg=0.3)
    result_l2 = minimize(l2.regularized_objective(mse), x0=np.zeros(n_features))
    coeffs_l2 = result_l2.x

    # Elastic Net
    elastic = ElasticNetRegularizer(lambda_l1=0.3, lambda_l2=0.3)
    result_elastic = minimize(elastic.regularized_objective(mse), x0=np.zeros(n_features))
    coeffs_elastic = result_elastic.x

    print(f"\nResults:")
    print(f"  L1 only:")
    print(f"    Non-zero coefficients: {np.sum(np.abs(coeffs_l1) > 0.01)}")
    print(f"    Selected from group 1: {np.sum(np.abs(coeffs_l1[0:3]) > 0.01)}")

    print(f"  L2 only:")
    print(f"    Non-zero coefficients: {np.sum(np.abs(coeffs_l2) > 0.01)}")
    print(f"    Selected from group 1: {np.sum(np.abs(coeffs_l2[0:3]) > 0.01)}")

    print(f"  Elastic Net:")
    print(f"    Non-zero coefficients: {np.sum(np.abs(coeffs_elastic) > 0.01)}")
    print(f"    Selected from group 1: {np.sum(np.abs(coeffs_elastic[0:3]) > 0.01)}")

    print(f"\n✓ Elastic Net balances sparsity and stability")
    print(f"✓ Better handles correlated features than L1 alone")


def demo_smoothness_volatility_surface():
    """Demonstrate smoothness regularization for volatility surfaces."""
    print("\n" + "=" * 80)
    print("SMOOTHNESS REGULARIZATION FOR VOLATILITY SURFACES")
    print("=" * 80)

    print("\nProblem: Calibrate implied volatility surface from sparse market data")
    print("Challenge: Market data is sparse and noisy")
    print("Solution: Smoothness penalty ensures realistic surface")

    # Create synthetic market data (sparse)
    strikes = np.array([80, 90, 100, 110, 120])
    maturities = np.array([0.25, 0.5, 1.0, 2.0])

    # True smooth surface: σ(K, T) = 0.2 + 0.1 * |log(K/S)| + 0.05 * T
    S = 100.0
    true_surface = np.zeros((len(maturities), len(strikes)))
    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            moneyness = np.log(K / S)
            true_surface[i, j] = 0.2 + 0.1 * abs(moneyness) + 0.05 * T

    # Add noise to simulate market observations
    np.random.seed(42)
    noisy_surface = true_surface + np.random.normal(0, 0.02, true_surface.shape)

    print(f"\nMarket data: {len(maturities)} × {len(strikes)} grid")
    print(f"Noise level: σ = 0.02 (2% vol points)")

    # Calibrate without smoothness
    surface_unsmoothed = noisy_surface.copy()

    # Calibrate with smoothness regularization
    smoothness = SmoothnessRegularizer(
        lambda_reg=1.0,
        order=2,
        direction="both",
    )

    def data_fit(surface_flat):
        surface = surface_flat.reshape(true_surface.shape)
        return jnp.sum((surface - noisy_surface) ** 2)

    reg_objective = smoothness.regularized_objective(data_fit, true_surface.shape)

    result = minimize(
        reg_objective,
        x0=noisy_surface.flatten(),
        method='L-BFGS-B',
        bounds=[(0.05, 1.0)] * len(noisy_surface.flatten()),
    )

    surface_smoothed = result.x.reshape(true_surface.shape)

    # Compute errors
    error_unsmoothed = np.mean((surface_unsmoothed - true_surface) ** 2)
    error_smoothed = np.mean((surface_smoothed - true_surface) ** 2)

    print(f"\nResults:")
    print(f"  MSE without smoothing: {error_unsmoothed:.6f}")
    print(f"  MSE with smoothing:    {error_smoothed:.6f}")
    print(f"  Improvement: {(error_unsmoothed - error_smoothed) / error_unsmoothed * 100:.1f}%")

    # Compute smoothness metrics
    smoothness_unsmoothed = smoothness.penalty_2d(surface_unsmoothed)
    smoothness_smoothed = smoothness.penalty_2d(surface_smoothed)

    print(f"\nSmoothness metric (lower is smoother):")
    print(f"  Without regularization: {smoothness_unsmoothed:.6f}")
    print(f"  With regularization:    {smoothness_smoothed:.6f}")

    print(f"\n✓ Smoothness regularization reduces overfitting to noise")
    print(f"✓ Produces realistic, tradeable volatility surfaces")


def demo_arbitrage_free_constraints():
    """Demonstrate arbitrage-free constraint enforcement."""
    print("\n" + "=" * 80)
    print("ARBITRAGE-FREE CONSTRAINTS")
    print("=" * 80)

    print("\nProblem: Ensure calibrated volatility surface is arbitrage-free")
    print("Constraints:")
    print("  1. Calendar spread: σ²(T)·T must increase with T")
    print("  2. Butterfly: Call prices convex in strike")

    # Create a surface with potential arbitrage
    strikes = np.array([90, 95, 100, 105, 110])
    maturities = np.array([0.25, 0.5, 1.0])

    # Initial surface with calendar arbitrage
    initial_surface = np.array([
        [0.25, 0.23, 0.20, 0.23, 0.25],  # 3M
        [0.22, 0.21, 0.19, 0.21, 0.22],  # 6M  (lower variance!)
        [0.28, 0.26, 0.24, 0.26, 0.28],  # 1Y
    ])

    print(f"\nInitial surface:")
    print(f"  3M ATM vol: {initial_surface[0, 2]:.2%}")
    print(f"  6M ATM vol: {initial_surface[1, 2]:.2%}")
    print(f"  1Y ATM vol: {initial_surface[2, 2]:.2%}")

    # Check for arbitrage
    constraints = ArbitrageFreeConstraints(
        lambda_calendar=1000.0,
        lambda_butterfly=1000.0,
    )

    S = 100.0
    r = 0.05

    initial_penalty = constraints.total_penalty(
        initial_surface, maturities, strikes, S, r
    )

    print(f"\nInitial arbitrage penalty: {initial_penalty:.2f}")

    if initial_penalty > 0:
        print(f"  ⚠ Surface contains arbitrage opportunities!")

    # Enforce arbitrage-free constraints
    def objective(surface_flat):
        # Just want to minimize arbitrage violations
        surface = surface_flat.reshape(initial_surface.shape)
        arb_penalty = constraints.total_penalty(surface, maturities, strikes, S, r)

        # Small penalty for deviation from initial
        deviation = jnp.sum((surface - initial_surface) ** 2)

        return arb_penalty + 0.1 * deviation

    result = minimize(
        objective,
        x0=initial_surface.flatten(),
        method='L-BFGS-B',
        bounds=[(0.05, 1.0)] * len(initial_surface.flatten()),
    )

    adjusted_surface = result.x.reshape(initial_surface.shape)

    final_penalty = constraints.total_penalty(
        adjusted_surface, maturities, strikes, S, r
    )

    print(f"\nAdjusted surface:")
    print(f"  3M ATM vol: {adjusted_surface[0, 2]:.2%}")
    print(f"  6M ATM vol: {adjusted_surface[1, 2]:.2%}")
    print(f"  1Y ATM vol: {adjusted_surface[2, 2]:.2%}")

    print(f"\nFinal arbitrage penalty: {final_penalty:.2f}")

    if final_penalty < 1.0:
        print(f"  ✓ Surface is now arbitrage-free!")

    print(f"\n✓ Arbitrage-free constraints prevent impossible prices")
    print(f"✓ Essential for model calibration and risk management")


def comparison_summary():
    """Provide comparison summary of regularization techniques."""
    print("\n" + "=" * 80)
    print("REGULARIZATION TECHNIQUES COMPARISON")
    print("=" * 80)

    summary = """
╔═══════════════════╦══════════════════╦═════════════════╦══════════════════════╗
║ Technique         ║ Purpose          ║ Effect          ║ Best For             ║
╠═══════════════════╬══════════════════╬═════════════════╬══════════════════════╣
║ Tikhonov (L2)     ║ Stability        ║ Smooth params   ║ Ill-posed problems,  ║
║                   ║                  ║                 ║ noisy data           ║
╠═══════════════════╬══════════════════╬═════════════════╬══════════════════════╣
║ L1 (Lasso)        ║ Sparsity         ║ Zero params     ║ Feature selection,   ║
║                   ║                  ║                 ║ parsimonious models  ║
╠═══════════════════╬══════════════════╬═════════════════╬══════════════════════╣
║ Elastic Net       ║ Balance          ║ Sparse + smooth ║ Correlated features, ║
║                   ║                  ║                 ║ grouped selection    ║
╠═══════════════════╬══════════════════╬═════════════════╬══════════════════════╣
║ Smoothness        ║ Surface quality  ║ Low curvature   ║ Vol surfaces, local  ║
║                   ║                  ║                 ║ vol, term structures ║
╠═══════════════════╬══════════════════╬═════════════════╬══════════════════════╣
║ Arbitrage-Free    ║ No arbitrage     ║ Valid prices    ║ Option pricing, vol  ║
║ Constraints       ║                  ║                 ║ calibration          ║
╚═══════════════════╩══════════════════╩═════════════════╩══════════════════════╝

Key Guidelines:

1. **Model Calibration:**
   - Start with strong regularization, then reduce
   - Use Tikhonov for initial stability
   - Add smoothness for surfaces

2. **Volatility Surfaces:**
   - Always use smoothness regularization
   - Enforce arbitrage-free constraints
   - Typical λ_smooth ∈ [0.01, 1.0]
   - Typical λ_arbitrage ∈ [100, 10000]

3. **Parameter Estimation:**
   - L1 for feature selection
   - L2 for stability
   - Elastic Net for balanced approach

4. **Practical Considerations:**
   - Cross-validation for λ selection
   - Monitor both fit quality and regularization penalty
   - Start conservative (strong regularization)
   - Ensure numerical stability (avoid λ = 0)

5. **Common Mistakes to Avoid:**
   - ❌ No regularization with sparse/noisy data
   - ❌ Ignoring arbitrage constraints
   - ❌ Over-regularizing (underfitting)
   - ❌ Not validating on out-of-sample data

6. **Financial Applications:**
   - **Local Volatility:** Smoothness + arbitrage-free
   - **Stochastic Volatility:** Tikhonov for parameters
   - **Interest Rate Models:** L2 + smoothness
   - **Credit Models:** L1 for factor selection
"""

    print(summary)


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 80)
    print(" " * 15 + "REGULARIZATION & STABILITY TECHNIQUES")
    print(" " * 20 + "Neutryx Demonstration")
    print("=" * 80)

    # Run demonstrations
    demo_tikhonov_regularization()
    demo_l1_sparsity()
    demo_elastic_net()
    demo_smoothness_volatility_surface()
    demo_arbitrage_free_constraints()

    # Show comparison
    comparison_summary()

    print("\n" + "=" * 80)
    print("All demonstrations completed successfully!")
    print("\nKey Takeaways:")
    print("  • Regularization is essential for stable calibration")
    print("  • Different techniques serve different purposes")
    print("  • Combine multiple regularizers for best results")
    print("  • Always enforce arbitrage-free constraints")
    print("  • Cross-validate regularization strength")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
