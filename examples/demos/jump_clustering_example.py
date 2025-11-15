"""Advanced Jump Clustering Models Demo.

This example demonstrates the advanced jump clustering features in neutryx:
1. Multivariate Hawkes processes (cross-asset contagion)
2. Regime-switching jump clustering (crisis vs normal periods)
3. Intensity-dependent jump sizes (magnitude clustering)
4. Hawkes parameter calibration from data

These models capture realistic jump clustering behavior observed in financial markets,
where jumps tend to occur in clusters and can propagate across assets.
"""
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

from neutryx.core.engine import MCConfig
from neutryx.models.jump_clustering import (
    HawkesJumpParams,
    IntensityDependentJumpParams,
    MultivariateHawkesParams,
    RegimeSwitchingHawkesParams,
    calibrate_hawkes_from_jumps,
    simulate_hawkes_jump_diffusion,
    simulate_intensity_dependent_hawkes,
    simulate_multivariate_hawkes,
    simulate_regime_switching_hawkes,
)


def demo_multivariate_hawkes():
    """Demo 1: Multivariate Hawkes - Cross-asset jump contagion."""
    print("=" * 80)
    print("DEMO 1: Multivariate Hawkes Process (Cross-Asset Contagion)")
    print("=" * 80)
    print()

    # Setup: Model two assets with asymmetric contagion
    # Asset 0 (e.g., large-cap index) influences Asset 1 (e.g., small-cap)
    # more strongly than vice versa
    params = MultivariateHawkesParams(
        n_assets=2,
        sigma=jnp.array([0.15, 0.25]),  # Higher vol for asset 1
        lambda0=jnp.array([5.0, 3.0]),  # Baseline jump rates
        alpha=jnp.array(
            [
                [1.5, 0.5],  # Asset 0: self-excites strongly, weak from asset 1
                [1.8, 1.2],  # Asset 1: strong contagion from asset 0
            ]
        ),
        beta=jnp.ones((2, 2)) * 2.5,
        mu_jump=jnp.array([-0.05, -0.06]),  # Negative jumps (crashes)
        sigma_jump=jnp.array([0.10, 0.12]),
        correlation=jnp.array([[1.0, 0.6], [0.6, 1.0]]),  # Correlated diffusion
    )

    print(f"Number of assets: {params.n_assets}")
    print(f"Spectral radius (stability): {params.spectral_radius():.3f}")
    print(f"Branching matrix:\n{params.branching_matrix()}")
    print()

    # Market parameters
    S0 = jnp.array([100.0, 50.0])
    q = jnp.array([0.02, 0.01])
    T = 1.0
    r = 0.03

    # Simulation
    cfg = MCConfig(paths=1000, steps=252, dtype=jnp.float32)
    key = jr.PRNGKey(42)

    print("Simulating multivariate Hawkes jump-diffusion...")
    paths = simulate_multivariate_hawkes(key, S0, T, r, q, params, cfg)

    print(f"Simulated paths shape: {paths.shape}")
    print(
        f"Asset 0 final prices: mean={paths[:, -1, 0].mean():.2f}, std={paths[:, -1, 0].std():.2f}"
    )
    print(
        f"Asset 1 final prices: mean={paths[:, -1, 1].mean():.2f}, std={paths[:, -1, 1].std():.2f}"
    )
    print()

    # Visualize a few sample paths
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot sample paths
    time_grid = jnp.linspace(0, T, cfg.steps + 1)
    for i in range(5):
        ax1.plot(
            time_grid, paths[i, :, 0], alpha=0.7, label=f"Path {i+1}" if i == 0 else ""
        )
        ax2.plot(time_grid, paths[i, :, 1], alpha=0.7)

    ax1.set_title("Asset 0: Large-Cap Index")
    ax1.set_xlabel("Time (years)")
    ax1.set_ylabel("Price")
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Asset 1: Small-Cap Index (with contagion)")
    ax2.set_xlabel("Time (years)")
    ax2.set_ylabel("Price")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("/tmp/multivariate_hawkes_demo.png", dpi=150)
    print("Plot saved to /tmp/multivariate_hawkes_demo.png")
    print()


def demo_regime_switching_hawkes():
    """Demo 2: Regime-Switching Hawkes - Normal vs Crisis periods."""
    print("=" * 80)
    print("DEMO 2: Regime-Switching Hawkes (Normal vs Crisis)")
    print("=" * 80)
    print()

    # Setup: Two regimes with very different clustering behavior
    params = RegimeSwitchingHawkesParams(
        # Regime 0: Normal market (low clustering)
        # Regime 1: Crisis market (high clustering, supercritical!)
        sigma=jnp.array([0.15, 0.35]),  # Higher vol in crisis
        lambda0=jnp.array([2.0, 10.0]),  # More jumps in crisis
        alpha=jnp.array([0.6, 4.0]),  # Extreme clustering in crisis
        beta=jnp.array([2.0, 2.5]),
        mu_jump=jnp.array([-0.02, -0.12]),  # Larger drops in crisis
        sigma_jump=jnp.array([0.05, 0.20]),
        q_matrix=jnp.array(
            [
                [-0.3, 0.3],  # Switch to crisis slowly
                [1.5, -1.5],  # Exit crisis faster
            ]
        ),
    )

    print("Regime 0 (Normal): alpha/beta = {:.3f}".format(params.branching_ratios()[0]))
    print(
        "Regime 1 (Crisis): alpha/beta = {:.3f} (supercritical!)".format(
            params.branching_ratios()[1]
        )
    )
    print()

    # Market parameters
    S0 = 100.0
    T = 2.0  # Longer horizon to see regime switches
    r = 0.03
    q = 0.02

    # Simulation
    cfg = MCConfig(paths=500, steps=504, dtype=jnp.float32)
    key = jr.PRNGKey(123)

    print("Simulating regime-switching Hawkes...")
    paths, regimes = simulate_regime_switching_hawkes(
        key, S0, T, r, q, params, cfg, initial_regime=0
    )

    print(f"Price paths shape: {paths.shape}")
    print(f"Regime paths shape: {regimes.shape}")
    print(
        f"Time in regime 0 (normal): {(regimes == 0).mean() * 100:.1f}%"
    )
    print(f"Time in regime 1 (crisis): {(regimes == 1).mean() * 100:.1f}%")
    print()

    # Visualize regime-dependent behavior
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    time_grid = jnp.linspace(0, T, cfg.steps + 1)

    # Plot a single path with regime shading
    path_idx = 0
    ax1.plot(time_grid, paths[path_idx, :], linewidth=1.5, color="navy")

    # Shade background by regime
    for i in range(len(time_grid) - 1):
        if regimes[path_idx, i] == 1:  # Crisis regime
            ax1.axvspan(
                time_grid[i],
                time_grid[i + 1],
                alpha=0.2,
                color="red",
                label="Crisis" if i == 0 else "",
            )

    ax1.set_ylabel("Asset Price")
    ax1.set_title("Price Path with Regime Shading (Red = Crisis)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot regime indicator
    ax2.plot(time_grid, regimes[path_idx, :], linewidth=2, color="darkred")
    ax2.set_xlabel("Time (years)")
    ax2.set_ylabel("Regime")
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["Normal", "Crisis"])
    ax2.set_title("Market Regime")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("/tmp/regime_switching_hawkes_demo.png", dpi=150)
    print("Plot saved to /tmp/regime_switching_hawkes_demo.png")
    print()


def demo_intensity_dependent_jumps():
    """Demo 3: Intensity-Dependent Jump Sizes - Magnitude clustering."""
    print("=" * 80)
    print("DEMO 3: Intensity-Dependent Jump Sizes (Magnitude Clustering)")
    print("=" * 80)
    print()

    # Compare standard vs intensity-dependent jump sizes
    # Standard: Jump sizes independent of clustering intensity
    # Intensity-dependent: Larger jumps during high-intensity periods

    params_standard = IntensityDependentJumpParams(
        sigma=0.20,
        lambda0=5.0,
        alpha=2.0,
        beta=2.5,
        mu_jump_base=-0.04,
        sigma_jump_base=0.10,
        intensity_sensitivity=0.0,  # No dependency
    )

    params_dependent = IntensityDependentJumpParams(
        sigma=0.20,
        lambda0=5.0,
        alpha=2.0,
        beta=2.5,
        mu_jump_base=-0.04,
        sigma_jump_base=0.10,
        intensity_sensitivity=0.25,  # Jumps grow with intensity
    )

    print("Standard Hawkes: Jump sizes constant")
    print(
        f"Intensity-dependent: Jump sizes scale with log(λ/λ₀) * {params_dependent.intensity_sensitivity}"
    )
    print()

    # Show how jump size changes with intensity
    intensities = jnp.array([5.0, 10.0, 20.0, 50.0])
    print("Jump size sensitivity:")
    print(f"{'Intensity':<12} {'Standard μ':<15} {'Dependent μ':<15} {'Difference'}")
    print("-" * 60)
    for intensity in intensities:
        mu_std, _ = params_standard.jump_size_params(intensity)
        mu_dep, _ = params_dependent.jump_size_params(intensity)
        print(
            f"{intensity:<12.1f} {mu_std:<15.4f} {mu_dep:<15.4f} {mu_dep - mu_std:.4f}"
        )
    print()

    # Market parameters
    S0 = 100.0
    T = 1.0
    r = 0.03
    q = 0.01

    # Simulation
    cfg = MCConfig(paths=1000, steps=252, dtype=jnp.float32)
    key = jr.PRNGKey(456)

    print("Simulating both models...")
    key1, key2 = jr.split(key)

    paths_standard = simulate_intensity_dependent_hawkes(
        key1, S0, T, r, q, params_standard, cfg
    )
    paths_dependent = simulate_intensity_dependent_hawkes(
        key2, S0, T, r, q, params_dependent, cfg
    )

    print(f"Standard model final price: mean={paths_standard[:, -1].mean():.2f}")
    print(f"Intensity-dependent final price: mean={paths_dependent[:, -1].mean():.2f}")
    print()

    # Compare distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Sample paths
    time_grid = jnp.linspace(0, T, cfg.steps + 1)
    for i in range(10):
        ax1.plot(time_grid, paths_standard[i, :], alpha=0.4, color="blue")
        ax1.plot(
            time_grid,
            paths_dependent[i, :],
            alpha=0.4,
            color="red",
            linestyle="--",
        )

    ax1.plot([], [], color="blue", label="Standard")
    ax1.plot([], [], color="red", linestyle="--", label="Intensity-Dependent")
    ax1.set_title("Sample Paths")
    ax1.set_xlabel("Time (years)")
    ax1.set_ylabel("Price")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Terminal distribution
    ax2.hist(
        paths_standard[:, -1],
        bins=50,
        alpha=0.5,
        color="blue",
        label="Standard",
        density=True,
    )
    ax2.hist(
        paths_dependent[:, -1],
        bins=50,
        alpha=0.5,
        color="red",
        label="Intensity-Dependent",
        density=True,
    )
    ax2.set_title("Terminal Price Distribution")
    ax2.set_xlabel("Price")
    ax2.set_ylabel("Density")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("/tmp/intensity_dependent_demo.png", dpi=150)
    print("Plot saved to /tmp/intensity_dependent_demo.png")
    print()


def demo_hawkes_calibration():
    """Demo 4: Hawkes Parameter Calibration from observed jump times."""
    print("=" * 80)
    print("DEMO 4: Hawkes Parameter Calibration")
    print("=" * 80)
    print()

    # Simulate "observed" jump times from a known Hawkes process
    # Then calibrate to see if we recover the parameters

    print("Step 1: Generate synthetic jump times from known Hawkes process")
    true_params = HawkesJumpParams(
        sigma=0.15,
        lambda0=4.0,
        alpha=2.5,
        beta=3.0,
        mu_jump=-0.03,
        sigma_jump=0.08,
    )

    print(f"True parameters:")
    print(f"  lambda0 = {true_params.lambda0}")
    print(f"  alpha = {true_params.alpha}")
    print(f"  beta = {true_params.beta}")
    print(f"  Branching ratio = {true_params.branching_ratio():.3f}")
    print()

    # Simulate and extract jump times
    S0 = 100.0
    T = 5.0  # Longer horizon for more jumps
    r = 0.03
    q = 0.01
    cfg = MCConfig(paths=1, steps=1000, dtype=jnp.float32)
    key = jr.PRNGKey(789)

    paths = simulate_hawkes_jump_diffusion(key, S0, T, r, q, true_params, cfg)

    # Detect jumps (simplified: look for large returns)
    returns = jnp.diff(jnp.log(paths[0, :]))
    jump_threshold = 0.02  # 2% move
    jump_indices = jnp.where(jnp.abs(returns) > jump_threshold)[0]

    time_grid = jnp.linspace(0, T, cfg.steps + 1)
    jump_times = time_grid[jump_indices + 1]

    print(f"Detected {len(jump_times)} jumps over {T} years")
    print(f"Average jump rate: {len(jump_times) / T:.2f} per year")
    print()

    # Calibrate using both methods
    print("Step 2: Calibrate parameters from jump times")
    print()

    print("Method 1: Method of Moments")
    params_moments = calibrate_hawkes_from_jumps(jump_times, method="moments")
    print(f"  lambda0 = {params_moments.lambda0:.3f} (true: {true_params.lambda0})")
    print(f"  alpha = {params_moments.alpha:.3f} (true: {true_params.alpha})")
    print(f"  beta = {params_moments.beta:.3f} (true: {true_params.beta})")
    print(
        f"  Branching ratio = {params_moments.branching_ratio():.3f} (true: {true_params.branching_ratio():.3f})"
    )
    print()

    print("Method 2: Maximum Likelihood Estimation")
    params_mle = calibrate_hawkes_from_jumps(jump_times, method="mle")
    print(f"  lambda0 = {params_mle.lambda0:.3f} (true: {true_params.lambda0})")
    print(f"  alpha = {params_mle.alpha:.3f} (true: {true_params.alpha})")
    print(f"  beta = {params_mle.beta:.3f} (true: {true_params.beta})")
    print(
        f"  Branching ratio = {params_mle.branching_ratio():.3f} (true: {true_params.branching_ratio():.3f})"
    )
    print()

    # Visualize jump clustering
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    ax.plot(time_grid, paths[0, :], linewidth=1.5, alpha=0.7, label="Price Path")
    ax.scatter(
        jump_times,
        paths[0, jump_indices + 1],
        color="red",
        s=50,
        zorder=5,
        label="Detected Jumps",
    )

    # Add vertical lines for jump clusters
    for jt in jump_times:
        ax.axvline(jt, color="red", alpha=0.2, linewidth=1)

    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Asset Price")
    ax.set_title(f"Hawkes Process with {len(jump_times)} Jumps (Clustering Visible)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("/tmp/hawkes_calibration_demo.png", dpi=150)
    print("Plot saved to /tmp/hawkes_calibration_demo.png")
    print()


def main():
    """Run all demos."""
    print()
    print("*" * 80)
    print("ADVANCED JUMP CLUSTERING MODELS - COMPREHENSIVE DEMO")
    print("*" * 80)
    print()
    print("This demo showcases:")
    print("  1. Multivariate Hawkes (cross-asset contagion)")
    print("  2. Regime-Switching Hawkes (crisis vs normal)")
    print("  3. Intensity-Dependent Jump Sizes (magnitude clustering)")
    print("  4. Hawkes Parameter Calibration (from observed data)")
    print()

    # Run all demos
    demo_multivariate_hawkes()
    demo_regime_switching_hawkes()
    demo_intensity_dependent_jumps()
    demo_hawkes_calibration()

    print("*" * 80)
    print("ALL DEMOS COMPLETED!")
    print("*" * 80)
    print()
    print("Key Takeaways:")
    print(
        "  - Multivariate Hawkes capture jump contagion across assets (systemic risk)"
    )
    print(
        "  - Regime-switching allows different clustering in normal vs crisis periods"
    )
    print(
        "  - Intensity-dependent jumps create more realistic crisis dynamics"
    )
    print(
        "  - Calibration methods enable fitting models to historical jump data"
    )
    print()


if __name__ == "__main__":
    main()
