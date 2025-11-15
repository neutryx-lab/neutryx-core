"""Adaptive Mesh Refinement (AMR) for PDE Solvers - Demonstration.

This example demonstrates the power of adaptive mesh refinement for option pricing using PDEs.
We'll compare uniform grid vs adaptive grid solutions, showing:
1. How AMR automatically refines near regions of interest (strike, barriers)
2. Convergence improvements with AMR
3. Computational efficiency gains
4. Grid visualization and error analysis

The demo prices European options and shows the automatic grid refinement process.
"""

import time

import jax.numpy as jnp
import matplotlib.pyplot as plt

from neutryx.models.amr import (
    AdaptivePDEGrid,
    adaptive_crank_nicolson,
    estimate_curvature_error,
    estimate_gradient_error,
)
from neutryx.models.pde import PDEGrid, crank_nicolson_european


def demo_basic_amr():
    """Demo 1: Basic AMR vs uniform grid comparison."""
    print("=" * 80)
    print("DEMO 1: Basic Adaptive Mesh Refinement")
    print("=" * 80)
    print()

    # Market parameters
    strike = 100.0
    r = 0.05
    q = 0.02
    sigma = 0.30  # High vol to create interesting dynamics
    T = 1.0

    print(f"Pricing European call option:")
    print(f"  Strike: {strike}")
    print(f"  Rate: {r:.1%}, Dividend: {q:.1%}, Volatility: {sigma:.1%}")
    print(f"  Maturity: {T} years")
    print()

    def payoff(S):
        return jnp.maximum(S - strike, 0.0)

    # Solve with uniform grid
    print("Solving with UNIFORM grid (101 points)...")
    uniform_grid = PDEGrid(S_min=50.0, S_max=150.0, T=T, N_S=101, N_T=100)
    start = time.time()
    V_uniform, S_uniform = crank_nicolson_european(
        uniform_grid, payoff, r, q, sigma, option_type="call", strike=strike
    )
    uniform_time = time.time() - start
    print(f"  Time: {uniform_time:.4f}s")
    print(f"  Grid points: {uniform_grid.N_S}")
    print()

    # Solve with adaptive grid
    print("Solving with ADAPTIVE grid (starting at 51 points)...")
    adaptive_grid_init = AdaptivePDEGrid.from_uniform(
        S_min=50.0, S_max=150.0, T=T, N_S=51, N_T=100
    )

    start = time.time()
    V_adaptive, adaptive_grid_final, history = adaptive_crank_nicolson(
        grid=adaptive_grid_init,
        payoff_fn=payoff,
        r=r,
        q=q,
        sigma=sigma,
        option_type="call",
        strike=strike,
        tolerance=0.05,
        max_refinements=3,
        verbose=True,
    )
    adaptive_time = time.time() - start
    print(f"\n  Total time: {adaptive_time:.4f}s")
    print(f"  Final grid points: {adaptive_grid_final.N_S}")
    print(f"  Refinement iterations: {len(history)}")
    print()

    # Compare results at strike
    strike_idx_uniform = jnp.argmin(jnp.abs(S_uniform - strike))
    strike_idx_adaptive = jnp.argmin(jnp.abs(adaptive_grid_final.S_grid - strike))

    value_uniform = V_uniform[strike_idx_uniform, 0]
    value_adaptive = V_adaptive[strike_idx_adaptive]

    print(f"Option value at strike:")
    print(f"  Uniform grid:  {value_uniform:.4f}")
    print(f"  Adaptive grid: {value_adaptive:.4f}")
    print(f"  Difference:    {abs(value_uniform - value_adaptive):.6f}")
    print()

    # Visualize grids
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Grid distribution
    ax = axes[0, 0]
    ax.plot(S_uniform, jnp.zeros_like(S_uniform), 'b.', alpha=0.5, label='Uniform')
    ax.plot(
        adaptive_grid_final.S_grid,
        jnp.zeros_like(adaptive_grid_final.S_grid),
        'r.',
        alpha=0.7,
        label='Adaptive (final)',
    )
    ax.axvline(strike, color='k', linestyle='--', alpha=0.5, label='Strike')
    ax.set_xlabel('Spot Price')
    ax.set_yticks([])
    ax.set_title('Grid Point Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Grid spacing
    ax = axes[0, 1]
    uniform_spacing = jnp.diff(S_uniform)
    adaptive_spacing = jnp.diff(adaptive_grid_final.S_grid)

    ax.plot(
        S_uniform[:-1],
        uniform_spacing,
        'b-',
        linewidth=2,
        label='Uniform',
        alpha=0.7,
    )
    ax.plot(
        adaptive_grid_final.S_grid[:-1],
        adaptive_spacing,
        'r-',
        linewidth=2,
        label='Adaptive',
        alpha=0.7,
    )
    ax.axvline(strike, color='k', linestyle='--', alpha=0.5, label='Strike')
    ax.set_xlabel('Spot Price')
    ax.set_ylabel('Local Grid Spacing')
    ax.set_title('Grid Spacing (smaller = finer resolution)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Option values
    ax = axes[1, 0]
    ax.plot(S_uniform, V_uniform[:, 0], 'b-', linewidth=2, label='Uniform', alpha=0.7)
    ax.plot(
        adaptive_grid_final.S_grid,
        V_adaptive,
        'r--',
        linewidth=2,
        label='Adaptive',
        alpha=0.7,
    )
    ax.axvline(strike, color='k', linestyle='--', alpha=0.5, label='Strike')
    ax.set_xlabel('Spot Price')
    ax.set_ylabel('Option Value')
    ax.set_title('Option Value Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Convergence history
    ax = axes[1, 1]
    if history:
        iterations = [h['iteration'] for h in history]
        n_points = [h['n_points'] for h in history]
        mean_errors = [h['mean_error'] for h in history]

        ax2 = ax.twinx()
        ax.plot(
            iterations, n_points, 'bo-', linewidth=2, markersize=8, label='Grid Points'
        )
        ax2.plot(
            iterations,
            mean_errors,
            'rs-',
            linewidth=2,
            markersize=8,
            label='Mean Error',
        )

        ax.set_xlabel('Refinement Iteration')
        ax.set_ylabel('Number of Grid Points', color='b')
        ax2.set_ylabel('Mean Error', color='r')
        ax.set_title('AMR Convergence History')
        ax.tick_params(axis='y', labelcolor='b')
        ax2.tick_params(axis='y', labelcolor='r')
        ax.grid(True, alpha=0.3)

        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    plt.savefig('/tmp/amr_basic_demo.png', dpi=150, bbox_inches='tight')
    print(f"Saved visualization to /tmp/amr_basic_demo.png")
    print()


def demo_error_indicators():
    """Demo 2: Error indicator visualization."""
    print("=" * 80)
    print("DEMO 2: Error Indicators and Refinement Criteria")
    print("=" * 80)
    print()

    # Simple example with known solution
    strike = 100.0
    S_grid = jnp.linspace(50.0, 150.0, 101)

    # Terminal payoff (call option)
    payoff = jnp.maximum(S_grid - strike, 0.0)

    print("Analyzing error indicators for call option payoff...")
    print()

    # Compute error indicators
    gradient_error = estimate_gradient_error(payoff, S_grid)
    curvature_error = estimate_curvature_error(payoff, S_grid)

    print(f"Maximum gradient error:  {jnp.max(gradient_error):.4f}")
    print(f"Maximum curvature error: {jnp.max(curvature_error):.4f}")
    print()

    # Find regions with high error
    high_grad_regions = jnp.where(gradient_error > 0.5)[0]
    high_curv_regions = jnp.where(curvature_error > 0.5)[0]

    print(f"Regions with high gradient error: {len(high_grad_regions)} points")
    print(f"Regions with high curvature error: {len(high_curv_regions)} points")
    print()

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Payoff function
    ax = axes[0, 0]
    ax.plot(S_grid, payoff, 'b-', linewidth=2)
    ax.axvline(strike, color='r', linestyle='--', alpha=0.7, label='Strike (kink)')
    ax.set_xlabel('Spot Price')
    ax.set_ylabel('Payoff')
    ax.set_title('Call Option Payoff (Kink at Strike)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Gradient error
    ax = axes[0, 1]
    ax.plot(S_grid, gradient_error, 'g-', linewidth=2)
    ax.axvline(strike, color='r', linestyle='--', alpha=0.7)
    ax.axhline(0.5, color='orange', linestyle=':', label='Refinement Threshold')
    ax.fill_between(
        S_grid,
        0,
        gradient_error,
        where=(gradient_error > 0.5),
        alpha=0.3,
        color='red',
        label='Needs Refinement',
    )
    ax.set_xlabel('Spot Price')
    ax.set_ylabel('Normalized Error')
    ax.set_title('Gradient-Based Error Indicator')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Curvature error
    ax = axes[1, 0]
    ax.plot(S_grid, curvature_error, 'm-', linewidth=2)
    ax.axvline(strike, color='r', linestyle='--', alpha=0.7)
    ax.axhline(0.5, color='orange', linestyle=':', label='Refinement Threshold')
    ax.fill_between(
        S_grid,
        0,
        curvature_error,
        where=(curvature_error > 0.5),
        alpha=0.3,
        color='red',
        label='Needs Refinement',
    )
    ax.set_xlabel('Spot Price')
    ax.set_ylabel('Normalized Error')
    ax.set_title('Curvature-Based Error Indicator')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Combined indicator
    ax = axes[1, 1]
    combined = 0.6 * gradient_error + 0.4 * curvature_error
    ax.plot(S_grid, combined, 'k-', linewidth=2, label='Combined Error')
    ax.axvline(strike, color='r', linestyle='--', alpha=0.7)
    ax.axhline(0.5, color='orange', linestyle=':', label='Refinement Threshold')
    ax.fill_between(
        S_grid,
        0,
        combined,
        where=(combined > 0.5),
        alpha=0.3,
        color='red',
        label='Needs Refinement',
    )
    ax.set_xlabel('Spot Price')
    ax.set_ylabel('Normalized Error')
    ax.set_title('Combined Error Indicator (60% grad + 40% curv)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/tmp/amr_error_indicators.png', dpi=150, bbox_inches='tight')
    print(f"Saved visualization to /tmp/amr_error_indicators.png")
    print()


def demo_strike_concentrated_grid():
    """Demo 3: Strike-concentrated initial grid."""
    print("=" * 80)
    print("DEMO 3: Strike-Concentrated Initial Grid")
    print("=" * 80)
    print()

    strike = 100.0
    S_min, S_max = 50.0, 150.0

    print("Comparing grid initializations:")
    print(f"  Uniform grid")
    print(f"  Strike-concentrated grid (strike={strike})")
    print()

    # Create grids
    grid_uniform = AdaptivePDEGrid.from_uniform(
        S_min=S_min, S_max=S_max, T=1.0, N_S=101, N_T=50
    )

    grid_concentrated = AdaptivePDEGrid.from_strike_concentrated(
        S_min=S_min,
        S_max=S_max,
        T=1.0,
        N_S=101,
        N_T=50,
        strike=strike,
        concentration=2.5,
    )

    print(f"Uniform grid points: {grid_uniform.N_S}")
    print(f"Concentrated grid points: {grid_concentrated.N_S}")
    print()

    # Analyze spacing near strike
    def analyze_spacing_near_strike(grid, label):
        strike_idx = jnp.argmin(jnp.abs(grid.S_grid - strike))
        if 10 < strike_idx < grid.N_S - 10:
            near_strike_spacing = jnp.mean(
                jnp.diff(grid.S_grid[strike_idx - 5 : strike_idx + 6])
            )
            far_spacing = jnp.mean(jnp.diff(grid.S_grid[:10]))
            print(f"{label}:")
            print(f"  Spacing near strike: {near_strike_spacing:.3f}")
            print(f"  Spacing far from strike: {far_spacing:.3f}")
            print(f"  Ratio (far/near): {far_spacing / near_strike_spacing:.2f}x")
            print()

    analyze_spacing_near_strike(grid_uniform, "Uniform grid")
    analyze_spacing_near_strike(grid_concentrated, "Concentrated grid")

    # Visualize
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Plot 1: Grid point distribution
    ax = axes[0]
    ax.plot(
        grid_uniform.S_grid,
        jnp.ones_like(grid_uniform.S_grid),
        'b.',
        markersize=6,
        alpha=0.6,
        label='Uniform',
    )
    ax.plot(
        grid_concentrated.S_grid,
        jnp.zeros_like(grid_concentrated.S_grid),
        'r.',
        markersize=6,
        alpha=0.6,
        label='Strike-Concentrated',
    )
    ax.axvline(strike, color='k', linestyle='--', alpha=0.5, label='Strike')
    ax.set_xlabel('Spot Price')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Concentrated', 'Uniform'])
    ax.set_title('Grid Point Distribution Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')

    # Plot 2: Local spacing comparison
    ax = axes[1]
    uniform_spacing = jnp.diff(grid_uniform.S_grid)
    concentrated_spacing = jnp.diff(grid_concentrated.S_grid)

    ax.plot(
        grid_uniform.S_grid[:-1],
        uniform_spacing,
        'b-',
        linewidth=2,
        label='Uniform',
        alpha=0.7,
    )
    ax.plot(
        grid_concentrated.S_grid[:-1],
        concentrated_spacing,
        'r-',
        linewidth=2,
        label='Strike-Concentrated',
        alpha=0.7,
    )
    ax.axvline(strike, color='k', linestyle='--', alpha=0.5, label='Strike')
    ax.set_xlabel('Spot Price')
    ax.set_ylabel('Local Grid Spacing')
    ax.set_title('Grid Spacing Comparison (smaller = finer resolution)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/tmp/amr_strike_concentrated.png', dpi=150, bbox_inches='tight')
    print(f"Saved visualization to /tmp/amr_strike_concentrated.png")
    print()


def main():
    """Run all AMR demos."""
    print("\n" + "=" * 80)
    print("ADAPTIVE MESH REFINEMENT FOR PDE SOLVERS - COMPREHENSIVE DEMO")
    print("=" * 80)
    print()
    print("This demo showcases adaptive mesh refinement (AMR) techniques for")
    print("option pricing using finite difference methods.")
    print()

    # Run demos
    demo_basic_amr()
    demo_error_indicators()
    demo_strike_concentrated_grid()

    print("=" * 80)
    print("ALL DEMOS COMPLETED!")
    print("=" * 80)
    print()
    print("Key Takeaways:")
    print("  - AMR automatically refines grids in regions needing high resolution")
    print("  - Error indicators identify where refinement is needed")
    print("  - Strike-concentrated grids provide good initial distributions")
    print("  - AMR can achieve better accuracy with fewer points than uniform grids")
    print()
    print("Output files saved:")
    print("  /tmp/amr_basic_demo.png")
    print("  /tmp/amr_error_indicators.png")
    print("  /tmp/amr_strike_concentrated.png")
    print()


if __name__ == "__main__":
    main()
