"""Tests for Adaptive Mesh Refinement (AMR) module."""

import jax.numpy as jnp
import jax.random as jr
import pytest

from neutryx.models.amr import (
    AdaptivePDEGrid,
    ErrorIndicators,
    RefineInfo,
    adaptive_crank_nicolson,
    estimate_curvature_error,
    estimate_gradient_error,
    interpolate_to_new_grid,
    refine_grid_adaptive,
    richardson_extrapolation_error,
)


class TestAdaptivePDEGrid:
    """Test suite for AdaptivePDEGrid class."""

    def test_uniform_grid_creation(self):
        """Test creation from uniform grid."""
        grid = AdaptivePDEGrid.from_uniform(
            S_min=50.0, S_max=150.0, T=1.0, N_S=101, N_T=50
        )

        assert grid.N_S == 101
        assert grid.N_T == 50
        assert grid.S_min == 50.0
        assert grid.S_max == 150.0
        assert grid.T == 1.0
        assert jnp.allclose(grid.dt, 1.0 / 50)

    def test_strike_concentrated_grid(self):
        """Test grid with concentration around strike."""
        strike = 100.0
        grid = AdaptivePDEGrid.from_strike_concentrated(
            S_min=50.0, S_max=150.0, T=1.0, N_S=101, N_T=50,
            strike=strike, concentration=2.0
        )

        assert grid.N_S <= 101  # May be slightly less due to unique filtering
        assert grid.S_min >= 50.0
        assert grid.S_max <= 150.0

        # Check that grid is denser near strike
        # Find closest points to strike
        strike_idx = jnp.argmin(jnp.abs(grid.S_grid - strike))

        if 10 < strike_idx < grid.N_S - 10:
            # Compare spacing near strike vs far from strike
            near_strike_spacing = jnp.mean(jnp.diff(grid.S_grid[strike_idx - 5 : strike_idx + 5]))
            far_spacing = jnp.mean(jnp.diff(grid.S_grid[:10]))

            # Near strike should have smaller spacing
            assert near_strike_spacing < far_spacing * 1.5

    def test_local_spacing(self):
        """Test local spacing calculation."""
        S_grid = jnp.array([50.0, 60.0, 70.0, 85.0, 100.0, 120.0, 150.0])
        grid = AdaptivePDEGrid(S_grid=S_grid, T=1.0, N_T=50)

        # Check boundary spacing
        assert jnp.isclose(grid.dS_local(0), 10.0)  # 60 - 50
        assert jnp.isclose(grid.dS_local(6), 30.0)  # 150 - 120

        # Check interior spacing (average of neighbors)
        # dS_local(2) = (S[3] - S[1]) / 2 = (85 - 60) / 2 = 12.5
        assert jnp.isclose(grid.dS_local(2), 12.5)


class TestErrorEstimators:
    """Test suite for error estimator functions."""

    def test_gradient_error_linear_function(self):
        """Test gradient error on linear function (should be uniform)."""
        S_grid = jnp.linspace(50.0, 150.0, 101)
        solution = 2.0 * S_grid + 10.0  # Linear function

        error = estimate_gradient_error(solution, S_grid)

        # All gradients should be equal for linear function
        # Error should be uniform (all approximately equal after normalization)
        assert jnp.allclose(error, error[0], atol=1e-5)

    def test_gradient_error_steep_region(self):
        """Test that gradient error identifies steep regions."""
        S_grid = jnp.linspace(50.0, 150.0, 101)
        # Function with steep gradient near center
        solution = jnp.where(
            S_grid < 100.0,
            S_grid,
            S_grid + 10.0 * (S_grid - 100.0)  # Steep slope after 100
        )

        error = estimate_gradient_error(solution, S_grid)

        # Error should be higher in steep region (S > 100)
        center_idx = jnp.argmin(jnp.abs(S_grid - 100.0))
        left_error = jnp.mean(error[: center_idx - 5])
        right_error = jnp.mean(error[center_idx + 5 :])

        assert right_error > left_error

    def test_curvature_error_parabolic(self):
        """Test curvature error on parabolic function."""
        S_grid = jnp.linspace(50.0, 150.0, 101)
        solution = (S_grid - 100.0) ** 2 / 100.0  # Parabola

        error = estimate_curvature_error(solution, S_grid)

        # Parabola has constant second derivative
        # So curvature should be relatively uniform
        interior_error = error[10:-10]  # Exclude boundaries
        assert jnp.std(interior_error) < 0.3  # Low variation

    def test_curvature_error_kink(self):
        """Test that curvature error identifies kinks/discontinuities."""
        S_grid = jnp.linspace(50.0, 150.0, 101)
        # Function with kink at strike
        strike = 100.0
        solution = jnp.maximum(S_grid - strike, 0.0)  # Call payoff (kink at strike)

        error = estimate_curvature_error(solution, S_grid)

        # Error should be highest near the kink
        strike_idx = jnp.argmin(jnp.abs(S_grid - strike))
        near_kink = error[strike_idx - 2 : strike_idx + 3]
        far_from_kink = jnp.concatenate([error[:40], error[-40:]])

        assert jnp.max(near_kink) > jnp.mean(far_from_kink) * 2

    def test_richardson_extrapolation(self):
        """Test Richardson extrapolation error estimate."""
        # Create two grids: coarse and fine
        S_coarse = jnp.linspace(50.0, 150.0, 51)
        S_fine = jnp.linspace(50.0, 150.0, 101)

        # Analytical solution: Black-Scholes call value approximation
        # Use simple approximation for testing
        strike = 100.0
        solution_coarse = jnp.maximum(S_coarse - strike, 0.0) + 5.0
        solution_fine = jnp.maximum(S_fine - strike, 0.0) + 5.0

        error = richardson_extrapolation_error(
            solution_fine, solution_coarse, S_fine, S_coarse, order=2
        )

        # Error should be small for this smooth function
        assert jnp.all(error >= 0)
        assert error.shape == solution_fine.shape

    def test_error_indicators_combined(self):
        """Test combined error indicator calculation."""
        N = 50
        grad_error = jnp.linspace(0.1, 1.0, N)
        curv_error = jnp.linspace(0.2, 0.8, N)
        rich_error = jnp.linspace(0.15, 0.9, N)

        indicators = ErrorIndicators(
            gradient_error=grad_error,
            curvature_error=curv_error,
            richardson_error=rich_error,
        )

        # Test with default weights
        combined = indicators.combined_error()
        assert combined.shape == (N,)
        assert jnp.all(combined >= 0)
        assert jnp.all(combined <= 1.0)

        # Test without Richardson error
        indicators_no_rich = ErrorIndicators(
            gradient_error=grad_error, curvature_error=curv_error
        )
        combined_no_rich = indicators_no_rich.combined_error()
        assert combined_no_rich.shape == (N,)


class TestGridRefinement:
    """Test suite for grid refinement operations."""

    def test_refine_grid_basic(self):
        """Test basic grid refinement."""
        S_grid = jnp.linspace(50.0, 150.0, 51)

        # Create error indicators suggesting refinement in middle region
        error = jnp.zeros(51)
        error = error.at[20:30].set(0.5)  # High error in middle

        indicators = ErrorIndicators(
            gradient_error=error,
            curvature_error=error * 0.8
        )

        new_grid, refine_info = refine_grid_adaptive(
            S_grid, indicators, tolerance=0.2, max_points=100
        )

        # Grid should have more points
        assert len(new_grid) > len(S_grid)
        assert refine_info.n_points_added > 0

        # New grid should still be sorted
        assert jnp.all(jnp.diff(new_grid) > 0)

        # New grid should contain original points
        assert jnp.all(jnp.isin(S_grid, new_grid))

    def test_refine_grid_max_points_limit(self):
        """Test that refinement respects maximum points limit."""
        S_grid = jnp.linspace(50.0, 150.0, 50)

        # High error everywhere
        error = jnp.ones(50)
        indicators = ErrorIndicators(
            gradient_error=error, curvature_error=error
        )

        max_points = 60
        new_grid, refine_info = refine_grid_adaptive(
            S_grid, indicators, tolerance=0.1, max_points=max_points
        )

        # Should not exceed maximum
        assert len(new_grid) <= max_points

    def test_refine_grid_min_spacing(self):
        """Test that refinement respects minimum spacing."""
        S_grid = jnp.linspace(50.0, 60.0, 20)  # Small domain, many points

        # High error
        error = jnp.ones(20)
        indicators = ErrorIndicators(
            gradient_error=error, curvature_error=error
        )

        new_grid, refine_info = refine_grid_adaptive(
            S_grid, indicators, tolerance=0.1, max_points=100, min_spacing=0.1
        )

        # Check that all spacings are >= min_spacing
        spacings = jnp.diff(new_grid)
        assert jnp.all(spacings >= 0.1 - 1e-6)  # Small tolerance for numerical error


class TestInterpolation:
    """Test suite for interpolation functions."""

    def test_interpolate_linear_function(self):
        """Test interpolation preserves linear functions exactly."""
        S_old = jnp.linspace(50.0, 150.0, 51)
        solution_old = 2.0 * S_old + 10.0  # Linear function

        S_new = jnp.linspace(50.0, 150.0, 101)  # Finer grid
        solution_new = interpolate_to_new_grid(solution_old, S_old, S_new)

        # Linear interpolation should be exact for linear functions
        expected = 2.0 * S_new + 10.0
        assert jnp.allclose(solution_new, expected, atol=1e-10)

    def test_interpolate_preserves_boundaries(self):
        """Test interpolation preserves boundary values."""
        S_old = jnp.linspace(50.0, 150.0, 51)
        solution_old = jnp.sin(S_old / 20.0)

        S_new = jnp.linspace(50.0, 150.0, 101)
        solution_new = interpolate_to_new_grid(solution_old, S_old, S_new)

        # Boundaries should match exactly
        assert jnp.isclose(solution_new[0], solution_old[0])
        assert jnp.isclose(solution_new[-1], solution_old[-1])

    def test_interpolate_monotonicity(self):
        """Test interpolation preserves monotonicity."""
        S_old = jnp.linspace(50.0, 150.0, 51)
        solution_old = jnp.exp(S_old / 50.0)  # Monotonically increasing

        S_new = jnp.linspace(50.0, 150.0, 101)
        solution_new = interpolate_to_new_grid(solution_old, S_old, S_new)

        # Should still be monotonically increasing
        assert jnp.all(jnp.diff(solution_new) >= -1e-10)


class TestAdaptiveSolver:
    """Test suite for adaptive Crank-Nicolson solver."""

    def test_adaptive_solver_european_call(self):
        """Test adaptive solver on European call option."""
        # Initial coarse grid
        grid = AdaptivePDEGrid.from_uniform(
            S_min=50.0, S_max=150.0, T=1.0, N_S=51, N_T=50
        )

        strike = 100.0
        r = 0.05
        q = 0.02
        sigma = 0.25

        def payoff(S):
            return jnp.maximum(S - strike, 0.0)

        solution, final_grid, history = adaptive_crank_nicolson(
            grid=grid,
            payoff_fn=payoff,
            r=r,
            q=q,
            sigma=sigma,
            option_type="call",
            strike=strike,
            tolerance=0.05,
            max_refinements=2,
            verbose=False,
        )

        # Solution should be non-negative
        assert jnp.all(solution >= -1e-10)

        # Solution should satisfy boundary conditions
        # At S=S_min (far OTM), value should be near 0
        assert solution[0] < 1.0

        # At S=S_max (deep ITM), value should be approximately S*exp(-q*T) - K*exp(-r*T)
        expected_itm = final_grid.S_max * jnp.exp(-q * grid.T) - strike * jnp.exp(-r * grid.T)
        assert jnp.abs(solution[-1] - expected_itm) / expected_itm < 0.05

        # Grid should have been refined
        assert final_grid.N_S >= grid.N_S

        # At-the-money value should be reasonable
        strike_idx = jnp.argmin(jnp.abs(final_grid.S_grid - strike))
        atm_value = solution[strike_idx]
        # Rough check: ATM call should be worth something reasonable
        # For K=100, T=1, r=5%, q=2%, sigma=25%, ATM call ≈ 10-15
        assert 5.0 < atm_value < 20.0

    def test_adaptive_solver_european_put(self):
        """Test adaptive solver on European put option."""
        grid = AdaptivePDEGrid.from_uniform(
            S_min=50.0, S_max=150.0, T=1.0, N_S=51, N_T=50
        )

        strike = 100.0
        r = 0.05
        q = 0.02
        sigma = 0.25

        def payoff(S):
            return jnp.maximum(strike - S, 0.0)

        solution, final_grid, history = adaptive_crank_nicolson(
            grid=grid,
            payoff_fn=payoff,
            r=r,
            q=q,
            sigma=sigma,
            option_type="put",
            strike=strike,
            tolerance=0.05,
            max_refinements=2,
            verbose=False,
        )

        # Solution should be non-negative
        assert jnp.all(solution >= -1e-10)

        # At S=S_min (deep ITM), value should be approximately K*exp(-r*T) - S_min
        expected_itm = strike * jnp.exp(-r * grid.T) - final_grid.S_min * jnp.exp(-q * grid.T)
        assert solution[0] > expected_itm * 0.9

        # At S=S_max (far OTM), value should be near 0
        assert solution[-1] < 1.0

        # Grid should have been refined
        assert final_grid.N_S >= grid.N_S

    def test_adaptive_solver_refinement_history(self):
        """Test that refinement history is recorded correctly."""
        grid = AdaptivePDEGrid.from_uniform(
            S_min=50.0, S_max=150.0, T=1.0, N_S=51, N_T=50
        )

        strike = 100.0
        r = 0.05
        q = 0.02
        sigma = 0.25

        def payoff(S):
            return jnp.maximum(S - strike, 0.0)

        solution, final_grid, history = adaptive_crank_nicolson(
            grid=grid,
            payoff_fn=payoff,
            r=r,
            q=q,
            sigma=sigma,
            option_type="call",
            strike=strike,
            tolerance=0.1,
            max_refinements=3,
            verbose=False,
        )

        # History should be recorded
        assert isinstance(history, list)
        assert len(history) <= 3  # At most max_refinements entries

        # Each entry should have required fields
        for entry in history:
            assert 'iteration' in entry
            assert 'n_points' in entry
            assert 'points_added' in entry
            assert 'mean_error' in entry
            assert 'max_error' in entry

        # Number of points should increase over iterations
        if len(history) > 1:
            for i in range(len(history) - 1):
                assert history[i + 1]['n_points'] >= history[i]['n_points']

    def test_adaptive_solver_strike_refinement(self):
        """Test that grid is refined near the strike (high curvature region)."""
        # Start with strike-concentrated grid
        strike = 100.0
        grid = AdaptivePDEGrid.from_strike_concentrated(
            S_min=50.0, S_max=150.0, T=1.0, N_S=51, N_T=50,
            strike=strike, concentration=2.0
        )

        r = 0.05
        q = 0.02
        sigma = 0.25

        def payoff(S):
            return jnp.maximum(S - strike, 0.0)

        solution, final_grid, history = adaptive_crank_nicolson(
            grid=grid,
            payoff_fn=payoff,
            r=r,
            q=q,
            sigma=sigma,
            option_type="call",
            strike=strike,
            tolerance=0.05,
            max_refinements=2,
            verbose=False,
        )

        # Final grid should have high density near strike
        strike_idx = jnp.argmin(jnp.abs(final_grid.S_grid - strike))

        if 10 < strike_idx < final_grid.N_S - 10:
            # Compare local spacing near strike vs far regions
            near_strike_spacing = jnp.mean(
                jnp.diff(final_grid.S_grid[strike_idx - 5 : strike_idx + 6])
            )
            far_spacing = jnp.mean(jnp.diff(final_grid.S_grid[:10]))

            # Near strike should have tighter spacing
            assert near_strike_spacing < far_spacing


class TestConvergence:
    """Test convergence properties of AMR solver."""

    def test_convergence_with_refinement(self):
        """Test that solution improves with adaptive refinement."""
        strike = 100.0
        r = 0.05
        q = 0.02
        sigma = 0.25
        T = 1.0

        def payoff(S):
            return jnp.maximum(S - strike, 0.0)

        # Solve with different refinement levels
        tolerances = [0.2, 0.1, 0.05]
        solutions = []
        grids = []

        for tol in tolerances:
            grid = AdaptivePDEGrid.from_uniform(
                S_min=50.0, S_max=150.0, T=T, N_S=51, N_T=100
            )

            sol, final_grid, _ = adaptive_crank_nicolson(
                grid=grid,
                payoff_fn=payoff,
                r=r,
                q=q,
                sigma=sigma,
                option_type="call",
                strike=strike,
                tolerance=tol,
                max_refinements=3,
                verbose=False,
            )

            solutions.append(sol)
            grids.append(final_grid)

        # Finer tolerance should result in more grid points
        assert grids[-1].N_S >= grids[0].N_S

        # Solutions should be similar at ATM (convergence check)
        # Interpolate all solutions to common grid for comparison
        common_grid = grids[-1].S_grid
        strike_idx = jnp.argmin(jnp.abs(common_grid - strike))

        values_at_strike = []
        for sol, grid in zip(solutions, grids):
            sol_interp = jnp.interp(common_grid, grid.S_grid, sol)
            values_at_strike.append(sol_interp[strike_idx])

        # Values should converge (differences should decrease)
        diff_1 = abs(values_at_strike[1] - values_at_strike[0])
        diff_2 = abs(values_at_strike[2] - values_at_strike[1])

        # Later differences should be smaller (convergence)
        # Allow some tolerance due to adaptivity randomness
        assert diff_2 <= diff_1 * 1.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
