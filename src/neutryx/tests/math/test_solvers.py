"""Tests for numerical solvers."""

import jax.numpy as jnp
import pytest

from neutryx.math.solvers import bisection, brent, newton_raphson


class TestNewtonRaphson:
    """Tests for Newton-Raphson solver."""

    def test_simple_quadratic(self):
        """Test finding root of x² - 4 = 0."""

        def f(x):
            return x**2 - 4.0

        root = newton_raphson(f, x0=1.0, tol=1e-10)
        assert jnp.isclose(root, 2.0, atol=1e-8)
        assert jnp.isclose(f(root), 0.0, atol=1e-8)

    def test_cubic_equation(self):
        """Test finding root of x³ - 2x - 5 = 0."""

        def f(x):
            return x**3 - 2 * x - 5

        root = newton_raphson(f, x0=2.0, tol=1e-10)
        assert jnp.isclose(f(root), 0.0, atol=1e-6)  # Relaxed for float32
        # Known root is approximately 2.0946
        assert jnp.isclose(root, 2.0946, atol=1e-3)

    def test_with_manual_derivative(self):
        """Test with manually provided derivative."""

        def f(x):
            return x**2 - 4.0

        def fprime(x):
            return 2 * x

        root = newton_raphson(f, x0=1.0, fprime=fprime, tol=1e-10)
        assert jnp.isclose(root, 2.0, atol=1e-8)

    def test_with_bounds(self):
        """Test that bounds are respected."""

        def f(x):
            return x**2 - 4.0

        # Should find root at 2.0 within bounds
        root = newton_raphson(f, x0=1.5, bounds=(1.0, 3.0), tol=1e-10)
        assert 1.0 <= root <= 3.0
        assert jnp.isclose(root, 2.0, atol=1e-8)

    def test_max_iterations_exceeded(self):
        """Test that max iterations is enforced."""

        def f(x):
            return x**2 - 4.0

        with pytest.raises(ValueError, match="Newton-Raphson failed to converge"):
            newton_raphson(f, x0=1.0, max_iter=2)  # Too few iterations

    def test_automatic_differentiation(self):
        """Test that automatic differentiation works."""

        def f(x):
            return jnp.sin(x) - 0.5

        # Find x where sin(x) = 0.5
        root = newton_raphson(f, x0=0.5, tol=1e-10)
        assert jnp.isclose(jnp.sin(root), 0.5, atol=1e-8)


class TestBisection:
    """Tests for bisection method."""

    def test_simple_quadratic(self):
        """Test finding root of x² - 4 = 0."""

        def f(x):
            return x**2 - 4.0

        root = bisection(f, a=0.0, b=3.0, tol=1e-8)
        assert jnp.isclose(root, 2.0, atol=1e-6)
        assert jnp.isclose(f(root), 0.0, atol=1e-6)

    def test_cubic_equation(self):
        """Test finding root of x³ - 2x - 5 = 0."""

        def f(x):
            return x**3 - 2 * x - 5

        root = bisection(f, a=1.0, b=3.0, tol=1e-8)
        assert jnp.isclose(f(root), 0.0, atol=1e-6)

    def test_invalid_bracket(self):
        """Test that invalid bracket raises error."""

        def f(x):
            return x**2 - 4.0

        # Both endpoints positive
        with pytest.raises(ValueError, match="Function must have opposite signs"):
            bisection(f, a=3.0, b=4.0)

    def test_max_iterations_exceeded(self):
        """Test that max iterations is enforced."""

        def f(x):
            return x**2 - 4.0

        with pytest.raises(ValueError, match="Bisection failed to converge"):
            bisection(f, a=0.0, b=3.0, max_iter=2, tol=1e-10)


class TestBrent:
    """Tests for Brent's method."""

    def test_simple_quadratic(self):
        """Test finding root of x² - 4 = 0."""

        def f(x):
            return x**2 - 4.0

        root = brent(f, a=0.0, b=3.0, tol=1e-10)
        assert jnp.isclose(root, 2.0, atol=1e-8)
        assert jnp.isclose(f(root), 0.0, atol=1e-8)

    def test_cubic_equation(self):
        """Test finding root of x³ - 2x - 5 = 0."""

        def f(x):
            return x**3 - 2 * x - 5

        root = brent(f, a=1.0, b=3.0, tol=1e-10)
        assert jnp.isclose(f(root), 0.0, atol=1e-8)
        assert jnp.isclose(root, 2.0946, atol=1e-3)

    def test_invalid_bracket(self):
        """Test that invalid bracket raises error."""

        def f(x):
            return x**2 - 4.0

        with pytest.raises(ValueError, match="Function must have opposite signs"):
            brent(f, a=3.0, b=4.0)

    def test_max_iterations_exceeded(self):
        """Test that max iterations is enforced."""

        def f(x):
            return x**2 - 4.0

        with pytest.raises(ValueError, match="Brent's method failed to converge"):
            brent(f, a=0.0, b=3.0, max_iter=2)


class TestSolverComparison:
    """Compare different solvers on the same problems."""

    def test_all_solvers_same_root(self):
        """All solvers should find the same root."""

        def f(x):
            return x**2 - 4.0

        nr_root = newton_raphson(f, x0=1.0, tol=1e-10)
        bisect_root = bisection(f, a=0.0, b=3.0, tol=1e-10)
        brent_root = brent(f, a=0.0, b=3.0, tol=1e-10)

        assert jnp.isclose(nr_root, 2.0, atol=1e-8)
        assert jnp.isclose(bisect_root, 2.0, atol=1e-8)
        assert jnp.isclose(brent_root, 2.0, atol=1e-8)

    def test_difficult_function(self):
        """Test on a function with slow convergence."""

        def f(x):
            return jnp.exp(x) - 10.0

        # Known root: ln(10) ≈ 2.302585
        expected = jnp.log(10.0)

        nr_root = newton_raphson(f, x0=2.0, tol=1e-10)
        bisect_root = bisection(f, a=0.0, b=4.0, tol=1e-10)
        brent_root = brent(f, a=0.0, b=4.0, tol=1e-6)  # More reasonable tolerance for Brent

        assert jnp.isclose(nr_root, expected, atol=1e-6)  # Relaxed for float32
        assert jnp.isclose(bisect_root, expected, atol=1e-6)
        assert jnp.isclose(brent_root, expected, atol=1e-6)


class TestFinancialApplications:
    """Test solvers on typical financial problems."""

    def test_implied_discount_factor(self):
        """Find discount factor that prices a swap at par."""
        par_rate = 0.0550
        known_dfs = [0.95, 0.92, 0.89]

        def swap_pv(df_final):
            """PV of swap as function of final DF."""
            fixed_leg = par_rate * sum(known_dfs) + par_rate * df_final
            float_leg = 1.0 - df_final
            return fixed_leg - float_leg

        # Newton-Raphson
        df_nr = newton_raphson(swap_pv, x0=0.85, tol=1e-10)
        assert jnp.isclose(swap_pv(df_nr), 0.0, atol=1e-6)  # Relaxed for float32
        assert 0.0 < df_nr < 1.0

        # Brent
        df_brent = brent(swap_pv, a=0.5, b=1.0, tol=1e-10)
        assert jnp.isclose(swap_pv(df_brent), 0.0, atol=1e-6)
        assert jnp.isclose(df_nr, df_brent, atol=1e-6)

    def test_forward_rate_from_dfs(self):
        """Calculate forward rate from discount factors."""
        df_start = 0.95
        df_end = 0.90
        accrual = 0.5

        # Direct calculation
        forward_rate = (df_start / df_end - 1.0) / accrual

        # Using solver (should get same result)
        def forward_pv(rate):
            implied_df_end = df_start / (1.0 + rate * accrual)
            return implied_df_end - df_end

        solved_rate = newton_raphson(forward_pv, x0=0.05, tol=1e-10)
        assert jnp.isclose(solved_rate, forward_rate, atol=1e-8)

    def test_futures_implied_rate(self):
        """Calculate implied rate from futures price."""
        futures_price = 95.25
        expected_rate = (100.0 - futures_price) / 100.0

        def price_error(rate):
            return 100.0 - rate * 100.0 - futures_price

        solved_rate = newton_raphson(price_error, x0=0.05, tol=1e-10)
        assert jnp.isclose(solved_rate, expected_rate, atol=1e-10)
        assert jnp.isclose(solved_rate, 0.0475, atol=1e-10)
