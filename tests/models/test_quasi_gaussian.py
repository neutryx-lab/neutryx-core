"""
Tests for Quasi-Gaussian Interest Rate Model
"""

import jax
import jax.numpy as jnp
import pytest

from neutryx.models.quasi_gaussian import (
    QuasiGaussianParams,
    zero_coupon_bond_price,
    simulate_path,
    simulate_paths,
    caplet_price_mc,
    swaption_price_mc,
    V_coefficient,
    create_piecewise_constant_qg,
)


@pytest.fixture
def constant_params():
    """Quasi-Gaussian with constant parameters (similar to G2++)."""
    return QuasiGaussianParams(
        alpha_fn=lambda t: 0.1,
        beta_fn=lambda t: 0.2,
        sigma_x_fn=lambda t: 0.01,
        sigma_y_fn=lambda t: 0.015,
        forward_curve_fn=lambda t: 0.03,
        rho=-0.7,
        r0=0.03
    )


@pytest.fixture
def time_dependent_params():
    """Quasi-Gaussian with time-dependent parameters."""
    # Higher volatility at short end, decaying over time
    return QuasiGaussianParams(
        alpha_fn=lambda t: 0.1 + 0.05 * jnp.exp(-t),
        beta_fn=lambda t: 0.2 + 0.03 * jnp.exp(-t),
        sigma_x_fn=lambda t: 0.02 * jnp.exp(-0.1 * t),
        sigma_y_fn=lambda t: 0.025 * jnp.exp(-0.1 * t),
        forward_curve_fn=lambda t: 0.03 + 0.005 * t,  # Upward sloping
        rho=-0.7,
        r0=0.03
    )


class TestQuasiGaussianParams:
    """Test parameter validation."""

    def test_valid_constant_params(self, constant_params):
        """Test that valid constant parameters are accepted."""
        assert constant_params.alpha_fn(0.0) == 0.1
        assert constant_params.get_rho(0.0) == -0.7

    def test_valid_time_dependent_params(self, time_dependent_params):
        """Test that time-dependent parameters are accepted."""
        assert time_dependent_params.alpha_fn(0.0) > 0
        assert time_dependent_params.sigma_x_fn(1.0) > 0

    def test_negative_alpha(self):
        """Test that negative alpha raises error."""
        with pytest.raises(ValueError, match="α.0. must be positive"):
            QuasiGaussianParams(
                alpha_fn=lambda t: -0.1,
                beta_fn=lambda t: 0.2,
                sigma_x_fn=lambda t: 0.01,
                sigma_y_fn=lambda t: 0.015,
                forward_curve_fn=lambda t: 0.03,
                rho=-0.7,
                r0=0.03
            )

    def test_negative_sigma(self):
        """Test that negative volatility raises error."""
        with pytest.raises(ValueError, match="σ_x.0. must be positive"):
            QuasiGaussianParams(
                alpha_fn=lambda t: 0.1,
                beta_fn=lambda t: 0.2,
                sigma_x_fn=lambda t: -0.01,
                sigma_y_fn=lambda t: 0.015,
                forward_curve_fn=lambda t: 0.03,
                rho=-0.7,
                r0=0.03
            )

    def test_invalid_correlation(self):
        """Test that invalid correlation raises error."""
        with pytest.raises(ValueError, match="Correlation"):
            QuasiGaussianParams(
                alpha_fn=lambda t: 0.1,
                beta_fn=lambda t: 0.2,
                sigma_x_fn=lambda t: 0.01,
                sigma_y_fn=lambda t: 0.015,
                forward_curve_fn=lambda t: 0.03,
                rho=2.0,  # Invalid
                r0=0.03
            )

    def test_time_dependent_correlation(self):
        """Test time-dependent correlation function."""
        params = QuasiGaussianParams(
            alpha_fn=lambda t: 0.1,
            beta_fn=lambda t: 0.2,
            sigma_x_fn=lambda t: 0.01,
            sigma_y_fn=lambda t: 0.015,
            forward_curve_fn=lambda t: 0.03,
            rho=lambda t: -0.5 - 0.2 * jnp.exp(-t),  # Time-varying
            r0=0.03
        )

        rho_0 = params.get_rho(0.0)
        rho_1 = params.get_rho(1.0)
        assert -1.0 <= rho_0 <= 1.0
        assert -1.0 <= rho_1 <= 1.0


class TestBondPricing:
    """Test zero-coupon bond pricing."""

    def test_bond_price_at_zero_maturity(self, constant_params):
        """Bond price at T=0 should be 1.0."""
        price = zero_coupon_bond_price(constant_params, T=0.0)
        assert jnp.isclose(price, 1.0)

    def test_bond_price_range(self, constant_params):
        """Bond prices should be in (0, 1]."""
        for T in [0.5, 1.0, 5.0]:
            price = zero_coupon_bond_price(constant_params, T)
            assert 0.0 < price <= 1.0, f"Price {price} out of range for T={T}"

    def test_bond_price_decreasing_with_maturity(self, constant_params):
        """Bond prices should decrease with maturity."""
        price_1y = zero_coupon_bond_price(constant_params, T=1.0)
        price_5y = zero_coupon_bond_price(constant_params, T=5.0)

        assert price_1y > price_5y

    def test_bond_price_time_dependent(self, time_dependent_params):
        """Test bond pricing with time-dependent parameters."""
        price = zero_coupon_bond_price(time_dependent_params, T=5.0)
        assert 0.0 < price < 1.0
        assert jnp.isfinite(price)

    def test_bond_price_with_factors(self, constant_params):
        """Test bond pricing with non-zero factor values."""
        price_neutral = zero_coupon_bond_price(constant_params, T=5.0, x_t=0.0, y_t=0.0)
        price_positive_x = zero_coupon_bond_price(constant_params, T=5.0, x_t=0.01, y_t=0.0)

        # Positive x increases rate, decreases bond price
        assert price_positive_x < price_neutral

    def test_V_coefficient_positive(self, constant_params):
        """Variance coefficient should be non-negative."""
        V = V_coefficient(constant_params, t=0.0, T=5.0)
        assert V >= 0.0
        assert jnp.isfinite(V)

    def test_V_coefficient_increasing(self, constant_params):
        """Variance should increase with maturity."""
        V_1 = V_coefficient(constant_params, t=0.0, T=1.0)
        V_5 = V_coefficient(constant_params, t=0.0, T=5.0)

        assert V_5 > V_1


class TestSimulation:
    """Test path simulation."""

    def test_simulate_path_shape(self, constant_params):
        """Test that simulated path has correct shape."""
        key = jax.random.PRNGKey(42)
        r_path, x_path, y_path = simulate_path(constant_params, T=1.0, n_steps=100, key=key)

        assert r_path.shape == (101,)
        assert x_path.shape == (101,)
        assert y_path.shape == (101,)

    def test_simulate_path_initial_conditions(self, constant_params):
        """Test that paths start at initial conditions."""
        key = jax.random.PRNGKey(42)
        r_path, x_path, y_path = simulate_path(constant_params, T=1.0, n_steps=100, key=key)

        assert jnp.isclose(r_path[0], constant_params.r0)
        assert jnp.isclose(x_path[0], constant_params.x0)
        assert jnp.isclose(y_path[0], constant_params.y0)

    def test_simulate_path_finite_values(self, constant_params):
        """Test that all values are finite."""
        key = jax.random.PRNGKey(42)
        r_path, x_path, y_path = simulate_path(constant_params, T=1.0, n_steps=100, key=key)

        assert jnp.all(jnp.isfinite(r_path))
        assert jnp.all(jnp.isfinite(x_path))
        assert jnp.all(jnp.isfinite(y_path))

    def test_simulate_path_time_dependent(self, time_dependent_params):
        """Test simulation with time-dependent parameters."""
        key = jax.random.PRNGKey(42)
        r_path, x_path, y_path = simulate_path(time_dependent_params, T=1.0, n_steps=100, key=key)

        assert r_path.shape == (101,)
        assert jnp.all(jnp.isfinite(r_path))

    def test_simulate_paths_shape(self, constant_params):
        """Test multiple path simulation."""
        key = jax.random.PRNGKey(42)
        n_paths = 100
        n_steps = 50

        r_paths, x_paths, y_paths = simulate_paths(
            constant_params, T=1.0, n_steps=n_steps, n_paths=n_paths, key=key
        )

        assert r_paths.shape == (n_paths, n_steps + 1)
        assert x_paths.shape == (n_paths, n_steps + 1)
        assert y_paths.shape == (n_paths, n_steps + 1)

    def test_simulate_paths_initial_conditions(self, constant_params):
        """Test that all paths start at initial condition."""
        key = jax.random.PRNGKey(42)
        r_paths, _, _ = simulate_paths(constant_params, T=1.0, n_steps=50, n_paths=100, key=key)

        assert jnp.allclose(r_paths[:, 0], constant_params.r0)


class TestDerivativePricing:
    """Test derivative pricing with Monte Carlo."""

    def test_caplet_price_mc_positive(self, constant_params):
        """Caplet price should be positive."""
        key = jax.random.PRNGKey(42)
        price = caplet_price_mc(
            constant_params,
            strike=0.03,
            maturity=1.0,
            tenor=0.25,
            n_paths=1000,
            key=key
        )
        assert price >= 0.0
        assert jnp.isfinite(price)

    def test_caplet_price_mc_otm_vs_itm(self, constant_params):
        """Out-of-money caplet should be cheaper than in-the-money."""
        key = jax.random.PRNGKey(42)

        # ITM caplet
        price_itm = caplet_price_mc(
            constant_params, strike=0.01, maturity=1.0, tenor=0.25, n_paths=1000, key=key
        )

        # OTM caplet
        price_otm = caplet_price_mc(
            constant_params, strike=0.10, maturity=1.0, tenor=0.25, n_paths=1000, key=key
        )

        assert price_itm > price_otm

    def test_swaption_price_mc_positive(self, constant_params):
        """Swaption price should be non-negative."""
        key = jax.random.PRNGKey(42)
        price = swaption_price_mc(
            constant_params,
            swap_rate=0.03,
            option_maturity=1.0,
            swap_maturity=3.0,
            n_paths=1000,
            key=key
        )
        assert price >= 0.0
        assert jnp.isfinite(price)

    def test_swaption_payer_vs_receiver(self, constant_params):
        """Test payer vs receiver swaption."""
        key = jax.random.PRNGKey(42)

        price_payer = swaption_price_mc(
            constant_params,
            swap_rate=0.03,
            option_maturity=1.0,
            swap_maturity=3.0,
            is_payer=True,
            n_paths=1000,
            key=key
        )

        price_receiver = swaption_price_mc(
            constant_params,
            swap_rate=0.03,
            option_maturity=1.0,
            swap_maturity=3.0,
            is_payer=False,
            n_paths=1000,
            key=key
        )

        # Both should be non-negative
        assert price_payer >= 0.0
        assert price_receiver >= 0.0


class TestPiecewiseConstantQG:
    """Test piecewise constant parameter construction."""

    def test_create_constant_qg(self):
        """Test creating QG with constant parameters."""
        forward_fn = lambda t: 0.03

        params = create_piecewise_constant_qg(
            forward_fn,
            alpha=0.1,
            beta=0.2,
            sigma_x=0.01,
            sigma_y=0.015,
            rho=-0.7
        )

        assert params.r0 == 0.03
        assert params.alpha_fn(0.0) == 0.1
        assert params.alpha_fn(5.0) == 0.1  # Constant

    def test_create_piecewise_qg(self):
        """Test creating QG with piecewise constant parameters."""
        forward_fn = lambda t: 0.03

        # Different volatilities for different time periods
        time_grid = jnp.array([0.0, 2.0, 5.0, 10.0])
        sigma_x = jnp.array([0.015, 0.010, 0.008])  # 3 intervals
        sigma_y = jnp.array([0.020, 0.012, 0.010])

        params = create_piecewise_constant_qg(
            forward_fn,
            alpha=0.1,
            beta=0.2,
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            time_grid=time_grid,
            rho=-0.7
        )

        # Check piecewise behavior
        assert params.sigma_x_fn(0.5) == 0.015  # First interval
        assert params.sigma_x_fn(3.0) == 0.010  # Second interval
        assert params.sigma_x_fn(7.0) == 0.008  # Third interval

    def test_piecewise_array_parameters(self):
        """Test piecewise with array mean reversion."""
        forward_fn = lambda t: 0.03

        time_grid = jnp.array([0.0, 1.0, 5.0])
        alpha = jnp.array([0.15, 0.10])  # Higher at short end
        beta = jnp.array([0.25, 0.20])

        params = create_piecewise_constant_qg(
            forward_fn,
            alpha=alpha,
            beta=beta,
            sigma_x=0.01,
            sigma_y=0.015,
            time_grid=time_grid
        )

        assert params.alpha_fn(0.5) == 0.15
        assert params.alpha_fn(2.0) == 0.10


class TestJITCompilation:
    """Test JAX JIT compilation."""

    def test_jit_bond_price(self, constant_params):
        """Test JIT compilation of bond pricing."""
        jitted_fn = jax.jit(lambda T: zero_coupon_bond_price(constant_params, T))

        price = jitted_fn(5.0)
        assert 0.0 < price < 1.0

    def test_jit_simulation(self, constant_params):
        """Test JIT compilation of simulation."""
        # Already JIT-compiled
        key = jax.random.PRNGKey(42)
        r_path, _, _ = simulate_path(constant_params, T=1.0, n_steps=100, key=key)
        assert r_path.shape == (101,)


class TestNumericalStability:
    """Test numerical stability."""

    def test_high_volatility(self):
        """Test with high volatility."""
        params = QuasiGaussianParams(
            alpha_fn=lambda t: 0.1,
            beta_fn=lambda t: 0.2,
            sigma_x_fn=lambda t: 0.05,  # High volatility
            sigma_y_fn=lambda t: 0.08,
            forward_curve_fn=lambda t: 0.03,
            rho=-0.9,
            r0=0.03
        )

        price = zero_coupon_bond_price(params, T=5.0)
        assert 0.0 < price < 1.0
        assert jnp.isfinite(price)

    def test_steep_term_structure(self):
        """Test with steep upward sloping curve."""
        params = QuasiGaussianParams(
            alpha_fn=lambda t: 0.1,
            beta_fn=lambda t: 0.2,
            sigma_x_fn=lambda t: 0.01,
            sigma_y_fn=lambda t: 0.015,
            forward_curve_fn=lambda t: 0.01 + 0.01 * t,  # Steep slope
            rho=-0.7,
            r0=0.01
        )

        price = zero_coupon_bond_price(params, T=10.0)
        assert 0.0 < price < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
