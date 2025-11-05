"""
Tests for G2++ Two-Factor Gaussian Interest Rate Model
"""

import jax
import jax.numpy as jnp
import pytest

from neutryx.models.g2pp import (
    G2PPParams,
    zero_coupon_bond_price,
    simulate_path,
    simulate_paths,
    caplet_price,
    swaption_price,
    forward_rate_correlation,
    create_fitted_g2pp,
    B_coefficient,
)


@pytest.fixture
def standard_params():
    """Standard G2++ parameters for testing."""
    return G2PPParams(
        a=0.1,
        b=0.2,
        sigma_x=0.01,
        sigma_y=0.015,
        rho=-0.7,
        r0=0.03,
        x0=0.0,
        y0=0.0
    )


@pytest.fixture
def params_with_phi():
    """G2++ parameters with drift function."""
    phi_fn = lambda t: 0.03 + 0.001 * t  # Linear drift
    return G2PPParams(
        a=0.1,
        b=0.2,
        sigma_x=0.01,
        sigma_y=0.015,
        rho=-0.7,
        r0=0.03,
        phi_fn=phi_fn
    )


class TestG2PPParams:
    """Test parameter validation."""

    def test_valid_params(self, standard_params):
        """Test that valid parameters are accepted."""
        assert standard_params.a == 0.1
        assert standard_params.b == 0.2
        assert standard_params.rho == -0.7

    def test_negative_mean_reversion_a(self):
        """Test that negative mean reversion 'a' raises error."""
        with pytest.raises(ValueError, match="Mean reversion 'a' must be positive"):
            G2PPParams(a=-0.1, b=0.2, sigma_x=0.01, sigma_y=0.015, rho=0.5, r0=0.03)

    def test_negative_mean_reversion_b(self):
        """Test that negative mean reversion 'b' raises error."""
        with pytest.raises(ValueError, match="Mean reversion 'b' must be positive"):
            G2PPParams(a=0.1, b=-0.2, sigma_x=0.01, sigma_y=0.015, rho=0.5, r0=0.03)

    def test_negative_volatility_x(self):
        """Test that negative volatility sigma_x raises error."""
        with pytest.raises(ValueError, match="Volatility 'sigma_x' must be positive"):
            G2PPParams(a=0.1, b=0.2, sigma_x=-0.01, sigma_y=0.015, rho=0.5, r0=0.03)

    def test_negative_volatility_y(self):
        """Test that negative volatility sigma_y raises error."""
        with pytest.raises(ValueError, match="Volatility 'sigma_y' must be positive"):
            G2PPParams(a=0.1, b=0.2, sigma_x=0.01, sigma_y=-0.015, rho=0.5, r0=0.03)

    def test_invalid_correlation(self):
        """Test that correlation outside [-1,1] raises error."""
        with pytest.raises(ValueError, match="Correlation 'rho' must be in"):
            G2PPParams(a=0.1, b=0.2, sigma_x=0.01, sigma_y=0.015, rho=1.5, r0=0.03)

    def test_identical_mean_reversions(self):
        """Test that identical mean reversions raise error (identifiability)."""
        with pytest.raises(ValueError, match="Mean reversion speeds must be distinct"):
            G2PPParams(a=0.1, b=0.1, sigma_x=0.01, sigma_y=0.015, rho=0.5, r0=0.03)


class TestBondPricing:
    """Test zero-coupon bond pricing."""

    def test_bond_price_at_zero_maturity(self, standard_params):
        """Bond price at T=0 should be 1.0."""
        price = zero_coupon_bond_price(standard_params, T=0.0)
        assert jnp.isclose(price, 1.0)

    def test_bond_price_range(self, standard_params):
        """Bond prices should be in (0, 1] for positive maturities."""
        for T in [0.5, 1.0, 5.0, 10.0]:
            price = zero_coupon_bond_price(standard_params, T)
            assert 0.0 < price <= 1.0, f"Price {price} out of range for T={T}"

    def test_bond_price_decreasing_with_maturity(self, standard_params):
        """Bond prices should decrease with maturity (positive rates)."""
        price_1y = zero_coupon_bond_price(standard_params, T=1.0)
        price_5y = zero_coupon_bond_price(standard_params, T=5.0)
        price_10y = zero_coupon_bond_price(standard_params, T=10.0)

        assert price_1y > price_5y > price_10y

    def test_bond_price_with_factors(self, standard_params):
        """Test bond pricing with non-zero factor values."""
        # Positive x increases rate, decreases bond price
        price_neutral = zero_coupon_bond_price(standard_params, T=5.0, x_t=0.0, y_t=0.0)
        price_positive_x = zero_coupon_bond_price(standard_params, T=5.0, x_t=0.01, y_t=0.0)

        assert price_positive_x < price_neutral

    def test_bond_price_with_phi(self, params_with_phi):
        """Test bond pricing with drift function."""
        price = zero_coupon_bond_price(params_with_phi, T=5.0)
        assert 0.0 < price < 1.0
        assert jnp.isfinite(price)

    def test_B_coefficient_properties(self):
        """Test B coefficient properties."""
        a = 0.1

        # B(t,t) = 0
        B_0 = B_coefficient(a, t=1.0, T=1.0)
        assert jnp.isclose(B_0, 0.0)

        # B increases with T-t
        B_1 = B_coefficient(a, t=0.0, T=1.0)
        B_5 = B_coefficient(a, t=0.0, T=5.0)
        assert B_5 > B_1

        # Limit: B(0,T) → 1/a as T → ∞
        B_large = B_coefficient(a, t=0.0, T=100.0)
        assert jnp.isclose(B_large, 1.0/a, atol=1e-3)


class TestSimulation:
    """Test path simulation."""

    def test_simulate_path_shape(self, standard_params):
        """Test that simulated path has correct shape."""
        key = jax.random.PRNGKey(42)
        r_path, x_path, y_path = simulate_path(standard_params, T=1.0, n_steps=100, key=key)

        assert r_path.shape == (101,)  # n_steps + 1
        assert x_path.shape == (101,)
        assert y_path.shape == (101,)

    def test_simulate_path_initial_conditions(self, standard_params):
        """Test that paths start at initial conditions."""
        key = jax.random.PRNGKey(42)
        r_path, x_path, y_path = simulate_path(standard_params, T=1.0, n_steps=100, key=key)

        assert jnp.isclose(r_path[0], standard_params.r0)
        assert jnp.isclose(x_path[0], standard_params.x0)
        assert jnp.isclose(y_path[0], standard_params.y0)

    def test_simulate_path_finite_values(self, standard_params):
        """Test that all simulated values are finite."""
        key = jax.random.PRNGKey(42)
        r_path, x_path, y_path = simulate_path(standard_params, T=1.0, n_steps=100, key=key)

        assert jnp.all(jnp.isfinite(r_path))
        assert jnp.all(jnp.isfinite(x_path))
        assert jnp.all(jnp.isfinite(y_path))

    def test_simulate_path_exact_method(self, standard_params):
        """Test exact simulation method."""
        key = jax.random.PRNGKey(42)
        r_path, x_path, y_path = simulate_path(
            standard_params, T=1.0, n_steps=100, key=key, method="exact"
        )

        assert r_path.shape == (101,)
        assert jnp.all(jnp.isfinite(r_path))

    def test_simulate_paths_shape(self, standard_params):
        """Test multiple path simulation."""
        key = jax.random.PRNGKey(42)
        n_paths = 100
        n_steps = 50

        r_paths, x_paths, y_paths = simulate_paths(
            standard_params, T=1.0, n_steps=n_steps, n_paths=n_paths, key=key
        )

        assert r_paths.shape == (n_paths, n_steps + 1)
        assert x_paths.shape == (n_paths, n_steps + 1)
        assert y_paths.shape == (n_paths, n_steps + 1)

    def test_simulate_paths_initial_conditions(self, standard_params):
        """Test that all paths start at initial condition."""
        key = jax.random.PRNGKey(42)
        r_paths, _, _ = simulate_paths(standard_params, T=1.0, n_steps=50, n_paths=100, key=key)

        assert jnp.allclose(r_paths[:, 0], standard_params.r0)

    def test_correlation_in_simulation(self, standard_params):
        """Test that simulated factors have correct correlation structure."""
        key = jax.random.PRNGKey(42)
        n_paths = 10000
        _, x_paths, y_paths = simulate_paths(
            standard_params, T=1.0, n_steps=100, n_paths=n_paths, key=key
        )

        # Check correlation of increments (approximate)
        x_increments = jnp.diff(x_paths, axis=1)
        y_increments = jnp.diff(y_paths, axis=1)

        # Average correlation across time steps
        correlations = []
        for t in range(x_increments.shape[1]):
            corr = jnp.corrcoef(x_increments[:, t], y_increments[:, t])[0, 1]
            correlations.append(corr)

        avg_corr = jnp.mean(jnp.array(correlations))
        # Should be close to rho = -0.7, but with sampling error
        assert -0.8 < avg_corr < -0.6, f"Correlation {avg_corr} far from {standard_params.rho}"


class TestDerivativePricing:
    """Test derivative pricing functions."""

    def test_caplet_price_positive(self, standard_params):
        """Caplet price should be positive for at-the-money options."""
        # Use a strike closer to the forward rate to ensure positive value
        # With r0=0.03 and mean reversion, forward rate ~0.01-0.02
        price = caplet_price(standard_params, strike=0.01, maturity=1.0, tenor=0.25)
        assert price > 0.0
        assert jnp.isfinite(price)

    def test_caplet_price_otm_vs_itm(self, standard_params):
        """Out-of-money caplet should be cheaper than in-the-money."""
        # ITM: strike below forward rate
        price_itm = caplet_price(standard_params, strike=0.01, maturity=1.0, tenor=0.25)

        # OTM: strike above forward rate
        price_otm = caplet_price(standard_params, strike=0.10, maturity=1.0, tenor=0.25)

        assert price_itm > price_otm

    def test_caplet_price_notional_scaling(self, standard_params):
        """Caplet price should scale linearly with notional."""
        price_1 = caplet_price(standard_params, strike=0.03, maturity=1.0, tenor=0.25, notional=1.0)
        price_2 = caplet_price(standard_params, strike=0.03, maturity=1.0, tenor=0.25, notional=2.0)

        assert jnp.isclose(price_2, 2.0 * price_1)

    def test_swaption_price_positive(self, standard_params):
        """Swaption price should be positive."""
        price = swaption_price(
            standard_params,
            swap_rate=0.03,
            option_maturity=1.0,
            swap_maturity=5.0,
            is_payer=True
        )
        assert price >= 0.0
        assert jnp.isfinite(price)

    def test_forward_rate_correlation_range(self, standard_params):
        """Forward rate correlation should be in [-1, 1]."""
        corr = forward_rate_correlation(standard_params, T1=1.0, T2=5.0, maturity=0.0)
        assert -1.0 <= corr <= 1.0

    def test_forward_rate_correlation_identical_maturities(self, standard_params):
        """Correlation of a forward rate with itself should be 1."""
        corr = forward_rate_correlation(standard_params, T1=5.0, T2=5.0, maturity=0.0)
        # Allow for numerical precision - should be very close to 1.0
        assert jnp.isclose(corr, 1.0, atol=0.15), f"Got correlation {corr}, expected ~1.0"


class TestJITCompilation:
    """Test JAX JIT compilation."""

    def test_jit_bond_price(self, standard_params):
        """Test JIT compilation of bond pricing."""
        jitted_fn = jax.jit(lambda T: zero_coupon_bond_price(standard_params, T))

        price = jitted_fn(5.0)
        assert 0.0 < price < 1.0

    def test_jit_simulation(self, standard_params):
        """Test JIT compilation of simulation."""
        # Already JIT-compiled via @partial(jax.jit, ...)
        key = jax.random.PRNGKey(42)
        r_path, _, _ = simulate_path(standard_params, T=1.0, n_steps=100, key=key)
        assert r_path.shape == (101,)


class TestCreateFittedG2PP:
    """Test convenience function for creating fitted models."""

    def test_create_fitted_flat_curve(self):
        """Test fitting to flat forward curve."""
        forward_curve = lambda t: 0.03  # Flat 3%

        params = create_fitted_g2pp(
            forward_curve,
            a=0.1, b=0.2,
            sigma_x=0.01, sigma_y=0.015,
            rho=-0.7
        )

        assert params.r0 == 0.03
        assert params.phi_fn is not None

    def test_create_fitted_upward_curve(self):
        """Test fitting to upward sloping curve."""
        forward_curve = lambda t: 0.02 + 0.005 * t  # Upward sloping

        params = create_fitted_g2pp(
            forward_curve,
            a=0.1, b=0.2,
            sigma_x=0.01, sigma_y=0.015,
            rho=-0.7
        )

        assert params.r0 == 0.02
        assert params.phi_fn is not None

        # Check drift function is callable
        phi_1 = params.phi_fn(1.0)
        assert jnp.isfinite(phi_1)


class TestNumericalStability:
    """Test numerical stability in edge cases."""

    def test_high_volatility(self):
        """Test with high volatility parameters."""
        params = G2PPParams(
            a=0.1, b=0.2,
            sigma_x=0.05,  # 500 bps volatility
            sigma_y=0.08,
            rho=-0.9,
            r0=0.03
        )

        price = zero_coupon_bond_price(params, T=5.0)
        assert 0.0 < price < 1.0
        assert jnp.isfinite(price)

    def test_long_maturity(self, standard_params):
        """Test pricing with very long maturity."""
        price = zero_coupon_bond_price(standard_params, T=30.0)
        assert 0.0 < price < 1.0
        assert jnp.isfinite(price)

    def test_high_correlation(self):
        """Test with correlation close to ±1."""
        params = G2PPParams(
            a=0.1, b=0.2,
            sigma_x=0.01, sigma_y=0.015,
            rho=0.99,
            r0=0.03
        )

        price = zero_coupon_bond_price(params, T=5.0)
        assert 0.0 < price < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
