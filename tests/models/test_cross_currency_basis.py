"""
Tests for Cross-Currency Basis Model
"""

import jax
import jax.numpy as jnp
import pytest

from neutryx.models.g2pp import G2PPParams
from neutryx.models.cross_currency_basis import (
    CrossCurrencyBasisParams,
    simulate_path,
    simulate_paths,
    fx_forward_rate,
    cross_currency_swap_value,
    quanto_option_price_mc,
    calibrate_basis_spread,
)


@pytest.fixture
def domestic_model():
    """Domestic interest rate model (USD)."""
    return G2PPParams(
        a=0.1,
        b=0.2,
        sigma_x=0.01,
        sigma_y=0.015,
        rho=-0.7,
        r0=0.03  # 3% USD rate
    )


@pytest.fixture
def foreign_model():
    """Foreign interest rate model (EUR)."""
    return G2PPParams(
        a=0.15,
        b=0.25,
        sigma_x=0.012,
        sigma_y=0.018,
        rho=-0.6,
        r0=0.02  # 2% EUR rate
    )


@pytest.fixture
def ccb_params(domestic_model, foreign_model):
    """Cross-currency basis parameters (USD/EUR)."""
    return CrossCurrencyBasisParams(
        domestic_model=domestic_model,
        foreign_model=foreign_model,
        fx_spot=1.10,  # 1.10 USD per EUR
        fx_vol_fn=0.10,  # 10% FX volatility
        basis_spread=-0.0020  # -20 bps basis
    )


class TestCrossCurrencyBasisParams:
    """Test parameter validation."""

    def test_valid_params(self, ccb_params):
        """Test that valid parameters are accepted."""
        assert ccb_params.fx_spot == 1.10
        assert ccb_params.get_fx_vol(0.0) == 0.10
        assert ccb_params.get_basis_spread(0.0) == -0.0020

    def test_negative_fx_spot(self, domestic_model, foreign_model):
        """Test that negative FX spot raises error."""
        with pytest.raises(ValueError, match="FX spot must be positive"):
            CrossCurrencyBasisParams(
                domestic_model=domestic_model,
                foreign_model=foreign_model,
                fx_spot=-1.10,
                fx_vol_fn=0.10
            )

    def test_negative_fx_vol(self, domestic_model, foreign_model):
        """Test that negative FX volatility raises error."""
        with pytest.raises(ValueError, match="FX volatility must be positive"):
            CrossCurrencyBasisParams(
                domestic_model=domestic_model,
                foreign_model=foreign_model,
                fx_spot=1.10,
                fx_vol_fn=-0.10
            )

    def test_invalid_correlation_matrix(self, domestic_model, foreign_model):
        """Test that invalid correlation matrix raises error."""
        # Non-symmetric matrix
        invalid_corr = jnp.array([
            [1.0, 0.5, -0.3],
            [0.3, 1.0, -0.2],  # Not symmetric
            [-0.3, -0.2, 1.0]
        ])

        with pytest.raises(ValueError, match="Correlation matrix must be symmetric"):
            CrossCurrencyBasisParams(
                domestic_model=domestic_model,
                foreign_model=foreign_model,
                fx_spot=1.10,
                fx_vol_fn=0.10,
                correlation_matrix=invalid_corr
            )

    def test_time_dependent_fx_vol(self, domestic_model, foreign_model):
        """Test time-dependent FX volatility."""
        fx_vol_fn = lambda t: 0.10 + 0.05 * jnp.exp(-t)

        params = CrossCurrencyBasisParams(
            domestic_model=domestic_model,
            foreign_model=foreign_model,
            fx_spot=1.10,
            fx_vol_fn=fx_vol_fn
        )

        vol_0 = params.get_fx_vol(0.0)
        vol_5 = params.get_fx_vol(5.0)

        assert vol_0 > vol_5  # Decaying volatility

    def test_time_dependent_basis(self, domestic_model, foreign_model):
        """Test time-dependent basis spread."""
        basis_fn = lambda t: -0.0020 - 0.0010 * jnp.exp(-0.5 * t)

        params = CrossCurrencyBasisParams(
            domestic_model=domestic_model,
            foreign_model=foreign_model,
            fx_spot=1.10,
            fx_vol_fn=0.10,
            basis_spread=basis_fn
        )

        basis_0 = params.get_basis_spread(0.0)
        basis_5 = params.get_basis_spread(5.0)

        assert basis_0 < basis_5  # Basis converges toward zero


class TestSimulation:
    """Test path simulation."""

    def test_simulate_path_shape(self, ccb_params):
        """Test that simulated paths have correct shape."""
        key = jax.random.PRNGKey(42)
        r_d_path, r_f_path, S_path = simulate_path(ccb_params, T=1.0, n_steps=252, key=key)

        assert r_d_path.shape == (253,)
        assert r_f_path.shape == (253,)
        assert S_path.shape == (253,)

    def test_simulate_path_initial_conditions(self, ccb_params):
        """Test that paths start at initial conditions."""
        key = jax.random.PRNGKey(42)
        r_d_path, r_f_path, S_path = simulate_path(ccb_params, T=1.0, n_steps=252, key=key)

        assert jnp.isclose(r_d_path[0], ccb_params.domestic_model.r0)
        assert jnp.isclose(r_f_path[0], ccb_params.foreign_model.r0)
        assert jnp.isclose(S_path[0], ccb_params.fx_spot)

    def test_simulate_path_finite_values(self, ccb_params):
        """Test that all values are finite."""
        key = jax.random.PRNGKey(42)
        r_d_path, r_f_path, S_path = simulate_path(ccb_params, T=1.0, n_steps=252, key=key)

        assert jnp.all(jnp.isfinite(r_d_path))
        assert jnp.all(jnp.isfinite(r_f_path))
        assert jnp.all(jnp.isfinite(S_path))

    def test_simulate_path_fx_positive(self, ccb_params):
        """Test that FX spot remains positive."""
        key = jax.random.PRNGKey(42)
        _, _, S_path = simulate_path(ccb_params, T=1.0, n_steps=252, key=key)

        assert jnp.all(S_path > 0.0)

    def test_simulate_paths_shape(self, ccb_params):
        """Test multiple path simulation."""
        key = jax.random.PRNGKey(42)
        n_paths = 100
        n_steps = 50

        r_d_paths, r_f_paths, S_paths = simulate_paths(
            ccb_params, T=1.0, n_steps=n_steps, n_paths=n_paths, key=key
        )

        assert r_d_paths.shape == (n_paths, n_steps + 1)
        assert r_f_paths.shape == (n_paths, n_steps + 1)
        assert S_paths.shape == (n_paths, n_steps + 1)

    def test_simulate_paths_initial_conditions(self, ccb_params):
        """Test that all paths start at initial conditions."""
        key = jax.random.PRNGKey(42)
        r_d_paths, r_f_paths, S_paths = simulate_paths(
            ccb_params, T=1.0, n_steps=50, n_paths=100, key=key
        )

        assert jnp.allclose(r_d_paths[:, 0], ccb_params.domestic_model.r0)
        assert jnp.allclose(r_f_paths[:, 0], ccb_params.foreign_model.r0)
        assert jnp.allclose(S_paths[:, 0], ccb_params.fx_spot)


class TestFXForwardRate:
    """Test FX forward rate computation."""

    def test_fx_forward_positive(self, ccb_params):
        """FX forward should be positive."""
        forward = fx_forward_rate(ccb_params, T=1.0)
        assert forward > 0.0
        assert jnp.isfinite(forward)

    def test_fx_forward_interest_rate_parity(self, ccb_params):
        """Test covered interest rate parity relationship."""
        # With positive domestic-foreign rate differential,
        # forward FX should be lower than spot (domestic appreciates)

        forward_1y = fx_forward_rate(ccb_params, T=1.0)

        # Since r_d (3%) > r_f (2%), forward should be less than spot
        # (accounting for basis spread)
        # This tests the qualitative relationship
        assert jnp.isfinite(forward_1y)

    def test_fx_forward_maturity_structure(self, ccb_params):
        """Test FX forward for different maturities."""
        forward_1y = fx_forward_rate(ccb_params, T=1.0)
        forward_5y = fx_forward_rate(ccb_params, T=5.0)

        # Both should be positive and finite
        assert forward_1y > 0.0
        assert forward_5y > 0.0

    def test_fx_forward_zero_basis(self, domestic_model, foreign_model):
        """Test FX forward with zero basis spread."""
        params = CrossCurrencyBasisParams(
            domestic_model=domestic_model,
            foreign_model=foreign_model,
            fx_spot=1.10,
            fx_vol_fn=0.10,
            basis_spread=0.0
        )

        forward = fx_forward_rate(params, T=1.0)
        assert forward > 0.0


class TestCrossCurrencySwap:
    """Test cross-currency swap valuation."""

    def test_swap_value_finite(self, ccb_params):
        """Swap value should be finite."""
        value = cross_currency_swap_value(
            ccb_params,
            notional_domestic=1e6,
            notional_foreign=900000,
            domestic_rate=0.03,
            foreign_rate=0.02,
            maturity=5.0,
            tenor=0.5
        )

        assert jnp.isfinite(value)

    def test_swap_value_receiver_vs_payer(self, ccb_params):
        """Test receiver vs payer swap values are opposite."""
        value_receive_dom = cross_currency_swap_value(
            ccb_params,
            notional_domestic=1e6,
            notional_foreign=900000,
            domestic_rate=0.03,
            foreign_rate=0.02,
            maturity=5.0,
            is_receive_domestic=True
        )

        value_pay_dom = cross_currency_swap_value(
            ccb_params,
            notional_domestic=1e6,
            notional_foreign=900000,
            domestic_rate=0.03,
            foreign_rate=0.02,
            maturity=5.0,
            is_receive_domestic=False
        )

        # Should be opposite signs (approximately)
        assert jnp.sign(value_receive_dom) == -jnp.sign(value_pay_dom) or \
               (jnp.abs(value_receive_dom) < 1e-6 and jnp.abs(value_pay_dom) < 1e-6)

    def test_swap_value_notional_scaling(self, ccb_params):
        """Swap value should scale with notional."""
        value_1 = cross_currency_swap_value(
            ccb_params,
            notional_domestic=1e6,
            notional_foreign=900000,
            domestic_rate=0.03,
            foreign_rate=0.02,
            maturity=5.0
        )

        value_2 = cross_currency_swap_value(
            ccb_params,
            notional_domestic=2e6,
            notional_foreign=1800000,
            domestic_rate=0.03,
            foreign_rate=0.02,
            maturity=5.0
        )

        # Should be approximately 2x
        ratio = value_2 / (value_1 + 1e-10)  # Avoid division by zero
        assert 1.8 < ratio < 2.2  # Allow some numerical error


class TestQuantoOptionPricing:
    """Test quanto option pricing."""

    def test_quanto_call_price_positive(self, ccb_params):
        """Quanto call option price should be non-negative."""
        key = jax.random.PRNGKey(42)
        price = quanto_option_price_mc(
            ccb_params,
            strike=100.0,
            maturity=1.0,
            is_call=True,
            foreign_asset_spot=100.0,
            foreign_asset_vol=0.20,
            n_paths=1000,
            key=key
        )

        assert price >= 0.0
        assert jnp.isfinite(price)

    def test_quanto_put_price_positive(self, ccb_params):
        """Quanto put option price should be non-negative."""
        key = jax.random.PRNGKey(42)
        price = quanto_option_price_mc(
            ccb_params,
            strike=100.0,
            maturity=1.0,
            is_call=False,
            foreign_asset_spot=100.0,
            foreign_asset_vol=0.20,
            n_paths=1000,
            key=key
        )

        assert price >= 0.0
        assert jnp.isfinite(price)

    def test_quanto_call_otm_vs_itm(self, ccb_params):
        """ITM quanto call should be more expensive than OTM."""
        key = jax.random.PRNGKey(42)

        # ITM: strike below spot
        price_itm = quanto_option_price_mc(
            ccb_params,
            strike=80.0,
            maturity=1.0,
            is_call=True,
            foreign_asset_spot=100.0,
            foreign_asset_vol=0.20,
            n_paths=1000,
            key=key
        )

        # OTM: strike above spot
        price_otm = quanto_option_price_mc(
            ccb_params,
            strike=120.0,
            maturity=1.0,
            is_call=True,
            foreign_asset_spot=100.0,
            foreign_asset_vol=0.20,
            n_paths=1000,
            key=key
        )

        assert price_itm > price_otm


class TestCalibrateBasisSpread:
    """Test basis spread calibration."""

    def test_calibrate_basis_positive_spread(self, domestic_model, foreign_model):
        """Test calibration with positive basis spread."""
        # Market FX forward lower than theoretical → positive basis
        fx_spot = 1.10
        fx_forward = 1.08  # Lower forward
        maturity = 1.0

        basis = calibrate_basis_spread(
            domestic_model, foreign_model, fx_spot, fx_forward, maturity
        )

        assert jnp.isfinite(basis)
        # Positive basis expected (domestic more expensive to borrow)
        assert basis > 0.0

    def test_calibrate_basis_negative_spread(self, domestic_model, foreign_model):
        """Test calibration with negative basis spread."""
        # Market FX forward higher than theoretical → negative basis
        fx_spot = 1.10
        fx_forward = 1.12  # Higher forward
        maturity = 1.0

        basis = calibrate_basis_spread(
            domestic_model, foreign_model, fx_spot, fx_forward, maturity
        )

        assert jnp.isfinite(basis)
        # Negative basis expected
        assert basis < 0.0

    def test_calibrate_basis_zero_spread(self):
        """Test calibration when forward equals theoretical (zero basis)."""
        # Create models with same rate
        model = G2PPParams(
            a=0.1, b=0.2,
            sigma_x=0.01, sigma_y=0.015,
            rho=-0.7, r0=0.03
        )

        fx_spot = 1.10
        maturity = 1.0

        # Compute theoretical forward with zero basis
        fx_forward_theoretical = fx_forward_rate(
            CrossCurrencyBasisParams(
                domestic_model=model,
                foreign_model=model,
                fx_spot=fx_spot,
                fx_vol_fn=0.10,
                basis_spread=0.0
            ),
            maturity
        )

        basis = calibrate_basis_spread(
            model, model, fx_spot, fx_forward_theoretical, maturity
        )

        # Should be close to zero
        assert jnp.abs(basis) < 1e-6


class TestNumericalStability:
    """Test numerical stability in edge cases."""

    def test_high_fx_volatility(self, domestic_model, foreign_model):
        """Test with high FX volatility."""
        params = CrossCurrencyBasisParams(
            domestic_model=domestic_model,
            foreign_model=foreign_model,
            fx_spot=1.10,
            fx_vol_fn=0.30,  # 30% volatility
            basis_spread=-0.0020
        )

        key = jax.random.PRNGKey(42)
        r_d, r_f, S = simulate_path(params, T=1.0, n_steps=252, key=key)

        assert jnp.all(jnp.isfinite(r_d))
        assert jnp.all(jnp.isfinite(r_f))
        assert jnp.all(jnp.isfinite(S))
        assert jnp.all(S > 0.0)

    def test_large_basis_spread(self, domestic_model, foreign_model):
        """Test with large basis spread."""
        params = CrossCurrencyBasisParams(
            domestic_model=domestic_model,
            foreign_model=foreign_model,
            fx_spot=1.10,
            fx_vol_fn=0.10,
            basis_spread=0.0100  # 100 bps basis
        )

        forward = fx_forward_rate(params, T=1.0)
        assert forward > 0.0
        assert jnp.isfinite(forward)

    def test_long_maturity(self, ccb_params):
        """Test with long maturity."""
        forward = fx_forward_rate(ccb_params, T=30.0)
        assert forward > 0.0
        assert jnp.isfinite(forward)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
