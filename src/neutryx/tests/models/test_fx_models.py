"""Tests for FX stochastic volatility models."""
import jax.numpy as jnp
import pytest

from neutryx.models.fx_models import (
    FXHestonModel,
    FXSABRModel,
    FXBatesModel,
    TwoFactorFXModel,
)


class TestFXHestonModel:
    """Test FX Heston stochastic volatility model."""

    def test_fx_heston_creation(self):
        """Test FX Heston model instantiation."""
        model = FXHestonModel(
            v0=0.04,
            kappa=2.0,
            theta=0.04,
            sigma=0.3,
            rho=-0.7,
            r_domestic=0.05,
            r_foreign=0.02,
        )
        assert model.v0 == 0.04
        assert model.kappa == 2.0
        assert model.theta == 0.04
        assert model.sigma == 0.3
        assert model.rho == -0.7
        assert model.r_domestic == 0.05
        assert model.r_foreign == 0.02

    def test_feller_condition_warning(self):
        """Test that Feller condition violation triggers warning."""
        import warnings

        # Violate Feller: 2κθ < σ²
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model = FXHestonModel(
                v0=0.04,
                kappa=0.5,   # Small kappa
                theta=0.01,  # Small theta
                sigma=0.8,   # Large sigma -> 2*0.5*0.01=0.01 < 0.64
                rho=-0.7,
                r_domestic=0.05,
                r_foreign=0.02,
            )
            assert len(w) == 1
            assert "Feller condition violated" in str(w[0].message)

    def test_parameter_validation(self):
        """Test parameter validation."""
        # Invalid rho
        with pytest.raises(ValueError, match="Correlation must be in"):
            FXHestonModel(
                v0=0.04, kappa=2.0, theta=0.04, sigma=0.3,
                rho=1.5,  # Invalid
                r_domestic=0.05, r_foreign=0.02
            )

        # Negative variance
        with pytest.raises(ValueError, match="must be positive"):
            FXHestonModel(
                v0=-0.04,  # Invalid
                kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7,
                r_domestic=0.05, r_foreign=0.02
            )

    def test_price_atm_call(self):
        """Test pricing ATM FX call option."""
        model = FXHestonModel(
            v0=0.04,
            kappa=2.0,
            theta=0.04,
            sigma=0.3,
            rho=-0.7,
            r_domestic=0.05,
            r_foreign=0.02,
        )

        S = 1.10  # EURUSD spot
        K = 1.10  # ATM
        T = 1.0

        price = model.price(S, K, T, is_call=True)

        # ATM call should be positive
        assert price > 0

        # Rough sanity check: ATM call ≈ 0.4 * S * σ * √T (rule of thumb)
        approx_price = 0.4 * S * jnp.sqrt(model.v0) * jnp.sqrt(T)
        assert 0.5 * approx_price < price < 2.0 * approx_price

    def test_price_otm_call(self):
        """Test pricing OTM call."""
        model = FXHestonModel(
            v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7,
            r_domestic=0.05, r_foreign=0.02
        )

        S = 1.10
        K_otm = 1.20  # OTM call
        T = 1.0

        price_otm = model.price(S, K_otm, T, is_call=True)

        # OTM should be cheaper than ATM
        price_atm = model.price(S, S, T, is_call=True)
        assert price_otm < price_atm

    def test_put_call_parity(self):
        """Test approximate put-call parity for European options."""
        model = FXHestonModel(
            v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7,
            r_domestic=0.05, r_foreign=0.02
        )

        S = 1.10
        K = 1.10
        T = 1.0

        call_price = model.price(S, K, T, is_call=True)
        put_price = model.price(S, K, T, is_call=False)

        # Put-call parity: C - P = S*exp(-r_f*T) - K*exp(-r_d*T)
        forward = S * jnp.exp((model.r_domestic - model.r_foreign) * T)
        discount = jnp.exp(-model.r_domestic * T)
        parity_diff = forward - K
        actual_diff = call_price - put_price

        # Should match within numerical tolerance
        assert jnp.abs(actual_diff - parity_diff * discount) < 0.01

    def test_implied_vol(self):
        """Test implied volatility calculation."""
        model = FXHestonModel(
            v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7,
            r_domestic=0.05, r_foreign=0.02
        )

        S = 1.10
        K = 1.10
        T = 1.0

        # Get implied vol for ATM
        impl_vol = model.implied_vol(S, K, T, is_call=True)

        # Should be reasonably close to spot vol √v0
        # (Can deviate due to vol-of-vol and correlation effects in Heston)
        spot_vol = jnp.sqrt(model.v0)
        assert jnp.abs(impl_vol - spot_vol) < 0.10

    def test_smile_calibration(self):
        """Test smile calibration to market data."""
        # Create model with initial guess
        model = FXHestonModel(
            v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5,
            r_domestic=0.05, r_foreign=0.02
        )

        S = 1.10
        T = 1.0

        # Synthetic market smile (increasing vol for OTM puts)
        strikes = jnp.array([0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25])
        market_vols = jnp.array([0.25, 0.22, 0.20, 0.20, 0.21, 0.23, 0.26])

        # Calibrate
        result = model.calibrate_to_smile(S, T, strikes, market_vols)

        # Check calibration succeeded
        assert result['success']
        assert result['rmse'] < 0.10  # Good fit (10% vol error is acceptable)

        # Check parameters are reasonable
        v0, kappa, theta, sigma, rho = result['params']
        assert 0.001 < v0 < 1.0
        assert 0.01 < kappa < 10.0
        assert 0.001 < theta < 1.0
        assert 0.01 < sigma < 2.0
        assert -0.99 < rho < 0.99


class TestFXSABRModel:
    """Test FX SABR model."""

    def test_fx_sabr_creation(self):
        """Test SABR model instantiation."""
        model = FXSABRModel(
            alpha=0.15,
            beta=0.5,
            rho=-0.25,
            nu=0.4,
        )
        assert model.alpha == 0.15
        assert model.beta == 0.5
        assert model.rho == -0.25
        assert model.nu == 0.4

    def test_parameter_validation(self):
        """Test parameter validation."""
        # Invalid alpha
        with pytest.raises(ValueError, match="Alpha must be positive"):
            FXSABRModel(alpha=-0.1, beta=0.5, rho=-0.25, nu=0.4)

        # Invalid beta
        with pytest.raises(ValueError, match="Beta must be in"):
            FXSABRModel(alpha=0.15, beta=1.5, rho=-0.25, nu=0.4)

        # Invalid rho
        with pytest.raises(ValueError, match="Rho must be in"):
            FXSABRModel(alpha=0.15, beta=0.5, rho=-1.5, nu=0.4)

        # Negative nu
        with pytest.raises(ValueError, match="Nu must be non-negative"):
            FXSABRModel(alpha=0.15, beta=0.5, rho=-0.25, nu=-0.1)

    def test_implied_vol_atm(self):
        """Test SABR implied vol at ATM."""
        model = FXSABRModel(
            alpha=0.15,
            beta=0.5,
            rho=-0.25,
            nu=0.4,
        )

        F = 1.10
        K = 1.10  # ATM
        T = 1.0

        impl_vol = model.implied_vol(F, K, T)

        # ATM vol should be close to alpha for β=0.5
        assert jnp.abs(impl_vol - model.alpha) < 0.02

    def test_smile_shape(self):
        """Test SABR smile shape (vol increases for OTM puts with ρ<0)."""
        model = FXSABRModel(
            alpha=0.15,
            beta=0.5,
            rho=-0.5,  # Negative correlation
            nu=0.4,
        )

        F = 1.10
        T = 1.0

        # Calculate vols across strikes
        K_low = 1.00   # OTM put
        K_atm = 1.10
        K_high = 1.20  # OTM call

        vol_low = model.implied_vol(F, K_low, T)
        vol_atm = model.implied_vol(F, K_atm, T)
        vol_high = model.implied_vol(F, K_high, T)

        # With ρ<0, OTM puts have higher vol than ATM
        assert vol_low > vol_atm

        # OTM calls have lower vol
        assert vol_high < vol_atm

    def test_calibration_fixed_beta(self):
        """Test SABR calibration with fixed beta."""
        model = FXSABRModel(
            alpha=0.15, beta=0.5, rho=-0.25, nu=0.4
        )

        F = 1.10
        T = 1.0

        # Synthetic market smile
        strikes = jnp.array([0.95, 1.00, 1.05, 1.10, 1.15, 1.20])
        market_vols = jnp.array([0.18, 0.16, 0.15, 0.15, 0.16, 0.17])

        # Calibrate with β=0.5 fixed
        result = model.calibrate_to_smile(F, T, strikes, market_vols, beta=0.5)

        # Check success (either optimizer succeeded or RMSE is good)
        assert result['success'] or result['rmse'] < 0.02
        assert result['rmse'] < 0.02

        # Check parameters
        alpha, beta, rho, nu = result['params']
        assert beta == 0.5  # Fixed
        assert alpha > 0
        assert -1 < rho < 1
        assert nu >= 0


class TestFXBatesModel:
    """Test FX Bates model (Heston + jumps)."""

    def test_fx_bates_creation(self):
        """Test Bates model instantiation."""
        model = FXBatesModel(
            v0=0.03, kappa=2.0, theta=0.03,
            sigma=0.3, rho=-0.7,
            r_domestic=0.05, r_foreign=0.001,
            lambda_jump=0.5,
            mu_jump=-0.02,
            sigma_jump=0.05,
        )
        assert model.v0 == 0.03
        assert model.lambda_jump == 0.5
        assert model.mu_jump == -0.02
        assert model.sigma_jump == 0.05

    def test_characteristic_function(self):
        """Test characteristic function computation."""
        model = FXBatesModel(
            v0=0.03, kappa=2.0, theta=0.03,
            sigma=0.3, rho=-0.7,
            r_domestic=0.05, r_foreign=0.001,
            lambda_jump=0.5,
            mu_jump=-0.02,
            sigma_jump=0.05,
        )

        u = 1.0 + 0.5j
        T = 1.0
        S0 = 100.0

        cf = model.characteristic_function(u, T, S0)

        # CF should be complex
        assert isinstance(cf, complex) or jnp.iscomplexobj(cf)

        # |CF| should be finite and reasonable
        assert jnp.abs(cf) < 1000


class TestTwoFactorFXModel:
    """Test two-factor FX volatility model."""

    def test_two_factor_creation(self):
        """Test two-factor model instantiation."""
        model = TwoFactorFXModel(
            v1_0=0.02, v2_0=0.02,
            kappa1=10.0, kappa2=0.5,
            theta1=0.015, theta2=0.025,
            sigma1=0.5, sigma2=0.2,
            rho12=0.3, rho1S=-0.5, rho2S=-0.3,
            r_domestic=0.05, r_foreign=0.02,
        )
        assert model.v1_0 == 0.02
        assert model.v2_0 == 0.02
        assert model.kappa1 == 10.0  # Fast
        assert model.kappa2 == 0.5   # Slow

    def test_price_monte_carlo(self):
        """Test Monte Carlo pricing."""
        model = TwoFactorFXModel(
            v1_0=0.02, v2_0=0.02,
            kappa1=10.0, kappa2=0.5,
            theta1=0.015, theta2=0.025,
            sigma1=0.5, sigma2=0.2,
            rho12=0.3, rho1S=-0.5, rho2S=-0.3,
            r_domestic=0.05, r_foreign=0.02,
        )

        S0 = 1.10
        K = 1.10
        T = 1.0

        # Price ATM call with reduced paths for speed
        price = model.price_monte_carlo(
            S0, K, T,
            is_call=True,
            n_paths=10000,  # Reduced for test speed
            n_steps=50
        )

        # Price should be positive
        assert price > 0

        # Rough sanity check
        total_vol = jnp.sqrt(model.v1_0 + model.v2_0)
        approx_price = 0.4 * S0 * total_vol * jnp.sqrt(T)
        assert 0.3 * approx_price < price < 3.0 * approx_price

    def test_put_call_consistency(self):
        """Test that put and call pricing is consistent."""
        model = TwoFactorFXModel(
            v1_0=0.02, v2_0=0.02,
            kappa1=10.0, kappa2=0.5,
            theta1=0.015, theta2=0.025,
            sigma1=0.5, sigma2=0.2,
            rho12=0.3, rho1S=-0.5, rho2S=-0.3,
            r_domestic=0.05, r_foreign=0.02,
        )

        S0 = 1.10
        K = 1.15  # OTM call / ITM put
        T = 1.0

        # Use same random key for both
        import jax
        key = jax.random.PRNGKey(42)

        call_price = model.price_monte_carlo(
            S0, K, T, is_call=True,
            n_paths=50000, n_steps=50,
            random_key=key
        )

        put_price = model.price_monte_carlo(
            S0, K, T, is_call=False,
            n_paths=50000, n_steps=50,
            random_key=key
        )

        # Check put-call parity (approximate due to MC)
        forward = S0 * jnp.exp((model.r_domestic - model.r_foreign) * T)
        discount = jnp.exp(-model.r_domestic * T)
        parity_diff = (forward - K) * discount

        actual_diff = call_price - put_price

        # Allow larger tolerance for MC (sampling error with 50k paths)
        assert jnp.abs(actual_diff - parity_diff) < 0.15


class TestFXModelsIntegration:
    """Integration tests across FX models."""

    def test_heston_vs_garman_kohlhagen_convergence(self):
        """Test that Heston converges to GK when vol is constant."""
        # Low vol-of-vol should make Heston ≈ GK
        model = FXHestonModel(
            v0=0.04, kappa=10.0, theta=0.04,
            sigma=0.01,  # Very low σ → deterministic vol
            rho=0.0,
            r_domestic=0.05, r_foreign=0.02
        )

        from neutryx.products.fx_options import garman_kohlhagen

        S = 1.10
        K = 1.10
        T = 1.0
        vol = jnp.sqrt(0.04)

        heston_price = model.price(S, K, T, is_call=True)
        gk_price = garman_kohlhagen(S, K, T, vol, 0.05, 0.02, is_call=True)

        # Should be very close
        assert jnp.abs(heston_price - gk_price) < 0.01

    def test_multi_model_smile_comparison(self):
        """Compare smile shapes across Heston and SABR."""
        # Create similar models
        heston = FXHestonModel(
            v0=0.04, kappa=2.0, theta=0.04,
            sigma=0.4, rho=-0.7,
            r_domestic=0.05, r_foreign=0.02
        )

        sabr = FXSABRModel(
            alpha=0.20, beta=0.5, rho=-0.5, nu=0.4
        )

        F = 1.10
        T = 1.0
        strikes = jnp.array([1.00, 1.05, 1.10, 1.15, 1.20])

        # Get smiles
        heston_vols = jnp.array([
            heston.implied_vol(F, K, T) for K in strikes
        ])

        sabr_vols = jnp.array([
            sabr.implied_vol(F, K, T) for K in strikes
        ])

        # Both should exhibit smile (higher vol at wings)
        # Check OTM puts have higher vol than ATM
        assert heston_vols[0] > heston_vols[2]
        assert sabr_vols[0] > sabr_vols[2]

        # Models should produce different but reasonable smiles
        assert not jnp.allclose(heston_vols, sabr_vols, atol=0.01)
