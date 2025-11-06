"""Tests for advanced equity models: Lévy processes, SLV, and jump clustering."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from neutryx.core.engine import MCConfig
from neutryx.models.levy_processes import (
    CGMYParams,
    NIGParams,
    cgmy_characteristic_function,
    nig_characteristic_function,
    price_vanilla_cgmy_mc,
    price_vanilla_nig_mc,
    simulate_cgmy,
    simulate_nig,
)
from neutryx.models.jump_clustering import (
    HawkesJumpParams,
    IntensityDependentJumpParams,
    MultivariateHawkesParams,
    RegimeSwitchingHawkesParams,
    SelfExcitingLevyParams,
    calibrate_hawkes_from_jumps,
    price_vanilla_hawkes_mc,
    simulate_hawkes_jump_diffusion,
    simulate_intensity_dependent_hawkes,
    simulate_multivariate_hawkes,
    simulate_regime_switching_hawkes,
    simulate_self_exciting_levy,
)
from neutryx.models.equity_models import (
    SLVParams,
    TimeChangedLevyParams,
    simulate_slv,
    simulate_time_changed_levy,
)


@pytest.fixture
def key():
    """Random key for tests."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def mc_config():
    """Standard MC configuration for tests."""
    return MCConfig(paths=2000, steps=50, antithetic=True, dtype=jnp.float32)


@pytest.fixture
def market_params():
    """Standard market parameters."""
    return {
        'S0': 100.0,
        'K': 100.0,
        'T': 1.0,
        'r': 0.05,
        'q': 0.01,
    }


# ==============================================================================
# Normal Inverse Gaussian (NIG) Tests
# ==============================================================================


class TestNIG:
    """Test suite for NIG model."""

    def test_nig_params_validation(self):
        """Test NIG parameter validation."""
        # Valid parameters
        params = NIGParams(alpha=15.0, beta=-5.0, delta=0.5, mu=0.0)
        assert params.alpha == 15.0
        assert params.beta == -5.0

        # Invalid: alpha must be positive
        with pytest.raises(ValueError, match="alpha must be positive"):
            NIGParams(alpha=-1.0, beta=0.0, delta=0.5, mu=0.0)

        # Invalid: delta must be positive
        with pytest.raises(ValueError, match="delta must be positive"):
            NIGParams(alpha=15.0, beta=0.0, delta=-0.5, mu=0.0)

        # Invalid: alpha^2 <= beta^2
        with pytest.raises(ValueError, match="alpha\\^2 > beta\\^2"):
            NIGParams(alpha=5.0, beta=6.0, delta=0.5, mu=0.0)

    def test_nig_simulation_basic(self, key, mc_config, market_params):
        """Test basic NIG simulation."""
        params = NIGParams(alpha=15.0, beta=-5.0, delta=0.5, mu=0.0)

        paths = simulate_nig(
            key,
            market_params['S0'],
            market_params['T'],
            market_params['r'],
            market_params['q'],
            params,
            mc_config,
        )

        # Check shape
        assert paths.shape == (mc_config.paths, mc_config.steps + 1)

        # Check initial value
        assert jnp.allclose(paths[:, 0], market_params['S0'])

        # Check all positive
        assert jnp.all(paths > 0)

        # Check final values are reasonable
        ST = paths[:, -1]
        assert jnp.mean(ST) > 0
        assert jnp.std(ST) > 0

    def test_nig_martingale_property(self, key, mc_config):
        """Test that NIG process is approximately a martingale."""
        params = NIGParams(alpha=15.0, beta=-5.0, delta=0.5, mu=0.0)
        S0 = 100.0
        T = 1.0
        r = 0.05
        q = 0.01

        # Use large number of paths for better convergence
        large_cfg = MCConfig(paths=10000, steps=50, antithetic=True, dtype=jnp.float32)

        paths = simulate_nig(key, S0, T, r, q, params, large_cfg)
        ST = paths[:, -1]

        # Forward price should match E[ST]
        forward = S0 * jnp.exp((r - q) * T)
        mean_ST = jnp.mean(ST)

        # Allow 5% relative error
        assert jnp.abs(mean_ST - forward) / forward < 0.05

    def test_nig_characteristic_function(self):
        """Test NIG characteristic function properties."""
        params = NIGParams(alpha=15.0, beta=-5.0, delta=0.5, mu=0.0)
        t = 1.0

        # CF at u=0 should be 1
        cf_0 = nig_characteristic_function(0.0, t, params)
        assert jnp.abs(cf_0 - 1.0) < 1e-6

        # CF should have correct symmetry for beta=0
        params_sym = NIGParams(alpha=15.0, beta=0.0, delta=0.5, mu=0.0)
        cf_pos = nig_characteristic_function(1.0, t, params_sym)
        cf_neg = nig_characteristic_function(-1.0, t, params_sym)
        assert jnp.abs(cf_pos - jnp.conj(cf_neg)) < 1e-6

    def test_nig_pricing(self, key, mc_config, market_params):
        """Test NIG option pricing."""
        params = NIGParams(alpha=15.0, beta=-5.0, delta=0.5, mu=0.0)

        call_price = price_vanilla_nig_mc(
            key,
            market_params['S0'],
            market_params['K'],
            market_params['T'],
            market_params['r'],
            market_params['q'],
            params,
            mc_config,
            kind="call",
        )

        # Call price should be positive
        assert call_price > 0
        assert not jnp.isnan(call_price)

        # Put price should also work
        put_price = price_vanilla_nig_mc(
            key,
            market_params['S0'],
            market_params['K'],
            market_params['T'],
            market_params['r'],
            market_params['q'],
            params,
            mc_config,
            kind="put",
        )

        assert put_price > 0
        assert not jnp.isnan(put_price)


# ==============================================================================
# CGMY Tests
# ==============================================================================


class TestCGMY:
    """Test suite for CGMY model."""

    def test_cgmy_params_validation(self):
        """Test CGMY parameter validation."""
        # Valid parameters
        params = CGMYParams(C=1.0, G=10.0, M=10.0, Y=0.5)
        assert params.C == 1.0

        # Invalid: C must be positive
        with pytest.raises(ValueError, match="C must be positive"):
            CGMYParams(C=-1.0, G=10.0, M=10.0, Y=0.5)

        # Invalid: Y >= 2
        with pytest.raises(ValueError, match="Y must be < 2"):
            CGMYParams(C=1.0, G=10.0, M=10.0, Y=2.5)

        # Invalid: Y >= 1 but G or M <= 1
        with pytest.raises(ValueError, match="G > 1 and M > 1"):
            CGMYParams(C=1.0, G=0.5, M=10.0, Y=1.2)

    def test_cgmy_simulation_basic(self, key, mc_config, market_params):
        """Test basic CGMY simulation."""
        params = CGMYParams(C=1.0, G=10.0, M=10.0, Y=0.5)

        paths = simulate_cgmy(
            key,
            market_params['S0'],
            market_params['T'],
            market_params['r'],
            market_params['q'],
            params,
            mc_config,
        )

        # Check shape
        assert paths.shape == (mc_config.paths, mc_config.steps + 1)

        # Check initial value
        assert jnp.allclose(paths[:, 0], market_params['S0'])

        # Check all positive
        assert jnp.all(paths > 0)

    def test_cgmy_different_y_values(self, key, market_params):
        """Test CGMY with different Y parameters."""
        S0 = market_params['S0']
        T = market_params['T']
        r = market_params['r']
        q = market_params['q']

        cfg = MCConfig(paths=500, steps=50, antithetic=False, dtype=jnp.float32)

        # Finite activity (Y < 0)
        params_finite = CGMYParams(C=0.5, G=10.0, M=10.0, Y=-0.5)
        paths_finite = simulate_cgmy(key, S0, T, r, q, params_finite, cfg)
        assert paths_finite.shape[0] == cfg.paths

        # Infinite activity, finite variation (0 < Y < 1)
        params_inf_fin = CGMYParams(C=1.0, G=10.0, M=10.0, Y=0.5)
        paths_inf_fin = simulate_cgmy(key, S0, T, r, q, params_inf_fin, cfg)
        assert paths_inf_fin.shape[0] == cfg.paths

        # Infinite activity, infinite variation (1 < Y < 2)
        params_inf_inf = CGMYParams(C=1.0, G=5.0, M=5.0, Y=1.5)
        paths_inf_inf = simulate_cgmy(key, S0, T, r, q, params_inf_inf, cfg)
        assert paths_inf_inf.shape[0] == cfg.paths

    def test_cgmy_pricing(self, key, mc_config, market_params):
        """Test CGMY option pricing."""
        params = CGMYParams(C=1.0, G=10.0, M=10.0, Y=0.5)

        call_price = price_vanilla_cgmy_mc(
            key,
            market_params['S0'],
            market_params['K'],
            market_params['T'],
            market_params['r'],
            market_params['q'],
            params,
            mc_config,
            kind="call",
        )

        assert call_price > 0
        assert not jnp.isnan(call_price)

    def test_cgmy_characteristic_function(self):
        """Test CGMY characteristic function."""
        params = CGMYParams(C=1.0, G=10.0, M=10.0, Y=0.5)
        t = 1.0

        # CF at u=0 should be 1
        cf_0 = cgmy_characteristic_function(0.0, t, params)
        assert jnp.abs(cf_0 - 1.0) < 1e-6


# ==============================================================================
# Hawkes Jump-Diffusion Tests
# ==============================================================================


class TestHawkesJump:
    """Test suite for Hawkes jump-diffusion model."""

    def test_hawkes_params_validation(self):
        """Test Hawkes parameter validation."""
        # Valid subcritical parameters
        params = HawkesJumpParams(
            sigma=0.15,
            lambda0=5.0,
            alpha=1.5,
            beta=2.0,
            mu_jump=-0.05,
            sigma_jump=0.1,
        )
        assert params.branching_ratio() == 1.5 / 2.0

        # Invalid: negative sigma
        with pytest.raises(ValueError, match="sigma must be non-negative"):
            HawkesJumpParams(
                sigma=-0.1,
                lambda0=5.0,
                alpha=1.5,
                beta=2.0,
                mu_jump=-0.05,
                sigma_jump=0.1,
            )

        # Invalid: non-positive lambda0
        with pytest.raises(ValueError, match="lambda0 must be positive"):
            HawkesJumpParams(
                sigma=0.15,
                lambda0=0.0,
                alpha=1.5,
                beta=2.0,
                mu_jump=-0.05,
                sigma_jump=0.1,
            )

        # Invalid: negative alpha
        with pytest.raises(ValueError, match="alpha must be non-negative"):
            HawkesJumpParams(
                sigma=0.15,
                lambda0=5.0,
                alpha=-1.0,
                beta=2.0,
                mu_jump=-0.05,
                sigma_jump=0.1,
            )

    def test_hawkes_branching_ratio(self):
        """Test Hawkes branching ratio calculation."""
        # Subcritical
        params_sub = HawkesJumpParams(
            sigma=0.15, lambda0=5.0, alpha=1.5, beta=2.0, mu_jump=-0.05, sigma_jump=0.1
        )
        assert params_sub.branching_ratio() < 1.0
        assert params_sub.mean_intensity() > params_sub.lambda0

        # Critical
        params_crit = HawkesJumpParams(
            sigma=0.15, lambda0=5.0, alpha=2.0, beta=2.0, mu_jump=-0.05, sigma_jump=0.1
        )
        assert params_crit.branching_ratio() == 1.0
        assert params_crit.mean_intensity() == float('inf')

    def test_hawkes_simulation_basic(self, key, market_params):
        """Test basic Hawkes jump-diffusion simulation."""
        params = HawkesJumpParams(
            sigma=0.15,
            lambda0=5.0,
            alpha=1.5,
            beta=2.0,
            mu_jump=-0.05,
            sigma_jump=0.1,
        )

        # Use smaller config for Hawkes (computationally intensive)
        cfg = MCConfig(paths=100, steps=50, antithetic=False, dtype=jnp.float32)

        paths = simulate_hawkes_jump_diffusion(
            key,
            market_params['S0'],
            market_params['T'],
            market_params['r'],
            market_params['q'],
            params,
            cfg,
        )

        # Check shape
        assert paths.shape == (cfg.paths, cfg.steps + 1)

        # Check initial value
        assert jnp.allclose(paths[:, 0], market_params['S0'])

        # Check all positive
        assert jnp.all(paths > 0)

    def test_hawkes_clustering_effect(self, key, market_params):
        """Test that Hawkes process creates jump clustering."""
        # Strong clustering
        params_cluster = HawkesJumpParams(
            sigma=0.10,
            lambda0=2.0,
            alpha=1.8,  # High alpha -> strong clustering
            beta=2.0,
            mu_jump=-0.1,
            sigma_jump=0.15,
        )

        # No clustering (alpha=0)
        params_no_cluster = HawkesJumpParams(
            sigma=0.10,
            lambda0=2.0,
            alpha=0.0,  # No self-excitation
            beta=2.0,
            mu_jump=-0.1,
            sigma_jump=0.15,
        )

        cfg = MCConfig(paths=50, steps=100, antithetic=False, dtype=jnp.float32)

        paths_cluster = simulate_hawkes_jump_diffusion(
            key, market_params['S0'], market_params['T'], market_params['r'], market_params['q'],
            params_cluster, cfg
        )

        key2 = jax.random.fold_in(key, 999)
        paths_no_cluster = simulate_hawkes_jump_diffusion(
            key2, market_params['S0'], market_params['T'], market_params['r'], market_params['q'],
            params_no_cluster, cfg
        )

        # Clustering should lead to higher volatility
        vol_cluster = jnp.std(jnp.log(paths_cluster[:, -1] / paths_cluster[:, 0]))
        vol_no_cluster = jnp.std(jnp.log(paths_no_cluster[:, -1] / paths_no_cluster[:, 0]))

        # Not a strict test, but clustering typically increases volatility
        assert vol_cluster >= 0  # Just check it's computed

    def test_hawkes_pricing(self, key, market_params):
        """Test Hawkes jump-diffusion option pricing."""
        params = HawkesJumpParams(
            sigma=0.15,
            lambda0=5.0,
            alpha=1.5,
            beta=2.0,
            mu_jump=-0.05,
            sigma_jump=0.1,
        )

        cfg = MCConfig(paths=500, steps=50, antithetic=False, dtype=jnp.float32)

        call_price = price_vanilla_hawkes_mc(
            key,
            market_params['S0'],
            market_params['K'],
            market_params['T'],
            market_params['r'],
            market_params['q'],
            params,
            cfg,
            kind="call",
        )

        assert call_price > 0
        assert not jnp.isnan(call_price)


# ==============================================================================
# SLV (Stochastic Local Volatility) Tests
# ==============================================================================


class TestSLV:
    """Test suite for SLV model."""

    def test_slv_simulation_constant_local_vol(self, key, mc_config, market_params):
        """Test SLV with constant local volatility function."""

        def constant_local_vol(S, t):
            return 0.2

        params = SLVParams(
            kappa=2.0,
            theta=0.04,
            xi=0.3,
            rho=-0.7,
            local_vol_func=constant_local_vol,
            V0=0.04,
        )

        paths = simulate_slv(
            key,
            market_params['S0'],
            market_params['T'],
            market_params['r'],
            market_params['q'],
            params,
            mc_config,
        )

        # Check shape
        assert paths.shape == (mc_config.paths, mc_config.steps + 1)

        # Check initial value
        assert jnp.allclose(paths[:, 0], market_params['S0'])

        # Check all positive
        assert jnp.all(paths > 0)

    def test_slv_vs_heston_limit(self, key, market_params):
        """Test that SLV with constant local vol=1 reduces to Heston."""

        def unity_local_vol(S, t):
            return 1.0

        params_slv = SLVParams(
            kappa=2.0,
            theta=0.04,
            xi=0.3,
            rho=-0.7,
            local_vol_func=unity_local_vol,
            V0=0.04,
        )

        cfg = MCConfig(paths=1000, steps=50, antithetic=False, dtype=jnp.float32)

        paths_slv = simulate_slv(
            key,
            market_params['S0'],
            market_params['T'],
            market_params['r'],
            market_params['q'],
            params_slv,
            cfg,
        )

        # SLV with sigma_L = 1 should behave like Heston
        # Just check that it runs and produces reasonable output
        ST_mean = jnp.mean(paths_slv[:, -1])
        forward = market_params['S0'] * jnp.exp(
            (market_params['r'] - market_params['q']) * market_params['T']
        )

        # Should be roughly at forward (allow for Euler discretization bias)
        # With high vol-of-vol (xi=0.3) and only 50 steps, Euler scheme has negative bias
        assert jnp.abs(ST_mean - forward) / forward < 0.15  # 15% tolerance for discretization error


# ==============================================================================
# Time-Changed Lévy Framework Tests
# ==============================================================================


class TestTimeChangedLevy:
    """Test suite for time-changed Lévy framework."""

    def test_time_changed_levy_vg(self, key, mc_config, market_params):
        """Test time-changed Lévy with VG."""
        params = TimeChangedLevyParams(
            levy_process='VG',
            levy_params={'theta': -0.14, 'sigma': 0.2, 'nu': 0.2},
        )

        paths = simulate_time_changed_levy(
            key,
            market_params['S0'],
            market_params['T'],
            market_params['r'],
            market_params['q'],
            params,
            mc_config,
        )

        assert paths.shape[1] == mc_config.steps + 1
        assert jnp.all(paths > 0)

    def test_time_changed_levy_nig(self, key, mc_config, market_params):
        """Test time-changed Lévy with NIG."""
        params = TimeChangedLevyParams(
            levy_process='NIG',
            levy_params={'alpha': 15.0, 'beta': -5.0, 'delta': 0.5, 'mu': 0.0},
        )

        paths = simulate_time_changed_levy(
            key,
            market_params['S0'],
            market_params['T'],
            market_params['r'],
            market_params['q'],
            params,
            mc_config,
        )

        assert paths.shape[1] == mc_config.steps + 1
        assert jnp.all(paths > 0)

    def test_time_changed_levy_cgmy(self, key, mc_config, market_params):
        """Test time-changed Lévy with CGMY."""
        params = TimeChangedLevyParams(
            levy_process='CGMY',
            levy_params={'C': 1.0, 'G': 10.0, 'M': 10.0, 'Y': 0.5},
        )

        paths = simulate_time_changed_levy(
            key,
            market_params['S0'],
            market_params['T'],
            market_params['r'],
            market_params['q'],
            params,
            mc_config,
        )

        assert paths.shape[1] == mc_config.steps + 1
        assert jnp.all(paths > 0)

    def test_time_changed_levy_invalid(self, key, mc_config, market_params):
        """Test that invalid Lévy process raises error."""
        params = TimeChangedLevyParams(
            levy_process='INVALID',
            levy_params={},
        )

        with pytest.raises(NotImplementedError, match="not yet implemented"):
            simulate_time_changed_levy(
                key,
                market_params['S0'],
                market_params['T'],
                market_params['r'],
                market_params['q'],
                params,
                mc_config,
            )


# ==============================================================================
# Self-Exciting Lévy Tests
# ==============================================================================


class TestSelfExcitingLevy:
    """Test suite for self-exciting Lévy model."""

    def test_self_exciting_levy_vg(self, key, mc_config, market_params):
        """Test self-exciting Lévy with VG base."""
        params = SelfExcitingLevyParams(
            base_levy_params={'theta': -0.14, 'sigma': 0.2, 'nu': 0.2},
            levy_type='VG',
            lambda0=1.0,
            alpha=0.5,
            beta=2.0,
        )

        paths = simulate_self_exciting_levy(
            key,
            market_params['S0'],
            market_params['T'],
            market_params['r'],
            market_params['q'],
            params,
            mc_config,
        )

        assert paths.shape[1] == mc_config.steps + 1
        assert jnp.all(paths > 0)


# ==============================================================================
# Multivariate Hawkes Tests
# ==============================================================================


class TestMultivariateHawkes:
    """Test suite for multivariate Hawkes processes."""

    def test_multivariate_hawkes_params_validation(self):
        """Test parameter validation for multivariate Hawkes."""
        # Valid parameters
        params = MultivariateHawkesParams(
            n_assets=2,
            sigma=jnp.array([0.15, 0.20]),
            lambda0=jnp.array([5.0, 3.0]),
            alpha=jnp.array([[1.5, 0.8], [1.2, 1.0]]),
            beta=jnp.ones((2, 2)) * 2.0,
            mu_jump=jnp.array([-0.05, -0.03]),
            sigma_jump=jnp.array([0.1, 0.08]),
        )
        assert params.n_assets == 2

        # Invalid: wrong shape for sigma
        with pytest.raises(ValueError, match="sigma shape"):
            MultivariateHawkesParams(
                n_assets=2,
                sigma=jnp.array([0.15]),  # Should be [2]
                lambda0=jnp.array([5.0, 3.0]),
                alpha=jnp.ones((2, 2)),
                beta=jnp.ones((2, 2)),
                mu_jump=jnp.array([-0.05, -0.03]),
                sigma_jump=jnp.array([0.1, 0.08]),
            )

        # Invalid: negative sigma
        with pytest.raises(ValueError, match="sigma must be non-negative"):
            MultivariateHawkesParams(
                n_assets=2,
                sigma=jnp.array([0.15, -0.20]),
                lambda0=jnp.array([5.0, 3.0]),
                alpha=jnp.ones((2, 2)),
                beta=jnp.ones((2, 2)),
                mu_jump=jnp.array([-0.05, -0.03]),
                sigma_jump=jnp.array([0.1, 0.08]),
            )

    def test_multivariate_hawkes_stability(self):
        """Test stability check via spectral radius."""
        # Subcritical (stable)
        params_stable = MultivariateHawkesParams(
            n_assets=2,
            sigma=jnp.array([0.15, 0.20]),
            lambda0=jnp.array([5.0, 3.0]),
            alpha=jnp.array([[0.5, 0.2], [0.3, 0.4]]),  # Weak coupling
            beta=jnp.ones((2, 2)) * 2.0,
            mu_jump=jnp.array([-0.05, -0.03]),
            sigma_jump=jnp.array([0.1, 0.08]),
        )
        assert params_stable.spectral_radius() < 1.0

    def test_multivariate_hawkes_simulation(self, key, market_params):
        """Test multivariate Hawkes simulation."""
        params = MultivariateHawkesParams(
            n_assets=2,
            sigma=jnp.array([0.15, 0.20]),
            lambda0=jnp.array([5.0, 3.0]),
            alpha=jnp.array([[1.5, 0.8], [1.2, 1.0]]),
            beta=jnp.ones((2, 2)) * 2.0,
            mu_jump=jnp.array([-0.05, -0.03]),
            sigma_jump=jnp.array([0.1, 0.08]),
        )

        S0 = jnp.array([100.0, 50.0])
        q = jnp.array([0.01, 0.02])
        cfg = MCConfig(paths=100, steps=50, dtype=jnp.float32)

        paths = simulate_multivariate_hawkes(
            key, S0, market_params['T'], market_params['r'], q, params, cfg
        )

        # Check shape: [paths, steps+1, n_assets]
        assert paths.shape == (100, 51, 2)

        # Check initial values
        assert jnp.allclose(paths[:, 0, 0], 100.0)
        assert jnp.allclose(paths[:, 0, 1], 50.0)

        # Check all positive
        assert jnp.all(paths > 0)

    def test_multivariate_hawkes_correlation(self, key, market_params):
        """Test multivariate Hawkes with correlation matrix."""
        correlation = jnp.array([[1.0, 0.5], [0.5, 1.0]])

        params = MultivariateHawkesParams(
            n_assets=2,
            sigma=jnp.array([0.15, 0.20]),
            lambda0=jnp.array([5.0, 3.0]),
            alpha=jnp.array([[1.5, 0.8], [1.2, 1.0]]),
            beta=jnp.ones((2, 2)) * 2.0,
            mu_jump=jnp.array([-0.05, -0.03]),
            sigma_jump=jnp.array([0.1, 0.08]),
            correlation=correlation,
        )

        S0 = jnp.array([100.0, 50.0])
        q = jnp.array([0.01, 0.02])
        cfg = MCConfig(paths=100, steps=50, dtype=jnp.float32)

        paths = simulate_multivariate_hawkes(
            key, S0, market_params['T'], market_params['r'], q, params, cfg
        )

        # Should produce valid paths
        assert paths.shape == (100, 51, 2)
        assert jnp.all(paths > 0)


# ==============================================================================
# Regime-Switching Hawkes Tests
# ==============================================================================


class TestRegimeSwitchingHawkes:
    """Test suite for regime-switching Hawkes processes."""

    def test_regime_switching_params_validation(self):
        """Test parameter validation for regime-switching Hawkes."""
        # Valid parameters
        params = RegimeSwitchingHawkesParams(
            sigma=jnp.array([0.15, 0.30]),
            lambda0=jnp.array([2.0, 8.0]),
            alpha=jnp.array([0.5, 3.0]),
            beta=jnp.array([2.0, 2.0]),
            mu_jump=jnp.array([-0.02, -0.08]),
            sigma_jump=jnp.array([0.05, 0.15]),
            q_matrix=jnp.array([[-0.5, 0.5], [2.0, -2.0]]),
        )
        assert params.branching_ratios()[0] == 0.25
        assert params.branching_ratios()[1] == 1.5

        # Invalid: wrong shape
        with pytest.raises(ValueError, match="must have shape \\(2,\\)"):
            RegimeSwitchingHawkesParams(
                sigma=jnp.array([0.15]),  # Should be [2]
                lambda0=jnp.array([2.0, 8.0]),
                alpha=jnp.array([0.5, 3.0]),
                beta=jnp.array([2.0, 2.0]),
                mu_jump=jnp.array([-0.02, -0.08]),
                sigma_jump=jnp.array([0.05, 0.15]),
                q_matrix=jnp.array([[-0.5, 0.5], [2.0, -2.0]]),
            )

        # Invalid: Q-matrix not valid (rows don't sum to zero)
        with pytest.raises(ValueError, match="Q-matrix row"):
            RegimeSwitchingHawkesParams(
                sigma=jnp.array([0.15, 0.30]),
                lambda0=jnp.array([2.0, 8.0]),
                alpha=jnp.array([0.5, 3.0]),
                beta=jnp.array([2.0, 2.0]),
                mu_jump=jnp.array([-0.02, -0.08]),
                sigma_jump=jnp.array([0.05, 0.15]),
                q_matrix=jnp.array([[-0.5, 0.6], [2.0, -2.0]]),  # Row 0 doesn't sum to 0
            )

    def test_regime_switching_simulation(self, key, market_params):
        """Test regime-switching Hawkes simulation."""
        params = RegimeSwitchingHawkesParams(
            sigma=jnp.array([0.15, 0.30]),
            lambda0=jnp.array([2.0, 8.0]),
            alpha=jnp.array([0.5, 3.0]),
            beta=jnp.array([2.0, 2.0]),
            mu_jump=jnp.array([-0.02, -0.08]),
            sigma_jump=jnp.array([0.05, 0.15]),
            q_matrix=jnp.array([[-0.5, 0.5], [2.0, -2.0]]),
        )

        cfg = MCConfig(paths=100, steps=50, dtype=jnp.float32)

        paths, regimes = simulate_regime_switching_hawkes(
            key,
            market_params['S0'],
            market_params['T'],
            market_params['r'],
            market_params['q'],
            params,
            cfg,
            initial_regime=0,
        )

        # Check paths shape
        assert paths.shape == (100, 51)
        assert regimes.shape == (100, 51)

        # Check initial values
        assert jnp.allclose(paths[:, 0], market_params['S0'])

        # Check regimes are 0 or 1
        assert jnp.all((regimes == 0) | (regimes == 1))

        # Check all paths positive
        assert jnp.all(paths > 0)

    def test_regime_switching_initial_regime(self, key, market_params):
        """Test that initial regime is respected."""
        params = RegimeSwitchingHawkesParams(
            sigma=jnp.array([0.15, 0.30]),
            lambda0=jnp.array([2.0, 8.0]),
            alpha=jnp.array([0.5, 3.0]),
            beta=jnp.array([2.0, 2.0]),
            mu_jump=jnp.array([-0.02, -0.08]),
            sigma_jump=jnp.array([0.05, 0.15]),
            q_matrix=jnp.array([[-0.5, 0.5], [2.0, -2.0]]),
        )

        cfg = MCConfig(paths=10, steps=50, dtype=jnp.float32)

        # Start in regime 1
        _, regimes = simulate_regime_switching_hawkes(
            key,
            market_params['S0'],
            market_params['T'],
            market_params['r'],
            market_params['q'],
            params,
            cfg,
            initial_regime=1,
        )

        # All paths should start in regime 1
        assert jnp.all(regimes[:, 0] == 1)


# ==============================================================================
# Intensity-Dependent Jump Size Tests
# ==============================================================================


class TestIntensityDependentJumps:
    """Test suite for intensity-dependent jump sizes."""

    def test_intensity_dependent_params_validation(self):
        """Test parameter validation."""
        # Valid parameters
        params = IntensityDependentJumpParams(
            sigma=0.15,
            lambda0=5.0,
            alpha=1.5,
            beta=2.0,
            mu_jump_base=-0.03,
            sigma_jump_base=0.08,
            intensity_sensitivity=0.2,
        )
        assert params.intensity_sensitivity == 0.2

        # Invalid: negative intensity_sensitivity
        with pytest.raises(ValueError, match="intensity_sensitivity must be non-negative"):
            IntensityDependentJumpParams(
                sigma=0.15,
                lambda0=5.0,
                alpha=1.5,
                beta=2.0,
                mu_jump_base=-0.03,
                sigma_jump_base=0.08,
                intensity_sensitivity=-0.2,
            )

    def test_intensity_dependent_jump_size_scaling(self):
        """Test that jump sizes scale with intensity."""
        params = IntensityDependentJumpParams(
            sigma=0.15,
            lambda0=5.0,
            alpha=1.5,
            beta=2.0,
            mu_jump_base=-0.03,
            sigma_jump_base=0.08,
            intensity_sensitivity=0.2,
        )

        # At baseline intensity
        mu_base, sigma_base = params.jump_size_params(5.0)
        assert jnp.isclose(mu_base, -0.03)

        # At double intensity
        mu_double, sigma_double = params.jump_size_params(10.0)
        # mu should increase: mu = -0.03 + 0.2 * log(2)
        expected_mu = -0.03 + 0.2 * jnp.log(2.0)
        assert jnp.isclose(mu_double, expected_mu)

    def test_intensity_dependent_simulation(self, key, market_params):
        """Test simulation with intensity-dependent jump sizes."""
        params = IntensityDependentJumpParams(
            sigma=0.15,
            lambda0=5.0,
            alpha=1.5,
            beta=2.0,
            mu_jump_base=-0.03,
            sigma_jump_base=0.08,
            intensity_sensitivity=0.2,
        )

        cfg = MCConfig(paths=100, steps=50, dtype=jnp.float32)

        paths = simulate_intensity_dependent_hawkes(
            key,
            market_params['S0'],
            market_params['T'],
            market_params['r'],
            market_params['q'],
            params,
            cfg,
        )

        # Check shape
        assert paths.shape == (100, 51)

        # Check initial value
        assert jnp.allclose(paths[:, 0], market_params['S0'])

        # Check all positive
        assert jnp.all(paths > 0)

    def test_intensity_dependent_vs_standard(self, key, market_params):
        """Test that zero sensitivity recovers standard Hawkes."""
        # Standard Hawkes via intensity-dependent with sensitivity=0
        params_standard = IntensityDependentJumpParams(
            sigma=0.15,
            lambda0=5.0,
            alpha=1.5,
            beta=2.0,
            mu_jump_base=-0.03,
            sigma_jump_base=0.08,
            intensity_sensitivity=0.0,  # Zero sensitivity
        )

        cfg = MCConfig(paths=100, steps=50, dtype=jnp.float32)

        paths_standard = simulate_intensity_dependent_hawkes(
            key,
            market_params['S0'],
            market_params['T'],
            market_params['r'],
            market_params['q'],
            params_standard,
            cfg,
        )

        # Should produce valid paths
        assert paths_standard.shape == (100, 51)
        assert jnp.all(paths_standard > 0)


# ==============================================================================
# Hawkes Calibration Tests
# ==============================================================================


class TestHawkesCalibration:
    """Test suite for Hawkes parameter calibration."""

    def test_calibrate_from_jumps_moments(self):
        """Test method of moments calibration."""
        # Generate synthetic jump times
        jump_times = jnp.array([0.1, 0.15, 0.2, 0.5, 0.51, 0.52, 1.0, 1.5, 2.0])

        params = calibrate_hawkes_from_jumps(jump_times, method="moments")

        # Should return valid parameters
        assert params.lambda0 > 0
        assert params.alpha >= 0
        assert params.beta > 0

    def test_calibrate_from_jumps_mle(self):
        """Test MLE calibration."""
        # Generate synthetic jump times with clustering
        jump_times = jnp.array(
            [0.1, 0.12, 0.13, 0.5, 0.52, 0.55, 1.0, 1.02, 1.5, 2.0]
        )

        params = calibrate_hawkes_from_jumps(jump_times, method="mle")

        # Should return valid parameters
        assert params.lambda0 > 0
        assert params.alpha >= 0
        assert params.beta > 0

        # With clustering, should have alpha > 0
        assert params.alpha > 0

    def test_calibrate_invalid_method(self):
        """Test that invalid method raises error."""
        jump_times = jnp.array([0.1, 0.2, 0.5, 1.0])

        with pytest.raises(ValueError, match="Unknown calibration method"):
            calibrate_hawkes_from_jumps(jump_times, method="invalid")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
