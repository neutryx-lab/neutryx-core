"""Comprehensive tests for XVA (CVA, FVA, MVA) calculations and capital metrics."""
import jax
import jax.numpy as jnp
import pytest

from neutryx.core.engine import MCConfig, simulate_gbm
from neutryx.valuations.xva import cva as cva_module
from neutryx.valuations.xva import fva as fva_module
from neutryx.valuations.xva import mva as mva_module
from neutryx.valuations.exposure import epe
from neutryx.valuations.xva.capital import CapitalCalculator


def test_epe_nonnegative():
    """Original test: EPE should be non-negative."""
    key = jax.random.PRNGKey(0)
    cfg = MCConfig(steps=8, paths=1000)
    paths = simulate_gbm(key, 100.0, 0.01, 0.2, 1.0, cfg)
    val = float(epe(paths, K=100.0, is_call=True))
    assert val >= 0.0


class TestCVA:
    """Test suite for Credit Valuation Adjustment."""

    def test_cva_positive_exposure(self):
        """Test CVA calculation with positive exposure."""
        epe_t = jnp.array([10.0, 10.0, 10.0, 10.0])
        df_t = jnp.array([0.99, 0.98, 0.97, 0.96])
        pd_t = jnp.array([0.01, 0.02, 0.03, 0.04])
        lgd = 0.6

        cva_value = cva_module.cva(epe_t, df_t, pd_t, lgd)

        assert cva_value > 0.0, "CVA should be positive for positive exposure"

    def test_cva_zero_exposure(self):
        """Test CVA calculation with zero exposure."""
        epe_t = jnp.zeros(5)
        df_t = jnp.array([0.99, 0.98, 0.97, 0.96, 0.95])
        pd_t = jnp.array([0.01, 0.02, 0.03, 0.04, 0.05])

        cva_value = cva_module.cva(epe_t, df_t, pd_t, lgd=0.6)

        assert abs(cva_value) < 1e-6, "CVA should be near zero for zero exposure"

    def test_cva_increasing_pd(self):
        """Test that CVA increases with default probability."""
        epe_t = jnp.array([10.0, 10.0, 10.0])
        df_t = jnp.array([0.99, 0.98, 0.97])

        pd_low = jnp.array([0.01, 0.02, 0.03])
        cva_low = cva_module.cva(epe_t, df_t, pd_low, lgd=0.6)

        pd_high = jnp.array([0.05, 0.10, 0.15])
        cva_high = cva_module.cva(epe_t, df_t, pd_high, lgd=0.6)

        assert cva_high > cva_low, "CVA should increase with default probability"


class TestFVA:
    """Test suite for Funding Valuation Adjustment."""

    def test_fva_positive_exposure(self):
        """Test FVA calculation with positive exposure."""
        epe_t = jnp.array([10.0, 10.0, 10.0, 10.0])
        df_t = jnp.array([0.99, 0.98, 0.97, 0.96])
        funding_spread = 0.02

        fva_value = fva_module.fva(epe_t, funding_spread, df_t)

        assert fva_value > 0.0, "FVA should be positive for positive exposure"

    def test_fva_zero_spread(self):
        """Test FVA calculation with zero funding spread."""
        epe_t = jnp.array([10.0, 10.0, 10.0])
        df_t = jnp.array([0.99, 0.98, 0.97])

        fva_value = fva_module.fva(epe_t, 0.0, df_t)

        assert abs(fva_value) < 1e-6, "FVA should be zero for zero funding spread"


class TestMVA:
    """Test suite for Margin Valuation Adjustment."""

    def test_mva_positive_margin(self):
        """Test MVA calculation with positive margin profile."""
        im_profile = jnp.array([100.0, 100.0, 100.0, 100.0])
        df_t = jnp.array([0.99, 0.98, 0.97, 0.96])
        spread = 0.015

        mva_value = mva_module.mva(im_profile, df_t, spread)

        assert mva_value > 0.0, "MVA should be positive for positive margin"

    def test_mva_zero_spread(self):
        """Test MVA calculation with zero spread."""
        im_profile = jnp.array([100.0, 100.0, 100.0])
        df_t = jnp.array([0.99, 0.98, 0.97])

        mva_value = mva_module.mva(im_profile, df_t, 0.0)

        assert abs(mva_value) < 1e-6, "MVA should be zero for zero spread"


class TestCapitalCalculator:
    """Test suite for Capital Calculator."""

    @pytest.fixture
    def setup_calculator(self):
        """Create a basic capital calculator setup."""
        discount_curve = jnp.array([0.99, 0.98, 0.97, 0.96])
        default_probabilities = jnp.array([0.01, 0.02, 0.03, 0.04])
        return CapitalCalculator(
            discount_curve=discount_curve,
            default_probabilities=default_probabilities,
            lgd=0.6,
            funding_spread=0.02,
            hurdle_rate=0.1,
        )

    def test_calculator_initialization(self, setup_calculator):
        """Test that calculator initializes correctly."""
        calc = setup_calculator
        assert calc.lgd == 0.6
        assert calc.funding_spread == 0.02

    def test_cva_calculation(self, setup_calculator):
        """Test CVA calculation through calculator."""
        calc = setup_calculator
        epe = jnp.array([10.0, 10.0, 10.0, 10.0])

        cva_value = calc.cva(epe)

        assert cva_value > 0.0, "CVA should be positive"
        assert isinstance(cva_value, float), "CVA should return float"

    def test_pfe_calculation(self, setup_calculator):
        """Test Potential Future Exposure calculation."""
        calc = setup_calculator

        pathwise = jnp.array(
            [
                [5.0, 10.0, 15.0, 20.0],
                [10.0, 20.0, 30.0, 40.0],
                [2.0, 4.0, 6.0, 8.0],
                [15.0, 30.0, 45.0, 60.0],
            ]
        )

        pfe = calc.pfe(pathwise, quantile=0.75)

        assert pfe.shape == (4,), f"PFE should have shape (4,), got {pfe.shape}"
        assert jnp.all(pfe >= 0.0), "PFE should be non-negative"

    def test_compute_all(self, setup_calculator):
        """Test comprehensive metric computation."""
        calc = setup_calculator
        epe_profile = jnp.array([10.0, 10.0, 10.0, 10.0])
        pathwise = jnp.array(
            [
                [5.0, 10.0, 15.0, 20.0],
                [10.0, 20.0, 30.0, 40.0],
                [2.0, 4.0, 6.0, 8.0],
                [15.0, 30.0, 45.0, 60.0],
            ]
        )

        metrics = calc.compute_all(epe=epe_profile, pathwise=pathwise)

        assert "cva" in metrics
        assert "funding_cost" in metrics
        assert "pfe" in metrics
        assert "kva" in metrics
        assert metrics["cva"] > 0.0
        assert metrics["funding_cost"] > 0.0
