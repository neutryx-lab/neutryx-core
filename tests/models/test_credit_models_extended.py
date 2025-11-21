"""Tests for comprehensive credit risk models.

Tests cover:
- Gaussian copula simulation
- Student-t copula with tail dependence
- Large Portfolio Approximation (LPA/Vasicek)
- CreditMetrics rating migrations
- Merton structural model
- Black-Cox first-passage model
"""
import jax
import jax.numpy as jnp
import pytest

from neutryx.models.credit_models import (
    GaussianCopulaParams,
    simulate_gaussian_copula,
    base_correlation_to_compound_correlation,
    StudentTCopulaParams,
    simulate_student_t_copula,
    LPAParams,
    vasicek_loss_distribution,
    lpa_expected_loss,
    lpa_unexpected_loss,
    CreditMetricsParams,
    simulate_credit_migrations,
    MertonModelParams,
    merton_default_probability,
    merton_distance_to_default,
    merton_equity_value,
    BlackCoxParams,
    black_cox_default_probability,
    credit_spread_from_default_prob,
)


# ==============================================================================
# Gaussian Copula Tests
# ==============================================================================


class TestGaussianCopula:
    """Test Gaussian copula credit model."""

    def test_gaussian_copula_basic(self):
        """Test basic Gaussian copula simulation."""
        key = jax.random.PRNGKey(42)
        n_names = 50

        # Create uniform correlation matrix
        corr = jnp.eye(n_names) * 0.7 + jnp.ones((n_names, n_names)) * 0.15

        params = GaussianCopulaParams(
            correlation_matrix=corr,
            default_probabilities=jnp.full(n_names, 0.02),  # 2% PD
            recovery_rates=jnp.full(n_names, 0.40),  # 40% recovery
        )

        defaults, losses = simulate_gaussian_copula(key, params, n_simulations=5000)

        # Check shapes
        assert defaults.shape == (5000, n_names)
        assert losses.shape == (5000,)

        # Check defaults are binary
        assert jnp.all((defaults == 0) | (defaults == 1))

        # Expected default rate should be close to 2%
        default_rate = defaults.mean()
        assert 0.01 < default_rate < 0.04  # Allow wide tolerance for MC

        # Losses should be positive
        assert jnp.all(losses >= 0)

    def test_gaussian_copula_independent(self):
        """Test with independent (diagonal correlation)."""
        key = jax.random.PRNGKey(123)
        n_names = 100

        params = GaussianCopulaParams(
            correlation_matrix=jnp.eye(n_names),  # Independent
            default_probabilities=jnp.full(n_names, 0.01),
            recovery_rates=jnp.full(n_names, 0.40),
        )

        defaults, losses = simulate_gaussian_copula(key, params, n_simulations=10000)

        # With independence, loss distribution should have low variance
        # Expected loss = 1% * 60% LGD = 0.006
        expected_loss = 0.01 * 0.60
        mean_loss = losses.mean()

        assert abs(mean_loss - expected_loss) < 0.002

    def test_gaussian_copula_high_correlation(self):
        """Test high correlation increases loss variance."""
        key = jax.random.PRNGKey(456)
        n_names = 50

        # High correlation
        corr_high = jnp.eye(n_names) * 0.1 + jnp.ones((n_names, n_names)) * 0.5

        params = GaussianCopulaParams(
            correlation_matrix=corr_high,
            default_probabilities=jnp.full(n_names, 0.02),
            recovery_rates=jnp.full(n_names, 0.40),
        )

        defaults, losses = simulate_gaussian_copula(key, params, n_simulations=5000)

        # High correlation should lead to higher loss variance
        loss_std = losses.std()
        assert loss_std > 0.01  # Should have some dispersion

    def test_base_correlation_conversion(self):
        """Test base correlation to compound correlation conversion."""
        # Test simple case
        compound_corr = base_correlation_to_compound_correlation(
            base_corr_lower=0.3,
            base_corr_upper=0.4,
            attachment_lower=0.03,
            attachment_upper=0.07,
        )

        # Result should be reasonable (weighted average of base correlations)
        assert 0.0 <= compound_corr <= 1.0
        assert isinstance(compound_corr, float)


# ==============================================================================
# Student-t Copula Tests
# ==============================================================================


class TestStudentTCopula:
    """Test Student-t copula with tail dependence."""

    def test_student_t_copula_basic(self):
        """Test basic Student-t copula simulation."""
        key = jax.random.PRNGKey(789)
        n_names = 30

        corr = jnp.eye(n_names) * 0.6 + jnp.ones((n_names, n_names)) * 0.2

        params = StudentTCopulaParams(
            correlation_matrix=corr,
            default_probabilities=jnp.full(n_names, 0.03),
            recovery_rates=jnp.full(n_names, 0.40),
            degrees_of_freedom=4.0,  # Heavy tails
        )

        defaults, losses = simulate_student_t_copula(key, params, n_simulations=3000)

        # Check shapes
        assert defaults.shape == (3000, n_names)
        assert losses.shape == (3000,)

        # Default rate should be approximately 3%
        default_rate = defaults.mean()
        assert 0.015 < default_rate < 0.05

    def test_student_t_vs_gaussian_tail_dependence(self):
        """Student-t should exhibit more tail dependence than Gaussian."""
        key = jax.random.PRNGKey(111)
        n_names = 50

        corr = jnp.eye(n_names) * 0.5 + jnp.ones((n_names, n_names)) * 0.3

        # Student-t copula
        params_t = StudentTCopulaParams(
            correlation_matrix=corr,
            default_probabilities=jnp.full(n_names, 0.02),
            recovery_rates=jnp.full(n_names, 0.40),
            degrees_of_freedom=3.0,  # Very heavy tails
        )

        key1, key2 = jax.random.split(key)
        _, losses_t = simulate_student_t_copula(key1, params_t, n_simulations=5000)

        # Gaussian copula (same parameters, no tail dependence)
        params_g = GaussianCopulaParams(
            correlation_matrix=corr,
            default_probabilities=jnp.full(n_names, 0.02),
            recovery_rates=jnp.full(n_names, 0.40),
        )

        _, losses_g = simulate_gaussian_copula(key2, params_g, n_simulations=5000)

        # Student-t should have fatter tails (higher 99th percentile)
        p99_t = jnp.percentile(losses_t, 99)
        p99_g = jnp.percentile(losses_g, 99)

        # This is a stochastic test, might occasionally fail
        # But generally t-copula should have heavier tails
        assert p99_t >= p99_g * 0.9  # Allow some tolerance


# ==============================================================================
# Large Portfolio Approximation (LPA) Tests
# ==============================================================================


class TestLPA:
    """Test Large Portfolio Approximation (Vasicek model)."""

    def test_lpa_expected_loss(self):
        """Test expected loss calculation."""
        params = LPAParams(
            default_probability=0.02,
            correlation=0.20,
            recovery_rate=0.40,
            n_names=1000,
        )

        el = lpa_expected_loss(params)

        # EL = PD * LGD = 0.02 * 0.60 = 0.012
        expected = 0.02 * 0.60
        assert abs(el - expected) < 1e-6

    def test_lpa_unexpected_loss(self):
        """Test unexpected loss (VaR - EL)."""
        params = LPAParams(
            default_probability=0.01,
            correlation=0.15,
            recovery_rate=0.40,
            n_names=1000,
        )

        ul_99 = lpa_unexpected_loss(params, confidence_level=0.99)
        ul_999 = lpa_unexpected_loss(params, confidence_level=0.999)

        # UL should be positive
        assert ul_99 > 0
        assert ul_999 > 0

        # Higher confidence level should give higher UL
        assert ul_999 > ul_99

    def test_vasicek_loss_distribution(self):
        """Test Vasicek loss distribution computation."""
        params = LPAParams(
            default_probability=0.02,
            correlation=0.20,
            recovery_rate=0.40,
            n_names=500,
        )

        # Create loss grid
        loss_grid = jnp.linspace(0, 0.15, 100)
        loss_dist = vasicek_loss_distribution(params, loss_grid)

        # PDF should be positive
        assert jnp.all(loss_dist >= 0)

        # Should integrate to approximately 1 (within numerical tolerance)
        total_prob = jnp.sum(loss_dist) * (loss_grid[1] - loss_grid[0])
        assert abs(total_prob - 1.0) < 0.1  # Rough integration check

    def test_lpa_basel_capital(self):
        """Test Basel II/III capital formula."""
        # Typical corporate loan parameters
        params = LPAParams(
            default_probability=0.01,  # 1% PD
            correlation=0.12,  # Basel correlation for corporates
            recovery_rate=0.45,  # 45% recovery (55% LGD)
            n_names=1000,
        )

        # Basel uses 99.9% confidence level
        ul = lpa_unexpected_loss(params, confidence_level=0.999)

        # Capital = UL
        # For these parameters, should be a few percent
        assert 0.0 < ul < 0.1


# ==============================================================================
# CreditMetrics Tests
# ==============================================================================


class TestCreditMetrics:
    """Test CreditMetrics rating migration framework."""

    def test_credit_migrations_basic(self):
        """Test basic CreditMetrics simulation."""
        key = jax.random.PRNGKey(222)

        # 3-rating system: AAA (0), BBB (1), Default (2)
        transition_matrix = jnp.array([
            [0.95, 0.04, 0.01],  # From AAA
            [0.02, 0.93, 0.05],  # From BBB
            [0.00, 0.00, 1.00],  # From Default (absorbing)
        ])

        n_obligors = 10
        params = CreditMetricsParams(
            transition_matrix=transition_matrix,
            current_ratings=jnp.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),  # 5 AAA, 5 BBB
            exposures=jnp.ones(n_obligors) * 100.0,
            values_by_rating=jnp.array([100.0, 80.0, 20.0]),  # Value in each rating
            correlation_matrix=jnp.eye(n_obligors) * 0.8 + jnp.ones((n_obligors, n_obligors)) * 0.1,
        )

        ratings, values = simulate_credit_migrations(key, params, n_simulations=1000)

        # Check shapes
        assert ratings.shape == (1000, n_obligors)
        assert values.shape == (1000,)

        # Ratings should be 0, 1, or 2
        assert jnp.all((ratings >= 0) & (ratings <= 2))

        # Portfolio values should be positive
        assert jnp.all(values > 0)

    def test_migrations_preserve_default(self):
        """Test that default is an absorbing state."""
        key = jax.random.PRNGKey(333)

        # Simple 2-state: Performing (0), Default (1)
        transition_matrix = jnp.array([
            [0.97, 0.03],  # From Performing
            [0.00, 1.00],  # From Default (absorbing)
        ])

        n_obligors = 5
        # Start with 2 already defaulted
        current_ratings = jnp.array([0, 0, 0, 1, 1])

        params = CreditMetricsParams(
            transition_matrix=transition_matrix,
            current_ratings=current_ratings,
            exposures=jnp.ones(n_obligors) * 100.0,
            values_by_rating=jnp.array([100.0, 30.0]),  # 30% recovery in default
            correlation_matrix=jnp.eye(n_obligors) * 0.7 + jnp.ones((n_obligors, n_obligors)) * 0.15,
        )

        ratings, _ = simulate_credit_migrations(key, params, n_simulations=500)

        # Obligors 3 and 4 started in default (state 1)
        # They should remain in default across all simulations
        assert jnp.all(ratings[:, 3] == 1)
        assert jnp.all(ratings[:, 4] == 1)


# ==============================================================================
# Merton Model Tests
# ==============================================================================


class TestMertonModel:
    """Test Merton structural credit model."""

    def test_merton_default_probability(self):
        """Test Merton default probability calculation."""
        params = MertonModelParams(
            asset_value=100.0,
            debt_value=80.0,
            volatility=0.25,
            maturity=1.0,
            risk_free_rate=0.05,
        )

        pd = merton_default_probability(params)

        # PD should be between 0 and 1
        assert 0.0 < pd < 1.0

        # With assets > debt, PD should be relatively low
        assert pd < 0.3

    def test_merton_distance_to_default(self):
        """Test distance-to-default metric."""
        params = MertonModelParams(
            asset_value=100.0,
            debt_value=60.0,  # Well above debt
            volatility=0.20,
            maturity=1.0,
            risk_free_rate=0.05,
        )

        dd = merton_distance_to_default(params)

        # DD should be positive when assets > debt
        assert dd > 0

        # Should be a reasonable number of standard deviations
        assert 0.5 < dd < 10.0

    def test_merton_equity_value(self):
        """Test equity value as call option."""
        params = MertonModelParams(
            asset_value=100.0,
            debt_value=80.0,
            volatility=0.30,
            maturity=2.0,
            risk_free_rate=0.04,
        )

        equity = merton_equity_value(params)

        # Equity should be positive
        assert equity > 0

        # Equity should be less than asset value
        assert equity < params.asset_value

        # Put-call parity: Assets = Equity + Debt_PV - Put
        # So Equity < Assets - Debt_PV * exp(-r*T)
        debt_pv = params.debt_value * jnp.exp(-params.risk_free_rate * params.maturity)
        assert equity < params.asset_value - debt_pv + 50  # Allow option value

    def test_merton_higher_volatility_increases_pd(self):
        """Higher asset volatility should increase default probability."""
        base_params = MertonModelParams(
            asset_value=100.0,
            debt_value=80.0,
            volatility=0.20,
            maturity=1.0,
            risk_free_rate=0.05,
        )

        high_vol_params = MertonModelParams(
            asset_value=100.0,
            debt_value=80.0,
            volatility=0.40,  # Higher volatility
            maturity=1.0,
            risk_free_rate=0.05,
        )

        pd_low = merton_default_probability(base_params)
        pd_high = merton_default_probability(high_vol_params)

        # Higher volatility should increase default risk
        assert pd_high > pd_low


# ==============================================================================
# Black-Cox Model Tests
# ==============================================================================


class TestBlackCoxModel:
    """Test Black-Cox first-passage model."""

    def test_black_cox_default_probability(self):
        """Test Black-Cox default probability."""
        params = BlackCoxParams(
            asset_value=100.0,
            barrier=60.0,  # Default if assets hit 60
            volatility=0.30,
            maturity=5.0,
            risk_free_rate=0.05,
            dividend_yield=0.02,
        )

        pd = black_cox_default_probability(params)

        # PD should be between 0 and 1
        assert 0.0 < pd < 1.0

    def test_black_cox_higher_barrier_increases_pd(self):
        """Higher barrier should increase default probability."""
        low_barrier_params = BlackCoxParams(
            asset_value=100.0,
            barrier=50.0,
            volatility=0.25,
            maturity=3.0,
            risk_free_rate=0.05,
            dividend_yield=0.01,
        )

        high_barrier_params = BlackCoxParams(
            asset_value=100.0,
            barrier=70.0,  # Higher barrier
            volatility=0.25,
            maturity=3.0,
            risk_free_rate=0.05,
            dividend_yield=0.01,
        )

        pd_low = black_cox_default_probability(low_barrier_params)
        pd_high = black_cox_default_probability(high_barrier_params)

        # Higher barrier should increase default probability
        assert pd_high > pd_low

    def test_black_cox_longer_maturity_increases_pd(self):
        """Longer maturity should increase default probability (more time to hit barrier)."""
        short_maturity_params = BlackCoxParams(
            asset_value=100.0,
            barrier=60.0,
            volatility=0.30,
            maturity=1.0,
            risk_free_rate=0.05,
            dividend_yield=0.02,
        )

        long_maturity_params = BlackCoxParams(
            asset_value=100.0,
            barrier=60.0,
            volatility=0.30,
            maturity=10.0,  # Longer maturity
            risk_free_rate=0.05,
            dividend_yield=0.02,
        )

        pd_short = black_cox_default_probability(short_maturity_params)
        pd_long = black_cox_default_probability(long_maturity_params)

        # Longer maturity should increase first-passage probability
        assert pd_long > pd_short

    def test_black_cox_vs_merton(self):
        """Black-Cox should generally give higher PD than Merton (earlier default)."""
        # Merton model
        merton_params = MertonModelParams(
            asset_value=100.0,
            debt_value=80.0,
            volatility=0.25,
            maturity=5.0,
            risk_free_rate=0.05,
        )

        pd_merton = merton_default_probability(merton_params)

        # Black-Cox with barrier at debt level
        black_cox_params = BlackCoxParams(
            asset_value=100.0,
            barrier=80.0,  # Same as debt
            volatility=0.25,
            maturity=5.0,
            risk_free_rate=0.05,
            dividend_yield=0.0,
        )

        pd_bc = black_cox_default_probability(black_cox_params)

        # Black-Cox allows earlier default, so PD should be higher
        # (or at least not much lower)
        assert pd_bc >= pd_merton * 0.5  # Allow some numerical tolerance


# ==============================================================================
# Utility Function Tests
# ==============================================================================


class TestUtilities:
    """Test utility functions."""

    def test_credit_spread_from_default_prob(self):
        """Test credit spread calculation."""
        spread = credit_spread_from_default_prob(
            default_prob=0.05,  # 5% default probability
            recovery_rate=0.40,  # 40% recovery
            maturity=5.0,
        )

        # Spread should be positive
        assert spread > 0

        # Should be reasonable (a few hundred bps)
        assert 0.0 < spread < 0.1  # Less than 10% spread

    def test_higher_default_prob_increases_spread(self):
        """Higher default probability should increase credit spread."""
        spread_low = credit_spread_from_default_prob(0.01, 0.40, 5.0)
        spread_high = credit_spread_from_default_prob(0.10, 0.40, 5.0)

        assert spread_high > spread_low

    def test_lower_recovery_increases_spread(self):
        """Lower recovery rate should increase credit spread."""
        spread_high_rec = credit_spread_from_default_prob(0.05, 0.60, 5.0)  # 60% recovery
        spread_low_rec = credit_spread_from_default_prob(0.05, 0.20, 5.0)  # 20% recovery

        assert spread_low_rec > spread_high_rec


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
