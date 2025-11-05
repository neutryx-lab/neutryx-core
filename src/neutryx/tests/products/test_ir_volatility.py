"""Tests for interest rate volatility trading products."""
import jax
import jax.numpy as jnp
import pytest

from neutryx.products.ir_volatility import (
    SwaptionStraddle,
    SwaptionStrangle,
    IRVarianceSwap,
    CapletVarianceSwap,
    ForwardIRVarianceSwap,
    RateCorrelationSwap,
    VolatilityDispersionSwap,
    compute_forward_variance_strike,
)


def test_swaption_straddle_basic():
    """Test swaption straddle payoff."""
    straddle = SwaptionStraddle(
        T=1.0,
        K=0.05,
        annuity=4.5,
        notional=1_000_000,
    )

    # At strike: zero payoff
    payoff_atm = straddle.payoff_terminal(0.05)
    assert abs(payoff_atm) < 1e-6

    # Above strike: positive payoff
    payoff_up = straddle.payoff_terminal(0.06)
    expected_up = 1_000_000 * 4.5 * 0.01  # 1% move
    assert abs(payoff_up - expected_up) < 1.0

    # Below strike: positive payoff (symmetric)
    payoff_down = straddle.payoff_terminal(0.04)
    expected_down = 1_000_000 * 4.5 * 0.01
    assert abs(payoff_down - expected_down) < 1.0

    # Payoffs should be approximately equal for equal moves
    assert abs(payoff_up - payoff_down) < 1.0  # Tolerance for floating point


def test_swaption_strangle_basic():
    """Test swaption strangle payoff."""
    strangle = SwaptionStrangle(
        T=1.0,
        K_low=0.045,
        K_high=0.055,
        annuity=4.5,
        notional=1_000_000,
    )

    # In the middle: zero payoff
    payoff_middle = strangle.payoff_terminal(0.05)
    assert abs(payoff_middle) < 1e-6

    # Below lower strike: positive payoff
    payoff_down = strangle.payoff_terminal(0.04)
    expected_down = 1_000_000 * 4.5 * (0.045 - 0.04)
    assert abs(payoff_down - expected_down) < 1.0

    # Above upper strike: positive payoff
    payoff_up = strangle.payoff_terminal(0.06)
    expected_up = 1_000_000 * 4.5 * (0.06 - 0.055)
    assert abs(payoff_up - expected_up) < 1.0


def test_swaption_strangle_vs_straddle():
    """Test that strangle is cheaper than straddle (zero payoff range)."""
    straddle = SwaptionStraddle(T=1.0, K=0.05, annuity=4.5, notional=1_000_000)
    strangle = SwaptionStrangle(
        T=1.0, K_low=0.048, K_high=0.052, annuity=4.5, notional=1_000_000
    )

    # Small move: straddle pays, strangle doesn't
    payoff_straddle = straddle.payoff_terminal(0.051)
    payoff_strangle = strangle.payoff_terminal(0.051)

    assert payoff_straddle > 0
    assert abs(payoff_strangle) < 1e-6  # In the zero range


def test_ir_variance_swap_basic():
    """Test IR variance swap payoff."""
    var_swap = IRVarianceSwap(
        T=1.0,
        strike_variance=0.04,  # 20% vol
        notional_per_variance_point=10_000,
    )

    # Generate rate path
    path = jnp.array([0.03, 0.032, 0.031, 0.033, 0.035])
    payoff = var_swap.payoff_path(path)

    # Should be a scalar
    assert jnp.isscalar(payoff) or payoff.shape == ()


def test_ir_variance_swap_zero_vol():
    """Test variance swap with zero volatility."""
    var_swap = IRVarianceSwap(
        T=1.0,
        strike_variance=0.04,
        notional_per_variance_point=10_000,
    )

    # Constant path (zero vol)
    path = jnp.ones(100) * 0.03
    payoff = var_swap.payoff_path(path)

    # Realized variance should be near zero, so payoff negative
    expected = -0.04 * 10_000
    assert payoff <= expected * 0.9  # Payoff should be close to -400


def test_caplet_variance_swap_basic():
    """Test caplet variance swap payoff."""
    caplet_var_swap = CapletVarianceSwap(
        T=1.0,
        strike_vol_variance=0.01,
        notional_per_vol_variance_point=50_000,
        caplet_tenor=0.25,
    )

    # Path of implied vols
    path = jnp.linspace(0.20, 0.25, 50)
    payoff = caplet_var_swap.payoff_path(path)

    # Should be a scalar
    assert jnp.isscalar(payoff) or payoff.shape == ()


def test_forward_variance_swap_validation():
    """Test forward variance swap validation."""
    # T must be > T1
    with pytest.raises(ValueError, match="T.*must be > T1"):
        ForwardIRVarianceSwap(T=1.0, T1=2.0, strike_variance=0.04)


def test_forward_variance_swap_payoff():
    """Test forward variance swap payoff."""
    fwd_var_swap = ForwardIRVarianceSwap(
        T=2.0,
        T1=1.0,
        strike_variance=0.0441,  # 21% forward vol
        notional_per_variance_point=10_000,
    )

    # Realized variance higher than strike
    realized_var = 0.05  # 22.36% vol
    payoff = fwd_var_swap.payoff_terminal(realized_var)

    expected = (0.05 - 0.0441) * 10_000
    assert abs(payoff - expected) < 1.0


def test_rate_correlation_swap_basic():
    """Test rate correlation swap payoff."""
    corr_swap = RateCorrelationSwap(
        T=1.0,
        strike_correlation=0.7,
        notional_per_correlation_point=100_000,
    )

    # Two perfectly correlated paths
    path1 = jnp.linspace(0.03, 0.04, 50)
    path2 = jnp.linspace(0.02, 0.03, 50)  # Parallel shift
    path = jnp.stack([path1, path2])

    payoff = corr_swap.payoff_path(path)

    # Should profit from high correlation
    # Realized correlation ~1.0, strike 0.7
    assert payoff > 0


def test_rate_correlation_swap_uncorrelated():
    """Test correlation swap with uncorrelated rates."""
    corr_swap = RateCorrelationSwap(
        T=1.0,
        strike_correlation=0.0,
        notional_per_correlation_point=100_000,
    )

    # Uncorrelated paths
    import jax.random as jrand
    key = jrand.PRNGKey(42)
    path1 = 0.03 + 0.001 * jnp.cumsum(jrand.normal(key, (100,)))
    path2 = 0.03 + 0.001 * jnp.cumsum(jrand.normal(jrand.split(key)[0], (100,)))
    path = jnp.stack([path1, path2])

    payoff = corr_swap.payoff_path(path)

    # Payoff depends on realized correlation
    # Just check it's finite
    assert jnp.isfinite(payoff)


def test_rate_correlation_swap_validation():
    """Test correlation swap path validation."""
    corr_swap = RateCorrelationSwap(
        T=1.0,
        strike_correlation=0.5,
        notional_per_correlation_point=100_000,
    )

    # Wrong path shape
    path = jnp.array([0.03, 0.032, 0.031])  # 1D instead of 2D

    with pytest.raises(ValueError, match="Path must have shape"):
        corr_swap.payoff_path(path)


def test_volatility_dispersion_swap_basic():
    """Test volatility dispersion swap payoff."""
    disp_swap = VolatilityDispersionSwap(
        T=1.0,
        strike_dispersion=0.05,
        notional_per_dispersion_point=10_000,
        n_rates=4,
    )

    # 4 rate paths with varying volatilities
    import jax.random as jrand
    key = jrand.PRNGKey(123)

    paths = []
    for i in range(4):
        vol_scale = 0.01 * (i + 1)  # Increasing vols
        returns = vol_scale * jrand.normal(jrand.split(key, 4)[i], (50,))
        path = 0.03 * jnp.exp(jnp.cumsum(returns))
        paths.append(path)

    path = jnp.stack(paths)
    payoff = disp_swap.payoff_path(path)

    # Should be finite
    assert jnp.isfinite(payoff)


def test_volatility_dispersion_swap_validation():
    """Test dispersion swap path validation."""
    disp_swap = VolatilityDispersionSwap(
        T=1.0,
        strike_dispersion=0.05,
        notional_per_dispersion_point=10_000,
    )

    # 1D path instead of 2D
    path = jnp.array([0.03, 0.032, 0.031])

    with pytest.raises(ValueError, match="Path must be 2D"):
        disp_swap.payoff_path(path)


def test_compute_forward_variance_strike():
    """Test forward variance strike calculation."""
    # 1Y vol = 20%, 2Y vol = 22%
    spot_var_1y = 0.20**2
    spot_var_2y = 0.22**2

    fwd_var = compute_forward_variance_strike(
        spot_variance_t1=spot_var_1y,
        spot_variance_t2=spot_var_2y,
        t1=1.0,
        t2=2.0,
    )

    # Forward vol should be above the lower bound
    fwd_vol = jnp.sqrt(fwd_var)
    assert fwd_vol > 0.20  # Must be higher than 1Y spot vol

    # Verify no-arbitrage relationship
    # Total variance at T2 = total variance at T1 + forward variance * (T2 - T1)
    total_var_2y = spot_var_2y * 2.0
    total_var_1y = spot_var_1y * 1.0
    fwd_contribution = fwd_var * (2.0 - 1.0)

    assert abs(total_var_2y - (total_var_1y + fwd_contribution)) < 1e-10


def test_compute_forward_variance_validation():
    """Test forward variance validation."""
    with pytest.raises(ValueError, match="t2.*must be > t1"):
        compute_forward_variance_strike(
            spot_variance_t1=0.04,
            spot_variance_t2=0.0484,
            t1=2.0,
            t2=1.0,  # Invalid
        )


def test_ir_variance_swap_varying_frequency():
    """Test variance swap with different observation frequencies."""
    # Daily observations
    var_swap_daily = IRVarianceSwap(
        T=1.0,
        strike_variance=0.04,
        notional_per_variance_point=10_000,
        observation_frequency=252,
    )

    # Weekly observations
    var_swap_weekly = IRVarianceSwap(
        T=1.0,
        strike_variance=0.04,
        notional_per_variance_point=10_000,
        observation_frequency=52,
    )

    # Same path, different frequencies
    path = jnp.linspace(0.03, 0.04, 100)

    payoff_daily = var_swap_daily.payoff_path(path)
    payoff_weekly = var_swap_weekly.payoff_path(path)

    # Both should be finite
    assert jnp.isfinite(payoff_daily)
    assert jnp.isfinite(payoff_weekly)


def test_swaption_straddle_array_input():
    """Test straddle with array input."""
    straddle = SwaptionStraddle(
        T=1.0,
        K=0.05,
        annuity=4.5,
        notional=1_000_000,
    )

    # Array of swap rates
    rates = jnp.array([0.04, 0.05, 0.06])
    payoffs = jax.vmap(straddle.payoff_terminal)(rates)

    assert len(payoffs) == 3
    assert payoffs[1] < payoffs[0]  # ATM has min payoff
    assert payoffs[1] < payoffs[2]


def test_swaption_strangle_array_input():
    """Test strangle with array input."""
    strangle = SwaptionStrangle(
        T=1.0,
        K_low=0.045,
        K_high=0.055,
        annuity=4.5,
        notional=1_000_000,
    )

    rates = jnp.linspace(0.03, 0.07, 21)
    payoffs = jax.vmap(strangle.payoff_terminal)(rates)

    # Zero payoff in the middle
    middle_idx = 10
    assert abs(payoffs[middle_idx]) < 100

    # Positive payoffs at extremes
    assert payoffs[0] > 0
    assert payoffs[-1] > 0
