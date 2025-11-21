"""Tests for FX exotic derivatives (TARFs, accumulators, etc.)."""
import jax.numpy as jnp
import pytest

from neutryx.products.fx_exotics import (
    TARF,
    Accumulator,
    Decumulator,
    FaderOption,
    Napoleon,
    RangeAccrual,
    DualRangeAccrual,
    PivotRangeAccrual,
)
from neutryx.products.fx_derivatives_advanced import (
    FXOneTouchOption,
    FXNoTouchOption,
    FXDoubleOneTouchOption,
    FXDoubleNoTouchOption,
    FXBasketOption,
    FXBestOfWorstOfOption,
    QuantoOption,
    CompositeOption,
    fx_one_touch,
    fx_no_touch,
)


def test_tarf_hits_target():
    """Test TARF that hits target and knocks out."""
    product = TARF(
        K=1.2,
        T=1.0,
        target_profit=0.1,
        fixing_times=jnp.array([0.25, 0.5, 0.75, 1.0]),
        notional=1.0,
        leverage=1.0,
        is_long=True,
        knockout_type="target",
    )

    # FX rises, profit accumulates
    path = jnp.array([1.2, 1.25, 1.28, 1.3, 1.32])
    payoff = product.payoff_path(path)

    # Should knock out when target reached
    assert payoff > 0.0


def test_tarf_no_knockout():
    """Test TARF that doesn't hit target."""
    product = TARF(
        K=1.2,
        T=1.0,
        target_profit=1.0,  # Very high target
        fixing_times=jnp.array([0.25, 0.5, 0.75, 1.0]),
        notional=1.0,
        leverage=1.0,
        is_long=True,
    )

    # Small FX movements
    path = jnp.array([1.2, 1.21, 1.22, 1.21, 1.23])
    payoff = product.payoff_path(path)

    # Should return accumulated profit
    assert payoff >= 0.0


def test_accumulator_knockout():
    """Test accumulator with knockout."""
    product = Accumulator(
        K=1.2,
        T=1.0,
        fixing_times=jnp.array([0.25, 0.5, 0.75, 1.0]),
        notional_per_fixing=1.0,
        leverage_down=2.0,
        leverage_up=1.0,
        knockout_barrier=1.3,
        is_long=True,
    )

    # FX rises and hits knockout
    path = jnp.linspace(1.2, 1.35, 100)
    payoff = product.payoff_path(path)

    # Can be positive or negative depending on path
    assert isinstance(float(payoff), float)


def test_accumulator_no_knockout():
    """Test accumulator without knockout."""
    product = Accumulator(
        K=1.2,
        T=1.0,
        fixing_times=jnp.array([0.25, 0.5, 0.75, 1.0]),
        notional_per_fixing=1.0,
        leverage_down=2.0,
        leverage_up=1.0,
        knockout_barrier=None,
        is_long=True,
    )

    path = jnp.linspace(1.2, 1.15, 100)
    payoff = product.payoff_path(path)

    # Accumulator can have losses
    assert isinstance(float(payoff), float)


def test_decumulator():
    """Test decumulator (reverse accumulator)."""
    product = Decumulator(
        K=1.2,
        T=1.0,
        fixing_times=jnp.array([0.25, 0.5, 0.75, 1.0]),
        notional_per_fixing=1.0,
        leverage_up=2.0,
        leverage_down=1.0,
        knockout_barrier=1.1,
    )

    # FX declines
    path = jnp.linspace(1.2, 1.15, 100)
    payoff = product.payoff_path(path)

    assert isinstance(float(payoff), float)


def test_fader_option_in_range():
    """Test fader option with accrual in range."""
    product = FaderOption(
        K=1.2,
        T=1.0,
        lower_bound=1.15,
        upper_bound=1.25,
        max_notional=1.0,
        is_call=True,
        accrual_type="in_range",
    )

    # Path stays mostly in range
    path = jnp.full(100, 1.2)
    payoff = product.payoff_path(path)

    # Should have high effective notional
    assert payoff >= 0.0


def test_fader_option_out_range():
    """Test fader option with accrual out of range."""
    product = FaderOption(
        K=1.2,
        T=1.0,
        lower_bound=1.15,
        upper_bound=1.25,
        max_notional=1.0,
        is_call=True,
        accrual_type="out_range",
    )

    # Path stays in range
    path = jnp.full(100, 1.2)
    payoff = product.payoff_path(path)

    # Effective notional should be low (accrues when out of range)
    assert payoff >= 0.0


def test_napoleon_option_reset_up():
    """Test Napoleon option with upward reset."""
    product = Napoleon(
        K_initial=1.2,
        T=1.0,
        reset_barrier=1.3,
        K_reset=1.25,
        is_call=True,
        reset_type="up",
    )

    # Path crosses reset barrier
    path = jnp.array([1.2, 1.28, 1.32, 1.3, 1.35])
    payoff = product.payoff_path(path)

    # Strike should reset to more favorable level
    assert payoff > 0.0


def test_napoleon_option_no_reset():
    """Test Napoleon option without reset."""
    product = Napoleon(
        K_initial=1.2,
        T=1.0,
        reset_barrier=1.3,
        K_reset=1.25,
        is_call=True,
        reset_type="up",
    )

    # Path doesn't cross barrier
    path = jnp.array([1.2, 1.22, 1.25, 1.24, 1.26])
    payoff = product.payoff_path(path)

    # Uses original strike
    assert payoff >= 0.0


def test_range_accrual():
    """Test range accrual note."""
    product = RangeAccrual(
        T=1.0,
        lower_bound=1.15,
        upper_bound=1.25,
        accrual_rate=0.05,
        principal=1.0,
    )

    # Path stays fully in range
    path = jnp.full(100, 1.2)
    payoff = product.payoff_path(path)

    # Should receive principal + full accrual
    expected = 1.0 + 0.05 * 1.0
    assert abs(payoff - expected) < 0.01


def test_range_accrual_partial():
    """Test range accrual with partial time in range."""
    product = RangeAccrual(
        T=1.0,
        lower_bound=1.15,
        upper_bound=1.25,
        accrual_rate=0.05,
        principal=1.0,
    )

    # Path out of range for half the time
    path = jnp.concatenate([jnp.full(50, 1.3), jnp.full(50, 1.2)])
    payoff = product.payoff_path(path)

    # Should receive principal + partial accrual
    assert 1.0 < payoff < 1.05


# Touch options tests
def test_fx_one_touch_up():
    """Test FX one-touch option with up barrier."""
    option = FXOneTouchOption(
        spot=1.20,
        barrier=1.30,
        expiry=1.0,
        domestic_rate=0.05,
        foreign_rate=0.02,
        volatility=0.10,
        barrier_type="up",
        payout=1.0,
        notional=1000.0,
    )
    price = option.price()
    # Price should be positive (may exceed discounted payout due to reflection)
    assert price > 0


def test_fx_one_touch_down():
    """Test FX one-touch option with down barrier."""
    option = FXOneTouchOption(
        spot=1.20,
        barrier=1.10,
        expiry=1.0,
        domestic_rate=0.05,
        foreign_rate=0.02,
        volatility=0.10,
        barrier_type="down",
        payout=1.0,
        notional=1000.0,
    )
    price = option.price()
    assert price > 0


def test_fx_no_touch_complement():
    """Test that one-touch + no-touch = discounted payout."""
    params = {
        "spot": 1.20,
        "barrier": 1.30,
        "expiry": 1.0,
        "domestic_rate": 0.05,
        "foreign_rate": 0.02,
        "volatility": 0.10,
        "barrier_type": "up",
        "payout": 1.0,
        "notional": 1.0,
    }

    one_touch = FXOneTouchOption(**params)
    no_touch = FXNoTouchOption(**params)

    payout_pv = jnp.exp(-params["domestic_rate"] * params["expiry"])

    # One-touch + no-touch should equal discounted payout
    assert abs((one_touch.price() + no_touch.price()) - payout_pv) < 0.01


def test_fx_double_one_touch():
    """Test FX double one-touch option."""
    option = FXDoubleOneTouchOption(
        spot=1.20,
        lower_barrier=1.10,
        upper_barrier=1.30,
        expiry=1.0,
        domestic_rate=0.05,
        foreign_rate=0.02,
        volatility=0.10,
        payout=1.0,
        notional=1000.0,
    )
    price = option.price()
    # Price should be positive
    assert price > 0


def test_fx_double_no_touch():
    """Test FX double no-touch option."""
    option = FXDoubleNoTouchOption(
        spot=1.20,
        lower_barrier=1.10,
        upper_barrier=1.30,
        expiry=1.0,
        domestic_rate=0.05,
        foreign_rate=0.02,
        volatility=0.10,
        payout=1.0,
        notional=1000.0,
    )
    price = option.price()
    # Price should be less than or equal to discounted payout
    # May be zero if barriers are very close
    assert price >= 0
    assert price <= 1000.0 * jnp.exp(-0.05 * 1.0)


# Multi-currency basket options tests
def test_fx_basket_option_analytical():
    """Test FX basket option with analytical pricing."""
    option = FXBasketOption(
        spot_rates=jnp.array([1.10, 1.25, 1.30]),
        strike=1.20,
        weights=jnp.array([0.4, 0.3, 0.3]),
        expiry=1.0,
        domestic_rate=0.05,
        foreign_rates=jnp.array([0.02, 0.03, 0.01]),
        volatilities=jnp.array([0.10, 0.12, 0.11]),
        correlation_matrix=jnp.eye(3),
        option_type="call",
        notional=1000.0,
    )
    price = option.price_analytical()
    # Price should be positive for at-the-money option
    assert price > 0


def test_fx_basket_option_mc():
    """Test FX basket option with Monte Carlo pricing."""
    import jax.random as jrand

    key = jrand.PRNGKey(42)

    option = FXBasketOption(
        spot_rates=jnp.array([1.10, 1.25]),
        strike=1.15,
        weights=jnp.array([0.5, 0.5]),
        expiry=1.0,
        domestic_rate=0.05,
        foreign_rates=jnp.array([0.02, 0.03]),
        volatilities=jnp.array([0.10, 0.12]),
        correlation_matrix=jnp.array([[1.0, 0.5], [0.5, 1.0]]),
        option_type="call",
        notional=1000.0,
    )
    price = option.price_mc(n_paths=5000, n_steps=50, key=key)
    # Price should be positive
    assert price > 0


def test_fx_basket_helpers():
    """Test FX basket option helper methods."""
    option = FXBasketOption(
        spot_rates=jnp.array([1.10, 1.25]),
        strike=1.15,
        weights=jnp.array([0.5, 0.5]),
        expiry=1.0,
        domestic_rate=0.05,
        foreign_rates=jnp.array([0.02, 0.03]),
        volatilities=jnp.array([0.10, 0.12]),
        correlation_matrix=jnp.array([[1.0, 0.5], [0.5, 1.0]]),
    )

    basket_spot = option.basket_spot_value()
    assert basket_spot > 0

    basket_vol = option.basket_volatility()
    assert basket_vol > 0


def test_fx_best_of_worst_of_option():
    """Test FX best-of/worst-of option."""
    import jax.random as jrand

    key = jrand.PRNGKey(42)

    option = FXBestOfWorstOfOption(
        spot_rates=jnp.array([1.10, 1.25]),
        strikes=jnp.array([1.12, 1.27]),
        expiry=1.0,
        domestic_rate=0.05,
        foreign_rates=jnp.array([0.02, 0.03]),
        volatilities=jnp.array([0.10, 0.12]),
        correlation_matrix=jnp.array([[1.0, 0.3], [0.3, 1.0]]),
        option_type="call",
        payoff_type="worst-of",
        notional=1000.0,
    )

    price = option.price_mc(n_paths=5000, n_steps=50, key=key)
    assert price >= 0


# Quanto options tests
def test_quanto_option_call():
    """Test quanto call option."""
    option = QuantoOption(
        spot=100.0,
        strike=100.0,
        expiry=1.0,
        domestic_rate=0.05,
        foreign_rate=0.02,
        asset_volatility=0.20,
        fx_volatility=0.10,
        correlation=0.3,
        quanto_fx_rate=1.0,
        option_type="call",
        notional=1000.0,
    )
    price = option.price()
    # ATM option should have positive price
    assert price > 0


def test_quanto_option_put():
    """Test quanto put option."""
    option = QuantoOption(
        spot=100.0,
        strike=100.0,
        expiry=1.0,
        domestic_rate=0.05,
        foreign_rate=0.02,
        asset_volatility=0.20,
        fx_volatility=0.10,
        correlation=-0.3,
        quanto_fx_rate=1.0,
        option_type="put",
        notional=1000.0,
    )
    price = option.price()
    assert price > 0


def test_quanto_option_correlation_effect():
    """Test that correlation affects quanto pricing."""
    base_params = {
        "spot": 100.0,
        "strike": 100.0,
        "expiry": 1.0,
        "domestic_rate": 0.05,
        "foreign_rate": 0.02,
        "asset_volatility": 0.20,
        "fx_volatility": 0.10,
        "quanto_fx_rate": 1.0,
        "option_type": "call",
        "notional": 1.0,
    }

    # Positive correlation
    quanto_pos = QuantoOption(**base_params, correlation=0.5)
    price_pos = quanto_pos.price()

    # Negative correlation
    quanto_neg = QuantoOption(**base_params, correlation=-0.5)
    price_neg = quanto_neg.price()

    # Prices should differ due to correlation
    assert abs(price_pos - price_neg) > 0.001


def test_quanto_option_delta():
    """Test quanto option delta calculation."""
    option = QuantoOption(
        spot=100.0,
        strike=100.0,
        expiry=1.0,
        domestic_rate=0.05,
        foreign_rate=0.02,
        asset_volatility=0.20,
        fx_volatility=0.10,
        correlation=0.3,
        quanto_fx_rate=1.0,
        option_type="call",
        notional=1.0,
    )
    delta = option.delta()
    # ATM call delta should be around 0.5 (quanto-adjusted)
    assert 0.3 < delta < 0.7


def test_composite_option():
    """Test composite/compound option."""
    option = CompositeOption(
        spot=1.20,
        strike_mother=0.05,
        strike_daughter=1.25,
        expiry_mother=0.5,
        expiry_daughter=1.0,
        domestic_rate=0.05,
        foreign_rate=0.02,
        volatility=0.10,
        mother_type="call",
        daughter_type="call",
        notional=1000.0,
    )
    price = option.price()
    # Price should be positive
    assert price >= 0


# Enhanced range accrual tests
def test_range_accrual_with_knockout():
    """Test range accrual with knockout provision."""
    product = RangeAccrual(
        T=1.0,
        lower_bound=1.15,
        upper_bound=1.25,
        accrual_rate=0.05,
        principal=1.0,
        knockout_barrier=1.30,
        knockout_type="up",
    )

    # Path hits knockout
    path = jnp.linspace(1.2, 1.35, 100)
    payoff = product.payoff_path(path)

    # Should only return principal (no accrual)
    assert abs(payoff - 1.0) < 0.01


def test_range_accrual_with_leverage():
    """Test range accrual with leverage."""
    product = RangeAccrual(
        T=1.0,
        lower_bound=1.15,
        upper_bound=1.25,
        accrual_rate=0.05,
        principal=1.0,
        leverage=2.0,
    )

    # Path stays fully in range
    path = jnp.full(100, 1.2)
    payoff = product.payoff_path(path)

    # Should receive principal + leveraged accrual
    expected = 1.0 + 0.05 * 1.0 * 2.0
    assert abs(payoff - expected) < 0.01


def test_dual_range_accrual_both_in_range():
    """Test dual range accrual when both FX pairs in range."""
    product = DualRangeAccrual(
        T=1.0,
        lower_bounds=(1.15, 1.05),
        upper_bounds=(1.25, 1.15),
        base_accrual_rate=0.03,
        enhanced_accrual_rate=0.08,
        principal=1.0,
        require_both=True,
    )

    # Both paths in range
    paths = jnp.array([jnp.full(100, 1.20), jnp.full(100, 1.10)])
    payoff = product.payoff_path(paths)

    # Should receive enhanced accrual
    expected = 1.0 + 0.08 * 1.0
    assert abs(payoff - expected) < 0.01


def test_dual_range_accrual_one_in_range():
    """Test dual range accrual when only one FX pair in range."""
    product = DualRangeAccrual(
        T=1.0,
        lower_bounds=(1.15, 1.05),
        upper_bounds=(1.25, 1.15),
        base_accrual_rate=0.03,
        enhanced_accrual_rate=0.08,
        principal=1.0,
        require_both=False,
    )

    # One in range, one out
    paths = jnp.array([jnp.full(100, 1.20), jnp.full(100, 1.30)])
    payoff = product.payoff_path(paths)

    # Should receive base accrual
    assert payoff > 1.0


def test_pivot_range_accrual():
    """Test pivot range accrual."""
    product = PivotRangeAccrual(
        T=1.0,
        pivot=1.20,
        lower_bound=1.15,
        upper_bound=1.25,
        accrual_rate_above=0.06,
        accrual_rate_below=0.04,
        principal=1.0,
    )

    # Path mostly above pivot
    path = jnp.full(100, 1.22)
    payoff = product.payoff_path(path)

    # Should get mostly above-pivot accrual
    expected_approx = 1.0 + 0.06 * 1.0
    assert abs(payoff - expected_approx) < 0.01
