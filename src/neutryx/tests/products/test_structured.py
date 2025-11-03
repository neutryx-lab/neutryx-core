"""Tests for structured products (autocallables, reverse convertibles, etc.)."""
import jax.numpy as jnp
import pytest

from neutryx.products.structured import (
    PhoenixAutocallable,
    SnowballAutocallable,
    StepDownAutocallable,
    ReverseConvertible,
    DualCurrencyInvestment,
    EquityLinkedNote,
    BonusEnhancedNote,
)


def test_phoenix_autocallable_autocalls():
    """Test Phoenix autocallable that autocalls early."""
    product = PhoenixAutocallable(
        K=100.0,
        T=1.0,
        autocall_barrier=1.0,
        coupon_barrier=0.85,
        coupon_rate=0.05,
        observation_times=jnp.array([0.25, 0.5, 0.75, 1.0]),
        put_strike=1.0,
    )

    # Price rises above autocall barrier at first observation
    path = jnp.array([100.0, 105.0, 102.0, 101.0])
    payoff = product.payoff_path(path)

    # Should autocall with principal + 1 coupon
    assert payoff > 1.0
    assert payoff < 1.1  # 1.0 + 0.05


def test_phoenix_autocallable_maturity():
    """Test Phoenix autocallable that goes to maturity."""
    product = PhoenixAutocallable(
        K=100.0,
        T=1.0,
        autocall_barrier=1.1,  # High barrier
        coupon_barrier=0.85,
        coupon_rate=0.05,
        observation_times=jnp.array([0.25, 0.5, 0.75, 1.0]),
        put_strike=1.0,
    )

    # Price stays flat
    path = jnp.linspace(100.0, 100.0, 100)
    payoff = product.payoff_path(path)

    # Should pay principal + accumulated coupons
    assert payoff > 1.0


def test_snowball_autocallable_increasing_coupons():
    """Test Snowball autocallable with increasing coupon rates."""
    product = SnowballAutocallable(
        K=100.0,
        T=1.0,
        autocall_barrier=1.05,
        coupon_barrier=0.9,
        initial_coupon_rate=0.05,
        coupon_step=0.01,
        observation_times=jnp.array([0.25, 0.5, 0.75, 1.0]),
        put_strike=1.0,
    )

    path = jnp.linspace(100.0, 95.0, 100)
    payoff = product.payoff_path(path)

    assert payoff >= 0.0


def test_step_down_autocallable():
    """Test step-down autocallable with decreasing barriers."""
    product = StepDownAutocallable(
        K=100.0,
        T=1.0,
        autocall_barriers=jnp.array([1.0, 0.95, 0.9, 0.85]),
        coupon_barrier=0.8,
        coupon_rate=0.06,
        observation_times=jnp.array([0.25, 0.5, 0.75, 1.0]),
        put_strike=1.0,
    )

    # Price declines but crosses a step-down barrier
    path = jnp.linspace(100.0, 90.0, 100)
    payoff = product.payoff_path(path)

    assert payoff > 0.0


def test_reverse_convertible_barrier_hit():
    """Test reverse convertible when barrier is hit."""
    product = ReverseConvertible(
        K=100.0,
        T=1.0,
        coupon_rate=0.15,
        barrier=0.7,
        principal=1.0,
    )

    # Path hits barrier then recovers
    path = jnp.array([100.0, 80.0, 65.0, 75.0])
    payoff = product.payoff_path(path)

    # Should receive stock + coupon
    assert payoff > 0.0


def test_reverse_convertible_barrier_not_hit():
    """Test reverse convertible when barrier is not hit."""
    product = ReverseConvertible(
        K=100.0,
        T=1.0,
        coupon_rate=0.15,
        barrier=0.7,
        principal=1.0,
    )

    # Path stays above barrier
    path = jnp.array([100.0, 95.0, 90.0, 92.0])
    payoff = product.payoff_path(path)

    # Should receive principal + coupon
    expected = 1.0 + 0.15
    assert abs(payoff - expected) < 0.01


def test_dual_currency_investment_call():
    """Test DCI with call option (short call)."""
    product = DualCurrencyInvestment(
        K=110.0,
        T=1.0,
        interest_rate=0.05,
        base_currency_amount=1.0,
        option_type="call",
    )

    # FX rises above strike
    path = jnp.array([100.0, 105.0, 112.0, 115.0])
    payoff = product.payoff_path(path)

    # Option exercised, receive alternate currency converted at strike
    assert payoff > 0.0


def test_equity_linked_note_capped():
    """Test ELN with cap."""
    product = EquityLinkedNote(
        K=100.0,
        T=1.0,
        participation_rate=1.5,
        cap=0.3,
        floor=1.0,
        principal=1.0,
    )

    # Stock rises significantly
    path = jnp.array([100.0, 120.0, 150.0, 160.0])
    payoff = product.payoff_path(path)

    # Should be capped at 1.0 + 0.3
    assert payoff <= 1.31  # Allow small tolerance


def test_equity_linked_note_floor():
    """Test ELN with floor protection."""
    product = EquityLinkedNote(
        K=100.0,
        T=1.0,
        participation_rate=1.0,
        cap=None,
        floor=0.95,
        principal=1.0,
    )

    # Stock declines
    path = jnp.array([100.0, 90.0, 80.0, 70.0])
    payoff = product.payoff_path(path)

    # Should be floored
    assert payoff >= 0.95 * 1.0


def test_bonus_enhanced_note():
    """Test bonus enhanced note."""
    product = BonusEnhancedNote(
        K=100.0,
        T=1.0,
        bonus_level=1.2,
        barrier=0.8,
        principal=1.0,
    )

    # Barrier not hit
    path = jnp.array([100.0, 95.0, 90.0, 92.0])
    payoff = product.payoff_path(path)

    # Should get max of final level and bonus
    assert payoff >= 1.0
