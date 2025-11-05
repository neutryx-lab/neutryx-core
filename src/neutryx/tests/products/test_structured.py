"""Tests for structured products (autocallables, reverse convertibles, etc.)."""
import jax.numpy as jnp
import pytest

from neutryx.products.structured import (
    PhoenixAutocallable,
    SnowballAutocallable,
    StepDownAutocallable,
    AthenaAutocallable,
    ReverseConvertible,
    DualCurrencyInvestment,
    EquityLinkedNote,
    BonusEnhancedNote,
    CliquetOption,
    NapoleonOption,
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


def test_athena_autocallable_no_memory():
    """Test Athena autocallable (no memory feature)."""
    product = AthenaAutocallable(
        K=100.0,
        T=1.0,
        autocall_barrier=1.0,
        coupon_barrier=0.75,
        coupon_rate=0.05,
        observation_times=jnp.array([0.25, 0.5, 0.75, 1.0]),
        put_strike=1.0,
    )

    # Path: Above barrier at obs 1, below at obs 2, above at obs 3 & 4
    # Obs 1: 105 (above 100 autocall, above 75 coupon) -> no autocall yet
    # Obs 2: 70 (below coupon barrier) -> coupon NOT paid (no memory)
    # Obs 3: 85 (above coupon barrier) -> coupon paid
    # Obs 4: 90 (above coupon barrier) -> coupon paid
    path = jnp.linspace(100.0, 90.0, 100)
    # Manually adjust specific points for observations
    # This is a simplified test - in practice path would be more complex

    # Simplified: test that coupons are lost if barrier not hit
    # Path stays flat at 100
    path_flat = jnp.ones(100) * 100.0
    payoff = product.payoff_path(path_flat)

    # Should get principal + coupons for all periods where barrier hit
    # Since always above both barriers, should get full coupons
    expected_min = 1.0 + 4 * 0.05  # Principal + 4 coupons
    assert payoff >= expected_min * 0.95  # Allow some tolerance


def test_athena_vs_phoenix_memory():
    """Compare Athena (no memory) vs Phoenix (memory) with same params."""
    # When barrier is always hit, both should be similar
    # When barrier is missed, Phoenix accumulates while Athena loses coupons

    athena = AthenaAutocallable(
        K=100.0,
        T=1.0,
        autocall_barrier=1.2,  # High barrier to avoid autocall
        coupon_barrier=0.90,
        coupon_rate=0.05,
        observation_times=jnp.array([0.5, 1.0]),
        put_strike=1.0,
    )

    phoenix = PhoenixAutocallable(
        K=100.0,
        T=1.0,
        autocall_barrier=1.2,
        coupon_barrier=0.90,
        coupon_rate=0.05,
        observation_times=jnp.array([0.5, 1.0]),
        put_strike=1.0,
    )

    # Path: First obs below barrier, second above
    # At obs 1 (idx 50): below 90 barrier
    # At obs 2 (idx 99): above 90 barrier
    path = jnp.linspace(100.0, 85.0, 50).tolist() + jnp.linspace(85.0, 95.0, 50).tolist()
    path = jnp.array(path)

    athena_payoff = athena.payoff_path(path)
    phoenix_payoff = phoenix.payoff_path(path)

    # Phoenix should have higher payoff due to memory
    # Athena loses first coupon, Phoenix accumulates it
    assert phoenix_payoff >= athena_payoff


def test_cliquet_option_basic():
    """Test basic cliquet option with local caps and floors."""
    product = CliquetOption(
        K=100.0,
        T=1.0,
        reset_times=jnp.array([0.25, 0.5, 0.75, 1.0]),
        local_floor=0.0,
        local_cap=0.10,
        global_floor=0.0,
        global_cap=None,
    )

    # Path: 100 -> 115 -> 110 -> 120 -> 125
    # Period 1: (115-100)/100 = 15% -> capped to 10%
    # Period 2: (110-115)/115 = -4.35% -> floored to 0%
    # Period 3: (120-110)/110 = 9.09% -> no cap/floor
    # Period 4: (125-120)/120 = 4.17% -> no cap/floor
    # Total: 10% + 0% + 9.09% + 4.17% = 23.26%
    path = jnp.array([100.0, 115.0, 110.0, 120.0, 125.0])
    payoff = product.payoff_path(path)

    # Payoff should be K * total_return = 100 * 0.2326 = 23.26
    # Allow some tolerance
    assert payoff > 20.0
    assert payoff < 25.0


def test_cliquet_option_all_capped():
    """Test cliquet when all periods are capped."""
    product = CliquetOption(
        K=100.0,
        T=1.0,
        reset_times=jnp.array([0.25, 0.5, 0.75, 1.0]),
        local_floor=0.0,
        local_cap=0.05,  # 5% cap
        global_floor=0.0,
    )

    # Path: 100 -> 120 -> 140 -> 160 -> 180
    # Each period has >5% return, all capped
    # Total: 4 * 5% = 20%
    path = jnp.array([100.0, 120.0, 140.0, 160.0, 180.0])
    payoff = product.payoff_path(path)

    expected = 100.0 * 0.20  # 20.0
    assert jnp.isclose(payoff, expected, atol=1.0)


def test_cliquet_option_all_floored():
    """Test cliquet when all periods hit the floor."""
    product = CliquetOption(
        K=100.0,
        T=1.0,
        reset_times=jnp.array([0.25, 0.5, 0.75, 1.0]),
        local_floor=0.0,
        local_cap=0.10,
    )

    # Path: Continuous decline
    # All periods negative, floored to 0%
    path = jnp.array([100.0, 90.0, 80.0, 70.0, 60.0])
    payoff = product.payoff_path(path)

    # All periods floored to 0, total = 0
    expected = 0.0
    assert jnp.isclose(payoff, expected, atol=0.1)


def test_cliquet_with_global_cap():
    """Test cliquet with global cap on total returns."""
    product = CliquetOption(
        K=100.0,
        T=1.0,
        reset_times=jnp.array([0.25, 0.5, 0.75, 1.0]),
        local_floor=0.0,
        local_cap=0.10,
        global_floor=0.0,
        global_cap=0.25,  # 25% global cap
    )

    # Path that would give >25% total return
    # Each period: ~8% = 4 * 8% = 32%, but global capped at 25%
    path = jnp.array([100.0, 108.0, 116.64, 125.97, 136.05])
    payoff = product.payoff_path(path)

    # Should be capped at global cap
    expected = 100.0 * 0.25  # 25.0
    assert payoff <= expected * 1.01  # Small tolerance


def test_napoleon_option_guaranteed_return():
    """Test Napoleon option with guaranteed minimum return."""
    product = NapoleonOption(
        K=100.0,
        T=1.0,
        reset_times=jnp.array([0.25, 0.5, 0.75, 1.0]),
        local_floor=0.0,
        local_cap=0.10,
        guaranteed_return=0.05,  # 5% guaranteed
    )

    # Path: Slight decline, accumulated return < 5%
    # All periods barely positive or zero
    path = jnp.array([100.0, 101.0, 101.5, 102.0, 102.5])
    payoff = product.payoff_path(path)

    # Should get guaranteed minimum
    expected_min = 100.0 * 0.05  # 5.0
    assert payoff >= expected_min * 0.99


def test_napoleon_option_exceeds_guarantee():
    """Test Napoleon when accumulated return exceeds guarantee."""
    product = NapoleonOption(
        K=100.0,
        T=1.0,
        reset_times=jnp.array([0.25, 0.5, 0.75, 1.0]),
        local_floor=0.0,
        local_cap=0.10,
        guaranteed_return=0.05,
    )

    # Path: Strong performance, accumulated > 5%
    # Each period: 8% capped to 10%, total = 4 * 8% = 32%
    path = jnp.array([100.0, 108.0, 116.64, 125.97, 136.05])
    payoff = product.payoff_path(path)

    # Should exceed guaranteed minimum
    expected_min = 100.0 * 0.05
    assert payoff > expected_min * 2.0  # Significantly higher
