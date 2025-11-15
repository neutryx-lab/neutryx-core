"""Tests for European swaption pricing."""

import jax.numpy as jnp
import pytest

from neutryx.products.swaptions import (
    EuropeanSwaption,
    SwaptionType,
    black_swaption_price,
    european_swaption_black,
    implied_swaption_volatility,
    swap_annuity,
    swaption_delta,
    swaption_vega,
    forward_swap_rate,
)


def test_swap_annuity():
    """Test swap annuity calculation."""
    # 5-year swap, semiannual payments
    n_periods = 10
    year_fractions = jnp.full(n_periods, 0.5, dtype=jnp.float32)
    discount_factors = jnp.array([
        jnp.exp(-0.05 * (i + 1) * 0.5) for i in range(n_periods)
    ], dtype=jnp.float32)

    annuity = swap_annuity(discount_factors, year_fractions)

    # Annuity should be positive
    assert annuity > 0

    # For a 5-year swap at 5%, annuity should be around 4.3-4.5
    assert 4.0 < annuity < 5.0


def test_forward_swap_rate_flat_curve():
    """Forward swap rate reduces to par rate under flat curve."""
    option_maturity = 1.0
    swap_maturity = 5.0
    payment_frequency = 2
    n_payments = int(swap_maturity * payment_frequency)
    year_fractions = jnp.full(n_payments, 1.0 / payment_frequency)

    discount_rate = 0.03
    payment_times = option_maturity + jnp.arange(1, n_payments + 1) / payment_frequency
    discount_factors = jnp.exp(-discount_rate * payment_times)
    df_option = jnp.exp(-discount_rate * option_maturity)

    fwd = forward_swap_rate(
        option_maturity,
        swap_maturity,
        discount_factors,
        year_fractions,
        df_option_maturity=float(df_option),
    )

    annuity = swap_annuity(discount_factors, year_fractions)
    expected = (df_option - jnp.exp(-discount_rate * (option_maturity + swap_maturity))) / annuity

    assert jnp.isclose(fwd, expected)


def test_black_swaption_atm():
    """Test at-the-money swaption pricing."""
    # ATM swaption: forward rate = strike
    forward_rate = 0.05
    strike = 0.05
    option_maturity = 1.0
    volatility = 0.20
    annuity = 4.5
    notional = 1_000_000.0

    # Price payer and receiver
    payer_price = black_swaption_price(
        forward_rate, strike, option_maturity, volatility, annuity, notional, is_payer=True
    )
    receiver_price = black_swaption_price(
        forward_rate, strike, option_maturity, volatility, annuity, notional, is_payer=False
    )

    # Both should be positive
    assert payer_price > 0
    assert receiver_price > 0

    # ATM: payer and receiver should have same price (put-call symmetry)
    assert jnp.isclose(payer_price, receiver_price, rtol=1e-6)


def test_black_swaption_itm_otm():
    """Test in-the-money and out-of-the-money swaptions."""
    option_maturity = 1.0
    volatility = 0.20
    annuity = 4.5
    notional = 1_000_000.0

    # ITM payer: forward > strike
    payer_itm = black_swaption_price(
        0.06, 0.05, option_maturity, volatility, annuity, notional, is_payer=True
    )

    # OTM payer: forward < strike
    payer_otm = black_swaption_price(
        0.04, 0.05, option_maturity, volatility, annuity, notional, is_payer=True
    )

    # ITM should be more expensive than OTM
    assert payer_itm > payer_otm

    # ITM receiver: strike > forward
    receiver_itm = black_swaption_price(
        0.04, 0.05, option_maturity, volatility, annuity, notional, is_payer=False
    )

    # OTM receiver: strike < forward
    receiver_otm = black_swaption_price(
        0.06, 0.05, option_maturity, volatility, annuity, notional, is_payer=False
    )

    assert receiver_itm > receiver_otm


def test_black_swaption_zero_vol():
    """Test swaption pricing with zero volatility."""
    option_maturity = 1.0
    annuity = 4.5
    notional = 1_000_000.0

    # ITM payer with zero vol should equal intrinsic value
    forward_rate = 0.06
    strike = 0.05
    price = black_swaption_price(
        forward_rate, strike, option_maturity, 0.0, annuity, notional, is_payer=True
    )

    # Intrinsic value = (forward - strike) * annuity * notional
    intrinsic = (forward_rate - strike) * annuity * notional
    assert jnp.isclose(price, intrinsic, rtol=1e-6)

    # OTM with zero vol should be worthless
    price_otm = black_swaption_price(
        0.04, 0.05, option_maturity, 0.0, annuity, notional, is_payer=True
    )
    assert jnp.isclose(price_otm, 0.0, atol=1e-6)


def test_black_swaption_volatility_impact():
    """Test that higher volatility increases swaption value."""
    forward_rate = 0.05
    strike = 0.05
    option_maturity = 1.0
    annuity = 4.5
    notional = 1_000_000.0

    price_low_vol = black_swaption_price(
        forward_rate, strike, option_maturity, 0.10, annuity, notional, is_payer=True
    )

    price_high_vol = black_swaption_price(
        forward_rate, strike, option_maturity, 0.30, annuity, notional, is_payer=True
    )

    # Higher volatility => higher price
    assert price_high_vol > price_low_vol


def test_black_swaption_maturity_impact():
    """Test that longer maturity increases swaption value."""
    forward_rate = 0.05
    strike = 0.05
    volatility = 0.20
    annuity = 4.5
    notional = 1_000_000.0

    price_short = black_swaption_price(
        forward_rate, strike, 0.5, volatility, annuity, notional, is_payer=True
    )

    price_long = black_swaption_price(
        forward_rate, strike, 2.0, volatility, annuity, notional, is_payer=True
    )

    # Longer maturity => higher price (more time value)
    assert price_long > price_short


def test_european_swaption_payoff_and_greeks():
    """EuropeanSwaption helper exposes payoff/greeks."""
    annuity = 4.5
    swaption = EuropeanSwaption(
        T=1.0,
        strike=0.05,
        annuity=annuity,
        notional=1_000_000.0,
        swaption_type=SwaptionType.PAYER,
    )

    # Payoff uses terminal forward rate
    payoff = swaption.payoff_terminal(0.06)
    expected_payoff = (0.06 - 0.05) * annuity * 1_000_000.0
    assert jnp.isclose(payoff, expected_payoff)

    # Black price/greeks reuse existing helpers
    forward_rate = 0.055
    volatility = 0.20
    price = swaption.price_black(forward_rate, volatility)
    delta = swaption.delta(forward_rate, volatility)
    vega = swaption.vega(forward_rate, volatility)

    assert price > 0.0
    assert delta > 0.0
    assert vega > 0.0


def test_european_swaption_black():
    """Test full European swaption pricing."""
    price = european_swaption_black(
        strike=0.05,
        option_maturity=1.0,
        swap_maturity=5.0,
        volatility=0.20,
        discount_rate=0.03,
        notional=1_000_000,
        payment_frequency=2,
        is_payer=True,
    )

    # Price should be positive
    assert price > 0

    # Should be reasonable for a $1M notional (typically tens to tens of thousands)
    assert 10 < price < 200_000


def test_european_swaption_payer_receiver():
    """Test payer vs receiver swaption."""
    payer_price = european_swaption_black(
        strike=0.05,
        option_maturity=1.0,
        swap_maturity=5.0,
        volatility=0.20,
        discount_rate=0.03,
        notional=1_000_000,
        is_payer=True,
    )

    receiver_price = european_swaption_black(
        strike=0.05,
        option_maturity=1.0,
        swap_maturity=5.0,
        volatility=0.20,
        discount_rate=0.03,
        notional=1_000_000,
        is_payer=False,
    )

    # Both should be positive
    assert payer_price > 0
    assert receiver_price > 0


def test_swaption_vega():
    """Test swaption vega calculation."""
    forward_rate = 0.05
    strike = 0.05
    option_maturity = 1.0
    volatility = 0.20
    annuity = 4.5
    notional = 1_000_000.0

    vega = swaption_vega(
        forward_rate, strike, option_maturity, volatility, annuity, notional
    )

    # Vega should be positive
    assert vega > 0

    # Verify vega numerically
    epsilon = 0.001
    price_base = black_swaption_price(
        forward_rate, strike, option_maturity, volatility, annuity, notional, True
    )
    price_up = black_swaption_price(
        forward_rate, strike, option_maturity, volatility + epsilon, annuity, notional, True
    )

    numerical_vega = (price_up - price_base) / epsilon

    # Should be close (vega is typically per 100bp/1% vol change)
    # Our vega is per unit change, so compare directly
    assert jnp.isclose(vega, numerical_vega, rtol=0.05)


def test_swaption_delta():
    """Test swaption delta calculation."""
    forward_rate = 0.05
    strike = 0.05
    option_maturity = 1.0
    volatility = 0.20
    annuity = 4.5
    notional = 1_000_000.0

    # Payer delta
    delta_payer = swaption_delta(
        forward_rate, strike, option_maturity, volatility, annuity, notional, is_payer=True
    )

    # Receiver delta
    delta_receiver = swaption_delta(
        forward_rate, strike, option_maturity, volatility, annuity, notional, is_payer=False
    )

    # Payer delta should be positive
    assert delta_payer > 0

    # Receiver delta should be negative
    assert delta_receiver < 0

    # For ATM, deltas should be in reasonable ranges
    # Note: Unlike prices, deltas are NOT symmetric for payer/receiver at ATM
    # because delta depends on Φ(d1) and Φ(-d1) which differ when d1 != 0
    assert abs(delta_payer) > 100_000  # Reasonable magnitude for $1M notional
    assert abs(delta_receiver) > 100_000


def test_implied_volatility():
    """Test implied volatility calculation."""
    forward_rate = 0.05
    strike = 0.05
    option_maturity = 1.0
    known_vol = 0.25
    annuity = 4.5
    notional = 1_000_000.0

    # Calculate price at known volatility
    market_price = float(
        black_swaption_price(
            forward_rate, strike, option_maturity, known_vol, annuity, notional, True
        )
    )

    # Recover implied vol
    implied_vol = implied_swaption_volatility(
        market_price, forward_rate, strike, option_maturity, annuity, notional, True
    )

    # Should recover the original volatility
    assert jnp.isclose(implied_vol, known_vol, atol=1e-4)


def test_implied_volatility_itm():
    """Test implied vol for ITM swaption."""
    forward_rate = 0.06
    strike = 0.05
    option_maturity = 1.0
    known_vol = 0.20
    annuity = 4.5
    notional = 1_000_000.0

    market_price = float(
        black_swaption_price(
            forward_rate, strike, option_maturity, known_vol, annuity, notional, True
        )
    )

    implied_vol = implied_swaption_volatility(
        market_price, forward_rate, strike, option_maturity, annuity, notional, True
    )

    assert jnp.isclose(implied_vol, known_vol, atol=1e-4)


def test_swaption_put_call_parity():
    """Test swaption put-call parity (payer-receiver parity).

    For ATM swaptions: Payer - Receiver ≈ 0 (small difference due to discounting)
    """
    forward_rate = 0.05
    strike = 0.05
    option_maturity = 1.0
    volatility = 0.20
    annuity = 4.5
    notional = 1_000_000.0

    payer_price = black_swaption_price(
        forward_rate, strike, option_maturity, volatility, annuity, notional, True
    )

    receiver_price = black_swaption_price(
        forward_rate, strike, option_maturity, volatility, annuity, notional, False
    )

    # For ATM, prices should be equal
    assert jnp.isclose(payer_price, receiver_price, rtol=1e-6)


def test_jit_compilation():
    """Test that key functions are JIT-compilable."""
    import jax

    # JIT compile key functions
    jitted_black = jax.jit(
        lambda: black_swaption_price(0.05, 0.05, 1.0, 0.20, 4.5, 1_000_000, True)
    )
    jitted_vega = jax.jit(lambda: swaption_vega(0.05, 0.05, 1.0, 0.20, 4.5, 1_000_000))

    # Execute
    price = jitted_black()
    vega = jitted_vega()

    # Should produce valid results
    assert jnp.isfinite(price)
    assert jnp.isfinite(vega)
    assert price > 0
    assert vega > 0
