"""Tests for basket option payoffs."""

import jax.numpy as jnp
import pytest

from neutryx.products.basket import (
    WorstOfCall,
    WorstOfPut,
    BestOfCall,
    BestOfPut,
    AverageBasketCall,
    AverageBasketPut,
    SpreadOption,
    RainbowOption,
)


def test_worst_of_call_itm():
    """Test worst-of call in-the-money."""
    option = WorstOfCall(K=100.0, T=1.0, n_assets=3)
    # Terminal prices: [110, 120, 130], worst = 110
    spots = jnp.array([110.0, 120.0, 130.0])

    payoff = option.payoff_terminal(spots)

    # Payoff = max(min(spots) - K, 0) = max(110 - 100, 0) = 10
    expected_payoff = 10.0
    assert jnp.isclose(payoff, expected_payoff)


def test_worst_of_call_otm():
    """Test worst-of call out-of-the-money."""
    option = WorstOfCall(K=100.0, T=1.0, n_assets=3)
    # Terminal prices: [95, 110, 120], worst = 95
    spots = jnp.array([95.0, 110.0, 120.0])

    payoff = option.payoff_terminal(spots)

    # Payoff = max(95 - 100, 0) = 0
    assert payoff == 0.0


def test_worst_of_put_itm():
    """Test worst-of put in-the-money."""
    option = WorstOfPut(K=100.0, T=1.0, n_assets=3)
    # Terminal prices: [85, 90, 95], worst = 85
    spots = jnp.array([85.0, 90.0, 95.0])

    payoff = option.payoff_terminal(spots)

    # Payoff = max(K - min(spots), 0) = max(100 - 85, 0) = 15
    expected_payoff = 15.0
    assert jnp.isclose(payoff, expected_payoff)


def test_worst_of_put_otm():
    """Test worst-of put out-of-the-money."""
    option = WorstOfPut(K=100.0, T=1.0, n_assets=3)
    # Terminal prices: [105, 110, 115], worst = 105
    spots = jnp.array([105.0, 110.0, 115.0])

    payoff = option.payoff_terminal(spots)

    # Payoff = max(100 - 105, 0) = 0
    assert payoff == 0.0


def test_best_of_call_itm():
    """Test best-of call in-the-money."""
    option = BestOfCall(K=100.0, T=1.0, n_assets=3)
    # Terminal prices: [95, 105, 110], best = 110
    spots = jnp.array([95.0, 105.0, 110.0])

    payoff = option.payoff_terminal(spots)

    # Payoff = max(max(spots) - K, 0) = max(110 - 100, 0) = 10
    expected_payoff = 10.0
    assert jnp.isclose(payoff, expected_payoff)


def test_best_of_call_otm():
    """Test best-of call out-of-the-money."""
    option = BestOfCall(K=100.0, T=1.0, n_assets=3)
    # Terminal prices: [85, 90, 95], best = 95
    spots = jnp.array([85.0, 90.0, 95.0])

    payoff = option.payoff_terminal(spots)

    # Payoff = max(95 - 100, 0) = 0
    assert payoff == 0.0


def test_best_of_put_itm():
    """Test best-of put in-the-money."""
    option = BestOfPut(K=100.0, T=1.0, n_assets=3)
    # Terminal prices: [85, 90, 110], best = 110
    spots = jnp.array([85.0, 90.0, 110.0])

    payoff = option.payoff_terminal(spots)

    # Payoff = max(K - max(spots), 0) = max(100 - 110, 0) = 0
    assert payoff == 0.0


def test_best_of_put_itm_all_below_strike():
    """Test best-of put when all assets below strike."""
    option = BestOfPut(K=100.0, T=1.0, n_assets=3)
    # Terminal prices: [85, 90, 95], best = 95
    spots = jnp.array([85.0, 90.0, 95.0])

    payoff = option.payoff_terminal(spots)

    # Payoff = max(100 - 95, 0) = 5
    expected_payoff = 5.0
    assert jnp.isclose(payoff, expected_payoff)


def test_average_basket_call_equal_weights():
    """Test average basket call with equal weights."""
    option = AverageBasketCall(K=100.0, T=1.0, n_assets=3, weights=None)
    # Terminal prices: [90, 100, 110], average = 100
    spots = jnp.array([90.0, 100.0, 110.0])

    payoff = option.payoff_terminal(spots)

    # Average = 100, Payoff = max(100 - 100, 0) = 0
    assert payoff == 0.0


def test_average_basket_call_custom_weights():
    """Test average basket call with custom weights."""
    weights = jnp.array([0.5, 0.3, 0.2])
    option = AverageBasketCall(K=100.0, T=1.0, n_assets=3, weights=weights)
    # Terminal prices: [90, 100, 110]
    spots = jnp.array([90.0, 100.0, 110.0])

    payoff = option.payoff_terminal(spots)

    # Weighted average = 0.5*90 + 0.3*100 + 0.2*110 = 45 + 30 + 22 = 97
    # Payoff = max(97 - 100, 0) = 0
    assert payoff == 0.0


def test_average_basket_call_custom_weights_itm():
    """Test average basket call with custom weights in-the-money."""
    weights = jnp.array([0.2, 0.3, 0.5])
    option = AverageBasketCall(K=100.0, T=1.0, n_assets=3, weights=weights)
    # Terminal prices: [90, 100, 120]
    spots = jnp.array([90.0, 100.0, 120.0])

    payoff = option.payoff_terminal(spots)

    # Weighted average = 0.2*90 + 0.3*100 + 0.5*120 = 18 + 30 + 60 = 108
    # Payoff = max(108 - 100, 0) = 8
    expected_payoff = 8.0
    assert jnp.isclose(payoff, expected_payoff)


def test_average_basket_put_equal_weights():
    """Test average basket put with equal weights."""
    option = AverageBasketPut(K=100.0, T=1.0, n_assets=3, weights=None)
    # Terminal prices: [85, 90, 95], average = 90
    spots = jnp.array([85.0, 90.0, 95.0])

    payoff = option.payoff_terminal(spots)

    # Average = 90, Payoff = max(100 - 90, 0) = 10
    expected_payoff = 10.0
    assert jnp.isclose(payoff, expected_payoff)


def test_spread_option_call_itm():
    """Test spread call option in-the-money."""
    option = SpreadOption(K=5.0, T=1.0, is_call=True)
    # S1 = 110, S2 = 100, spread = 10
    spots = jnp.array([110.0, 100.0])

    payoff = option.payoff_terminal(spots)

    # Payoff = max(110 - 100 - 5, 0) = max(5, 0) = 5
    expected_payoff = 5.0
    assert jnp.isclose(payoff, expected_payoff)


def test_spread_option_call_otm():
    """Test spread call option out-of-the-money."""
    option = SpreadOption(K=15.0, T=1.0, is_call=True)
    # S1 = 110, S2 = 100, spread = 10
    spots = jnp.array([110.0, 100.0])

    payoff = option.payoff_terminal(spots)

    # Payoff = max(10 - 15, 0) = 0
    assert payoff == 0.0


def test_spread_option_put_itm():
    """Test spread put option in-the-money."""
    option = SpreadOption(K=15.0, T=1.0, is_call=False)
    # S1 = 110, S2 = 100, spread = 10
    spots = jnp.array([110.0, 100.0])

    payoff = option.payoff_terminal(spots)

    # Payoff = max(15 - 10, 0) = 5
    expected_payoff = 5.0
    assert jnp.isclose(payoff, expected_payoff)


def test_spread_option_negative_spread():
    """Test spread option with negative spread (S1 < S2)."""
    option = SpreadOption(K=0.0, T=1.0, is_call=True)
    # S1 = 95, S2 = 105, spread = -10
    spots = jnp.array([95.0, 105.0])

    payoff = option.payoff_terminal(spots)

    # Payoff = max(-10 - 0, 0) = 0
    assert payoff == 0.0


def test_rainbow_option_best_of_call():
    """Test rainbow option best-of call."""
    strikes = jnp.array([100.0, 105.0])
    option = RainbowOption(
        strikes=strikes, T=1.0, option_type="best_of", is_call=True, n_assets=2
    )
    # S1 = 110, S2 = 100
    spots = jnp.array([110.0, 100.0])

    payoff = option.payoff_terminal(spots)

    # Individual payoffs: (110-100)=10, (100-105)=-5
    # Best = max(10, -5) = 10
    # Payoff = max(10, 0) = 10
    expected_payoff = 10.0
    assert jnp.isclose(payoff, expected_payoff)


def test_rainbow_option_worst_of_call():
    """Test rainbow option worst-of call."""
    strikes = jnp.array([100.0, 105.0])
    option = RainbowOption(
        strikes=strikes, T=1.0, option_type="worst_of", is_call=True, n_assets=2
    )
    # S1 = 110, S2 = 100
    spots = jnp.array([110.0, 100.0])

    payoff = option.payoff_terminal(spots)

    # Individual payoffs: (110-100)=10, (100-105)=-5
    # Worst = min(10, -5) = -5
    # Payoff = max(-5, 0) = 0
    assert payoff == 0.0


def test_rainbow_option_best_of_put():
    """Test rainbow option best-of put."""
    strikes = jnp.array([100.0, 105.0])
    option = RainbowOption(
        strikes=strikes, T=1.0, option_type="best_of", is_call=False, n_assets=2
    )
    # S1 = 95, S2 = 100
    spots = jnp.array([95.0, 100.0])

    payoff = option.payoff_terminal(spots)

    # Individual payoffs: (100-95)=5, (105-100)=5
    # Best = max(5, 5) = 5
    # Payoff = max(5, 0) = 5
    expected_payoff = 5.0
    assert jnp.isclose(payoff, expected_payoff)


def test_worst_of_vs_best_of_relationship():
    """Test that worst-of <= best-of for same strike."""
    K = 100.0
    spots = jnp.array([90.0, 105.0, 110.0])

    worst_call = WorstOfCall(K=K, T=1.0, n_assets=3)
    best_call = BestOfCall(K=K, T=1.0, n_assets=3)

    worst_payoff = worst_call.payoff_terminal(spots)
    best_payoff = best_call.payoff_terminal(spots)

    # Best-of should always be >= worst-of
    assert best_payoff >= worst_payoff


def test_basket_options_non_negative():
    """Test that all basket option payoffs are non-negative."""
    spots = jnp.array([90.0, 100.0, 110.0])

    options = [
        WorstOfCall(K=100.0, T=1.0, n_assets=3),
        BestOfCall(K=100.0, T=1.0, n_assets=3),
        WorstOfPut(K=100.0, T=1.0, n_assets=3),
        BestOfPut(K=100.0, T=1.0, n_assets=3),
        AverageBasketCall(K=100.0, T=1.0, n_assets=3),
        AverageBasketPut(K=100.0, T=1.0, n_assets=3),
    ]

    for option in options:
        payoff = option.payoff_terminal(spots)
        assert payoff >= 0.0, f"Negative payoff for {option.__class__.__name__}"


def test_basket_option_supports_pde():
    """Test that basket options support PDE pricing (not path-dependent)."""
    option = WorstOfCall(K=100.0, T=1.0, n_assets=2)
    assert option.requires_path is False
    assert option.supports_pde is True
