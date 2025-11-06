"""Tests for composite FX options."""
from __future__ import annotations

import jax.numpy as jnp
import pytest

from neutryx.products.fx_complex.composite import (
    BasketFXOption,
    BestOfFXOption,
    RainbowFXOption,
    SpreadFXOption,
    WorstOfFXOption,
    basket_fx_approximation,
)


class TestBasketFXOption:
    """Test cases for Basket FX Options."""

    def test_basket_call_payoff(self):
        """Test basket call option payoff."""
        weights = jnp.array([0.5, 0.3, 0.2])
        basket = BasketFXOption(
            T=1.0,
            strike=1.20,
            weights=weights,
            is_call=True,
        )

        spots = jnp.array([1.25, 1.18, 1.22])
        payoff = basket.payoff_terminal(spots)

        # Basket value = 0.5*1.25 + 0.3*1.18 + 0.2*1.22 = 1.223
        # Payoff = max(1.223 - 1.20, 0) = 0.023
        expected_basket = jnp.dot(weights, spots)
        expected_payoff = max(expected_basket - 1.20, 0)

        assert abs(payoff - expected_payoff) < 1e-6, "Payoff should match basket formula"

    def test_basket_put_payoff(self):
        """Test basket put option payoff."""
        weights = jnp.array([0.6, 0.4])
        basket = BasketFXOption(
            T=1.0,
            strike=1.30,
            weights=weights,
            is_call=False,
        )

        spots = jnp.array([1.25, 1.28])
        payoff = basket.payoff_terminal(spots)

        # Basket = 0.6*1.25 + 0.4*1.28 = 1.262
        # Payoff = max(1.30 - 1.262, 0) = 0.038
        expected_basket = jnp.dot(weights, spots)
        expected_payoff = max(1.30 - expected_basket, 0)

        assert abs(payoff - expected_payoff) < 1e-6, "Put payoff should be correct"

    def test_basket_volatility(self):
        """Test basket volatility calculation."""
        weights = jnp.array([0.5, 0.5])
        vols = jnp.array([0.10, 0.12])
        correlation_matrix = jnp.array([[1.0, 0.7], [0.7, 1.0]])

        basket = BasketFXOption(
            T=1.0,
            strike=1.20,
            weights=weights,
            vols=vols,
            correlation_matrix=correlation_matrix,
        )

        basket_vol = basket.basket_volatility()

        # Should be between min and max individual vols
        assert 0.10 <= basket_vol <= 0.12, "Basket vol should be in reasonable range"

        # With perfect correlation, should be weighted average
        basket_perfect = BasketFXOption(
            T=1.0,
            strike=1.20,
            weights=weights,
            vols=vols,
            correlation_matrix=jnp.ones((2, 2)),
        )

        basket_vol_perfect = basket_perfect.basket_volatility()
        expected_perfect = jnp.dot(weights, vols)

        assert (
            abs(basket_vol_perfect - expected_perfect) < 1e-6
        ), "Perfect correlation should give weighted average"


class TestSpreadFXOption:
    """Test cases for Spread FX Options."""

    def test_spread_call_payoff(self):
        """Test spread call option payoff."""
        spread = SpreadFXOption(
            T=1.0,
            strike=0.05,
            is_call=True,
            spread_weight_1=1.0,
            spread_weight_2=-1.0,
        )

        spots = jnp.array([1.30, 1.20])
        payoff = spread.payoff_terminal(spots)

        # Spread = 1.30 - 1.20 = 0.10
        # Payoff = max(0.10 - 0.05, 0) = 0.05
        assert abs(payoff - 0.05) < 1e-6, "Spread payoff should be correct"

    def test_spread_put_payoff(self):
        """Test spread put option payoff."""
        spread = SpreadFXOption(
            T=1.0,
            strike=0.10,
            is_call=False,
        )

        spots = jnp.array([1.28, 1.22])
        payoff = spread.payoff_terminal(spots)

        # Spread = 1.28 - 1.22 = 0.06
        # Put payoff = max(0.10 - 0.06, 0) = 0.04
        assert abs(payoff - 0.04) < 1e-6, "Spread put payoff should be correct"

    def test_spread_volatility(self):
        """Test spread volatility calculation."""
        spread = SpreadFXOption(
            T=1.0,
            strike=0.0,
            vol_1=0.10,
            vol_2=0.12,
            correlation=0.5,
            spread_weight_1=1.0,
            spread_weight_2=-1.0,
        )

        spread_vol = spread.spread_volatility()

        # Var(X - Y) = Var(X) + Var(Y) - 2*Corr*σ_X*σ_Y
        expected_var = 0.10**2 + 0.12**2 - 2 * 0.5 * 0.10 * 0.12
        expected_vol = jnp.sqrt(expected_var)

        assert abs(spread_vol - expected_vol) < 1e-6, "Spread vol should match formula"


class TestBestOfFXOption:
    """Test cases for Best-of FX Options."""

    def test_best_of_assets_call(self):
        """Test best-of assets call payoff."""
        best_of = BestOfFXOption(
            T=1.0,
            strikes=jnp.array([1.20]),
            is_call=True,
            payoff_type="best_of_assets",
        )

        spots = jnp.array([1.25, 1.18, 1.22])
        payoff = best_of.payoff_terminal(spots)

        # Best spot = 1.25
        # Payoff = max(1.25 - 1.20, 0) = 0.05
        assert abs(payoff - 0.05) < 1e-6, "Best-of payoff should be correct"

    def test_best_of_options(self):
        """Test best-of options payoff."""
        strikes = jnp.array([1.20, 1.25, 1.22])
        best_of = BestOfFXOption(
            T=1.0,
            strikes=strikes,
            is_call=True,
            payoff_type="best_of_options",
        )

        spots = jnp.array([1.28, 1.30, 1.24])
        payoff = best_of.payoff_terminal(spots)

        # Individual payoffs: max(1.28-1.20, 0)=0.08, max(1.30-1.25, 0)=0.05, max(1.24-1.22, 0)=0.02
        # Best payoff = 0.08
        assert abs(payoff - 0.08) < 1e-6, "Best-of options should select max payoff"

    def test_best_of_put(self):
        """Test best-of put (actually worst-of for put)."""
        best_of = BestOfFXOption(
            T=1.0,
            strikes=jnp.array([1.30]),
            is_call=False,
            payoff_type="best_of_assets",
        )

        spots = jnp.array([1.25, 1.28, 1.22])
        payoff = best_of.payoff_terminal(spots)

        # For put, uses worst spot = 1.22
        # Payoff = max(1.30 - 1.22, 0) = 0.08
        assert payoff > 0, "Best-of put should have positive payoff"


class TestWorstOfFXOption:
    """Test cases for Worst-of FX Options."""

    def test_worst_of_assets_call(self):
        """Test worst-of assets call payoff."""
        worst_of = WorstOfFXOption(
            T=1.0,
            strikes=jnp.array([1.20]),
            is_call=True,
            payoff_type="worst_of_assets",
        )

        spots = jnp.array([1.28, 1.25, 1.30])
        payoff = worst_of.payoff_terminal(spots)

        # Worst spot = 1.25
        # Payoff = max(1.25 - 1.20, 0) = 0.05
        assert abs(payoff - 0.05) < 1e-6, "Worst-of call should use minimum spot"

    def test_worst_of_options(self):
        """Test worst-of options payoff."""
        strikes = jnp.array([1.20, 1.22, 1.18])
        worst_of = WorstOfFXOption(
            T=1.0,
            strikes=strikes,
            is_call=True,
            payoff_type="worst_of_options",
        )

        spots = jnp.array([1.28, 1.30, 1.25])
        payoff = worst_of.payoff_terminal(spots)

        # Individual payoffs: 0.08, 0.08, 0.07
        # Worst payoff = 0.07
        assert abs(payoff - 0.07) < 1e-6, "Worst-of options should select min payoff"

    def test_worst_of_otm(self):
        """Test worst-of option when OTM."""
        worst_of = WorstOfFXOption(
            T=1.0,
            strikes=jnp.array([1.30]),
            is_call=True,
            payoff_type="worst_of_assets",
        )

        spots = jnp.array([1.28, 1.25, 1.22])
        payoff = worst_of.payoff_terminal(spots)

        # Worst spot = 1.22 < 1.30, so OTM
        # Payoff = 0
        assert payoff == 0.0, "OTM worst-of should have zero payoff"


class TestRainbowFXOption:
    """Test cases for Rainbow FX Options."""

    def test_rainbow_best_performer(self):
        """Test rainbow option on best performer."""
        rainbow = RainbowFXOption(
            T=1.0,
            strikes=jnp.array([1.20, 1.22, 1.18]),
            ranks=jnp.array([1]),  # Best performer only
            weights=jnp.array([1.0]),
            is_call=True,
        )

        spots = jnp.array([1.28, 1.30, 1.24])
        payoff = rainbow.payoff_terminal(spots)

        # Best spot = 1.30, strike for rank 1 = strikes[rank-1] after sorting
        # After sorting descending: [1.30, 1.28, 1.24]
        # Rank 1 (index 0) = 1.30, corresponding strike (need to track original indices)
        # Simplified: should give positive payoff
        assert payoff > 0, "Rainbow on best performer should have positive payoff"

    def test_rainbow_2nd_best(self):
        """Test rainbow option on 2nd best performer."""
        rainbow = RainbowFXOption(
            T=1.0,
            strikes=jnp.array([1.20, 1.20, 1.20]),
            ranks=jnp.array([2]),  # 2nd best
            weights=jnp.array([1.0]),
            is_call=True,
        )

        spots = jnp.array([1.28, 1.25, 1.30])
        payoff = rainbow.payoff_terminal(spots)

        # Sorted descending: [1.30, 1.28, 1.25]
        # 2nd best = 1.28
        # Payoff = max(1.28 - 1.20, 0) = 0.08
        assert payoff > 0, "Rainbow on 2nd best should have positive payoff"

    def test_rainbow_multiple_ranks(self):
        """Test rainbow option on multiple ranks."""
        rainbow = RainbowFXOption(
            T=1.0,
            strikes=jnp.array([1.20, 1.20, 1.20]),
            ranks=jnp.array([1, 2]),  # Best and 2nd best
            weights=jnp.array([0.6, 0.4]),
            is_call=True,
        )

        spots = jnp.array([1.30, 1.25, 1.28])
        payoff = rainbow.payoff_terminal(spots)

        # Should be weighted average of best and 2nd best payoffs
        assert payoff > 0, "Rainbow on multiple ranks should have positive payoff"


class TestBasketApproximation:
    """Test basket FX approximation function."""

    def test_basket_approximation_basic(self):
        """Test basic basket approximation."""
        spots = jnp.array([1.20, 1.25])
        weights = jnp.array([0.5, 0.5])
        vols = jnp.array([0.10, 0.12])
        correlation_matrix = jnp.array([[1.0, 0.6], [0.6, 1.0]])
        domestic_rates = jnp.array([0.03, 0.03])
        foreign_rates = jnp.array([0.02, 0.025])

        price = basket_fx_approximation(
            spots=spots,
            weights=weights,
            strike=1.22,
            time_to_maturity=1.0,
            vols=vols,
            correlation_matrix=correlation_matrix,
            domestic_rates=domestic_rates,
            foreign_rates=foreign_rates,
            is_call=True,
        )

        assert price > 0, "Basket call should have positive price"
        assert price < 1.22, "Price should be less than strike for OTM basket"

    def test_basket_put_approximation(self):
        """Test basket put approximation."""
        spots = jnp.array([1.30, 1.28])
        weights = jnp.array([0.5, 0.5])
        vols = jnp.array([0.10, 0.10])
        correlation_matrix = jnp.eye(2)
        domestic_rates = jnp.zeros(2)
        foreign_rates = jnp.zeros(2)

        price = basket_fx_approximation(
            spots=spots,
            weights=weights,
            strike=1.20,
            time_to_maturity=1.0,
            vols=vols,
            correlation_matrix=correlation_matrix,
            domestic_rates=domestic_rates,
            foreign_rates=foreign_rates,
            is_call=False,
        )

        # Put with basket at 1.29 and strike at 1.20 should have low value
        assert price >= 0, "Put price should be non-negative"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
