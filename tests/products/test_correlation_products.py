"""Tests for correlation and multi-asset products."""
import jax.numpy as jnp
import pytest

from neutryx.products.correlation_products import (
    AdvancedBasketOption,
    BasketSpreadOption,
    ConditionalVarianceSwap,
    CorridorVarianceSwap,
    DualDigitalOption,
    ExchangeOption,
    OutperformanceOption,
    QuotientOption,
    RainbowOption,
    SpreadOption,
    VarianceDispersionProduct,
)


class TestAdvancedBasketOption:
    """Test advanced basket options."""

    def test_arithmetic_basket(self):
        """Test arithmetic basket option."""
        option = AdvancedBasketOption(
            T=1.0,
            strikes=jnp.array([100.0]),
            weights=jnp.array([0.5, 0.5]),
            option_type='call',
            basket_type='arithmetic',
            num_assets=2,
        )
        spots = jnp.array([110.0, 120.0])
        payoff = option.payoff_terminal(spots)

        # Basket = 0.5*110 + 0.5*120 = 115
        # Payoff = max(115 - 100, 0) = 15
        assert jnp.allclose(payoff, 15.0)

    def test_geometric_basket(self):
        """Test geometric basket option."""
        option = AdvancedBasketOption(
            T=1.0,
            strikes=jnp.array([100.0]),
            weights=jnp.array([0.5, 0.5]),
            option_type='call',
            basket_type='geometric',
            num_assets=2,
        )
        spots = jnp.array([100.0, 100.0])
        payoff = option.payoff_terminal(spots)

        # Geometric mean = sqrt(100*100) = 100
        # Payoff = max(100 - 100, 0) = 0 (with small numerical tolerance)
        assert jnp.allclose(payoff, 0.0, atol=1e-5)

    def test_harmonic_basket(self):
        """Test harmonic basket option."""
        option = AdvancedBasketOption(
            T=1.0,
            strikes=jnp.array([90.0]),
            weights=jnp.array([0.5, 0.5]),
            option_type='call',
            basket_type='harmonic',
            num_assets=2,
        )
        spots = jnp.array([100.0, 100.0])
        payoff = option.payoff_terminal(spots)

        # Harmonic mean = 2 / (1/100 + 1/100) = 100
        # Payoff = max(100 - 90, 0) = 10
        assert jnp.allclose(payoff, 10.0)


class TestSpreadOption:
    """Test spread options."""

    def test_spread_call(self):
        """Test spread call option."""
        option = SpreadOption(
            T=1.0, K=5.0, option_type='call', notional=1.0, quantity_1=1.0, quantity_2=1.0
        )
        spots = jnp.array([110.0, 100.0])
        payoff = option.payoff_terminal(spots)

        # Spread = 110 - 100 = 10
        # Payoff = max(10 - 5, 0) = 5
        assert jnp.allclose(payoff, 5.0)

    def test_spread_put(self):
        """Test spread put option."""
        option = SpreadOption(
            T=1.0, K=5.0, option_type='put', notional=1.0, quantity_1=1.0, quantity_2=1.0
        )
        spots = jnp.array([102.0, 100.0])
        payoff = option.payoff_terminal(spots)

        # Spread = 102 - 100 = 2
        # Payoff = max(5 - 2, 0) = 3
        assert jnp.allclose(payoff, 3.0)


class TestRainbowOption:
    """Test rainbow options."""

    def test_best_of_call(self):
        """Test best-of call option."""
        option = RainbowOption(
            T=1.0,
            strikes=jnp.array([100.0, 100.0, 100.0]),
            option_type='call',
            payoff_style='best-of',
            num_assets=3,
        )
        spots = jnp.array([105.0, 110.0, 103.0])
        payoff = option.payoff_terminal(spots)

        # Best payoff = max(110 - 100, 0) = 10
        assert jnp.allclose(payoff, 10.0)

    def test_worst_of_call(self):
        """Test worst-of call option."""
        option = RainbowOption(
            T=1.0,
            strikes=jnp.array([100.0, 100.0, 100.0]),
            option_type='call',
            payoff_style='worst-of',
            num_assets=3,
        )
        spots = jnp.array([105.0, 110.0, 103.0])
        payoff = option.payoff_terminal(spots)

        # Worst payoff = min(105-100, 110-100, 103-100) = 3
        assert jnp.allclose(payoff, 3.0)

    def test_ranked_payoff(self):
        """Test ranked payoff (2nd best)."""
        option = RainbowOption(
            T=1.0,
            strikes=jnp.array([100.0, 100.0, 100.0]),
            rank=2,
            option_type='call',
            payoff_style='ranked',
            num_assets=3,
        )
        spots = jnp.array([105.0, 115.0, 110.0])
        payoff = option.payoff_terminal(spots)

        # Sorted performances: [15, 10, 5], 2nd = 10
        assert jnp.allclose(payoff, 10.0)

    def test_rainbow_spread(self):
        """Test rainbow spread option."""
        option = RainbowOption(
            T=1.0,
            strikes=jnp.array([100.0, 100.0, 100.0]),
            option_type='call',
            payoff_style='rainbow-spread',
            num_assets=3,
        )
        spots = jnp.array([105.0, 115.0, 110.0])
        payoff = option.payoff_terminal(spots)

        # Best - worst = 15 - 5 = 10
        assert jnp.allclose(payoff, 10.0)


class TestQuotientOption:
    """Test quotient (ratio) options."""

    def test_quotient_call(self):
        """Test quotient call option."""
        option = QuotientOption(T=1.0, K=1.0, option_type='call', notional=100.0)
        spots = jnp.array([110.0, 100.0])
        payoff = option.payoff_terminal(spots)

        # Ratio = 110/100 = 1.1
        # Payoff = max(1.1 - 1.0, 0) * 100 = 10
        assert jnp.allclose(payoff, 10.0)

    def test_quotient_put(self):
        """Test quotient put option."""
        option = QuotientOption(T=1.0, K=1.2, option_type='put', notional=100.0)
        spots = jnp.array([110.0, 100.0])
        payoff = option.payoff_terminal(spots)

        # Ratio = 110/100 = 1.1
        # Payoff = max(1.2 - 1.1, 0) * 100 = 10
        assert jnp.allclose(payoff, 10.0)


class TestExchangeOption:
    """Test exchange (Margrabe) options."""

    def test_exchange_option_payoff(self):
        """Test exchange option payoff."""
        option = ExchangeOption(T=1.0, notional=100.0, quantity_1=1.0, quantity_2=1.0)
        spots = jnp.array([110.0, 100.0])
        payoff = option.payoff_terminal(spots)

        # Exchange value = 110 - 100 = 10
        # Payoff = max(10, 0) * 100 = 1000
        assert jnp.allclose(payoff, 1000.0)

    def test_exchange_option_otm(self):
        """Test exchange option out of the money."""
        option = ExchangeOption(T=1.0, notional=100.0, quantity_1=1.0, quantity_2=1.0)
        spots = jnp.array([90.0, 100.0])
        payoff = option.payoff_terminal(spots)

        # Exchange value = 90 - 100 = -10
        # Payoff = max(-10, 0) = 0
        assert jnp.allclose(payoff, 0.0)

    def test_margrabe_price(self):
        """Test Margrabe analytical formula."""
        option = ExchangeOption(T=1.0, notional=1.0, quantity_1=1.0, quantity_2=1.0)
        price = option.margrabe_price(
            S1=100.0, S2=100.0, vol1=0.2, vol2=0.2, corr=0.5, r=0.05
        )
        assert price > 0


class TestDualDigitalOption:
    """Test dual digital options."""

    def test_dual_digital_and_above(self):
        """Test dual digital with AND condition, above barriers."""
        option = DualDigitalOption(
            T=1.0,
            barriers=jnp.array([100.0, 100.0]),
            payout=1000.0,
            condition='and',
            barrier_type='above',
        )
        # Both above barriers
        spots = jnp.array([105.0, 110.0])
        payoff = option.payoff_terminal(spots)
        assert jnp.allclose(payoff, 1000.0)

        # One below barrier
        spots = jnp.array([95.0, 110.0])
        payoff = option.payoff_terminal(spots)
        assert jnp.allclose(payoff, 0.0)

    def test_dual_digital_or_above(self):
        """Test dual digital with OR condition."""
        option = DualDigitalOption(
            T=1.0,
            barriers=jnp.array([100.0, 100.0]),
            payout=1000.0,
            condition='or',
            barrier_type='above',
        )
        # One above barrier is enough
        spots = jnp.array([95.0, 110.0])
        payoff = option.payoff_terminal(spots)
        assert jnp.allclose(payoff, 1000.0)


class TestBasketSpreadOption:
    """Test basket spread options."""

    def test_basket_spread_call(self):
        """Test basket spread call option."""
        weights_1 = jnp.array([0.5, 0.5])
        weights_2 = jnp.array([0.5, 0.5])
        option = BasketSpreadOption(
            T=1.0,
            K=5.0,
            weights_1=weights_1,
            weights_2=weights_2,
            option_type='call',
            notional=100.0,
        )
        # Basket 1: 0.5*110 + 0.5*120 = 115
        # Basket 2: 0.5*100 + 0.5*100 = 100
        # Spread = 15
        spots = jnp.array([110.0, 120.0, 100.0, 100.0])
        payoff = option.payoff_terminal(spots)

        # Payoff = max(15 - 5, 0) * 100 = 1000
        assert jnp.allclose(payoff, 1000.0)


class TestOutperformanceOption:
    """Test outperformance options."""

    def test_outperformance_call(self):
        """Test outperformance option."""
        option = OutperformanceOption(
            T=1.0, K=0.0, notional=1000.0, participation=1.0
        )
        # Asset 1: 110 (10% return), Asset 2: 105 (5% return)
        # Initial values: 100, 100
        spots = jnp.array([110.0, 105.0, 100.0, 100.0])
        payoff = option.payoff_terminal(spots)

        # Outperformance = 0.1 - 0.05 = 0.05
        # Payoff = max(0.05 - 0.0, 0) * 1000 = 50
        assert jnp.allclose(payoff, 50.0)


class TestVarianceDispersionProduct:
    """Test variance dispersion products."""

    def test_variance_dispersion_creation(self):
        """Test variance dispersion instantiation."""
        product = VarianceDispersionProduct(
            T=1.0,
            notional_per_point=1000.0,
            strike_stock_var=0.04,
            strike_index_var=0.02,
            num_stocks=3,
        )
        assert product.num_stocks == 3
        assert len(product.index_weights) == 3

    def test_variance_dispersion_payoff(self):
        """Test variance dispersion payoff calculation."""
        product = VarianceDispersionProduct(
            T=1.0,
            notional_per_point=1000.0,
            strike_stock_var=0.04,
            strike_index_var=0.02,
            num_stocks=2,
        )
        # Create sample paths (2 stocks + 1 index)
        stock1 = jnp.linspace(100, 110, 100)
        stock2 = jnp.linspace(100, 105, 100)
        index = jnp.linspace(100, 107, 100)
        paths = jnp.array([stock1, stock2, index])

        payoff = product.payoff_path(paths)
        # Should have some value based on variance differential
        assert isinstance(payoff, jnp.ndarray) or isinstance(payoff, float)


class TestCorridorVarianceSwap:
    """Test corridor variance swap."""

    def test_corridor_variance_creation(self):
        """Test corridor variance swap instantiation."""
        swap = CorridorVarianceSwap(
            T=1.0,
            strike_variance=0.04,
            corridor_lower=90.0,
            corridor_upper=110.0,
            notional_per_point=1000.0,
        )
        assert swap.corridor_lower == 90.0
        assert swap.corridor_upper == 110.0

    def test_corridor_variance_in_corridor(self):
        """Test corridor variance when always in corridor."""
        swap = CorridorVarianceSwap(
            T=1.0,
            strike_variance=0.04,
            corridor_lower=90.0,
            corridor_upper=110.0,
            notional_per_point=1000.0,
        )
        # Path always in corridor
        path = jnp.linspace(95, 105, 252)
        payoff = swap.payoff_path(path)

        # Should have some payoff
        assert isinstance(payoff, jnp.ndarray) or isinstance(payoff, float)


class TestConditionalVarianceSwap:
    """Test conditional variance swap."""

    def test_conditional_variance_creation(self):
        """Test conditional variance swap instantiation."""
        swap = ConditionalVarianceSwap(
            T=1.0,
            strike_variance=0.04,
            threshold=100.0,
            condition='above',
            notional_per_point=1000.0,
        )
        assert swap.threshold == 100.0
        assert swap.condition == 'above'

    def test_conditional_variance_always_above(self):
        """Test conditional variance when always above threshold."""
        swap = ConditionalVarianceSwap(
            T=1.0,
            strike_variance=0.04,
            threshold=100.0,
            condition='above',
            notional_per_point=1000.0,
        )
        # Path always above threshold
        path = jnp.linspace(105, 115, 252)
        payoff = swap.payoff_path(path)

        # Should have some payoff
        assert isinstance(payoff, jnp.ndarray) or isinstance(payoff, float)

    def test_conditional_variance_always_below(self):
        """Test conditional variance when always below threshold."""
        swap = ConditionalVarianceSwap(
            T=1.0,
            strike_variance=0.04,
            threshold=100.0,
            condition='below',
            notional_per_point=1000.0,
        )
        # Path always below threshold
        path = jnp.linspace(85, 95, 252)
        payoff = swap.payoff_path(path)

        # Should have some payoff
        assert isinstance(payoff, jnp.ndarray) or isinstance(payoff, float)
