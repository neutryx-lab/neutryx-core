"""Advanced correlation and multi-asset products.

This module implements sophisticated correlation products:
- Advanced basket options
- Dispersion trading products
- Complete rainbow options suite
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

from neutryx.products.base import Product, PathProduct

Array = jnp.ndarray


@dataclass
class AdvancedBasketOption(Product):
    """Advanced basket option with various weighting schemes.

    Args:
        T: Time to maturity (years)
        strikes: Strike prices for each asset (or single basket strike)
        weights: Basket weights (can be dynamic)
        option_type: 'call' or 'put'
        basket_type: 'arithmetic', 'geometric', 'harmonic', 'quadratic'
        notional: Notional amount
        num_assets: Number of assets in basket
    """

    T: float
    strikes: Array
    weights: Array
    option_type: Literal['call', 'put'] = 'call'
    basket_type: Literal['arithmetic', 'geometric', 'harmonic', 'quadratic'] = 'arithmetic'
    notional: float = 1.0
    num_assets: int = 2

    def calculate_basket_value(self, spots: Array) -> float:
        """Calculate basket value based on basket type.

        Args:
            spots: Array of spot prices

        Returns:
            Basket value
        """
        if self.basket_type == 'arithmetic':
            # Standard weighted average
            basket = jnp.sum(self.weights * spots)

        elif self.basket_type == 'geometric':
            # Geometric average
            log_basket = jnp.sum(self.weights * jnp.log(spots))
            basket = jnp.exp(log_basket)

        elif self.basket_type == 'harmonic':
            # Harmonic average
            basket = 1.0 / jnp.sum(self.weights / spots)

        elif self.basket_type == 'quadratic':
            # Quadratic basket (used in some structured products)
            basket = jnp.sqrt(jnp.sum(self.weights * spots**2))

        else:
            basket = jnp.sum(self.weights * spots)

        return basket

    def payoff_terminal(self, spots: Array) -> Array:
        """Calculate payoff given terminal spot prices.

        Args:
            spots: Array of terminal prices for each asset

        Returns:
            Basket option payoff
        """
        basket_value = self.calculate_basket_value(spots)

        # Use first strike if strikes is array, otherwise use scalar
        strike = self.strikes[0] if self.strikes.ndim > 0 else self.strikes

        if self.option_type == 'call':
            payoff = jnp.maximum(basket_value - strike, 0.0)
        else:
            payoff = jnp.maximum(strike - basket_value, 0.0)

        return self.notional * payoff


@dataclass
class SpreadOption(Product):
    """Spread option on the difference between two assets.

    Args:
        T: Time to maturity (years)
        K: Strike spread
        option_type: 'call' or 'put'
        notional: Notional amount
        quantity_1: Quantity of asset 1
        quantity_2: Quantity of asset 2
    """

    T: float
    K: float
    option_type: Literal['call', 'put'] = 'call'
    notional: float = 1.0
    quantity_1: float = 1.0
    quantity_2: float = 1.0

    def payoff_terminal(self, spots: Array) -> Array:
        """Calculate spread option payoff.

        Args:
            spots: Array [S1, S2] of terminal prices

        Returns:
            Spread option payoff
        """
        spread = self.quantity_1 * spots[0] - self.quantity_2 * spots[1]

        if self.option_type == 'call':
            payoff = jnp.maximum(spread - self.K, 0.0)
        else:
            payoff = jnp.maximum(self.K - spread, 0.0)

        return self.notional * payoff


@dataclass
class RainbowOption(Product):
    """Complete rainbow option suite.

    Options on ranked asset performances.

    Args:
        T: Time to maturity (years)
        strikes: Strike for each asset
        rank: Which rank to pay (1=best, 2=second best, etc.)
        option_type: 'call' or 'put'
        payoff_style: 'best-of', 'worst-of', 'ranked', 'rainbow-spread'
        notional: Notional amount
        num_assets: Number of assets
    """

    T: float
    strikes: Array
    rank: int = 1
    option_type: Literal['call', 'put'] = 'call'
    payoff_style: Literal['best-of', 'worst-of', 'ranked', 'rainbow-spread'] = 'ranked'
    notional: float = 1.0
    num_assets: int = 3

    def payoff_terminal(self, spots: Array) -> Array:
        """Calculate rainbow option payoff.

        Args:
            spots: Array of terminal prices for each asset

        Returns:
            Rainbow option payoff
        """
        if self.payoff_style == 'best-of':
            # Best performing asset
            if self.option_type == 'call':
                individual_payoffs = jnp.maximum(spots - self.strikes, 0.0)
            else:
                individual_payoffs = jnp.maximum(self.strikes - spots, 0.0)
            payoff = jnp.max(individual_payoffs)

        elif self.payoff_style == 'worst-of':
            # Worst performing asset
            if self.option_type == 'call':
                individual_payoffs = jnp.maximum(spots - self.strikes, 0.0)
            else:
                individual_payoffs = jnp.maximum(self.strikes - spots, 0.0)
            payoff = jnp.min(individual_payoffs)

        elif self.payoff_style == 'ranked':
            # Specific rank
            if self.option_type == 'call':
                performances = spots - self.strikes
            else:
                performances = self.strikes - spots

            # Sort descending for calls, ascending for puts
            sorted_perfs = jnp.sort(performances)
            if self.option_type == 'call':
                sorted_perfs = sorted_perfs[::-1]

            ranked_perf = sorted_perfs[self.rank - 1]
            payoff = jnp.maximum(ranked_perf, 0.0)

        elif self.payoff_style == 'rainbow-spread':
            # Spread between best and worst
            if self.option_type == 'call':
                performances = spots - self.strikes
            else:
                performances = self.strikes - spots

            spread = jnp.max(performances) - jnp.min(performances)
            payoff = jnp.maximum(spread, 0.0)

        else:
            payoff = 0.0

        return self.notional * payoff


@dataclass
class QuotientOption(Product):
    """Quotient option - option on ratio of two assets.

    Args:
        T: Time to maturity (years)
        K: Strike ratio
        option_type: 'call' or 'put'
        notional: Notional amount
    """

    T: float
    K: float
    option_type: Literal['call', 'put'] = 'call'
    notional: float = 1.0

    def payoff_terminal(self, spots: Array) -> Array:
        """Calculate quotient option payoff.

        Args:
            spots: Array [S1, S2] of terminal prices

        Returns:
            Quotient option payoff
        """
        ratio = spots[0] / spots[1]

        if self.option_type == 'call':
            payoff = jnp.maximum(ratio - self.K, 0.0)
        else:
            payoff = jnp.maximum(self.K - ratio, 0.0)

        return self.notional * payoff


@dataclass
class ExchangeOption(Product):
    """Exchange option (Margrabe) - option to exchange one asset for another.

    Args:
        T: Time to maturity (years)
        notional: Notional amount
        quantity_1: Quantity of asset to receive
        quantity_2: Quantity of asset to give
    """

    T: float
    notional: float = 1.0
    quantity_1: float = 1.0
    quantity_2: float = 1.0

    def payoff_terminal(self, spots: Array) -> Array:
        """Calculate exchange option payoff.

        Args:
            spots: Array [S1, S2] of terminal prices

        Returns:
            Exchange option payoff
        """
        exchange_value = self.quantity_1 * spots[0] - self.quantity_2 * spots[1]
        payoff = jnp.maximum(exchange_value, 0.0)

        return self.notional * payoff

    def margrabe_price(self, S1: float, S2: float, vol1: float, vol2: float,
                      corr: float, r: float) -> float:
        """Analytical Margrabe formula for exchange option.

        Args:
            S1: Current price of asset 1
            S2: Current price of asset 2
            vol1: Volatility of asset 1
            vol2: Volatility of asset 2
            corr: Correlation between assets
            r: Risk-free rate

        Returns:
            Exchange option price
        """
        # Composite volatility
        sigma = jnp.sqrt(vol1**2 + vol2**2 - 2*corr*vol1*vol2)

        # Margrabe formula
        d1 = (jnp.log(S1/S2) + 0.5*sigma**2*self.T) / (sigma*jnp.sqrt(self.T))
        d2 = d1 - sigma*jnp.sqrt(self.T)

        price = self.quantity_1 * S1 * norm.cdf(d1) - self.quantity_2 * S2 * norm.cdf(d2)

        return self.notional * price


@dataclass
class DualDigitalOption(Product):
    """Dual digital option - digital payoff on two assets.

    Args:
        T: Time to maturity (years)
        barriers: Barrier levels for each asset
        payout: Digital payout amount
        condition: 'and' (both barriers hit) or 'or' (either barrier hit)
        barrier_type: 'above' or 'below'
        notional: Notional amount
    """

    T: float
    barriers: Array
    payout: float = 1.0
    condition: Literal['and', 'or'] = 'and'
    barrier_type: Literal['above', 'below'] = 'above'
    notional: float = 1.0

    def payoff_terminal(self, spots: Array) -> Array:
        """Calculate dual digital payoff.

        Args:
            spots: Array of terminal prices

        Returns:
            Digital payoff
        """
        if self.barrier_type == 'above':
            hits = spots > self.barriers
        else:
            hits = spots < self.barriers

        if self.condition == 'and':
            trigger = jnp.all(hits)
        else:  # 'or'
            trigger = jnp.any(hits)

        payoff = jnp.where(trigger, self.payout, 0.0)

        return self.notional * payoff


@dataclass
class BasketSpreadOption(Product):
    """Spread option between two baskets.

    Args:
        T: Time to maturity (years)
        K: Strike spread
        weights_1: Weights for first basket
        weights_2: Weights for second basket
        option_type: 'call' or 'put'
        notional: Notional amount
    """

    T: float
    K: float
    weights_1: Array
    weights_2: Array
    option_type: Literal['call', 'put'] = 'call'
    notional: float = 1.0

    def payoff_terminal(self, spots: Array) -> Array:
        """Calculate basket spread option payoff.

        Args:
            spots: Array of all asset prices (concatenated for both baskets)

        Returns:
            Basket spread payoff
        """
        n1 = len(self.weights_1)
        basket_1 = jnp.sum(self.weights_1 * spots[:n1])
        basket_2 = jnp.sum(self.weights_2 * spots[n1:])

        spread = basket_1 - basket_2

        if self.option_type == 'call':
            payoff = jnp.maximum(spread - self.K, 0.0)
        else:
            payoff = jnp.maximum(self.K - spread, 0.0)

        return self.notional * payoff


@dataclass
class OutperformanceOption(Product):
    """Outperformance option - pays when one asset outperforms another.

    Args:
        T: Time to maturity (years)
        K: Outperformance threshold
        notional: Notional amount
        participation: Participation rate in outperformance
    """

    T: float
    K: float = 0.0
    notional: float = 1.0
    participation: float = 1.0

    def payoff_terminal(self, spots: Array) -> Array:
        """Calculate outperformance payoff.

        Args:
            spots: Array [S1, S2, initial_S1, initial_S2]

        Returns:
            Outperformance payoff
        """
        # Calculate returns
        if spots.shape[0] >= 4:
            return_1 = (spots[0] - spots[2]) / spots[2]
            return_2 = (spots[1] - spots[3]) / spots[3]
        else:
            # Assume equal initial values
            return_1 = spots[0] - 1.0
            return_2 = spots[1] - 1.0

        outperformance = return_1 - return_2

        payoff = jnp.maximum(outperformance - self.K, 0.0)

        return self.notional * self.participation * payoff


@dataclass
class VarianceDispersionProduct(PathProduct):
    """Variance dispersion product.

    Long individual stock variance, short index variance.

    Args:
        T: Time to maturity (years)
        notional_per_point: Notional per variance point
        strike_stock_var: Strike for average stock variance
        strike_index_var: Strike for index variance
        num_stocks: Number of stocks
        index_weights: Weights of stocks in index
    """

    T: float
    notional_per_point: float = 1000.0
    strike_stock_var: float = 0.04
    strike_index_var: float = 0.02
    num_stocks: int = 50
    index_weights: Optional[Array] = None

    def __post_init__(self):
        if self.index_weights is None:
            object.__setattr__(self, 'index_weights',
                             jnp.ones(self.num_stocks) / self.num_stocks)

    def payoff_path(self, paths: Array) -> Array:
        """Calculate dispersion payoff.

        Args:
            paths: Array of shape (num_stocks+1, num_steps)
                  Last row is index, others are individual stocks

        Returns:
            Dispersion payoff
        """
        if paths.ndim != 2:
            return 0.0

        # Calculate realized variances
        stock_paths = paths[:-1, :]
        index_path = paths[-1, :]

        # Returns
        stock_returns = jnp.diff(jnp.log(stock_paths), axis=1)
        index_returns = jnp.diff(jnp.log(index_path))

        # Realized variances (annualized)
        stock_vars = jnp.var(stock_returns, axis=1) * 252
        index_var = jnp.var(index_returns) * 252

        # Average stock variance
        avg_stock_var = jnp.mean(stock_vars)

        # Long stock variance leg
        stock_var_pnl = (avg_stock_var - self.strike_stock_var)

        # Short index variance leg
        index_var_pnl = -(index_var - self.strike_index_var)

        # Total PnL
        total_pnl = stock_var_pnl + index_var_pnl

        return self.notional_per_point * total_pnl * 10000  # Convert to bps


@dataclass
class CorridorVarianceSwap(PathProduct):
    """Corridor variance swap - variance swap with corridor.

    Only counts variance when underlying is within corridor.

    Args:
        T: Time to maturity (years)
        strike_variance: Strike variance
        corridor_lower: Lower corridor boundary
        corridor_upper: Upper corridor boundary
        notional_per_point: Notional per variance point
    """

    T: float
    strike_variance: float
    corridor_lower: float
    corridor_upper: float
    notional_per_point: float = 1000.0

    def payoff_path(self, path: Array) -> Array:
        """Calculate corridor variance swap payoff.

        Args:
            path: Price path

        Returns:
            Corridor variance swap payoff
        """
        # Check which observations are in corridor
        in_corridor = (path >= self.corridor_lower) & (path <= self.corridor_upper)

        # Calculate returns
        log_returns = jnp.diff(jnp.log(path))

        # Only include returns when in corridor
        corridor_returns = jnp.where(in_corridor[:-1], log_returns, 0.0)

        # Realized variance (annualized)
        corridor_fraction = jnp.mean(in_corridor)
        if corridor_fraction > 0:
            realized_var = jnp.sum(corridor_returns**2) / len(corridor_returns) * 252
        else:
            realized_var = 0.0

        # Payoff
        variance_diff = realized_var - self.strike_variance

        return self.notional_per_point * variance_diff * 10000  # Convert to bps


@dataclass
class ConditionalVarianceSwap(PathProduct):
    """Conditional variance swap.

    Variance swap conditional on asset being above/below threshold.

    Args:
        T: Time to maturity (years)
        strike_variance: Strike variance
        threshold: Threshold level
        condition: 'above' or 'below'
        notional_per_point: Notional per variance point
    """

    T: float
    strike_variance: float
    threshold: float
    condition: Literal['above', 'below'] = 'above'
    notional_per_point: float = 1000.0

    def payoff_path(self, path: Array) -> Array:
        """Calculate conditional variance swap payoff.

        Args:
            path: Price path

        Returns:
            Conditional variance swap payoff
        """
        # Determine which observations meet condition
        if self.condition == 'above':
            meets_condition = path >= self.threshold
        else:
            meets_condition = path <= self.threshold

        # Calculate returns only when condition is met
        log_returns = jnp.diff(jnp.log(path))
        conditional_returns = jnp.where(meets_condition[:-1], log_returns, 0.0)

        # Realized variance
        condition_fraction = jnp.mean(meets_condition)
        if condition_fraction > 0:
            realized_var = jnp.sum(conditional_returns**2) / len(conditional_returns) * 252
        else:
            realized_var = 0.0

        # Payoff
        variance_diff = realized_var - self.strike_variance

        return self.notional_per_point * variance_diff * 10000
