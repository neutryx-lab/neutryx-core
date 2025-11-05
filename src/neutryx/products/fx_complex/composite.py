"""Composite FX options - multi-asset options on FX rates.

Implements options on baskets, spreads, and combinations of FX pairs:
- Basket options: weighted average of FX rates
- Spread options: difference between two FX rates
- Best-of/Worst-of: max/min of multiple FX rates
- Rainbow options: complex multi-asset payoffs
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import jax.numpy as jnp
from jax import jit

from ..base import Product, PathProduct

Array = jnp.ndarray


class CompositeType(Enum):
    """Type of composite payoff."""

    BASKET = "basket"
    SPREAD = "spread"
    BEST_OF = "best_of"
    WORST_OF = "worst_of"
    RAINBOW = "rainbow"


@dataclass
class BasketFXOption(Product):
    """Basket option on multiple FX rates.

    Option on a weighted average of FX rates. Common for portfolios
    of currency exposures.

    Attributes:
        T: Maturity in years
        strike: Strike price for basket
        weights: Array of weights for each FX rate (must sum to 1.0)
        is_call: True for call, False for put
        domestic_rates: Array of domestic rates for each currency
        foreign_rates: Array of foreign rates for each currency
        vols: Array of volatilities for each FX rate
        correlation_matrix: Correlation matrix between FX rates
    """

    strike: float
    weights: Array
    is_call: bool = True
    domestic_rates: Array | None = None
    foreign_rates: Array | None = None
    vols: Array | None = None
    correlation_matrix: Array | None = None

    def __post_init__(self):
        """Initialize default parameters."""
        n = len(self.weights)

        if self.domestic_rates is None:
            self.domestic_rates = jnp.zeros(n)
        if self.foreign_rates is None:
            self.foreign_rates = jnp.zeros(n)
        if self.vols is None:
            self.vols = jnp.full(n, 0.10)
        if self.correlation_matrix is None:
            self.correlation_matrix = jnp.eye(n)

    @property
    def requires_path(self) -> bool:
        return False

    def payoff_terminal(self, spots: Array) -> Array:
        """Calculate basket option payoff.

        Args:
            spots: Array of FX spot rates at maturity

        Returns:
            Option payoff on weighted basket
        """
        # Calculate basket value
        basket_value = jnp.dot(self.weights, spots)

        # Option payoff
        if self.is_call:
            return jnp.maximum(basket_value - self.strike, 0.0)
        else:
            return jnp.maximum(self.strike - basket_value, 0.0)

    def basket_volatility(self) -> float:
        """Calculate effective volatility of basket.

        Returns:
            Basket volatility accounting for correlations
        """
        # Covariance matrix
        vol_matrix = jnp.diag(self.vols)
        cov_matrix = vol_matrix @ self.correlation_matrix @ vol_matrix

        # Basket variance = w^T Σ w
        basket_var = self.weights @ cov_matrix @ self.weights

        return float(jnp.sqrt(basket_var))


@dataclass
class SpreadFXOption(Product):
    """Spread option on difference between two FX rates.

    Option on the spread between two FX rates. Popular for
    relative value trades.

    Attributes:
        T: Maturity in years
        strike: Strike on spread
        is_call: True for call on spread, False for put
        domestic_rate_1: Domestic rate for FX1
        domestic_rate_2: Domestic rate for FX2
        foreign_rate_1: Foreign rate for FX1
        foreign_rate_2: Foreign rate for FX2
        vol_1: Volatility of FX1
        vol_2: Volatility of FX2
        correlation: Correlation between FX1 and FX2
        spread_weight_1: Weight on FX1 in spread (default 1.0)
        spread_weight_2: Weight on FX2 in spread (default -1.0)
    """

    strike: float
    is_call: bool = True
    domestic_rate_1: float = 0.0
    domestic_rate_2: float = 0.0
    foreign_rate_1: float = 0.0
    foreign_rate_2: float = 0.0
    vol_1: float = 0.10
    vol_2: float = 0.10
    correlation: float = 0.5
    spread_weight_1: float = 1.0
    spread_weight_2: float = -1.0

    @property
    def requires_path(self) -> bool:
        return False

    def payoff_terminal(self, spots: Array) -> Array:
        """Calculate spread option payoff.

        Args:
            spots: Array [spot_1, spot_2]

        Returns:
            Option payoff on spread
        """
        spot_1, spot_2 = spots[0], spots[1]

        # Spread value
        spread = self.spread_weight_1 * spot_1 + self.spread_weight_2 * spot_2

        # Option payoff
        if self.is_call:
            return jnp.maximum(spread - self.strike, 0.0)
        else:
            return jnp.maximum(self.strike - spread, 0.0)

    def spread_volatility(self) -> float:
        """Calculate volatility of spread.

        Returns:
            Spread volatility accounting for correlation
        """
        # Variance of weighted spread
        var_spread = (
            self.spread_weight_1**2 * self.vol_1**2
            + self.spread_weight_2**2 * self.vol_2**2
            + 2
            * self.spread_weight_1
            * self.spread_weight_2
            * self.correlation
            * self.vol_1
            * self.vol_2
        )

        return float(jnp.sqrt(var_spread))


@dataclass
class BestOfFXOption(PathProduct):
    """Best-of option on multiple FX rates.

    Option that pays based on the best performing FX rate.
    Can be structured as:
    - Best-of call: max(max(S_i) - K, 0)
    - Best-of on calls: max(max(S_i - K_i), 0)

    Attributes:
        T: Maturity in years
        strikes: Array of strikes (one per FX rate, or single strike)
        is_call: True for call, False for put
        payoff_type: 'best_of_assets' or 'best_of_options'
    """

    strikes: Array
    is_call: bool = True
    payoff_type: str = "best_of_assets"

    @property
    def requires_path(self) -> bool:
        return False

    def payoff_terminal(self, spots: Array) -> Array:
        """Calculate best-of option payoff.

        Args:
            spots: Array of FX spot rates at maturity

        Returns:
            Best-of payoff
        """
        if self.payoff_type == "best_of_assets":
            # Best of assets, then apply option
            if self.is_call:
                best_spot = jnp.max(spots)
                strike = self.strikes[0] if len(self.strikes) == 1 else jnp.mean(self.strikes)
                return jnp.maximum(best_spot - strike, 0.0)
            else:
                worst_spot = jnp.min(spots)
                strike = self.strikes[0] if len(self.strikes) == 1 else jnp.mean(self.strikes)
                return jnp.maximum(strike - worst_spot, 0.0)
        else:
            # Best of options on individual assets
            if self.is_call:
                payoffs = jnp.maximum(spots - self.strikes, 0.0)
            else:
                payoffs = jnp.maximum(self.strikes - spots, 0.0)

            return jnp.max(payoffs)

    def payoff_path(self, paths: Array) -> Array:
        """Best-of using terminal values."""
        if paths.ndim == 1:
            return self.payoff_terminal(paths[-1:])
        else:
            # Multiple paths: use terminal values
            terminal_spots = paths[:, -1]
            return self.payoff_terminal(terminal_spots)


@dataclass
class WorstOfFXOption(PathProduct):
    """Worst-of option on multiple FX rates.

    Option that pays based on the worst performing FX rate.

    Attributes:
        T: Maturity in years
        strikes: Array of strikes (one per FX rate)
        is_call: True for call, False for put
        payoff_type: 'worst_of_assets' or 'worst_of_options'
    """

    strikes: Array
    is_call: bool = True
    payoff_type: str = "worst_of_assets"

    @property
    def requires_path(self) -> bool:
        return False

    def payoff_terminal(self, spots: Array) -> Array:
        """Calculate worst-of option payoff.

        Args:
            spots: Array of FX spot rates at maturity

        Returns:
            Worst-of payoff
        """
        if self.payoff_type == "worst_of_assets":
            # Worst of assets, then apply option
            if self.is_call:
                worst_spot = jnp.min(spots)
                strike = self.strikes[0] if len(self.strikes) == 1 else jnp.mean(self.strikes)
                return jnp.maximum(worst_spot - strike, 0.0)
            else:
                best_spot = jnp.max(spots)
                strike = self.strikes[0] if len(self.strikes) == 1 else jnp.mean(self.strikes)
                return jnp.maximum(strike - best_spot, 0.0)
        else:
            # Worst of options on individual assets
            if self.is_call:
                payoffs = jnp.maximum(spots - self.strikes, 0.0)
            else:
                payoffs = jnp.maximum(self.strikes - spots, 0.0)

            return jnp.min(payoffs)

    def payoff_path(self, paths: Array) -> Array:
        """Worst-of using terminal values."""
        if paths.ndim == 1:
            return self.payoff_terminal(paths[-1:])
        else:
            terminal_spots = paths[:, -1]
            return self.payoff_terminal(terminal_spots)


@dataclass
class RainbowFXOption(PathProduct):
    """Rainbow option with complex multi-asset payoff.

    Flexible structure for complex multi-FX payoffs:
    - nth-best performance
    - Combination payoffs
    - Conditional structures

    Attributes:
        T: Maturity in years
        strikes: Array of strikes
        ranks: Array of ranks to pay (e.g., [1, 2] for best and 2nd best)
        weights: Weights for each rank
        is_call: True for call, False for put
    """

    strikes: Array
    ranks: Array  # Which ranks to include (1=best, 2=2nd best, etc.)
    weights: Array | None = None
    is_call: bool = True

    def __post_init__(self):
        """Initialize weights if not provided."""
        if self.weights is None:
            self.weights = jnp.ones(len(self.ranks)) / len(self.ranks)

    @property
    def requires_path(self) -> bool:
        return False

    def payoff_terminal(self, spots: Array) -> Array:
        """Calculate rainbow option payoff.

        Args:
            spots: Array of FX spot rates

        Returns:
            Rainbow payoff based on ranked performance
        """
        # Sort spots (descending for best first)
        if self.is_call:
            sorted_spots = jnp.sort(spots)[::-1]
        else:
            sorted_spots = jnp.sort(spots)

        # Get ranks (convert from 1-indexed to 0-indexed)
        rank_indices = (self.ranks - 1).astype(int)

        # Clip indices to valid range
        rank_indices = jnp.clip(rank_indices, 0, len(spots) - 1)

        # Select ranked spots
        ranked_spots = sorted_spots[rank_indices]

        # Calculate payoffs for ranked spots
        if self.is_call:
            individual_payoffs = jnp.maximum(ranked_spots - self.strikes[rank_indices], 0.0)
        else:
            individual_payoffs = jnp.maximum(self.strikes[rank_indices] - ranked_spots, 0.0)

        # Weighted sum
        total_payoff = jnp.dot(self.weights, individual_payoffs)

        return total_payoff

    def payoff_path(self, paths: Array) -> Array:
        """Rainbow using terminal values."""
        if paths.ndim == 1:
            return self.payoff_terminal(paths[-1:])
        else:
            terminal_spots = paths[:, -1]
            return self.payoff_terminal(terminal_spots)


@jit
def basket_fx_approximation(
    spots: Array,
    weights: Array,
    strike: float,
    time_to_maturity: float,
    vols: Array,
    correlation_matrix: Array,
    domestic_rates: Array,
    foreign_rates: Array,
    is_call: bool = True,
) -> float:
    """Approximate basket FX option price using moment matching.

    Args:
        spots: Current FX spot rates
        weights: Basket weights
        strike: Strike price
        time_to_maturity: Time to maturity
        vols: FX volatilities
        correlation_matrix: FX correlation matrix
        domestic_rates: Domestic rates
        foreign_rates: Foreign rates
        is_call: True for call

    Returns:
        Approximate option price
    """
    from jax.scipy.stats import norm

    # Forward basket
    forwards = spots * jnp.exp((domestic_rates - foreign_rates) * time_to_maturity)
    forward_basket = jnp.dot(weights, forwards)

    # Basket volatility
    vol_matrix = jnp.diag(vols)
    cov_matrix = vol_matrix @ correlation_matrix @ vol_matrix
    basket_var = weights @ cov_matrix @ weights
    basket_vol = jnp.sqrt(basket_var)

    # Black-Scholes on basket (Curran approximation)
    d1 = (jnp.log(forward_basket / strike) + 0.5 * basket_vol**2 * time_to_maturity) / (
        basket_vol * jnp.sqrt(time_to_maturity)
    )
    d2 = d1 - basket_vol * jnp.sqrt(time_to_maturity)

    # Discount factor (use average domestic rate)
    avg_domestic_rate = jnp.mean(domestic_rates)
    df = jnp.exp(-avg_domestic_rate * time_to_maturity)

    # Calculate both call and put prices (use jnp.where for JAX compatibility)
    price_call = df * (forward_basket * norm.cdf(d1) - strike * norm.cdf(d2))
    price_put = df * (strike * norm.cdf(-d2) - forward_basket * norm.cdf(-d1))

    price = jnp.where(is_call, price_call, price_put)

    return price


__all__ = [
    "CompositeType",
    "BasketFXOption",
    "SpreadFXOption",
    "BestOfFXOption",
    "WorstOfFXOption",
    "RainbowFXOption",
    "basket_fx_approximation",
]
