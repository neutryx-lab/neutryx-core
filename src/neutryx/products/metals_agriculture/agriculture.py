"""Agricultural commodity derivatives - Grains, Softs, etc."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import jax.numpy as jnp

from ..base import Product

Array = jnp.ndarray


class AgriculturalType(Enum):
    """Type of agricultural commodity."""

    CORN = "corn"
    WHEAT = "wheat"
    SOYBEANS = "soybeans"
    SUGAR = "sugar"
    COFFEE = "coffee"
    COTTON = "cotton"
    COCOA = "cocoa"


@dataclass
class CornFuture(Product):
    """Corn futures contract.

    Corn characteristics:
    - Strong seasonality (harvest in fall)
    - Weather-dependent
    - High storage costs
    - Moderate volatility (~25-30%)

    Attributes:
        T: Time to maturity
        forward_price: Forward price ($/bushel)
        notional: Contract size (5,000 bushels standard)
        storage_cost: Annual storage cost (higher than metals)
        convenience_yield: Convenience yield
    """

    forward_price: float
    notional: float = 5_000.0  # 5,000 bushels standard
    storage_cost: float = 0.06  # Higher storage for grains
    convenience_yield: float = 0.04

    @property
    def requires_path(self) -> bool:
        return False

    def payoff_terminal(self, spot: Array) -> Array:
        """Calculate corn futures payoff."""
        return self.notional * (spot - self.forward_price)

    def fair_forward_price(self, spot: float, risk_free_rate: float) -> float:
        """Calculate fair forward price."""
        carry_rate = risk_free_rate + self.storage_cost - self.convenience_yield
        return float(spot * jnp.exp(carry_rate * self.T))


@dataclass
class WheatFuture(Product):
    """Wheat futures contract.

    Wheat characteristics:
    - Global staple crop
    - Multiple contract specs (different wheat types)
    - Seasonal patterns
    - Moderate to high volatility (~28%)

    Attributes:
        T: Maturity
        forward_price: Forward price ($/bushel)
        notional: Contract size (5,000 bushels)
        storage_cost: Storage cost rate
        convenience_yield: Convenience yield
    """

    forward_price: float
    notional: float = 5_000.0
    storage_cost: float = 0.05
    convenience_yield: float = 0.04

    @property
    def requires_path(self) -> bool:
        return False

    def payoff_terminal(self, spot: Array) -> Array:
        """Calculate wheat futures payoff."""
        return self.notional * (spot - self.forward_price)


@dataclass
class SoybeanFuture(Product):
    """Soybean futures contract.

    Soybean characteristics:
    - Crush spread (soybeans -> soy oil + soy meal)
    - Strong China demand
    - Seasonal (harvest fall)
    - Moderate volatility (~25%)

    Attributes:
        T: Maturity
        forward_price: Forward price ($/bushel)
        notional: Contract size (5,000 bushels)
        storage_cost: Storage cost
        convenience_yield: Convenience yield
    """

    forward_price: float
    notional: float = 5_000.0
    storage_cost: float = 0.06
    convenience_yield: float = 0.04

    @property
    def requires_path(self) -> bool:
        return False

    def payoff_terminal(self, spot: Array) -> Array:
        """Calculate soybean futures payoff."""
        return self.notional * (spot - self.forward_price)


@dataclass
class AgriculturalOption(Product):
    """Generic agricultural commodity option.

    Can be used for any agricultural commodity with appropriate parameters.

    Attributes:
        T: Maturity
        strike: Strike price
        commodity_type: Type of agricultural commodity
        is_call: True for call
        notional: Contract size
        volatility: Implied volatility (typically 25-40% for ag)
        risk_free_rate: Risk-free rate
        convenience_yield: Convenience yield
    """

    strike: float
    commodity_type: AgriculturalType = AgriculturalType.CORN
    is_call: bool = True
    notional: float = 5_000.0  # 5,000 bushels default
    volatility: float = 0.30  # 30% default for ag
    risk_free_rate: float = 0.03
    convenience_yield: float = 0.04

    @property
    def requires_path(self) -> bool:
        return False

    def payoff_terminal(self, spot: Array) -> Array:
        """Calculate agricultural option payoff."""
        if self.is_call:
            payoff = jnp.maximum(spot - self.strike, 0.0)
        else:
            payoff = jnp.maximum(self.strike - spot, 0.0)
        return self.notional * payoff

    def price(self, spot: float) -> float:
        """Price using Black-Scholes with convenience yield."""
        from .precious_metals import black_scholes_commodity

        return float(
            black_scholes_commodity(
                S=spot,
                K=self.strike,
                T=self.T,
                r=self.risk_free_rate,
                q=self.convenience_yield,
                sigma=self.volatility,
                is_call=self.is_call,
            )
            * self.notional
        )


__all__ = [
    "AgriculturalType",
    "CornFuture",
    "WheatFuture",
    "SoybeanFuture",
    "AgriculturalOption",
]
