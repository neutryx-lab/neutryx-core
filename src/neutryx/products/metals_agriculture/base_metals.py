"""Base metals derivatives - Copper, Aluminum, etc."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import jax.numpy as jnp

from ..base import Product

Array = jnp.ndarray


class BaseMetalType(Enum):
    """Type of base metal."""

    COPPER = "copper"
    ALUMINUM = "aluminum"
    ZINC = "zinc"
    NICKEL = "nickel"
    LEAD = "lead"


@dataclass
class CopperFuture(Product):
    """Copper futures contract.

    Copper characteristics:
    - Key industrial metal (construction, electronics)
    - Economic indicator ("Dr. Copper")
    - Moderate volatility (~20-25%)
    - Significant storage costs

    Attributes:
        T: Time to maturity
        forward_price: Forward price ($/lb or $/tonne)
        notional: Contract size (25,000 lbs typical)
        storage_cost: Annual storage cost rate
        convenience_yield: Convenience yield
    """

    forward_price: float
    notional: float = 25_000.0  # 25,000 lbs standard contract
    storage_cost: float = 0.02
    convenience_yield: float = 0.03

    @property
    def requires_path(self) -> bool:
        return False

    def payoff_terminal(self, spot: Array) -> Array:
        """Calculate copper futures payoff."""
        return self.notional * (spot - self.forward_price)

    def fair_forward_price(self, spot: float, risk_free_rate: float) -> float:
        """Calculate fair forward price."""
        carry_rate = risk_free_rate + self.storage_cost - self.convenience_yield
        return float(spot * jnp.exp(carry_rate * self.T))


@dataclass
class AluminumFuture(Product):
    """Aluminum futures contract.

    Aluminum characteristics:
    - Most widely used non-ferrous metal
    - Lower volatility than copper (~18-22%)
    - High correlation with energy prices (energy-intensive production)
    - Moderate storage costs

    Attributes:
        T: Time to maturity
        forward_price: Forward price
        notional: Contract size (typically 25 metric tonnes)
        storage_cost: Storage cost rate
        convenience_yield: Convenience yield
    """

    forward_price: float
    notional: float = 25_000.0  # 25 metric tonnes (kg)
    storage_cost: float = 0.02
    convenience_yield: float = 0.025

    @property
    def requires_path(self) -> bool:
        return False

    def payoff_terminal(self, spot: Array) -> Array:
        """Calculate aluminum futures payoff."""
        return self.notional * (spot - self.forward_price)

    def fair_forward_price(self, spot: float, risk_free_rate: float) -> float:
        """Calculate fair forward price."""
        carry_rate = risk_free_rate + self.storage_cost - self.convenience_yield
        return float(spot * jnp.exp(carry_rate * self.T))


@dataclass
class BaseMetalOption(Product):
    """Generic base metal option.

    Can be used for any base metal with appropriate parameters.

    Attributes:
        T: Maturity
        strike: Strike price
        metal_type: Type of base metal
        is_call: True for call
        notional: Contract size
        volatility: Implied volatility (varies by metal: 18-30%)
        risk_free_rate: Risk-free rate
        convenience_yield: Convenience yield
    """

    strike: float
    metal_type: BaseMetalType = BaseMetalType.COPPER
    is_call: bool = True
    notional: float = 25_000.0
    volatility: float = 0.22  # Default ~22% for copper
    risk_free_rate: float = 0.03
    convenience_yield: float = 0.03

    @property
    def requires_path(self) -> bool:
        return False

    def payoff_terminal(self, spot: Array) -> Array:
        """Calculate base metal option payoff."""
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
    "BaseMetalType",
    "CopperFuture",
    "AluminumFuture",
    "BaseMetalOption",
]
