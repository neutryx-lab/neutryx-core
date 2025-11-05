"""Precious metals derivatives - Gold, Silver, Platinum, Palladium."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import jax.numpy as jnp
from jax import jit
from jax.scipy.stats import norm

from ..base import Product

Array = jnp.ndarray


class PreciousMetalType(Enum):
    """Type of precious metal."""

    GOLD = "gold"
    SILVER = "silver"
    PLATINUM = "platinum"
    PALLADIUM = "palladium"


@dataclass
class PreciousMetalFuture(Product):
    """Precious metal futures contract.

    Precious metals typically have:
    - Low storage costs
    - Minimal convenience yield
    - Monetary value (especially gold)
    - Lower volatility than energy commodities

    Attributes:
        T: Time to maturity
        forward_price: Forward price
        metal_type: Type of metal
        notional: Contract size (troy ounces typically)
        storage_cost: Annual storage cost rate (very low for metals)
        lease_rate: Lease rate (similar to dividend for metals)
    """

    forward_price: float
    metal_type: PreciousMetalType = PreciousMetalType.GOLD
    notional: float = 100.0  # 100 troy ounces typical contract
    storage_cost: float = 0.005  # 0.5% typical
    lease_rate: float = 0.001  # Metals can be leased

    @property
    def requires_path(self) -> bool:
        return False

    def payoff_terminal(self, spot: Array) -> Array:
        """Calculate futures payoff."""
        return self.notional * (spot - self.forward_price)

    def fair_forward_price(self, spot: float, risk_free_rate: float) -> float:
        """Calculate fair forward price.

        Forward = Spot × exp((r + storage - lease) × T)
        """
        carry_rate = risk_free_rate + self.storage_cost - self.lease_rate
        return float(spot * jnp.exp(carry_rate * self.T))


@dataclass
class GoldOption(Product):
    """Gold option.

    Gold is the most liquid precious metal with:
    - Lower volatility than silver (~15%)
    - Often used as store of value
    - Low storage costs
    - Can be leased (like dividend)

    Attributes:
        T: Maturity
        strike: Strike price ($/troy oz)
        is_call: True for call
        notional: Contract size (troy ounces)
        volatility: Implied volatility (typically 12-18% for gold)
        risk_free_rate: Risk-free rate
        lease_rate: Gold lease rate (like dividend yield)
    """

    strike: float
    is_call: bool = True
    notional: float = 100.0  # 100 oz contract
    volatility: float = 0.15
    risk_free_rate: float = 0.03
    lease_rate: float = 0.001

    @property
    def requires_path(self) -> bool:
        return False

    def payoff_terminal(self, spot: Array) -> Array:
        """Calculate gold option payoff."""
        if self.is_call:
            payoff = jnp.maximum(spot - self.strike, 0.0)
        else:
            payoff = jnp.maximum(self.strike - spot, 0.0)
        return self.notional * payoff

    def price(self, spot: float) -> float:
        """Price using Black-Scholes with lease rate as dividend yield."""
        return float(
            black_scholes_commodity(
                S=spot,
                K=self.strike,
                T=self.T,
                r=self.risk_free_rate,
                q=self.lease_rate,  # Treat lease rate as dividend
                sigma=self.volatility,
                is_call=self.is_call,
            )
            * self.notional
        )


@dataclass
class SilverOption(Product):
    """Silver option.

    Silver characteristics:
    - Higher volatility than gold (~25%)
    - Both monetary and industrial uses
    - More volatile due to smaller market
    - Higher storage costs than gold

    Attributes:
        T: Maturity
        strike: Strike price ($/troy oz)
        is_call: True for call
        notional: Contract size (typically 5000 oz for silver)
        volatility: Implied volatility (typically 20-30%)
        risk_free_rate: Risk-free rate
        lease_rate: Silver lease rate
    """

    strike: float
    is_call: bool = True
    notional: float = 5000.0  # 5000 oz standard contract
    volatility: float = 0.25
    risk_free_rate: float = 0.03
    lease_rate: float = 0.005

    @property
    def requires_path(self) -> bool:
        return False

    def payoff_terminal(self, spot: Array) -> Array:
        """Calculate silver option payoff."""
        if self.is_call:
            payoff = jnp.maximum(spot - self.strike, 0.0)
        else:
            payoff = jnp.maximum(self.strike - spot, 0.0)
        return self.notional * payoff

    def price(self, spot: float) -> float:
        """Price using Black-Scholes."""
        return float(
            black_scholes_commodity(
                S=spot,
                K=self.strike,
                T=self.T,
                r=self.risk_free_rate,
                q=self.lease_rate,
                sigma=self.volatility,
                is_call=self.is_call,
            )
            * self.notional
        )


@dataclass
class PlatinumOption(Product):
    """Platinum option.

    Platinum characteristics:
    - Primarily industrial use (automotive catalysts)
    - Moderate volatility (~20%)
    - Limited supply (rarer than gold)
    - Tied to auto industry demand

    Attributes:
        T: Maturity
        strike: Strike price
        is_call: True for call
        notional: Contract size (typically 50 oz)
        volatility: Implied volatility (18-25%)
        risk_free_rate: Risk-free rate
        lease_rate: Platinum lease rate
    """

    strike: float
    is_call: bool = True
    notional: float = 50.0  # 50 oz contract
    volatility: float = 0.20
    risk_free_rate: float = 0.03
    lease_rate: float = 0.01

    @property
    def requires_path(self) -> bool:
        return False

    def payoff_terminal(self, spot: Array) -> Array:
        """Calculate platinum option payoff."""
        if self.is_call:
            payoff = jnp.maximum(spot - self.strike, 0.0)
        else:
            payoff = jnp.maximum(self.strike - spot, 0.0)
        return self.notional * payoff

    def price(self, spot: float) -> float:
        """Price using Black-Scholes."""
        return float(
            black_scholes_commodity(
                S=spot,
                K=self.strike,
                T=self.T,
                r=self.risk_free_rate,
                q=self.lease_rate,
                sigma=self.volatility,
                is_call=self.is_call,
            )
            * self.notional
        )


@jit
def black_scholes_commodity(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,  # Convenience yield / lease rate
    sigma: float,
    is_call: bool = True,
) -> float:
    """Black-Scholes for commodity with convenience yield.

    Args:
        S: Spot price
        K: Strike price
        T: Time to maturity
        r: Risk-free rate
        q: Convenience yield / lease rate
        sigma: Volatility
        is_call: True for call

    Returns:
        Option price
    """
    # Handle zero time case (use jnp.where for JAX compatibility)
    intrinsic_call = jnp.maximum(S - K, 0.0)
    intrinsic_put = jnp.maximum(K - S, 0.0)
    intrinsic = jnp.where(is_call, intrinsic_call, intrinsic_put)

    # Black-Scholes formula
    sqrt_t = jnp.sqrt(jnp.maximum(T, 1e-10))
    d1 = (jnp.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t

    price_call = S * jnp.exp(-q * T) * norm.cdf(d1) - K * jnp.exp(-r * T) * norm.cdf(d2)
    price_put = K * jnp.exp(-r * T) * norm.cdf(-d2) - S * jnp.exp(-q * T) * norm.cdf(-d1)

    bs_price = jnp.where(is_call, price_call, price_put)

    # Use intrinsic if T <= 0, else use BS price
    return jnp.where(T <= 0, intrinsic, bs_price)


__all__ = [
    "PreciousMetalType",
    "PreciousMetalFuture",
    "GoldOption",
    "SilverOption",
    "PlatinumOption",
    "black_scholes_commodity",
]
