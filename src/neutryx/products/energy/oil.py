"""Oil derivatives - WTI and Brent crude futures and options."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import jax.numpy as jnp
from jax import jit
from jax.scipy.stats import norm

from ..base import Product

Array = jnp.ndarray


class OilType(Enum):
    """Type of crude oil."""

    WTI = "wti"  # West Texas Intermediate
    BRENT = "brent"  # Brent Crude
    DUBAI = "dubai"  # Dubai/Oman
    URALS = "urals"  # Urals Blend


@dataclass
class CrudeOilFuture(Product):
    """Crude oil futures contract.

    Attributes:
        T: Time to maturity in years
        forward_price: Forward price of oil
        oil_type: Type of crude (WTI, Brent, etc.)
        notional: Contract size (barrels, typically 1000)
        storage_cost: Annual storage cost rate
        convenience_yield: Convenience yield rate
    """

    forward_price: float
    oil_type: OilType = OilType.WTI
    notional: float = 1000.0  # 1 contract = 1000 barrels
    storage_cost: float = 0.02
    convenience_yield: float = 0.03

    @property
    def requires_path(self) -> bool:
        return False

    def payoff_terminal(self, spot: Array) -> Array:
        """Calculate futures payoff."""
        return self.notional * (spot - self.forward_price)

    def fair_forward_price(self, spot: float, risk_free_rate: float) -> float:
        """Calculate fair forward price using cost-of-carry.

        Forward = Spot × exp((r + storage - convenience) × T)
        """
        carry_rate = risk_free_rate + self.storage_cost - self.convenience_yield
        return float(spot * jnp.exp(carry_rate * self.T))


@dataclass
class WTICrudeOption(Product):
    """WTI Crude oil option.

    European-style option on WTI crude futures. Uses Black-76 model
    for futures option pricing.

    Attributes:
        T: Time to maturity
        strike: Strike price ($/barrel)
        is_call: True for call, False for put
        notional: Contract size in barrels
        volatility: Implied volatility
        risk_free_rate: Risk-free rate
    """

    strike: float
    is_call: bool = True
    notional: float = 1000.0
    volatility: float = 0.35
    risk_free_rate: float = 0.03

    @property
    def requires_path(self) -> bool:
        return False

    def payoff_terminal(self, spot: Array) -> Array:
        """Calculate option payoff."""
        if self.is_call:
            payoff = jnp.maximum(spot - self.strike, 0.0)
        else:
            payoff = jnp.maximum(self.strike - spot, 0.0)
        return self.notional * payoff

    def price(self, futures_price: float) -> float:
        """Price option using Black-76 model for futures.

        Args:
            futures_price: Current futures price

        Returns:
            Option price
        """
        return float(
            black_76_option(
                F=futures_price,
                K=self.strike,
                T=self.T,
                r=self.risk_free_rate,
                sigma=self.volatility,
                is_call=self.is_call,
            )
            * self.notional
        )


@dataclass
class BrentCrudeOption(Product):
    """Brent Crude oil option.

    Similar to WTI but for Brent crude futures. Typically has different
    volatility and liquidity characteristics.

    Attributes:
        T: Time to maturity
        strike: Strike price ($/barrel)
        is_call: True for call
        notional: Contract size
        volatility: Implied volatility (typically 30-40% for Brent)
        risk_free_rate: Risk-free rate
    """

    strike: float
    is_call: bool = True
    notional: float = 1000.0
    volatility: float = 0.35
    risk_free_rate: float = 0.03

    @property
    def requires_path(self) -> bool:
        return False

    def payoff_terminal(self, spot: Array) -> Array:
        """Calculate Brent option payoff."""
        if self.is_call:
            payoff = jnp.maximum(spot - self.strike, 0.0)
        else:
            payoff = jnp.maximum(self.strike - spot, 0.0)
        return self.notional * payoff

    def price(self, futures_price: float) -> float:
        """Price using Black-76 model."""
        return float(
            black_76_option(
                F=futures_price,
                K=self.strike,
                T=self.T,
                r=self.risk_free_rate,
                sigma=self.volatility,
                is_call=self.is_call,
            )
            * self.notional
        )


@dataclass
class OilSpreadOption(Product):
    """Option on the spread between two oil types.

    Common structures:
    - WTI-Brent spread (arb between US and European markets)
    - Calendar spreads (near vs far futures)
    - Location spreads

    Attributes:
        T: Maturity
        strike: Strike on the spread
        is_call: True for call on spread
        notional: Notional size
        vol_1: Volatility of first oil
        vol_2: Volatility of second oil
        correlation: Correlation between the two oils
    """

    strike: float
    is_call: bool = True
    notional: float = 1000.0
    vol_1: float = 0.35
    vol_2: float = 0.35
    correlation: float = 0.85  # WTI and Brent typically highly correlated
    risk_free_rate: float = 0.03

    @property
    def requires_path(self) -> bool:
        return False

    def payoff_terminal(self, spots: Array) -> Array:
        """Calculate spread option payoff.

        Args:
            spots: Array [price_1, price_2]
        """
        spread = spots[0] - spots[1]

        if self.is_call:
            payoff = jnp.maximum(spread - self.strike, 0.0)
        else:
            payoff = jnp.maximum(self.strike - spread, 0.0)

        return self.notional * payoff

    def spread_volatility(self) -> float:
        """Calculate volatility of the spread.

        Var(X - Y) = Var(X) + Var(Y) - 2*Corr*σ_X*σ_Y
        """
        var_spread = self.vol_1**2 + self.vol_2**2 - 2 * self.correlation * self.vol_1 * self.vol_2
        return float(jnp.sqrt(var_spread))


@jit
def black_76_option(
    F: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    is_call: bool = True,
) -> float:
    """Black-76 model for futures options.

    Args:
        F: Futures price
        K: Strike price
        T: Time to maturity
        r: Risk-free rate
        sigma: Volatility
        is_call: True for call

    Returns:
        Option price
    """
    # Handle zero time case (use jnp.where for JAX compatibility)
    intrinsic_call = jnp.maximum(F - K, 0.0)
    intrinsic_put = jnp.maximum(K - F, 0.0)
    intrinsic = jnp.where(is_call, intrinsic_call, intrinsic_put)

    # Black-76 formula
    sqrt_t = jnp.sqrt(jnp.maximum(T, 1e-10))
    d1 = (jnp.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t

    df = jnp.exp(-r * T)

    price_call = df * (F * norm.cdf(d1) - K * norm.cdf(d2))
    price_put = df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))

    black_price = jnp.where(is_call, price_call, price_put)

    # Use intrinsic if T <= 0, else use Black price
    return jnp.where(T <= 0, intrinsic, black_price)


__all__ = [
    "OilType",
    "CrudeOilFuture",
    "WTICrudeOption",
    "BrentCrudeOption",
    "OilSpreadOption",
    "black_76_option",
]
