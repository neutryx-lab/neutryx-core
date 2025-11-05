"""Transaction cost modeling for realistic execution simulation.

This module provides models for:
- Bid-ask spread costs
- Slippage
- Market impact
- Total execution costs
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp
import numpy as np
from jax import Array


@dataclass
class CostComponents:
    """Components of transaction costs."""

    spread_cost: float
    slippage_cost: float
    market_impact_cost: float
    commission: float

    @property
    def total_cost(self) -> float:
        """Total transaction cost."""
        return self.spread_cost + self.slippage_cost + self.market_impact_cost + self.commission

    @property
    def cost_bps(self) -> float:
        """Total cost in basis points (relative to notional)."""
        return self.total_cost * 10000


class TransactionCostModel(ABC):
    """Abstract base class for transaction cost models."""

    @abstractmethod
    def calculate_cost(
        self,
        quantity: float,
        price: float,
        volume: float,
        volatility: float,
    ) -> CostComponents:
        """Calculate transaction costs.

        Args:
            quantity: Trade quantity (positive for buy, negative for sell)
            price: Current market price
            volume: Average daily volume
            volatility: Price volatility

        Returns:
            CostComponents with breakdown of costs
        """
        pass


class SpreadModel:
    """Bid-ask spread cost model."""

    def __init__(
        self,
        fixed_spread_bps: float = 5.0,
        volatility_multiplier: float = 0.1,
    ):
        """Initialize spread model.

        Args:
            fixed_spread_bps: Fixed spread component in basis points
            volatility_multiplier: Multiplier for volatility-dependent spread
        """
        self.fixed_spread_bps = fixed_spread_bps
        self.volatility_multiplier = volatility_multiplier

    def calculate_spread_cost(
        self,
        quantity: float,
        price: float,
        volatility: float,
    ) -> float:
        """Calculate bid-ask spread cost.

        Args:
            quantity: Trade quantity (absolute value)
            price: Current price
            volatility: Price volatility

        Returns:
            Spread cost as fraction of notional
        """
        # Base spread
        base_spread = self.fixed_spread_bps / 10000.0

        # Volatility-dependent spread
        vol_spread = volatility * self.volatility_multiplier

        # Total spread (half-spread crossed per trade)
        total_spread = (base_spread + vol_spread) / 2

        return total_spread


class SlippageModel:
    """Slippage cost model."""

    def __init__(
        self,
        base_slippage_bps: float = 2.0,
        volatility_factor: float = 0.5,
        volume_factor: float = 0.1,
    ):
        """Initialize slippage model.

        Args:
            base_slippage_bps: Base slippage in basis points
            volatility_factor: Factor for volatility impact
            volume_factor: Factor for volume impact
        """
        self.base_slippage_bps = base_slippage_bps
        self.volatility_factor = volatility_factor
        self.volume_factor = volume_factor

    def calculate_slippage(
        self,
        quantity: float,
        price: float,
        volume: float,
        volatility: float,
    ) -> float:
        """Calculate slippage cost.

        Args:
            quantity: Trade quantity (absolute value)
            price: Current price
            volume: Average daily volume
            volatility: Price volatility

        Returns:
            Slippage cost as fraction of notional
        """
        # Base slippage
        base = self.base_slippage_bps / 10000.0

        # Volatility component
        vol_component = volatility * self.volatility_factor

        # Volume component (more slippage for larger orders relative to volume)
        volume_ratio = min(abs(quantity) / max(volume, 1.0), 1.0)
        volume_component = volume_ratio * self.volume_factor

        return base + vol_component + volume_component


class MarketImpactModel:
    """Market impact cost model using square-root law."""

    def __init__(
        self,
        permanent_impact_coef: float = 0.1,
        temporary_impact_coef: float = 0.05,
        daily_volume_fraction: float = 0.1,
    ):
        """Initialize market impact model.

        Args:
            permanent_impact_coef: Coefficient for permanent impact
            temporary_impact_coef: Coefficient for temporary impact
            daily_volume_fraction: Assumed fraction of daily volume tradeable
        """
        self.permanent_impact_coef = permanent_impact_coef
        self.temporary_impact_coef = temporary_impact_coef
        self.daily_volume_fraction = daily_volume_fraction

    def calculate_market_impact(
        self,
        quantity: float,
        price: float,
        volume: float,
        volatility: float,
    ) -> tuple[float, float]:
        """Calculate market impact using square-root law.

        Based on Almgren-Chriss model.

        Args:
            quantity: Trade quantity (absolute value)
            price: Current price
            volume: Average daily volume
            volatility: Price volatility

        Returns:
            Tuple of (permanent_impact, temporary_impact) as fractions
        """
        # Participation rate
        tradeable_volume = volume * self.daily_volume_fraction
        participation = abs(quantity) / max(tradeable_volume, 1.0)

        # Square-root impact
        impact_factor = np.sqrt(participation)

        # Permanent impact (affects price permanently)
        permanent_impact = (
            self.permanent_impact_coef * volatility * impact_factor
        )

        # Temporary impact (temporary price movement during execution)
        temporary_impact = (
            self.temporary_impact_coef * volatility * impact_factor
        )

        return permanent_impact, temporary_impact


class TotalCostModel(TransactionCostModel):
    """Complete transaction cost model combining all components."""

    def __init__(
        self,
        spread_model: Optional[SpreadModel] = None,
        slippage_model: Optional[SlippageModel] = None,
        market_impact_model: Optional[MarketImpactModel] = None,
        commission_bps: float = 1.0,
    ):
        """Initialize total cost model.

        Args:
            spread_model: Spread cost model
            slippage_model: Slippage cost model
            market_impact_model: Market impact model
            commission_bps: Commission in basis points
        """
        self.spread_model = spread_model or SpreadModel()
        self.slippage_model = slippage_model or SlippageModel()
        self.market_impact_model = market_impact_model or MarketImpactModel()
        self.commission_bps = commission_bps

    def calculate_cost(
        self,
        quantity: float,
        price: float,
        volume: float,
        volatility: float,
    ) -> CostComponents:
        """Calculate total transaction costs.

        Args:
            quantity: Trade quantity (positive for buy, negative for sell)
            price: Current market price
            volume: Average daily volume
            volatility: Price volatility

        Returns:
            CostComponents with breakdown of all costs
        """
        abs_quantity = abs(quantity)
        notional = abs_quantity * price

        # Calculate each component
        spread_cost = self.spread_model.calculate_spread_cost(
            abs_quantity, price, volatility
        )

        slippage_cost = self.slippage_model.calculate_slippage(
            abs_quantity, price, volume, volatility
        )

        permanent_impact, temporary_impact = (
            self.market_impact_model.calculate_market_impact(
                abs_quantity, price, volume, volatility
            )
        )
        market_impact_cost = permanent_impact + temporary_impact

        commission = (self.commission_bps / 10000.0)

        return CostComponents(
            spread_cost=spread_cost,
            slippage_cost=slippage_cost,
            market_impact_cost=market_impact_cost,
            commission=commission,
        )

    def calculate_cost_absolute(
        self,
        quantity: float,
        price: float,
        volume: float,
        volatility: float,
    ) -> CostComponents:
        """Calculate costs in absolute dollars.

        Args:
            quantity: Trade quantity
            price: Current price
            volume: Average daily volume
            volatility: Price volatility

        Returns:
            CostComponents with costs in dollars
        """
        costs = self.calculate_cost(quantity, price, volume, volatility)
        notional = abs(quantity) * price

        return CostComponents(
            spread_cost=costs.spread_cost * notional,
            slippage_cost=costs.slippage_cost * notional,
            market_impact_cost=costs.market_impact_cost * notional,
            commission=costs.commission * notional,
        )


class AlmgrenChrissModel(TransactionCostModel):
    """Almgren-Chriss optimal execution cost model."""

    def __init__(
        self,
        sigma: float = 0.3,  # Daily volatility
        eta: float = 1e-6,  # Temporary impact parameter
        gamma: float = 1e-7,  # Permanent impact parameter
        lambda_risk: float = 1e-6,  # Risk aversion parameter
    ):
        """Initialize Almgren-Chriss model.

        Args:
            sigma: Daily volatility
            eta: Temporary market impact parameter
            gamma: Permanent market impact parameter
            lambda_risk: Risk aversion parameter
        """
        self.sigma = sigma
        self.eta = eta
        self.gamma = gamma
        self.lambda_risk = lambda_risk

    def calculate_cost(
        self,
        quantity: float,
        price: float,
        volume: float,
        volatility: float,
    ) -> CostComponents:
        """Calculate optimal execution cost.

        Args:
            quantity: Total quantity to execute
            price: Current price
            volume: Daily volume
            volatility: Volatility (overrides default if provided)

        Returns:
            CostComponents
        """
        vol = volatility if volatility > 0 else self.sigma
        notional = abs(quantity) * price

        # Permanent impact cost
        permanent_cost = self.gamma * abs(quantity)

        # Temporary impact cost (depends on execution strategy)
        # For simplicity, assume optimal VWAP execution
        temporary_cost = self.eta * abs(quantity)

        # Risk cost (from price volatility during execution)
        risk_cost = self.lambda_risk * (vol ** 2) * (abs(quantity) ** 2)

        return CostComponents(
            spread_cost=0.0,
            slippage_cost=temporary_cost / notional,
            market_impact_cost=(permanent_cost + risk_cost) / notional,
            commission=0.0,
        )


def estimate_execution_cost(
    quantity: float,
    price: float,
    volume: float,
    volatility: float,
    model: str = "total",
) -> CostComponents:
    """Convenience function to estimate execution costs.

    Args:
        quantity: Trade quantity
        price: Current price
        volume: Average daily volume
        volatility: Price volatility
        model: Model type ('total', 'almgren-chriss')

    Returns:
        CostComponents with cost estimates
    """
    if model == "almgren-chriss":
        cost_model = AlmgrenChrissModel()
    else:
        cost_model = TotalCostModel()

    return cost_model.calculate_cost(quantity, price, volume, volatility)


__all__ = [
    "TransactionCostModel",
    "SpreadModel",
    "SlippageModel",
    "MarketImpactModel",
    "TotalCostModel",
    "AlmgrenChrissModel",
    "CostComponents",
    "estimate_execution_cost",
]
