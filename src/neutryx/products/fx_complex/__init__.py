"""Complex FX structures and derivatives.

This module implements sophisticated FX derivatives including:
- FX Variance Swaps with correlation and quanto adjustments
- Quanto Products (options, forwards, swaps)
- Composite Options (basket, spread, best-of, worst-of)
"""
from __future__ import annotations

from .fx_variance import (
    FXVarianceSwap,
    CorridorVarianceSwap,
    ConditionalVarianceSwap,
)
from .quanto import (
    QuantoOption,
    QuantoForward,
    QuantoSwap,
    QuantoDrift,
)
from .composite import (
    BasketFXOption,
    SpreadFXOption,
    BestOfFXOption,
    WorstOfFXOption,
    RainbowFXOption,
)

__all__ = [
    # FX Variance Swaps
    "FXVarianceSwap",
    "CorridorVarianceSwap",
    "ConditionalVarianceSwap",
    # Quanto Products
    "QuantoOption",
    "QuantoForward",
    "QuantoSwap",
    "QuantoDrift",
    # Composite Options
    "BasketFXOption",
    "SpreadFXOption",
    "BestOfFXOption",
    "WorstOfFXOption",
    "RainbowFXOption",
]
