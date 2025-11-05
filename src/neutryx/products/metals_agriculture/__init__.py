"""Metals and Agriculture derivatives.

This module implements:
- Precious metals options (Gold, Silver, Platinum)
- Base metals futures (Copper, Aluminum)
- Agricultural commodity options (Corn, Wheat, Soybeans)
- Weather derivatives (already in commodity_exotics, re-exported here)
"""
from __future__ import annotations

from .precious_metals import (
    GoldOption,
    PlatinumOption,
    PreciousMetalFuture,
    SilverOption,
)
from .base_metals import (
    AluminumFuture,
    BaseMetalOption,
    CopperFuture,
)
from .agriculture import (
    AgriculturalOption,
    CornFuture,
    SoybeanFuture,
    WheatFuture,
)

# Re-export weather derivatives from commodity_exotics
from ..commodity_exotics import (
    HeatingDegreeDays,
    CoolingDegreeDays,
    RainfallDerivative,
)

__all__ = [
    # Precious metals
    "PreciousMetalFuture",
    "GoldOption",
    "SilverOption",
    "PlatinumOption",
    # Base metals
    "CopperFuture",
    "AluminumFuture",
    "BaseMetalOption",
    # Agriculture
    "CornFuture",
    "WheatFuture",
    "SoybeanFuture",
    "AgriculturalOption",
    # Weather derivatives (re-exported)
    "HeatingDegreeDays",
    "CoolingDegreeDays",
    "RainfallDerivative",
]
