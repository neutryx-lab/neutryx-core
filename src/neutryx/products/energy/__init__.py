"""Energy derivatives - Oil, Gas, and Power products.

This module implements comprehensive energy derivatives:
- Oil futures and options (WTI, Brent)
- Natural gas swaps and structured options
- Power derivatives with peak/off-peak scheduling
- Energy spreads (spark, dark, crack)
"""
from __future__ import annotations

from .oil import (
    BrentCrudeOption,
    CrudeOilFuture,
    OilSpreadOption,
    WTICrudeOption,
)
from .natural_gas import (
    GasStorageContract,
    GasSwap,
    NaturalGasOption,
    SeasonalGasOption,
)
from .power import (
    OffPeakPowerOption,
    PeakPowerOption,
    PowerForward,
    PowerShapingContract,
)

__all__ = [
    # Oil products
    "CrudeOilFuture",
    "WTICrudeOption",
    "BrentCrudeOption",
    "OilSpreadOption",
    # Natural gas products
    "NaturalGasOption",
    "GasSwap",
    "SeasonalGasOption",
    "GasStorageContract",
    # Power products
    "PowerForward",
    "PeakPowerOption",
    "OffPeakPowerOption",
    "PowerShapingContract",
]
