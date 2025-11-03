"""Portfolio management and aggregation.

This module provides:
- Portfolio hierarchy management (Portfolio, NettingSet)
- Trade and counterparty organization
- Risk aggregation and netting calculations
- Exposure and MTM calculations
"""

from __future__ import annotations

from . import aggregation, allocations, netting
from .netting_set import NettingSet
from .portfolio import Portfolio

__all__ = [
    "aggregation",
    "allocations",
    "netting",
    "NettingSet",
    "Portfolio",
]
