"""Portfolio management and aggregation.

This module provides:
- Portfolio hierarchy management (Portfolio, NettingSet)
- Trade and counterparty organization
- Risk aggregation and netting calculations
- Exposure and MTM calculations
- Contract management (CSA, counterparty, master agreements)
"""

from __future__ import annotations

from . import aggregation, allocations, contracts, netting
from .netting_set import NettingSet
from .portfolio import Portfolio

__all__ = [
    "aggregation",
    "allocations",
    "contracts",
    "netting",
    "NettingSet",
    "Portfolio",
]
