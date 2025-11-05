"""Linear interest rate derivatives.

This module implements linear interest rate products including:
- Interest Rate Swaps (IRS) with multi-curve framework
- Overnight Index Swaps (OIS)
- Cross-Currency Swaps (CCS) with FX reset
- Basis Swaps (tenor and currency basis)
- Forward Rate Agreements (FRA)
- Caps, Floors, and Collars
"""
from __future__ import annotations

from .caps_floors import Cap, Collar, Floor, InterestRateCapFloorCollar
from .fra import ForwardRateAgreement
from .swaps import (
    BasisSwap,
    CrossCurrencySwap,
    InterestRateSwap,
    OvernightIndexSwap,
)

__all__ = [
    "InterestRateSwap",
    "OvernightIndexSwap",
    "CrossCurrencySwap",
    "BasisSwap",
    "ForwardRateAgreement",
    "InterestRateCapFloorCollar",
    "Cap",
    "Floor",
    "Collar",
]
