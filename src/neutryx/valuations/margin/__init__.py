"""Margin calculation for OTC derivatives.

This module provides calculations for:
- Variation Margin (VM): Mark-to-market collateral
- Initial Margin (IM): Risk-based collateral (SIMM and other models)
"""

from neutryx.valuations.margin.initial_margin import (
    InitialMarginModel,
    calculate_grid_im,
    calculate_schedule_im,
)
from neutryx.valuations.margin.variation_margin import (
    calculate_variation_margin,
    calculate_vm_call,
)

__all__ = [
    "calculate_variation_margin",
    "calculate_vm_call",
    "InitialMarginModel",
    "calculate_grid_im",
    "calculate_schedule_im",
]
