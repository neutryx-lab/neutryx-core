"""
Product-Specific Trade Generators

This module provides high-level convenience functions for generating
specific product types using market conventions.

Each generator wraps the TradeFactory and provides product-specific
APIs that are easier to use for common use cases.
"""

from neutryx.portfolio.trade_generation.generators.irs import (
    generate_irs_trade,
    IRSGenerator,
)
from neutryx.portfolio.trade_generation.generators.ois import (
    generate_ois_trade,
    OISGenerator,
)
from neutryx.portfolio.trade_generation.generators.fra import (
    generate_fra_trade,
    FRAGenerator,
)
from neutryx.portfolio.trade_generation.generators.basis import (
    generate_basis_swap_trade,
    BasisSwapGenerator,
)
from neutryx.portfolio.trade_generation.generators.ccs import (
    generate_ccs_trade,
    CCSGenerator,
)
from neutryx.portfolio.trade_generation.generators.capfloor import (
    generate_cap_trade,
    generate_floor_trade,
    CapFloorGenerator,
)
from neutryx.portfolio.trade_generation.generators.swaption import (
    generate_swaption_trade,
    SwaptionGenerator,
)

__all__ = [
    "generate_irs_trade",
    "IRSGenerator",
    "generate_ois_trade",
    "OISGenerator",
    "generate_fra_trade",
    "FRAGenerator",
    "generate_basis_swap_trade",
    "BasisSwapGenerator",
    "generate_ccs_trade",
    "CCSGenerator",
    "generate_cap_trade",
    "generate_floor_trade",
    "CapFloorGenerator",
    "generate_swaption_trade",
    "SwaptionGenerator",
]
