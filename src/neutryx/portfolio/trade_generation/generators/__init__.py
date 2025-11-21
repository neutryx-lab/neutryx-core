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

__all__ = [
    "generate_irs_trade",
    "IRSGenerator",
    "generate_ois_trade",
    "OISGenerator",
]
