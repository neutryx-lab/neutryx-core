"""Product catalogue and helper utilities."""
from .base import PathProduct, Product
from .vanilla import European
from .asian import AsianArithmetic
from .barrier import UpAndOutCall
from .lookback import LookbackFloatStrikeCall

PAYOFF_CATALOGUE = {
    "european": European,
    "asian_arithmetic": AsianArithmetic,
    "up_and_out_call": UpAndOutCall,
    "lookback_float_strike_call": LookbackFloatStrikeCall,
}

__all__ = [
    "PAYOFF_CATALOGUE",
    "Product",
    "PathProduct",
    "European",
    "AsianArithmetic",
    "UpAndOutCall",
    "LookbackFloatStrikeCall",
]
