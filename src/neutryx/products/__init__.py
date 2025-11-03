"""Product catalogue and helper utilities."""
from .asian import AsianArithmetic
from .barrier import UpAndOutCall
from .base import PathProduct, Product
from .fx_options import (
    FXBarrierOption,
    FXVanillaOption,
    fx_delta,
    fx_gamma,
    fx_theta,
    fx_vega,
    garman_kohlhagen,
)
from .lookback import LookbackFloatStrikeCall
from .vanilla import European
from . import basket
from . import bonds
from . import digital

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
    # FX products
    "FXVanillaOption",
    "FXBarrierOption",
    "garman_kohlhagen",
    "fx_delta",
    "fx_gamma",
    "fx_vega",
    "fx_theta",
    # Modules
    "basket",
    "bonds",
    "digital",
]
