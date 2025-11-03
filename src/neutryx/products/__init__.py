"""Product catalogue and helper utilities."""
from .asian import AsianArithmetic
from .barrier import (
    DoubleBarrierCall,
    DoubleBarrierPut,
    DownAndInCall,
    DownAndInPut,
    DownAndOutCall,
    DownAndOutPut,
    UpAndInCall,
    UpAndInPut,
    UpAndOutCall,
    UpAndOutPut,
)
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
from . import commodity
from . import convertible
from . import digital
from . import equity
from . import inflation
from . import swaptions
from . import volatility

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
    "LookbackFloatStrikeCall",
    # Barrier options
    "UpAndOutCall",
    "UpAndOutPut",
    "DownAndOutCall",
    "DownAndOutPut",
    "UpAndInCall",
    "UpAndInPut",
    "DownAndInCall",
    "DownAndInPut",
    "DoubleBarrierCall",
    "DoubleBarrierPut",
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
    "commodity",
    "convertible",
    "digital",
    "equity",
    "inflation",
    "swaptions",
    "volatility",
]
