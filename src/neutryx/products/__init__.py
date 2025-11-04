"""Product catalogue and helper utilities."""
from .asian import (
    AsianArithmetic,
    AsianGeometric,
    AsianArithmeticFloatingStrike,
    AsianGeometricFloatingStrike,
)
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
from .basket import (
    WorstOfCall,
    WorstOfPut,
    BestOfCall,
    BestOfPut,
    AverageBasketCall,
    AverageBasketPut,
    SpreadOption,
    RainbowOption,
)
from .fx_options import (
    FXBarrierOption,
    FXVanillaOption,
    fx_delta,
    fx_gamma,
    fx_theta,
    fx_vega,
    garman_kohlhagen,
)
from .lookback import (
    LookbackFloatStrikeCall,
    LookbackFloatStrikePut,
    LookbackFixedStrikeCall,
    LookbackFixedStrikePut,
    LookbackPartialFixedStrikeCall,
    LookbackPartialFixedStrikePut,
)
from .vanilla import European
from . import basket
from . import bonds
from . import commodity
from . import commodity_exotics
from . import convertible
from . import digital
from . import equity
from . import fx_exotics
from . import inflation
from . import structured
from . import swaptions
from . import volatility
from . import credit_derivatives
from . import hybrid_products
from . import advanced_rates
from . import correlation_products

PAYOFF_CATALOGUE = {
    "european": European,
    "asian_arithmetic": AsianArithmetic,
    "asian_geometric": AsianGeometric,
    "asian_arithmetic_floating": AsianArithmeticFloatingStrike,
    "asian_geometric_floating": AsianGeometricFloatingStrike,
    "up_and_out_call": UpAndOutCall,
    "lookback_float_strike_call": LookbackFloatStrikeCall,
    "lookback_float_strike_put": LookbackFloatStrikePut,
    "lookback_fixed_strike_call": LookbackFixedStrikeCall,
    "lookback_fixed_strike_put": LookbackFixedStrikePut,
    "worst_of_call": WorstOfCall,
    "best_of_call": BestOfCall,
    "average_basket_call": AverageBasketCall,
    "spread_option": SpreadOption,
}

__all__ = [
    "PAYOFF_CATALOGUE",
    "Product",
    "PathProduct",
    "European",
    # Asian options
    "AsianArithmetic",
    "AsianGeometric",
    "AsianArithmeticFloatingStrike",
    "AsianGeometricFloatingStrike",
    # Lookback options
    "LookbackFloatStrikeCall",
    "LookbackFloatStrikePut",
    "LookbackFixedStrikeCall",
    "LookbackFixedStrikePut",
    "LookbackPartialFixedStrikeCall",
    "LookbackPartialFixedStrikePut",
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
    # Basket options
    "WorstOfCall",
    "WorstOfPut",
    "BestOfCall",
    "BestOfPut",
    "AverageBasketCall",
    "AverageBasketPut",
    "SpreadOption",
    "RainbowOption",
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
    "commodity_exotics",
    "convertible",
    "digital",
    "equity",
    "fx_exotics",
    "inflation",
    "structured",
    "swaptions",
    "volatility",
    "credit_derivatives",
    "hybrid_products",
    "advanced_rates",
    "correlation_products",
]
