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
from .fx_vanilla_exotic import (
    FXForward,
    FXAmericanOption,
    FXDigitalAssetOrNothing,
    FXAsianArithmetic,
    FXAsianGeometric,
    FXAsianArithmeticFloatingStrike,
    FXAsianGeometricFloatingStrike,
    FXLookbackFloatingStrikeCall,
    FXLookbackFloatingStrikePut,
    FXLookbackFixedStrikeCall,
    FXLookbackFixedStrikePut,
    FXLookbackPartialFixedStrikeCall,
    FXLookbackPartialFixedStrikePut,
)
from .lookback import (
    LookbackFloatStrikeCall,
    LookbackFloatStrikePut,
    LookbackFixedStrikeCall,
    LookbackFixedStrikePut,
    LookbackPartialFixedStrikeCall,
    LookbackPartialFixedStrikePut,
)
from .ladder import (
    LadderCall,
    LadderPut,
    PercentageLadderCall,
    PercentageLadderPut,
)
from .vanilla import European, American
from . import basket
from . import bonds
from . import commodity
from . import commodity_exotics
from . import convertible
from . import digital
from . import equity
from . import fx_exotics
from . import inflation
from . import ladder
from . import structured
from .structured import (
    AthenaAutocallable,
    CliquetOption,
    NapoleonOption,
)
from . import swaptions
from . import volatility
from . import credit_derivatives
from . import hybrid_products
from . import advanced_rates
from . import correlation_products

PAYOFF_CATALOGUE = {
    "european": European,
    "american": American,
    "asian_arithmetic": AsianArithmetic,
    "asian_geometric": AsianGeometric,
    "asian_arithmetic_floating": AsianArithmeticFloatingStrike,
    "asian_geometric_floating": AsianGeometricFloatingStrike,
    "up_and_out_call": UpAndOutCall,
    "lookback_float_strike_call": LookbackFloatStrikeCall,
    "lookback_float_strike_put": LookbackFloatStrikePut,
    "lookback_fixed_strike_call": LookbackFixedStrikeCall,
    "lookback_fixed_strike_put": LookbackFixedStrikePut,
    "ladder_call": LadderCall,
    "ladder_put": LadderPut,
    "worst_of_call": WorstOfCall,
    "best_of_call": BestOfCall,
    "average_basket_call": AverageBasketCall,
    "spread_option": SpreadOption,
    "athena_autocallable": AthenaAutocallable,
    "cliquet_option": CliquetOption,
    "napoleon_option": NapoleonOption,
}

__all__ = [
    "PAYOFF_CATALOGUE",
    "Product",
    "PathProduct",
    # Vanilla options
    "European",
    "American",
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
    # Ladder options
    "LadderCall",
    "LadderPut",
    "PercentageLadderCall",
    "PercentageLadderPut",
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
    # Structured products
    "AthenaAutocallable",
    "CliquetOption",
    "NapoleonOption",
    # FX products
    "FXVanillaOption",
    "FXBarrierOption",
    "garman_kohlhagen",
    "fx_delta",
    "fx_gamma",
    "fx_vega",
    "fx_theta",
    # FX Vanilla & Exotic
    "FXForward",
    "FXAmericanOption",
    "FXDigitalAssetOrNothing",
    "FXAsianArithmetic",
    "FXAsianGeometric",
    "FXAsianArithmeticFloatingStrike",
    "FXAsianGeometricFloatingStrike",
    "FXLookbackFloatingStrikeCall",
    "FXLookbackFloatingStrikePut",
    "FXLookbackFixedStrikeCall",
    "FXLookbackFixedStrikePut",
    "FXLookbackPartialFixedStrikeCall",
    "FXLookbackPartialFixedStrikePut",
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
    "ladder",
    "structured",
    "swaptions",
    "volatility",
    "credit_derivatives",
    "hybrid_products",
    "advanced_rates",
    "correlation_products",
]
