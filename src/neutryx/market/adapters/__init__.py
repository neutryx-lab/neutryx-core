"""Market data adapters for vendor integrations."""

from .base import AdapterConfig, BaseMarketDataAdapter
from .bloomberg import BloombergAdapter
from .corporate_actions import (
    BloombergCorporateActionParser,
    CorporateActionParser,
    RefinitivCorporateActionParser,
    SimulatedCorporateActionParser,
)
from .refinitiv import RefinitivAdapter
from .simulated import SimulatedAdapter, SimulatedConfig
from .corporate_actions import (
    CorporateActionParser,
    BloombergCorporateActionParser,
    RefinitivCorporateActionParser,
    normalize_events,
)

__all__ = [
    "BaseMarketDataAdapter",
    "AdapterConfig",
    "BloombergAdapter",
    "CorporateActionParser",
    "BloombergCorporateActionParser",
    "RefinitivCorporateActionParser",
    "SimulatedCorporateActionParser",
    "RefinitivAdapter",
    "SimulatedAdapter",
    "SimulatedConfig",
    "CorporateActionParser",
    "BloombergCorporateActionParser",
    "RefinitivCorporateActionParser",
    "normalize_events",
]
