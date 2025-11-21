"""Market data adapters for vendor integrations."""

from .base import AdapterConfig, BaseMarketDataAdapter
from .bloomberg import BloombergAdapter
from .corporate_actions import (
    BloombergCorporateActionParser,
    CorporateActionParser,
    RefinitivCorporateActionParser,
    SimulatedCorporateActionParser,
    normalize_events,
)
from .refinitiv import RefinitivAdapter
from .simulated import SimulatedAdapter, SimulatedConfig
from .ice import ICEDataServicesAdapter, ICEDataServicesConfig
from .cme import CMEMarketDataAdapter, CMEMarketDataConfig

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
    "normalize_events",
    "ICEDataServicesAdapter",
    "ICEDataServicesConfig",
    "CMEMarketDataAdapter",
    "CMEMarketDataConfig",
]
