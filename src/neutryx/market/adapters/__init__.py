"""Market data adapters for vendor integrations."""

from .base import AdapterConfig, BaseMarketDataAdapter
from .bloomberg import BloombergAdapter
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
    "RefinitivAdapter",
    "SimulatedAdapter",
    "SimulatedConfig",
    "CorporateActionParser",
    "BloombergCorporateActionParser",
    "RefinitivCorporateActionParser",
    "normalize_events",
]
