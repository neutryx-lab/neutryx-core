"""
Market data adapters for various data vendors.

Supports Bloomberg, Refinitiv, IHS Markit, ICE, and exchange data feeds.
"""

from .base import AdapterConfig, BaseMarketDataAdapter
from .bloomberg import BloombergAdapter
from .refinitiv import RefinitivAdapter
from .ihs_markit import IHSMarkitAdapter
from .ice import ICEAdapter
from .exchange import ExchangeAdapter, CMEAdapter, EurexAdapter, JSEAdapter
from .simulated import SimulatedAdapter, SimulatedConfig

__all__ = [
    "BaseMarketDataAdapter",
    "AdapterConfig",
    "BloombergAdapter",
    "RefinitivAdapter",
    "IHSMarkitAdapter",
    "ICEAdapter",
    "ExchangeAdapter",
    "CMEAdapter",
    "EurexAdapter",
    "JSEAdapter",
    "SimulatedAdapter",
    "SimulatedConfig",
]
