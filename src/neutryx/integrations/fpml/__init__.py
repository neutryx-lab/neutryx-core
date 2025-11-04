"""FpML integration for Neutryx.

This package provides comprehensive FpML (Financial products Markup Language) support,
enabling bidirectional conversion between FpML XML documents and Neutryx pricing models.

Key Features:
- Parse FpML 5.x XML documents
- Convert FpML trades to Neutryx pricing requests
- Serialize Neutryx results back to FpML
- Support for Equity Options, FX Options, and Interest Rate Swaps

Quick Start:
    >>> from neutryx.bridge import fpml
    >>>
    >>> # Parse FpML XML
    >>> doc = fpml.parse_fpml(xml_string)
    >>>
    >>> # Convert to Neutryx pricing request
    >>> market_data = {"spot": 100.0, "volatility": 0.25, "rate": 0.05}
    >>> request = fpml.fpml_to_neutryx(doc, market_data)
    >>>
    >>> # Price with Neutryx
    >>> from neutryx.core.engine import price_vanilla_mc, MCConfig
    >>> import jax
    >>> key = jax.random.PRNGKey(42)
    >>> cfg = MCConfig(steps=252, paths=100_000)
    >>> price = price_vanilla_mc(
    ...     key, request.spot, request.strike, request.maturity,
    ...     request.rate, request.dividend, request.volatility, cfg
    ... )
    >>>
    >>> # Convert back to FpML
    >>> fpml_doc = fpml.neutryx_to_fpml(request, "US0378331005")
    >>> xml_output = fpml.serialize_fpml(fpml_doc)

Supported Products:
    - Equity Options (European and American)
    - FX Options
    - Interest Rate Swaps (vanilla fixed-floating)

Module Structure:
    - schemas: Pydantic models for FpML data structures
    - parser: XML parsing functionality
    - serializer: XML generation functionality
    - mappings: Bidirectional conversion logic
"""
from __future__ import annotations

from neutryx.integrations.fpml.mappings import (
    FpMLMappingError,
    FpMLToNeutryxMapper,
    NeutryxToFpMLMapper,
    fpml_to_neutryx,
    neutryx_to_fpml,
)
from neutryx.integrations.fpml.parser import FpMLParseError, FpMLParser, parse_fpml
from neutryx.integrations.fpml.schemas import (
    CurrencyCode,
    EquityExercise,
    EquityOption,
    EquityStrike,
    EquityUnderlyer,
    FpMLDocument,
    FxExercise,
    FxOption,
    FxStrike,
    InterestRateSwap,
    Money,
    OptionTypeEnum,
    Party,
    PartyReference,
    PutCallEnum,
    SwapStream,
    Trade,
    TradeHeader,
)
from neutryx.integrations.fpml.serializer import FpMLSerializer, serialize_fpml

__version__ = "0.1.0"

__all__ = [
    # High-level API
    "parse_fpml",
    "serialize_fpml",
    "fpml_to_neutryx",
    "neutryx_to_fpml",
    # Core classes
    "FpMLParser",
    "FpMLSerializer",
    "FpMLToNeutryxMapper",
    "NeutryxToFpMLMapper",
    # Exceptions
    "FpMLParseError",
    "FpMLMappingError",
    # Schema models (commonly used)
    "FpMLDocument",
    "Trade",
    "TradeHeader",
    "EquityOption",
    "FxOption",
    "InterestRateSwap",
    "Party",
    "PartyReference",
    "Money",
    "CurrencyCode",
    "PutCallEnum",
    "OptionTypeEnum",
    "EquityUnderlyer",
    "EquityExercise",
    "EquityStrike",
    "FxExercise",
    "FxStrike",
    "SwapStream",
]
