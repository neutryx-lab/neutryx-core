"""High-level FpML adapter for Neutryx.

This adapter provides a simplified interface for working with FpML documents,
combining parsing, mapping, pricing, and serialization in a single workflow.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Optional

import jax
import jax.numpy as jnp

from neutryx.bridge import fpml
from neutryx.core.engine import MCConfig, price_vanilla_mc


class FpMLPricingAdapter:
    """High-level adapter for FpML pricing workflows.

    This class simplifies the end-to-end process of:
    1. Parsing FpML XML
    2. Converting to Neutryx pricing requests
    3. Executing pricing
    4. Formatting results

    Example:
        >>> adapter = FpMLPricingAdapter()
        >>> result = adapter.price_from_xml(
        ...     fpml_xml_string,
        ...     market_data={"spot": 100.0, "volatility": 0.25, "rate": 0.05}
        ... )
        >>> print(f"Price: {result['price']:.4f}")
    """

    def __init__(self, default_mc_config: Optional[MCConfig] = None, seed: int = 42):
        """Initialize adapter.

        Args:
            default_mc_config: Default Monte Carlo configuration
            seed: Random seed for reproducibility
        """
        self.default_mc_config = default_mc_config or MCConfig(steps=252, paths=100_000)
        self.seed = seed
        self.key = jax.random.PRNGKey(seed)

    def parse_xml(self, xml_content: str) -> fpml.FpMLDocument:
        """Parse FpML XML to structured document.

        Args:
            xml_content: FpML XML string

        Returns:
            Parsed FpML document

        Raises:
            FpMLParseError: If parsing fails
        """
        return fpml.parse_fpml(xml_content)

    def price_from_xml(
        self, xml_content: str, market_data: dict[str, Any], mc_config: Optional[MCConfig] = None
    ) -> dict[str, Any]:
        """Price an FpML trade from XML.

        Args:
            xml_content: FpML XML string
            market_data: Market data dictionary with keys like 'spot', 'volatility', etc.
            mc_config: Optional Monte Carlo configuration (overrides default)

        Returns:
            Pricing result dictionary with keys:
                - price: Computed option price
                - trade: Parsed trade information
                - request: Neutryx pricing request used

        Raises:
            FpMLParseError: If XML parsing fails
            FpMLMappingError: If conversion to Neutryx request fails
        """
        # Parse XML
        fpml_doc = self.parse_xml(xml_content)

        # Convert to Neutryx request
        request = fpml.fpml_to_neutryx(fpml_doc, market_data)

        # Price
        cfg = mc_config or self.default_mc_config
        price = price_vanilla_mc(
            self.key,
            request.spot,
            request.strike,
            request.maturity,
            request.rate,
            request.dividend,
            request.volatility,
            cfg,
            is_call=request.call,
        )

        # Extract trade info
        trade = fpml_doc.primary_trade
        trade_info = {}
        if trade.equityOption:
            trade_info = {
                "product_type": "EquityOption",
                "option_type": trade.equityOption.optionType.value,
                "strike": float(trade.equityOption.strike.strikePrice),
                "underlyer": trade.equityOption.underlyer.instrumentId,
            }
        elif trade.fxOption:
            trade_info = {
                "product_type": "FxOption",
                "strike": float(trade.fxOption.strike.rate),
            }

        return {
            "price": float(price),
            "trade": fpml_doc.primary_trade,
            "request": request,
            "trade_info": trade_info,
        }

    def price_from_document(
        self, fpml_doc: fpml.FpMLDocument, market_data: dict[str, Any], mc_config: Optional[MCConfig] = None
    ) -> dict[str, Any]:
        """Price from already-parsed FpML document.

        Args:
            fpml_doc: Parsed FpML document
            market_data: Market data
            mc_config: Optional MC configuration

        Returns:
            Pricing result dictionary
        """
        request = fpml.fpml_to_neutryx(fpml_doc, market_data)

        cfg = mc_config or self.default_mc_config
        price = price_vanilla_mc(
            self.key,
            request.spot,
            request.strike,
            request.maturity,
            request.rate,
            request.dividend,
            request.volatility,
            cfg,
            is_call=request.call,
        )

        # Extract trade info
        trade = fpml_doc.primary_trade
        trade_info = {}
        if trade.equityOption:
            trade_info = {
                "product_type": "EquityOption",
                "option_type": trade.equityOption.optionType.value,
                "strike": float(trade.equityOption.strike.strikePrice),
                "underlyer": trade.equityOption.underlyer.instrumentId,
            }
        elif trade.fxOption:
            trade_info = {
                "product_type": "FxOption",
                "strike": float(trade.fxOption.strike.rate),
            }

        return {
            "price": float(price),
            "trade": fpml_doc.primary_trade,
            "request": request,
            "trade_info": trade_info,
        }

    def export_to_fpml(
        self,
        spot: float,
        strike: float,
        maturity: float,
        volatility: float,
        rate: float = 0.0,
        dividend: float = 0.0,
        is_call: bool = True,
        instrument_id: str = "UNKNOWN",
        trade_date: Optional[date] = None,
    ) -> str:
        """Export pricing parameters to FpML XML.

        Args:
            spot: Spot price
            strike: Strike price
            maturity: Time to maturity in years
            volatility: Volatility
            rate: Risk-free rate
            dividend: Dividend yield
            is_call: True for call, False for put
            instrument_id: Instrument identifier
            trade_date: Trade date (defaults to today)

        Returns:
            FpML XML string
        """
        from neutryx.api.rest import VanillaOptionRequest

        request = VanillaOptionRequest(
            spot=spot,
            strike=strike,
            maturity=maturity,
            rate=rate,
            dividend=dividend,
            volatility=volatility,
            call=is_call,
        )

        fpml_doc = fpml.neutryx_to_fpml(request, instrument_id, trade_date)
        return fpml.serialize_fpml(fpml_doc)


class FpMLBatchPricer:
    """Batch pricer for multiple FpML trades.

    Efficiently prices multiple trades using vectorized operations where possible.
    """

    def __init__(self, seed: int = 42):
        """Initialize batch pricer."""
        self.adapter = FpMLPricingAdapter(seed=seed)

    def price_multiple_xml(
        self, xml_documents: list[str], market_data_list: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Price multiple FpML documents.

        Args:
            xml_documents: List of FpML XML strings
            market_data_list: List of market data dictionaries (one per document)

        Returns:
            List of pricing result dictionaries

        Raises:
            ValueError: If lengths don't match
        """
        if len(xml_documents) != len(market_data_list):
            raise ValueError("Number of documents and market data entries must match")

        results = []
        for xml_doc, market_data in zip(xml_documents, market_data_list):
            try:
                result = self.adapter.price_from_xml(xml_doc, market_data)
                results.append(result)
            except Exception as e:
                results.append({"error": str(e), "price": None})

        return results


# Convenience functions


def quick_price_fpml(xml_content: str, market_data: dict[str, Any]) -> float:
    """Quick one-line FpML pricing.

    Args:
        xml_content: FpML XML string
        market_data: Market data dictionary

    Returns:
        Option price

    Example:
        >>> price = quick_price_fpml(
        ...     fpml_xml,
        ...     {"spot": 100, "volatility": 0.25, "rate": 0.05}
        ... )
    """
    adapter = FpMLPricingAdapter()
    result = adapter.price_from_xml(xml_content, market_data)
    return result["price"]


def validate_fpml(xml_content: str) -> bool:
    """Validate FpML XML structure.

    Args:
        xml_content: FpML XML string

    Returns:
        True if valid, False otherwise
    """
    try:
        fpml.parse_fpml(xml_content)
        return True
    except fpml.FpMLParseError:
        return False


__all__ = [
    "FpMLPricingAdapter",
    "FpMLBatchPricer",
    "quick_price_fpml",
    "validate_fpml",
]
