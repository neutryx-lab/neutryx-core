"""Mappings between FpML and Neutryx internal representations.

This module provides conversion functions to translate FpML trade representations
into Neutryx pricing request models, and vice versa.
"""
from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal
from typing import Any, Optional

from neutryx.api.rest import VanillaOptionRequest
from neutryx.bridge.fpml import schemas
from neutryx.products.swap import (
    DayCountConvention,
    PaymentFrequency,
    SwapLeg,
    price_vanilla_swap,
)


class FpMLMappingError(Exception):
    """Exception raised when FpML mapping fails."""

    pass


class FpMLToNeutryxMapper:
    """Maps FpML documents to Neutryx pricing requests."""

    def __init__(self, reference_date: Optional[date] = None):
        """Initialize mapper.

        Args:
            reference_date: Reference date for calculating time to maturity.
                          Defaults to trade date if not specified.
        """
        self.reference_date = reference_date

    def map_equity_option(
        self,
        fpml_option: schemas.EquityOption,
        trade_date: date,
        spot: Optional[float] = None,
        volatility: Optional[float] = None,
        rate: float = 0.0,
        dividend: float = 0.0,
        steps: int = 252,
        paths: int = 100_000,
    ) -> VanillaOptionRequest:
        """Map FpML equity option to Neutryx vanilla option request.

        Args:
            fpml_option: FpML equity option
            trade_date: Trade date from trade header
            spot: Current spot price (required for pricing)
            volatility: Volatility (required for pricing)
            rate: Risk-free rate
            dividend: Dividend yield
            steps: Number of Monte Carlo time steps
            paths: Number of Monte Carlo paths

        Returns:
            VanillaOptionRequest ready for pricing

        Raises:
            FpMLMappingError: If required market data is missing
        """
        if spot is None:
            raise FpMLMappingError("Spot price is required for option pricing")
        if volatility is None:
            raise FpMLMappingError("Volatility is required for option pricing")

        # Determine reference date
        ref_date = self.reference_date or trade_date

        # Calculate time to maturity
        expiration_date = fpml_option.equityExercise.expirationDate.unadjustedDate
        maturity_days = (expiration_date - ref_date).days
        if maturity_days < 0:
            raise FpMLMappingError(
                f"Option has expired: expiration={expiration_date}, reference={ref_date}"
            )
        maturity = maturity_days / 365.0  # Convert to years

        # Strike
        strike = float(fpml_option.strike.strikePrice)

        # Put or Call
        is_call = fpml_option.optionType == schemas.PutCallEnum.CALL

        return VanillaOptionRequest(
            spot=spot,
            strike=strike,
            maturity=maturity,
            rate=rate,
            dividend=dividend,
            volatility=volatility,
            steps=steps,
            paths=paths,
            call=is_call,
        )

    def map_fx_option(
        self,
        fpml_option: schemas.FxOption,
        trade_date: date,
        spot_rate: Optional[float] = None,
        volatility: Optional[float] = None,
        domestic_rate: float = 0.0,
        foreign_rate: float = 0.0,
        steps: int = 252,
        paths: int = 100_000,
    ) -> VanillaOptionRequest:
        """Map FpML FX option to Neutryx vanilla option request.

        FX options are mapped to vanilla options where:
        - Spot = current FX rate
        - Strike = FX option strike rate
        - Dividend = foreign interest rate (cost of carry)
        - Rate = domestic interest rate

        Args:
            fpml_option: FpML FX option
            trade_date: Trade date
            spot_rate: Current FX spot rate
            volatility: FX volatility
            domestic_rate: Domestic interest rate
            foreign_rate: Foreign interest rate (treated as dividend yield)
            steps: Number of time steps
            paths: Number of paths

        Returns:
            VanillaOptionRequest for FX option pricing

        Raises:
            FpMLMappingError: If required data is missing
        """
        if spot_rate is None:
            raise FpMLMappingError("Spot FX rate is required")
        if volatility is None:
            raise FpMLMappingError("FX volatility is required")

        ref_date = self.reference_date or trade_date
        expiry_date = fpml_option.fxExercise.expiryDate
        maturity_days = (expiry_date - ref_date).days
        if maturity_days < 0:
            raise FpMLMappingError(f"FX option has expired: expiry={expiry_date}")
        maturity = maturity_days / 365.0

        strike = float(fpml_option.strike.rate)

        # Determine if call or put based on currency amounts
        # In FX options, call currency is what you receive, put currency is what you pay
        # This is a simplified mapping - real FX options need more sophisticated handling
        call_amount = float(fpml_option.callCurrencyAmount.amount)
        put_amount = float(fpml_option.putCurrencyAmount.amount)
        is_call = call_amount > put_amount  # Simplified heuristic

        return VanillaOptionRequest(
            spot=spot_rate,
            strike=strike,
            maturity=maturity,
            rate=domestic_rate,
            dividend=foreign_rate,  # Foreign rate as dividend yield
            volatility=volatility,
            steps=steps,
            paths=paths,
            call=is_call,
        )

    def map_swap(
        self,
        fpml_swap: schemas.InterestRateSwap,
        trade_date: date,
        discount_rate: float = 0.05,
    ) -> dict[str, Any]:
        """Map FpML interest rate swap to pricing result.

        Args:
            fpml_swap: FpML swap
            trade_date: Trade date
            discount_rate: Discount rate for PV calculation

        Returns:
            Dictionary with swap valuation details

        Raises:
            FpMLMappingError: If swap structure is invalid
        """
        if len(fpml_swap.swapStream) != 2:
            raise FpMLMappingError("Swap must have exactly 2 streams")

        # Identify fixed and floating legs
        fixed_leg = None
        floating_leg = None

        for stream in fpml_swap.swapStream:
            calc = stream.calculationPeriodAmount
            if calc.fixedRateSchedule:
                fixed_leg = stream
            elif calc.floatingRateCalculation:
                floating_leg = stream

        if not fixed_leg or not floating_leg:
            raise FpMLMappingError("Swap must have one fixed and one floating leg")

        # Extract parameters
        notional = float(fixed_leg.calculationPeriodAmount.notionalSchedule.amount)
        fixed_rate = float(fixed_leg.calculationPeriodAmount.fixedRateSchedule.initialValue)

        # Calculate maturity
        ref_date = self.reference_date or trade_date
        term_date = fixed_leg.calculationPeriodDates.terminationDate.unadjustedDate
        maturity_days = (term_date - ref_date).days
        if maturity_days < 0:
            raise FpMLMappingError(f"Swap has matured: termination={term_date}")
        maturity = maturity_days / 365.0

        # Payment frequency
        freq = fixed_leg.paymentDates.paymentFrequency
        if freq.period == "M":
            payment_freq = freq.periodMultiplier  # Monthly
        elif freq.period == "Y":
            payment_freq = 1  # Annual
        else:
            # Assume semiannual for other cases
            payment_freq = 2

        # For floating leg, use discount rate as proxy
        floating_rate = discount_rate

        # Check if paying fixed or floating
        # Simple heuristic: if fixed leg is paying, pay_fixed=True
        pay_fixed = True  # Default assumption

        # Price the swap
        value = price_vanilla_swap(
            notional=notional,
            fixed_rate=fixed_rate,
            floating_rate=floating_rate,
            maturity=maturity,
            payment_frequency=payment_freq,
            discount_rate=discount_rate,
            pay_fixed=pay_fixed,
        )

        return {
            "value": value,
            "notional": notional,
            "fixed_rate": fixed_rate,
            "floating_rate": floating_rate,
            "maturity": maturity,
            "payment_frequency": payment_freq,
            "termination_date": term_date.isoformat(),
        }

    def map_trade(
        self,
        fpml_trade: schemas.Trade,
        market_data: Optional[dict[str, Any]] = None,
    ) -> VanillaOptionRequest | dict[str, Any]:
        """Map any FpML trade to appropriate Neutryx request.

        Args:
            fpml_trade: FpML trade
            market_data: Market data dictionary with keys like 'spot', 'volatility', etc.

        Returns:
            Neutryx pricing request (VanillaOptionRequest for options, dict for swaps)

        Raises:
            FpMLMappingError: If trade type not supported or data missing
        """
        market_data = market_data or {}
        trade_date = fpml_trade.tradeHeader.tradeDate

        if fpml_trade.equityOption:
            return self.map_equity_option(
                fpml_trade.equityOption,
                trade_date,
                spot=market_data.get("spot"),
                volatility=market_data.get("volatility"),
                rate=market_data.get("rate", 0.0),
                dividend=market_data.get("dividend", 0.0),
                steps=market_data.get("steps", 252),
                paths=market_data.get("paths", 100_000),
            )
        elif fpml_trade.fxOption:
            return self.map_fx_option(
                fpml_trade.fxOption,
                trade_date,
                spot_rate=market_data.get("spot_rate"),
                volatility=market_data.get("volatility"),
                domestic_rate=market_data.get("domestic_rate", 0.0),
                foreign_rate=market_data.get("foreign_rate", 0.0),
                steps=market_data.get("steps", 252),
                paths=market_data.get("paths", 100_000),
            )
        elif fpml_trade.swap:
            return self.map_swap(
                fpml_trade.swap,
                trade_date,
                discount_rate=market_data.get("discount_rate", 0.05),
            )
        else:
            raise FpMLMappingError("Unknown or unsupported trade type")


class NeutryxToFpMLMapper:
    """Maps Neutryx pricing requests back to FpML documents."""

    def map_vanilla_option_to_equity(
        self,
        request: VanillaOptionRequest,
        instrument_id: str,
        trade_date: date,
        buyer_party_id: str = "party1",
        seller_party_id: str = "party2",
    ) -> schemas.FpMLDocument:
        """Convert Neutryx vanilla option request to FpML equity option.

        Args:
            request: Neutryx vanilla option request
            instrument_id: Underlying instrument identifier (e.g., ISIN)
            trade_date: Trade date
            buyer_party_id: Buyer party identifier
            seller_party_id: Seller party identifier

        Returns:
            FpMLDocument containing the equity option trade
        """
        # Create parties
        parties = [
            schemas.Party(id=buyer_party_id, name="Buyer"),
            schemas.Party(id=seller_party_id, name="Seller"),
        ]

        # Calculate expiration date from maturity
        expiration_date = trade_date + timedelta(days=int(request.maturity * 365))

        # Build equity option
        option_type = schemas.PutCallEnum.CALL if request.call else schemas.PutCallEnum.PUT

        underlyer = schemas.EquityUnderlyer(
            instrumentId=instrument_id, description="Underlying equity"
        )

        strike = schemas.EquityStrike(strikePrice=Decimal(str(request.strike)))

        exercise = schemas.EquityExercise(
            optionType=schemas.OptionTypeEnum.EUROPEAN,
            expirationDate=schemas.AdjustableDate(unadjustedDate=expiration_date),
        )

        equity_option = schemas.EquityOption(
            buyerPartyReference=schemas.PartyReference(href=buyer_party_id),
            sellerPartyReference=schemas.PartyReference(href=seller_party_id),
            optionType=option_type,
            underlyer=underlyer,
            strike=strike,
            equityExercise=exercise,
            numberOfOptions=Decimal("1"),
        )

        # Build trade
        trade_header = schemas.TradeHeader(tradeDate=trade_date)
        trade = schemas.Trade(tradeHeader=trade_header, equityOption=equity_option)

        return schemas.FpMLDocument(party=parties, trade=[trade])


def fpml_to_neutryx(
    fpml_doc: schemas.FpMLDocument, market_data: Optional[dict[str, Any]] = None
) -> VanillaOptionRequest:
    """Convenience function to map FpML document to Neutryx request.

    Args:
        fpml_doc: Parsed FpML document
        market_data: Market data dictionary

    Returns:
        Neutryx pricing request for the primary trade

    Example:
        >>> from neutryx.bridge.fpml import parse_fpml, fpml_to_neutryx
        >>> fpml_doc = parse_fpml(xml_string)
        >>> market_data = {"spot": 100.0, "volatility": 0.25, "rate": 0.05}
        >>> request = fpml_to_neutryx(fpml_doc, market_data)
        >>> # Now use request with Neutryx pricing engine
    """
    mapper = FpMLToNeutryxMapper()
    return mapper.map_trade(fpml_doc.primary_trade, market_data)


def neutryx_to_fpml(
    request: VanillaOptionRequest,
    instrument_id: str,
    trade_date: Optional[date] = None,
) -> schemas.FpMLDocument:
    """Convenience function to map Neutryx request to FpML document.

    Args:
        request: Neutryx vanilla option request
        instrument_id: Underlying instrument identifier
        trade_date: Trade date (defaults to today)

    Returns:
        FpML document

    Example:
        >>> from neutryx.api.rest import VanillaOptionRequest
        >>> from neutryx.bridge.fpml import neutryx_to_fpml
        >>> request = VanillaOptionRequest(
        ...     spot=100, strike=105, maturity=1.0, volatility=0.2, call=True
        ... )
        >>> fpml_doc = neutryx_to_fpml(request, "US0378331005")
        >>> # Now serialize to XML using serializer module
    """
    if trade_date is None:
        trade_date = date.today()

    mapper = NeutryxToFpMLMapper()
    return mapper.map_vanilla_option_to_equity(request, instrument_id, trade_date)


__all__ = [
    "FpMLToNeutryxMapper",
    "NeutryxToFpMLMapper",
    "FpMLMappingError",
    "fpml_to_neutryx",
    "neutryx_to_fpml",
]
