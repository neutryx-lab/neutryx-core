"""FpML XML parser.

Converts FpML XML documents into Pydantic models for downstream processing.
"""
from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any, Optional
from xml.etree import ElementTree as ET

from neutryx.bridge.fpml import schemas


class FpMLParseError(Exception):
    """Exception raised when FpML parsing fails."""

    pass


class FpMLParser:
    """Parser for FpML XML documents.

    Supports FpML 5.x namespace conventions and extracts:
    - Equity options
    - FX options
    - Interest rate swaps
    """

    # Common FpML 5.x namespaces
    NAMESPACES = {
        "fpml": "http://www.fpml.org/FpML-5/confirmation",
        "xsi": "http://www.w3.org/2001/XMLSchema-instance",
    }

    def __init__(self, namespace: Optional[dict[str, str]] = None):
        """Initialize parser with optional custom namespaces."""
        self.ns = namespace or self.NAMESPACES

    def parse(self, xml_content: str) -> schemas.FpMLDocument:
        """Parse FpML XML string into structured document.

        Args:
            xml_content: FpML XML document as string

        Returns:
            FpMLDocument with parsed trades

        Raises:
            FpMLParseError: If parsing fails
        """
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            raise FpMLParseError(f"Invalid XML: {e}") from e

        # Handle both namespaced and non-namespaced documents
        if root.tag.startswith("{"):
            # Namespaced
            ns_prefix = "fpml:"
        else:
            # Non-namespaced (simpler test documents)
            ns_prefix = ""
            self.ns = {"fpml": ""}

        try:
            parties = self._parse_parties(root, ns_prefix)
            trades = self._parse_trades(root, ns_prefix)

            return schemas.FpMLDocument(party=parties, trade=trades)
        except Exception as e:
            raise FpMLParseError(f"Failed to parse FpML document: {e}") from e

    def _find(self, element: ET.Element, path: str, ns_prefix: str = "") -> Optional[ET.Element]:
        """Find element with namespace handling."""
        if ns_prefix:
            path = path.replace("/", f"/{ns_prefix}").replace(f"//{ns_prefix}", "//")
            if not path.startswith("//"):
                path = ns_prefix + path
        return element.find(path, self.ns)

    def _findall(self, element: ET.Element, path: str, ns_prefix: str = "") -> list[ET.Element]:
        """Find all elements with namespace handling."""
        if ns_prefix:
            path = path.replace("/", f"/{ns_prefix}").replace(f"//{ns_prefix}", "//")
            if not path.startswith("//"):
                path = ns_prefix + path
        return element.findall(path, self.ns)

    def _get_text(
        self, element: ET.Element, path: str, ns_prefix: str = "", required: bool = True
    ) -> Optional[str]:
        """Extract text from element."""
        elem = self._find(element, path, ns_prefix)
        if elem is None:
            if required:
                raise FpMLParseError(f"Required element not found: {path}")
            return None
        return elem.text

    def _parse_date(self, date_str: Optional[str]) -> Optional[date]:
        """Parse ISO date string."""
        if not date_str:
            return None
        try:
            return datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            raise FpMLParseError(f"Invalid date format: {date_str}")

    def _parse_parties(self, root: ET.Element, ns_prefix: str) -> list[schemas.Party]:
        """Parse party definitions."""
        parties = []
        for party_elem in self._findall(root, ".//party", ns_prefix):
            party_id = party_elem.get("id")
            if not party_id:
                continue

            name_elem = self._find(party_elem, "partyName", ns_prefix)
            name = name_elem.text if name_elem is not None else None

            parties.append(schemas.Party(id=party_id, name=name))

        return parties

    def _parse_trades(self, root: ET.Element, ns_prefix: str) -> list[schemas.Trade]:
        """Parse all trades in document."""
        trades = []
        for trade_elem in self._findall(root, ".//trade", ns_prefix):
            trade = self._parse_trade(trade_elem, ns_prefix)
            if trade:
                trades.append(trade)

        if not trades:
            raise FpMLParseError("No valid trades found in document")

        return trades

    def _parse_trade(self, trade_elem: ET.Element, ns_prefix: str) -> Optional[schemas.Trade]:
        """Parse a single trade element."""
        # Parse trade header
        trade_date_str = self._get_text(trade_elem, "tradeHeader/tradeDate", ns_prefix)
        trade_date = self._parse_date(trade_date_str)

        header = schemas.TradeHeader(tradeDate=trade_date)

        # Determine product type
        equity_option = self._find(trade_elem, ".//equityOption", ns_prefix)
        fx_option = self._find(trade_elem, ".//fxOption", ns_prefix)
        swap = self._find(trade_elem, ".//swap", ns_prefix)

        if equity_option is not None:
            product = self._parse_equity_option(equity_option, ns_prefix)
            return schemas.Trade(tradeHeader=header, equityOption=product)
        elif fx_option is not None:
            product = self._parse_fx_option(fx_option, ns_prefix)
            return schemas.Trade(tradeHeader=header, fxOption=product)
        elif swap is not None:
            product = self._parse_swap(swap, ns_prefix)
            return schemas.Trade(tradeHeader=header, swap=product)
        else:
            # Unknown product type, skip
            return None

    def _parse_party_reference(
        self, element: ET.Element, path: str, ns_prefix: str
    ) -> schemas.PartyReference:
        """Parse party reference."""
        ref_elem = self._find(element, path, ns_prefix)
        if ref_elem is None:
            raise FpMLParseError(f"Party reference not found: {path}")
        href = ref_elem.get("href")
        if not href:
            raise FpMLParseError(f"Party reference missing href: {path}")
        return schemas.PartyReference(href=href)

    def _parse_money(self, element: ET.Element, ns_prefix: str) -> schemas.Money:
        """Parse money amount."""
        currency = self._get_text(element, "currency", ns_prefix, required=True)
        amount_str = self._get_text(element, "amount", ns_prefix, required=True)
        return schemas.Money(
            currency=schemas.CurrencyCode(currency), amount=Decimal(amount_str)
        )

    def _parse_adjustable_date(
        self, element: ET.Element, ns_prefix: str
    ) -> schemas.AdjustableDate:
        """Parse adjustable date."""
        date_str = self._get_text(element, "unadjustedDate", ns_prefix, required=True)
        return schemas.AdjustableDate(unadjustedDate=self._parse_date(date_str))

    def _parse_equity_option(
        self, option_elem: ET.Element, ns_prefix: str
    ) -> schemas.EquityOption:
        """Parse equity option product."""
        # Party references
        buyer = self._parse_party_reference(option_elem, "buyerPartyReference", ns_prefix)
        seller = self._parse_party_reference(option_elem, "sellerPartyReference", ns_prefix)

        # Option type (Put/Call)
        option_type_str = self._get_text(option_elem, "optionType", ns_prefix)
        option_type = schemas.PutCallEnum(option_type_str)

        # Underlyer
        underlyer_elem = self._find(option_elem, "underlyer", ns_prefix)
        instrument_id = self._get_text(underlyer_elem, "instrumentId", ns_prefix)
        description = self._get_text(underlyer_elem, "description", ns_prefix, required=False)

        underlyer = schemas.EquityUnderlyer(
            instrumentId=instrument_id, description=description
        )

        # Strike
        strike_elem = self._find(option_elem, "strike", ns_prefix)
        strike_price_str = self._get_text(strike_elem, "strikePrice", ns_prefix)
        strike = schemas.EquityStrike(strikePrice=Decimal(strike_price_str))

        # Exercise
        exercise_elem = self._find(option_elem, "equityExercise", ns_prefix)
        exercise_type_str = self._get_text(
            exercise_elem, "equityEuropeanExercise/expirationDate/adjustableDate/unadjustedDate",
            ns_prefix, required=False
        )
        if not exercise_type_str:
            exercise_type_str = self._get_text(
                exercise_elem, "equityAmericanExercise/expirationDate/adjustableDate/unadjustedDate",
                ns_prefix, required=False
            )
            exercise_type = schemas.OptionTypeEnum.AMERICAN
        else:
            exercise_type = schemas.OptionTypeEnum.EUROPEAN

        expiration_date = schemas.AdjustableDate(unadjustedDate=self._parse_date(exercise_type_str))
        exercise = schemas.EquityExercise(
            optionType=exercise_type,
            expirationDate=expiration_date,
        )

        # Number of options
        num_options_str = self._get_text(option_elem, "numberOfOptions", ns_prefix)
        num_options = Decimal(num_options_str)

        return schemas.EquityOption(
            buyerPartyReference=buyer,
            sellerPartyReference=seller,
            optionType=option_type,
            underlyer=underlyer,
            equityExercise=exercise,
            strike=strike,
            numberOfOptions=num_options,
        )

    def _parse_fx_option(self, option_elem: ET.Element, ns_prefix: str) -> schemas.FxOption:
        """Parse FX option product."""
        buyer = self._parse_party_reference(option_elem, "buyerPartyReference", ns_prefix)
        seller = self._parse_party_reference(option_elem, "sellerPartyReference", ns_prefix)

        # Put and call currency amounts
        put_elem = self._find(option_elem, "putCurrencyAmount", ns_prefix)
        call_elem = self._find(option_elem, "callCurrencyAmount", ns_prefix)

        put_amount = self._parse_money(put_elem, ns_prefix)
        call_amount = self._parse_money(call_elem, ns_prefix)

        # Strike
        strike_elem = self._find(option_elem, "strike", ns_prefix)
        strike_rate_str = self._get_text(strike_elem, "rate", ns_prefix)
        strike = schemas.FxStrike(rate=Decimal(strike_rate_str))

        # Exercise
        exercise_elem = self._find(option_elem, "europeanExercise", ns_prefix)
        if not exercise_elem:
            exercise_elem = self._find(option_elem, "americanExercise", ns_prefix)
            exercise_type = schemas.OptionTypeEnum.AMERICAN
        else:
            exercise_type = schemas.OptionTypeEnum.EUROPEAN

        expiry_date_str = self._get_text(exercise_elem, "expiryDate", ns_prefix)
        expiry_date = self._parse_date(expiry_date_str)

        exercise = schemas.FxExercise(optionType=exercise_type, expiryDate=expiry_date)

        return schemas.FxOption(
            buyerPartyReference=buyer,
            sellerPartyReference=seller,
            putCurrencyAmount=put_amount,
            callCurrencyAmount=call_amount,
            strike=strike,
            fxExercise=exercise,
        )

    def _parse_swap(self, swap_elem: ET.Element, ns_prefix: str) -> schemas.InterestRateSwap:
        """Parse interest rate swap product."""
        streams = []
        for stream_elem in self._findall(swap_elem, "swapStream", ns_prefix):
            stream = self._parse_swap_stream(stream_elem, ns_prefix)
            streams.append(stream)

        if len(streams) != 2:
            raise FpMLParseError(f"Swap must have exactly 2 streams, found {len(streams)}")

        return schemas.InterestRateSwap(swapStream=streams)

    def _parse_swap_stream(self, stream_elem: ET.Element, ns_prefix: str) -> schemas.SwapStream:
        """Parse single swap stream."""
        payer = self._parse_party_reference(stream_elem, "payerPartyReference", ns_prefix)
        receiver = self._parse_party_reference(stream_elem, "receiverPartyReference", ns_prefix)

        # Calculation period dates
        calc_period_elem = self._find(stream_elem, "calculationPeriodDates", ns_prefix)
        effective_elem = self._find(calc_period_elem, "effectiveDate/unadjustedDate", ns_prefix)
        termination_elem = self._find(calc_period_elem, "terminationDate/unadjustedDate", ns_prefix)

        effective_date = schemas.AdjustableDate(
            unadjustedDate=self._parse_date(effective_elem.text)
        )
        termination_date = schemas.AdjustableDate(
            unadjustedDate=self._parse_date(termination_elem.text)
        )

        freq_elem = self._find(calc_period_elem, "calculationPeriodFrequency", ns_prefix)
        period_mult = int(self._get_text(freq_elem, "periodMultiplier", ns_prefix))
        period = self._get_text(freq_elem, "period", ns_prefix)

        calc_freq = schemas.PaymentFrequency(periodMultiplier=period_mult, period=period)
        calc_period_dates = schemas.CalculationPeriodDates(
            effectiveDate=effective_date,
            terminationDate=termination_date,
            calculationPeriodFrequency=calc_freq,
        )

        # Payment dates
        payment_dates_elem = self._find(stream_elem, "paymentDates", ns_prefix)
        pay_freq_elem = self._find(payment_dates_elem, "paymentFrequency", ns_prefix)
        pay_period_mult = int(self._get_text(pay_freq_elem, "periodMultiplier", ns_prefix))
        pay_period = self._get_text(pay_freq_elem, "period", ns_prefix)
        pay_freq = schemas.PaymentFrequency(periodMultiplier=pay_period_mult, period=pay_period)

        payment_dates = schemas.PaymentDates(paymentFrequency=pay_freq)

        # Calculation
        calc_elem = self._find(stream_elem, "calculationPeriodAmount/calculation", ns_prefix)
        notional_elem = self._find(calc_elem, "notionalSchedule/notionalStepSchedule/initialValue", ns_prefix)
        currency_elem = self._find(calc_elem, "notionalSchedule/notionalStepSchedule/currency", ns_prefix)

        notional = schemas.Money(
            currency=schemas.CurrencyCode(currency_elem.text),
            amount=Decimal(notional_elem.text),
        )

        day_count = self._get_text(calc_elem, "dayCountFraction", ns_prefix)

        # Fixed or floating
        fixed_rate_elem = self._find(calc_elem, "fixedRateSchedule/initialValue", ns_prefix)
        floating_rate_elem = self._find(calc_elem, "floatingRateCalculation", ns_prefix)

        if fixed_rate_elem is not None:
            fixed_rate = schemas.FixedRateSchedule(initialValue=Decimal(fixed_rate_elem.text))
            calculation = schemas.Calculation(
                notionalSchedule=notional,
                fixedRateSchedule=fixed_rate,
                dayCountFraction=schemas.DayCountFraction(day_count),
            )
        elif floating_rate_elem is not None:
            index_str = self._get_text(floating_rate_elem, "floatingRateIndex", ns_prefix)
            floating_calc = schemas.FloatingRateCalculation(
                floatingRateIndex=schemas.FloatingRateIndex(index_str),
                dayCountFraction=schemas.DayCountFraction(day_count),
            )
            calculation = schemas.Calculation(
                notionalSchedule=notional,
                floatingRateCalculation=floating_calc,
                dayCountFraction=schemas.DayCountFraction(day_count),
            )
        else:
            raise FpMLParseError("Swap stream must have fixed or floating rate")

        return schemas.SwapStream(
            payerPartyReference=payer,
            receiverPartyReference=receiver,
            calculationPeriodDates=calc_period_dates,
            paymentDates=payment_dates,
            calculationPeriodAmount=calculation,
        )


def parse_fpml(xml_content: str) -> schemas.FpMLDocument:
    """Convenience function to parse FpML XML.

    Args:
        xml_content: FpML XML document as string

    Returns:
        Parsed FpMLDocument

    Example:
        >>> fpml_doc = parse_fpml(xml_string)
        >>> trade = fpml_doc.primary_trade
        >>> if trade.equityOption:
        ...     print(f"Strike: {trade.equityOption.strike.strikePrice}")
    """
    parser = FpMLParser()
    return parser.parse(xml_content)


__all__ = ["FpMLParser", "FpMLParseError", "parse_fpml"]
