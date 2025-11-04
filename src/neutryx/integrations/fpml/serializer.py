"""FpML XML serializer.

Converts Pydantic FpML models into XML documents compliant with FpML 5.x standard.
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal

import defusedxml.minidom as minidom
from xml.etree import ElementTree as ET
from xml.etree.ElementTree import Element, SubElement

from neutryx.integrations.fpml import schemas


class FpMLSerializer:
    """Serializes FpML Pydantic models to XML.

    Generates XML documents compliant with FpML 5.x confirmation view.
    """

    FPML_NAMESPACE = "http://www.fpml.org/FpML-5/confirmation"
    XSI_NAMESPACE = "http://www.w3.org/2001/XMLSchema-instance"

    def __init__(self, pretty_print: bool = True):
        """Initialize serializer.

        Args:
            pretty_print: Whether to format XML with indentation
        """
        self.pretty_print = pretty_print

    def serialize(self, fpml_doc: schemas.FpMLDocument) -> str:
        """Serialize FpML document to XML string.

        Args:
            fpml_doc: FpML document to serialize

        Returns:
            XML string representation
        """
        # Register namespaces
        ET.register_namespace("", self.FPML_NAMESPACE)
        ET.register_namespace("xsi", self.XSI_NAMESPACE)

        # Create root element
        root = Element(
            "dataDocument",
            attrib={
                "xmlns": self.FPML_NAMESPACE,
                f"{{{self.XSI_NAMESPACE}}}schemaLocation": (
                    f"{self.FPML_NAMESPACE} "
                    "http://www.fpml.org/spec/fpml-5-0-0-confirmation.xsd"
                ),
                "fpmlVersion": "5-0",
            },
        )

        # Add parties
        for party in fpml_doc.party:
            self._add_party(root, party)

        # Add trades
        for trade in fpml_doc.trade:
            self._add_trade(root, trade)

        # Convert to string
        xml_str = ET.tostring(root, encoding="unicode")

        if self.pretty_print:
            return self._prettify_xml(xml_str)
        return xml_str

    def _prettify_xml(self, xml_str: str) -> str:
        """Format XML with proper indentation."""
        dom = minidom.parseString(xml_str)
        return dom.toprettyxml(indent="  ", encoding=None)

    def _add_party(self, parent: Element, party: schemas.Party) -> None:
        """Add party element."""
        party_elem = SubElement(parent, "party", attrib={"id": party.id})
        if party.name:
            name_elem = SubElement(party_elem, "partyName")
            name_elem.text = party.name

    def _add_trade(self, parent: Element, trade: schemas.Trade) -> None:
        """Add trade element."""
        trade_elem = SubElement(parent, "trade")

        # Trade header
        header_elem = SubElement(trade_elem, "tradeHeader")
        trade_date_elem = SubElement(header_elem, "tradeDate")
        trade_date_elem.text = trade.tradeHeader.tradeDate.isoformat()

        # Product
        if trade.equityOption:
            self._add_equity_option(trade_elem, trade.equityOption)
        elif trade.fxOption:
            self._add_fx_option(trade_elem, trade.fxOption)
        elif trade.swap:
            self._add_swap(trade_elem, trade.swap)

    def _add_party_reference(self, parent: Element, name: str, ref: schemas.PartyReference) -> None:
        """Add party reference element."""
        ref_elem = SubElement(parent, name, attrib={"href": ref.href})

    def _add_money(self, parent: Element, name: str, money: schemas.Money) -> None:
        """Add money element."""
        money_elem = SubElement(parent, name)
        curr_elem = SubElement(money_elem, "currency")
        curr_elem.text = money.currency.value
        amt_elem = SubElement(money_elem, "amount")
        amt_elem.text = str(money.amount)

    def _add_adjustable_date(
        self, parent: Element, name: str, adj_date: schemas.AdjustableDate
    ) -> None:
        """Add adjustable date element."""
        adj_elem = SubElement(parent, name)
        unadj_elem = SubElement(adj_elem, "unadjustedDate")
        unadj_elem.text = adj_date.unadjustedDate.isoformat()

    def _add_equity_option(self, parent: Element, option: schemas.EquityOption) -> None:
        """Add equity option product."""
        option_elem = SubElement(parent, "equityOption")

        # Party references
        self._add_party_reference(option_elem, "buyerPartyReference", option.buyerPartyReference)
        self._add_party_reference(
            option_elem, "sellerPartyReference", option.sellerPartyReference
        )

        # Option type
        opt_type_elem = SubElement(option_elem, "optionType")
        opt_type_elem.text = option.optionType.value

        # Underlyer
        underlyer_elem = SubElement(option_elem, "underlyer")
        inst_id_elem = SubElement(underlyer_elem, "instrumentId")
        inst_id_elem.text = option.underlyer.instrumentId
        if option.underlyer.description:
            desc_elem = SubElement(underlyer_elem, "description")
            desc_elem.text = option.underlyer.description

        # Number of options
        num_opt_elem = SubElement(option_elem, "numberOfOptions")
        num_opt_elem.text = str(option.numberOfOptions)

        # Strike
        strike_elem = SubElement(option_elem, "strike")
        strike_price_elem = SubElement(strike_elem, "strikePrice")
        strike_price_elem.text = str(option.strike.strikePrice)

        # Exercise
        exercise_elem = SubElement(option_elem, "equityExercise")
        if option.equityExercise.optionType == schemas.OptionTypeEnum.EUROPEAN:
            euro_elem = SubElement(exercise_elem, "equityEuropeanExercise")
            exp_date_elem = SubElement(euro_elem, "expirationDate")
            self._add_adjustable_date(
                exp_date_elem, "adjustableDate", option.equityExercise.expirationDate
            )
        else:
            amer_elem = SubElement(exercise_elem, "equityAmericanExercise")
            exp_date_elem = SubElement(amer_elem, "expirationDate")
            self._add_adjustable_date(
                exp_date_elem, "adjustableDate", option.equityExercise.expirationDate
            )

    def _add_fx_option(self, parent: Element, option: schemas.FxOption) -> None:
        """Add FX option product."""
        option_elem = SubElement(parent, "fxOption")

        # Party references
        self._add_party_reference(option_elem, "buyerPartyReference", option.buyerPartyReference)
        self._add_party_reference(
            option_elem, "sellerPartyReference", option.sellerPartyReference
        )

        # Currency amounts
        self._add_money(option_elem, "putCurrencyAmount", option.putCurrencyAmount)
        self._add_money(option_elem, "callCurrencyAmount", option.callCurrencyAmount)

        # Strike
        strike_elem = SubElement(option_elem, "strike")
        rate_elem = SubElement(strike_elem, "rate")
        rate_elem.text = str(option.strike.rate)

        # Exercise
        if option.fxExercise.optionType == schemas.OptionTypeEnum.EUROPEAN:
            exercise_elem = SubElement(option_elem, "europeanExercise")
        else:
            exercise_elem = SubElement(option_elem, "americanExercise")

        expiry_elem = SubElement(exercise_elem, "expiryDate")
        expiry_elem.text = option.fxExercise.expiryDate.isoformat()

    def _add_swap(self, parent: Element, swap: schemas.InterestRateSwap) -> None:
        """Add interest rate swap product."""
        swap_elem = SubElement(parent, "swap")

        for stream in swap.swapStream:
            self._add_swap_stream(swap_elem, stream)

    def _add_swap_stream(self, parent: Element, stream: schemas.SwapStream) -> None:
        """Add swap stream."""
        stream_elem = SubElement(parent, "swapStream")

        # Party references
        self._add_party_reference(stream_elem, "payerPartyReference", stream.payerPartyReference)
        self._add_party_reference(
            stream_elem, "receiverPartyReference", stream.receiverPartyReference
        )

        # Calculation period dates
        calc_period_elem = SubElement(stream_elem, "calculationPeriodDates")
        eff_date_elem = SubElement(calc_period_elem, "effectiveDate")
        unadj_eff = SubElement(eff_date_elem, "unadjustedDate")
        unadj_eff.text = stream.calculationPeriodDates.effectiveDate.unadjustedDate.isoformat()

        term_date_elem = SubElement(calc_period_elem, "terminationDate")
        unadj_term = SubElement(term_date_elem, "unadjustedDate")
        unadj_term.text = stream.calculationPeriodDates.terminationDate.unadjustedDate.isoformat()

        freq_elem = SubElement(calc_period_elem, "calculationPeriodFrequency")
        mult_elem = SubElement(freq_elem, "periodMultiplier")
        mult_elem.text = str(stream.calculationPeriodDates.calculationPeriodFrequency.periodMultiplier)
        period_elem = SubElement(freq_elem, "period")
        period_elem.text = stream.calculationPeriodDates.calculationPeriodFrequency.period

        # Payment dates
        pay_dates_elem = SubElement(stream_elem, "paymentDates")
        pay_freq_elem = SubElement(pay_dates_elem, "paymentFrequency")
        pay_mult_elem = SubElement(pay_freq_elem, "periodMultiplier")
        pay_mult_elem.text = str(stream.paymentDates.paymentFrequency.periodMultiplier)
        pay_period_elem = SubElement(pay_freq_elem, "period")
        pay_period_elem.text = stream.paymentDates.paymentFrequency.period

        # Calculation period amount
        calc_amt_elem = SubElement(stream_elem, "calculationPeriodAmount")
        calc_elem = SubElement(calc_amt_elem, "calculation")

        # Notional
        notional_sched_elem = SubElement(calc_elem, "notionalSchedule")
        notional_step_elem = SubElement(notional_sched_elem, "notionalStepSchedule")
        curr_elem = SubElement(notional_step_elem, "currency")
        curr_elem.text = stream.calculationPeriodAmount.notionalSchedule.currency.value
        init_val_elem = SubElement(notional_step_elem, "initialValue")
        init_val_elem.text = str(stream.calculationPeriodAmount.notionalSchedule.amount)

        # Fixed or floating
        if stream.calculationPeriodAmount.fixedRateSchedule:
            fixed_sched_elem = SubElement(calc_elem, "fixedRateSchedule")
            fixed_init_elem = SubElement(fixed_sched_elem, "initialValue")
            fixed_init_elem.text = str(
                stream.calculationPeriodAmount.fixedRateSchedule.initialValue
            )
        elif stream.calculationPeriodAmount.floatingRateCalculation:
            float_calc_elem = SubElement(calc_elem, "floatingRateCalculation")
            index_elem = SubElement(float_calc_elem, "floatingRateIndex")
            index_elem.text = stream.calculationPeriodAmount.floatingRateCalculation.floatingRateIndex.value

        # Day count
        day_count_elem = SubElement(calc_elem, "dayCountFraction")
        day_count_elem.text = stream.calculationPeriodAmount.dayCountFraction.value


def serialize_fpml(fpml_doc: schemas.FpMLDocument, pretty_print: bool = True) -> str:
    """Convenience function to serialize FpML document to XML.

    Args:
        fpml_doc: FpML document
        pretty_print: Whether to format with indentation

    Returns:
        XML string

    Example:
        >>> from neutryx.integrations.fpml import neutryx_to_fpml, serialize_fpml
        >>> fpml_doc = neutryx_to_fpml(request, "US0378331005")
        >>> xml_string = serialize_fpml(fpml_doc)
        >>> print(xml_string)
    """
    serializer = FpMLSerializer(pretty_print=pretty_print)
    return serializer.serialize(fpml_doc)


__all__ = ["FpMLSerializer", "serialize_fpml"]
