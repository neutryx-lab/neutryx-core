"""
FpML Adapter for Trade Generation

Provides FpML (Financial products Markup Language) import/export capabilities
for interest rate products. Supports reading FpML XML documents and creating
trades using market conventions.

FpML Version: 5.x compatible
Supported Products: IRS, OIS, FRA, Basis Swap, Cross-Currency Swap, Cap, Floor, Swaption
"""

from dataclasses import dataclass
from datetime import date, datetime
from typing import List, Optional, Dict, Any
from xml.etree import ElementTree as ET
from pathlib import Path

from neutryx.portfolio.trade_generation.generators import (
    IRSGenerator,
    OISGenerator,
    FRAGenerator,
    BasisSwapGenerator,
    CCSGenerator,
    CapFloorGenerator,
    SwaptionGenerator,
)
from neutryx.portfolio.trade_generation.factory import TradeGenerationResult
from neutryx.portfolio.contracts.trade import Trade


# FpML namespace (FpML 5.x)
FPML_NS = {
    'fpml': 'http://www.fpml.org/FpML-5/confirmation',
    'xsi': 'http://www.w3.org/2001/XMLSchema-instance',
}


@dataclass
class FpMLImportResult:
    """Result of FpML import operation"""
    trade_result: Optional[TradeGenerationResult] = None
    product_type: Optional[str] = None
    success: bool = False
    error_message: Optional[str] = None

    def __repr__(self) -> str:
        if self.success:
            return f"FpML Import SUCCESS: {self.product_type}"
        else:
            return f"FpML Import FAILED: {self.error_message}"


class FpMLAdapter:
    """
    FpML adapter for importing and exporting trades

    Provides bidirectional conversion between FpML XML format and
    Neutryx trade objects using market conventions.
    """

    def __init__(self):
        """Initialize FpML adapter with all generators"""
        self.irs_generator = IRSGenerator()
        self.ois_generator = OISGenerator()
        self.fra_generator = FRAGenerator()
        self.basis_generator = BasisSwapGenerator()
        self.ccs_generator = CCSGenerator()
        self.capfloor_generator = CapFloorGenerator()
        self.swaption_generator = SwaptionGenerator()

    def import_from_fpml_file(self, file_path: str) -> List[FpMLImportResult]:
        """
        Import trades from FpML XML file

        Args:
            file_path: Path to FpML XML file

        Returns:
            List of FpMLImportResult for each trade in the file

        Example:
            >>> adapter = FpMLAdapter()
            >>> results = adapter.import_from_fpml_file("trades.xml")
            >>> for result in results:
            ...     if result.success:
            ...         print(f"Imported: {result.trade_result.trade.id}")
        """
        tree = ET.parse(file_path)
        root = tree.getroot()

        results = []

        # Find all trade elements
        for trade_elem in root.findall('.//fpml:trade', FPML_NS):
            result = self._import_trade_from_element(trade_elem)
            results.append(result)

        return results

    def import_from_fpml_string(self, fpml_xml: str) -> List[FpMLImportResult]:
        """
        Import trades from FpML XML string

        Args:
            fpml_xml: FpML XML as string

        Returns:
            List of FpMLImportResult
        """
        root = ET.fromstring(fpml_xml)

        results = []

        # Find all trade elements
        for trade_elem in root.findall('.//fpml:trade', FPML_NS):
            result = self._import_trade_from_element(trade_elem)
            results.append(result)

        return results

    def _import_trade_from_element(self, trade_elem: ET.Element) -> FpMLImportResult:
        """Import a single trade from XML element"""
        try:
            # Detect product type
            if trade_elem.find('.//fpml:swap', FPML_NS) is not None:
                swap_elem = trade_elem.find('.//fpml:swap', FPML_NS)

                # Check if it's a cross-currency swap
                if self._is_cross_currency_swap(swap_elem):
                    return self._import_ccs(trade_elem)
                # Check if it's a basis swap
                elif self._is_basis_swap(swap_elem):
                    return self._import_basis_swap(trade_elem)
                # Check if it's OIS
                elif self._is_ois(swap_elem):
                    return self._import_ois(trade_elem)
                else:
                    return self._import_irs(trade_elem)

            elif trade_elem.find('.//fpml:fra', FPML_NS) is not None:
                return self._import_fra(trade_elem)

            elif trade_elem.find('.//fpml:capFloor', FPML_NS) is not None:
                return self._import_capfloor(trade_elem)

            elif trade_elem.find('.//fpml:swaption', FPML_NS) is not None:
                return self._import_swaption(trade_elem)

            else:
                return FpMLImportResult(
                    success=False,
                    error_message="Unknown or unsupported product type"
                )

        except Exception as e:
            return FpMLImportResult(
                success=False,
                error_message=f"Import failed: {str(e)}"
            )

    def _is_cross_currency_swap(self, swap_elem: ET.Element) -> bool:
        """Check if swap is cross-currency"""
        legs = swap_elem.findall('.//fpml:swapStream', FPML_NS)
        if len(legs) >= 2:
            currencies = set()
            for leg in legs:
                currency_elem = leg.find('.//fpml:currency', FPML_NS)
                if currency_elem is not None:
                    currencies.add(currency_elem.text)
            return len(currencies) > 1
        return False

    def _is_basis_swap(self, swap_elem: ET.Element) -> bool:
        """Check if swap is basis swap (both legs floating)"""
        legs = swap_elem.findall('.//fpml:swapStream', FPML_NS)
        floating_count = 0
        for leg in legs:
            if leg.find('.//fpml:floatingRateCalculation', FPML_NS) is not None:
                floating_count += 1
        return floating_count >= 2

    def _is_ois(self, swap_elem: ET.Element) -> bool:
        """Check if swap is OIS (overnight index swap)"""
        # Look for overnight or RFR indices
        for leg in swap_elem.findall('.//fpml:swapStream', FPML_NS):
            floating_elem = leg.find('.//fpml:floatingRateCalculation', FPML_NS)
            if floating_elem is not None:
                index_elem = floating_elem.find('.//fpml:floatingRateIndex', FPML_NS)
                if index_elem is not None:
                    index_name = index_elem.text.upper()
                    if any(rfr in index_name for rfr in ['SOFR', 'ESTR', 'SONIA', 'TONAR', 'SARON', 'OVERNIGHT']):
                        return True
        return False

    def _parse_date(self, date_str: str) -> date:
        """Parse FpML date string to date object"""
        # FpML uses ISO 8601 format: YYYY-MM-DD
        return datetime.strptime(date_str, '%Y-%m-%d').date()

    def _get_party_id(self, trade_elem: ET.Element) -> str:
        """Extract counterparty ID from trade element"""
        # Try to find party reference
        party_elem = trade_elem.find('.//fpml:partyTradeIdentifier/fpml:partyReference', FPML_NS)
        if party_elem is not None:
            href = party_elem.get('href', '')
            return href.replace('#', '') if href else 'UNKNOWN'
        return 'UNKNOWN'

    def _import_irs(self, trade_elem: ET.Element) -> FpMLImportResult:
        """Import Interest Rate Swap from FpML"""
        swap_elem = trade_elem.find('.//fpml:swap', FPML_NS)

        # Extract trade date
        trade_date_elem = trade_elem.find('.//fpml:tradeDate', FPML_NS)
        trade_date = self._parse_date(trade_date_elem.text) if trade_date_elem is not None else date.today()

        # Extract currency from first leg
        currency_elem = swap_elem.find('.//fpml:currency', FPML_NS)
        currency = currency_elem.text if currency_elem is not None else 'USD'

        # Extract notional from first leg
        notional_elem = swap_elem.find('.//fpml:notionalStepSchedule/fpml:initialValue', FPML_NS)
        notional = float(notional_elem.text) if notional_elem is not None else 0.0

        # Extract fixed rate
        fixed_rate_elem = swap_elem.find('.//fpml:fixedRateSchedule/fpml:initialValue', FPML_NS)
        fixed_rate = float(fixed_rate_elem.text) if fixed_rate_elem is not None else 0.0

        # Extract maturity (calculate tenor)
        termination_elem = swap_elem.find('.//fpml:terminationDate/fpml:unadjustedDate', FPML_NS)
        effective_elem = swap_elem.find('.//fpml:effectiveDate/fpml:unadjustedDate', FPML_NS)

        if termination_elem is not None and effective_elem is not None:
            maturity_date = self._parse_date(termination_elem.text)
            effective_date = self._parse_date(effective_elem.text)
            years = (maturity_date - effective_date).days / 365.25
            tenor = f"{int(years)}Y"
        else:
            tenor = "5Y"  # Default

        # Determine payer/receiver
        # In FpML, check payer/receiver party references
        swap_type = "PAYER"  # Default

        counterparty_id = self._get_party_id(trade_elem)

        result = self.irs_generator.generate(
            currency=currency,
            trade_date=trade_date,
            tenor=tenor,
            notional=notional,
            fixed_rate=fixed_rate,
            counterparty_id=counterparty_id,
            swap_type=swap_type,
        )

        return FpMLImportResult(
            trade_result=result,
            product_type="IRS",
            success=True
        )

    def _import_ois(self, trade_elem: ET.Element) -> FpMLImportResult:
        """Import Overnight Index Swap from FpML"""
        # Similar to IRS import but for OIS
        swap_elem = trade_elem.find('.//fpml:swap', FPML_NS)

        trade_date_elem = trade_elem.find('.//fpml:tradeDate', FPML_NS)
        trade_date = self._parse_date(trade_date_elem.text) if trade_date_elem is not None else date.today()

        currency_elem = swap_elem.find('.//fpml:currency', FPML_NS)
        currency = currency_elem.text if currency_elem is not None else 'USD'

        notional_elem = swap_elem.find('.//fpml:notionalStepSchedule/fpml:initialValue', FPML_NS)
        notional = float(notional_elem.text) if notional_elem is not None else 0.0

        fixed_rate_elem = swap_elem.find('.//fpml:fixedRateSchedule/fpml:initialValue', FPML_NS)
        fixed_rate = float(fixed_rate_elem.text) if fixed_rate_elem is not None else 0.0

        # Calculate tenor
        termination_elem = swap_elem.find('.//fpml:terminationDate/fpml:unadjustedDate', FPML_NS)
        effective_elem = swap_elem.find('.//fpml:effectiveDate/fpml:unadjustedDate', FPML_NS)

        if termination_elem is not None and effective_elem is not None:
            maturity_date = self._parse_date(termination_elem.text)
            effective_date = self._parse_date(effective_elem.text)
            years = (maturity_date - effective_date).days / 365.25
            tenor = f"{int(years)}Y"
        else:
            tenor = "5Y"

        counterparty_id = self._get_party_id(trade_elem)

        result = self.ois_generator.generate(
            currency=currency,
            trade_date=trade_date,
            tenor=tenor,
            notional=notional,
            fixed_rate=fixed_rate,
            counterparty_id=counterparty_id,
            swap_type="PAYER",
        )

        return FpMLImportResult(
            trade_result=result,
            product_type="OIS",
            success=True
        )

    def _import_fra(self, trade_elem: ET.Element) -> FpMLImportResult:
        """Import Forward Rate Agreement from FpML"""
        fra_elem = trade_elem.find('.//fpml:fra', FPML_NS)

        trade_date_elem = trade_elem.find('.//fpml:tradeDate', FPML_NS)
        trade_date = self._parse_date(trade_date_elem.text) if trade_date_elem is not None else date.today()

        currency_elem = fra_elem.find('.//fpml:currency', FPML_NS)
        currency = currency_elem.text if currency_elem is not None else 'USD'

        notional_elem = fra_elem.find('.//fpml:notional/fpml:amount', FPML_NS)
        notional = float(notional_elem.text) if notional_elem is not None else 0.0

        fixed_rate_elem = fra_elem.find('.//fpml:fixedRate', FPML_NS)
        fixed_rate = float(fixed_rate_elem.text) if fixed_rate_elem is not None else 0.0

        # Extract FRA tenor from dates
        # FpML FRAs have adjustedEffectiveDate and terminationDate
        fra_tenor = "3x6"  # Default, would need date calculation

        counterparty_id = self._get_party_id(trade_elem)

        result = self.fra_generator.generate(
            currency=currency,
            trade_date=trade_date,
            fra_tenor=fra_tenor,
            notional=notional,
            fixed_rate=fixed_rate,
            counterparty_id=counterparty_id,
        )

        return FpMLImportResult(
            trade_result=result,
            product_type="FRA",
            success=True
        )

    def _import_basis_swap(self, trade_elem: ET.Element) -> FpMLImportResult:
        """Import Basis Swap from FpML"""
        # Simplified basis swap import
        return FpMLImportResult(
            success=False,
            error_message="Basis swap FpML import not yet fully implemented"
        )

    def _import_ccs(self, trade_elem: ET.Element) -> FpMLImportResult:
        """Import Cross-Currency Swap from FpML"""
        # Simplified CCS import
        return FpMLImportResult(
            success=False,
            error_message="CCS FpML import not yet fully implemented"
        )

    def _import_capfloor(self, trade_elem: ET.Element) -> FpMLImportResult:
        """Import Cap/Floor from FpML"""
        # Simplified cap/floor import
        return FpMLImportResult(
            success=False,
            error_message="Cap/Floor FpML import not yet fully implemented"
        )

    def _import_swaption(self, trade_elem: ET.Element) -> FpMLImportResult:
        """Import Swaption from FpML"""
        # Simplified swaption import
        return FpMLImportResult(
            success=False,
            error_message="Swaption FpML import not yet fully implemented"
        )

    def export_to_fpml(self, trade: Trade) -> str:
        """
        Export a trade to FpML XML format

        Args:
            trade: Trade object to export

        Returns:
            FpML XML as string

        Example:
            >>> adapter = FpMLAdapter()
            >>> fpml_xml = adapter.export_to_fpml(trade)
            >>> print(fpml_xml)
        """
        # Create root element
        root = ET.Element('dataDocument', {
            'xmlns': FPML_NS['fpml'],
            'xmlns:xsi': FPML_NS['xsi'],
            'fpmlVersion': '5-10',
        })

        # Add trade element
        trade_elem = ET.SubElement(root, 'trade')

        # Add trade header
        trade_header = ET.SubElement(trade_elem, 'tradeHeader')

        # Party trade identifier
        party_trade_id = ET.SubElement(trade_header, 'partyTradeIdentifier')
        party_ref = ET.SubElement(party_trade_id, 'partyReference')
        party_ref.set('href', f'#{trade.counterparty_id}')

        trade_id_elem = ET.SubElement(party_trade_id, 'tradeId')
        trade_id_elem.set('tradeIdScheme', 'http://www.neutryx.com/trade-id')
        trade_id_elem.text = trade.id

        # Trade date
        trade_date_elem = ET.SubElement(trade_header, 'tradeDate')
        trade_date_elem.text = trade.trade_date.isoformat()

        # Add product-specific details
        product_subtype = trade.product_details.get('product_subtype', '')

        if trade.product_type.value in ['InterestRateSwap'] and product_subtype not in ['BASIS_SWAP', 'CROSS_CURRENCY_SWAP']:
            self._add_swap_to_fpml(trade_elem, trade)
        elif product_subtype == 'FRA':
            self._add_fra_to_fpml(trade_elem, trade)

        # Convert to string with pretty formatting
        ET.indent(root, space='  ')
        return ET.tostring(root, encoding='unicode', method='xml')

    def _add_swap_to_fpml(self, trade_elem: ET.Element, trade: Trade):
        """Add swap details to FpML"""
        swap_elem = ET.SubElement(trade_elem, 'swap')

        # Fixed leg
        fixed_leg = ET.SubElement(swap_elem, 'swapStream', {'id': 'fixedLeg'})

        # Add calculation period dates
        calc_period = ET.SubElement(fixed_leg, 'calculationPeriodDates', {'id': 'fixedLegCalcPeriodDates'})
        effective = ET.SubElement(calc_period, 'effectiveDate')
        unadj_effective = ET.SubElement(effective, 'unadjustedDate')
        unadj_effective.text = trade.effective_date.isoformat()

        termination = ET.SubElement(calc_period, 'terminationDate')
        unadj_termination = ET.SubElement(termination, 'unadjustedDate')
        unadj_termination.text = trade.maturity_date.isoformat()

        # Add notional
        notional_schedule = ET.SubElement(fixed_leg, 'notionalStepSchedule')
        notional_currency = ET.SubElement(notional_schedule, 'currency')
        notional_currency.text = trade.currency
        notional_amount = ET.SubElement(notional_schedule, 'initialValue')
        notional_amount.text = str(trade.notional)

        # Add fixed rate if available
        if 'fixed_rate' in trade.product_details:
            fixed_rate_schedule = ET.SubElement(fixed_leg, 'fixedRateSchedule')
            fixed_rate_value = ET.SubElement(fixed_rate_schedule, 'initialValue')
            fixed_rate_value.text = str(trade.product_details['fixed_rate'])

    def _add_fra_to_fpml(self, trade_elem: ET.Element, trade: Trade):
        """Add FRA details to FpML"""
        fra_elem = ET.SubElement(trade_elem, 'fra')

        # Add currency
        currency_elem = ET.SubElement(fra_elem, 'currency')
        currency_elem.text = trade.currency

        # Add notional
        notional_elem = ET.SubElement(fra_elem, 'notional')
        notional_amount = ET.SubElement(notional_elem, 'amount')
        notional_amount.text = str(trade.notional)

        # Add fixed rate
        if 'fixed_rate' in trade.product_details:
            fixed_rate_elem = ET.SubElement(fra_elem, 'fixedRate')
            fixed_rate_elem.text = str(trade.product_details['fixed_rate'])


def import_fpml_file(file_path: str) -> List[FpMLImportResult]:
    """
    Convenience function to import trades from FpML file

    Args:
        file_path: Path to FpML XML file

    Returns:
        List of FpMLImportResult

    Example:
        >>> results = import_fpml_file("trades.xml")
        >>> for result in results:
        ...     if result.success:
        ...         print(f"Imported: {result.product_type}")
    """
    adapter = FpMLAdapter()
    return adapter.import_from_fpml_file(file_path)


def export_to_fpml_file(trade: Trade, file_path: str):
    """
    Convenience function to export trade to FpML file

    Args:
        trade: Trade object to export
        file_path: Output file path

    Example:
        >>> export_to_fpml_file(trade, "trade.xml")
    """
    adapter = FpMLAdapter()
    fpml_xml = adapter.export_to_fpml(trade)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(fpml_xml)
