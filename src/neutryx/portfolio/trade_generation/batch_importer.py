"""
CSV/Excel Batch Trade Importer

High-level API for batch importing trades from CSV or Excel files.
Supports all trade types with market convention application.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import csv

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
from neutryx.products.linear_rates.swaps import Tenor


class ImportStatus(Enum):
    """Status of import operation"""
    SUCCESS = "success"
    FAILED = "failed"
    WARNING = "warning"


@dataclass
class ImportResult:
    """Result of importing a single trade"""
    row_number: int
    status: ImportStatus
    trade_result: Optional[TradeGenerationResult] = None
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        if self.status == ImportStatus.SUCCESS:
            trade_id = self.trade_result.trade.id if self.trade_result else "N/A"
            return f"Row {self.row_number}: SUCCESS (Trade ID: {trade_id})"
        elif self.status == ImportStatus.FAILED:
            return f"Row {self.row_number}: FAILED - {self.error_message}"
        else:
            return f"Row {self.row_number}: WARNING - {', '.join(self.warnings)}"


@dataclass
class BatchImportSummary:
    """Summary of batch import operation"""
    total_rows: int
    successful: int
    failed: int
    warnings: int
    results: List[ImportResult]

    def print_summary(self):
        """Print a formatted summary"""
        print("\n" + "="*60)
        print("BATCH IMPORT SUMMARY")
        print("="*60)
        print(f"Total rows processed: {self.total_rows}")
        print(f"Successful imports:   {self.successful} ({self.successful/self.total_rows*100:.1f}%)")
        print(f"Failed imports:       {self.failed} ({self.failed/self.total_rows*100:.1f}%)")
        print(f"Warnings:             {self.warnings} ({self.warnings/self.total_rows*100:.1f}%)")
        print("="*60)

        if self.failed > 0:
            print("\nFAILED IMPORTS:")
            for result in self.results:
                if result.status == ImportStatus.FAILED:
                    print(f"  {result}")

        if self.warnings > 0:
            print("\nWARNINGS:")
            for result in self.results:
                if result.status == ImportStatus.WARNING:
                    print(f"  {result}")


class BatchTradeImporter:
    """
    Batch importer for trades from CSV/Excel files

    Supports importing multiple trade types with automatic generator selection
    and market convention application.
    """

    def __init__(self):
        """Initialize batch importer with all generators"""
        self.irs_generator = IRSGenerator()
        self.ois_generator = OISGenerator()
        self.fra_generator = FRAGenerator()
        self.basis_generator = BasisSwapGenerator()
        self.ccs_generator = CCSGenerator()
        self.capfloor_generator = CapFloorGenerator()
        self.swaption_generator = SwaptionGenerator()

    def import_from_csv(
        self,
        file_path: Union[str, Path],
        date_format: str = "%Y-%m-%d",
    ) -> BatchImportSummary:
        """
        Import trades from a CSV file

        Args:
            file_path: Path to CSV file
            date_format: Date format string (default: YYYY-MM-DD)

        Returns:
            BatchImportSummary with results of all imports

        CSV Format:
            Required columns:
            - product_type: IRS, OIS, FRA, BASIS, CCS, CAP, FLOOR, SWAPTION
            - currency: USD, EUR, GBP, JPY, CHF (or currency pair for CCS)
            - trade_date: Trade date in specified format
            - notional: Notional amount
            - counterparty_id: Counterparty identifier

            Product-specific columns (see documentation for each product type)
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        results = []
        row_number = 1  # Start from 1 (header is row 0)

        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for row in reader:
                row_number += 1
                result = self._import_single_trade(row, row_number, date_format)
                results.append(result)

        # Create summary
        summary = BatchImportSummary(
            total_rows=len(results),
            successful=sum(1 for r in results if r.status == ImportStatus.SUCCESS),
            failed=sum(1 for r in results if r.status == ImportStatus.FAILED),
            warnings=sum(1 for r in results if r.status == ImportStatus.WARNING),
            results=results,
        )

        return summary

    def _import_single_trade(
        self,
        row: Dict[str, str],
        row_number: int,
        date_format: str,
    ) -> ImportResult:
        """Import a single trade from CSV row"""
        try:
            # Get product type
            product_type = row.get('product_type', '').upper().strip()
            if not product_type:
                return ImportResult(
                    row_number=row_number,
                    status=ImportStatus.FAILED,
                    error_message="Missing product_type column"
                )

            # Route to appropriate generator
            if product_type == 'IRS':
                result = self._import_irs(row, date_format)
            elif product_type == 'OIS':
                result = self._import_ois(row, date_format)
            elif product_type == 'FRA':
                result = self._import_fra(row, date_format)
            elif product_type == 'BASIS':
                result = self._import_basis(row, date_format)
            elif product_type == 'CCS':
                result = self._import_ccs(row, date_format)
            elif product_type in ['CAP', 'FLOOR']:
                result = self._import_capfloor(row, date_format, is_cap=(product_type == 'CAP'))
            elif product_type == 'SWAPTION':
                result = self._import_swaption(row, date_format)
            else:
                return ImportResult(
                    row_number=row_number,
                    status=ImportStatus.FAILED,
                    error_message=f"Unknown product_type: {product_type}"
                )

            # Check for convention warnings
            warnings = []
            if result.validation_result and result.validation_result.has_warnings():
                for warning in result.validation_result.warnings:
                    warnings.append(f"{warning.field}: {warning.message}")

            return ImportResult(
                row_number=row_number,
                status=ImportStatus.WARNING if warnings else ImportStatus.SUCCESS,
                trade_result=result,
                warnings=warnings,
            )

        except Exception as e:
            return ImportResult(
                row_number=row_number,
                status=ImportStatus.FAILED,
                error_message=str(e)
            )

    def _parse_date(self, date_str: str, date_format: str) -> date:
        """Parse date string to date object"""
        return datetime.strptime(date_str.strip(), date_format).date()

    def _parse_float(self, value: str) -> float:
        """Parse float from string, handling various formats"""
        return float(value.strip().replace(',', ''))

    def _parse_bool(self, value: str) -> bool:
        """Parse boolean from string"""
        return value.strip().upper() in ['TRUE', 'T', 'YES', 'Y', '1']

    def _import_irs(self, row: Dict[str, str], date_format: str) -> TradeGenerationResult:
        """Import Interest Rate Swap"""
        # IRS uses 'tenor' instead of 'maturity_years' (e.g., "5Y", "10Y")
        tenor = row.get('tenor', row.get('maturity_years', ''))
        if tenor and not tenor.endswith('Y'):
            # Convert numeric to tenor format (e.g., "5.0" -> "5Y")
            tenor = f"{int(float(tenor))}Y"
        swap_type = "PAYER" if self._parse_bool(row.get('is_payer', 'true')) else "RECEIVER"

        return self.irs_generator.generate(
            currency=row['currency'].strip(),
            trade_date=self._parse_date(row['trade_date'], date_format),
            tenor=tenor.strip(),
            notional=self._parse_float(row['notional']),
            fixed_rate=self._parse_float(row['fixed_rate']),
            counterparty_id=row['counterparty_id'].strip(),
            swap_type=swap_type,
            trade_number=row.get('trade_number'),
            book_id=row.get('book_id'),
            desk_id=row.get('desk_id'),
            trader_id=row.get('trader_id'),
        )

    def _import_ois(self, row: Dict[str, str], date_format: str) -> TradeGenerationResult:
        """Import Overnight Index Swap"""
        # OIS uses 'tenor' instead of 'maturity_years' (e.g., "5Y", "10Y")
        tenor = row.get('tenor', row.get('maturity_years', ''))
        if tenor and not tenor.endswith('Y'):
            # Convert numeric to tenor format (e.g., "5.0" -> "5Y")
            tenor = f"{int(float(tenor))}Y"
        swap_type = "PAYER" if self._parse_bool(row.get('is_payer', 'true')) else "RECEIVER"

        return self.ois_generator.generate(
            currency=row['currency'].strip(),
            trade_date=self._parse_date(row['trade_date'], date_format),
            tenor=tenor.strip(),
            notional=self._parse_float(row['notional']),
            fixed_rate=self._parse_float(row['fixed_rate']),
            counterparty_id=row['counterparty_id'].strip(),
            swap_type=swap_type,
            trade_number=row.get('trade_number'),
            book_id=row.get('book_id'),
            desk_id=row.get('desk_id'),
            trader_id=row.get('trader_id'),
        )

    def _import_fra(self, row: Dict[str, str], date_format: str) -> TradeGenerationResult:
        """Import Forward Rate Agreement"""
        return self.fra_generator.generate(
            currency=row['currency'].strip(),
            trade_date=self._parse_date(row['trade_date'], date_format),
            fra_tenor=row['fra_tenor'].strip(),
            notional=self._parse_float(row['notional']),
            fixed_rate=self._parse_float(row['fixed_rate']),
            counterparty_id=row['counterparty_id'].strip(),
            is_payer=self._parse_bool(row.get('is_payer', 'true')),
            trade_number=row.get('trade_number'),
            book_id=row.get('book_id'),
            desk_id=row.get('desk_id'),
            trader_id=row.get('trader_id'),
        )

    def _import_basis(self, row: Dict[str, str], date_format: str) -> TradeGenerationResult:
        """Import Basis Swap"""
        # Parse optional tenor overrides
        tenor_1 = None
        tenor_2 = None
        if 'tenor_1' in row and row['tenor_1'].strip():
            tenor_1 = self._parse_tenor(row['tenor_1'])
        if 'tenor_2' in row and row['tenor_2'].strip():
            tenor_2 = self._parse_tenor(row['tenor_2'])

        return self.basis_generator.generate(
            currency=row['currency'].strip(),
            trade_date=self._parse_date(row['trade_date'], date_format),
            maturity_years=self._parse_float(row['maturity_years']),
            notional=self._parse_float(row['notional']),
            basis_spread=self._parse_float(row['basis_spread']),
            counterparty_id=row['counterparty_id'].strip(),
            tenor_1=tenor_1,
            tenor_2=tenor_2,
            trade_number=row.get('trade_number'),
            book_id=row.get('book_id'),
            desk_id=row.get('desk_id'),
            trader_id=row.get('trader_id'),
        )

    def _import_ccs(self, row: Dict[str, str], date_format: str) -> TradeGenerationResult:
        """Import Cross-Currency Swap"""
        return self.ccs_generator.generate(
            currency_pair=row['currency'].strip(),  # e.g., "USDEUR"
            trade_date=self._parse_date(row['trade_date'], date_format),
            maturity_years=self._parse_float(row['maturity_years']),
            notional_domestic=self._parse_float(row['notional_domestic']),
            notional_foreign=self._parse_float(row['notional_foreign']),
            domestic_rate=self._parse_float(row['domestic_rate']),
            foreign_rate=self._parse_float(row['foreign_rate']),
            fx_spot=self._parse_float(row['fx_spot']),
            counterparty_id=row['counterparty_id'].strip(),
            fx_reset=self._parse_bool(row.get('fx_reset', 'true')),
            trade_number=row.get('trade_number'),
            book_id=row.get('book_id'),
            desk_id=row.get('desk_id'),
            trader_id=row.get('trader_id'),
        )

    def _import_capfloor(
        self,
        row: Dict[str, str],
        date_format: str,
        is_cap: bool
    ) -> TradeGenerationResult:
        """Import Cap or Floor"""
        return self.capfloor_generator.generate(
            currency=row['currency'].strip(),
            trade_date=self._parse_date(row['trade_date'], date_format),
            maturity_years=self._parse_float(row['maturity_years']),
            notional=self._parse_float(row['notional']),
            strike=self._parse_float(row['strike']),
            counterparty_id=row['counterparty_id'].strip(),
            is_cap=is_cap,
            volatility=self._parse_float(row.get('volatility', '0.20')),
            trade_number=row.get('trade_number'),
            book_id=row.get('book_id'),
            desk_id=row.get('desk_id'),
            trader_id=row.get('trader_id'),
        )

    def _import_swaption(self, row: Dict[str, str], date_format: str) -> TradeGenerationResult:
        """Import Swaption"""
        return self.swaption_generator.generate(
            currency=row['currency'].strip(),
            trade_date=self._parse_date(row['trade_date'], date_format),
            option_maturity_years=self._parse_float(row['option_maturity_years']),
            swap_maturity_years=self._parse_float(row['swap_maturity_years']),
            notional=self._parse_float(row['notional']),
            strike=self._parse_float(row['strike']),
            counterparty_id=row['counterparty_id'].strip(),
            is_payer=self._parse_bool(row.get('is_payer', 'true')),
            volatility=self._parse_float(row.get('volatility', '0.20')),
            trade_number=row.get('trade_number'),
            book_id=row.get('book_id'),
            desk_id=row.get('desk_id'),
            trader_id=row.get('trader_id'),
        )

    def _parse_tenor(self, tenor_str: str) -> Tenor:
        """Parse tenor string to Tenor enum"""
        tenor_map = {
            'ON': Tenor.OVERNIGHT,
            '1M': Tenor.ONE_MONTH,
            '3M': Tenor.THREE_MONTH,
            '6M': Tenor.SIX_MONTH,
            '12M': Tenor.TWELVE_MONTH,
        }
        tenor_upper = tenor_str.strip().upper()
        if tenor_upper not in tenor_map:
            raise ValueError(f"Invalid tenor: {tenor_str}. Valid values: {list(tenor_map.keys())}")
        return tenor_map[tenor_upper]


def import_trades_from_csv(
    file_path: Union[str, Path],
    date_format: str = "%Y-%m-%d",
) -> BatchImportSummary:
    """
    Convenience function to import trades from CSV file

    Args:
        file_path: Path to CSV file
        date_format: Date format string (default: YYYY-MM-DD)

    Returns:
        BatchImportSummary with results

    Example:
        >>> summary = import_trades_from_csv("trades.csv")
        >>> summary.print_summary()
    """
    importer = BatchTradeImporter()
    return importer.import_from_csv(file_path, date_format)
