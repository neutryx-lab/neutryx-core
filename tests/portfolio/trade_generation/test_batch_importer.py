"""Tests for Batch Trade Importer"""

import pytest
from datetime import date
from pathlib import Path
import tempfile
import csv

from neutryx.portfolio.trade_generation.batch_importer import (
    BatchTradeImporter,
    import_trades_from_csv,
    ImportStatus,
)


class TestBatchTradeImporter:
    """Test Batch Trade Importer"""

    def test_importer_creation(self):
        """Test creating batch importer"""
        importer = BatchTradeImporter()
        assert importer is not None
        assert importer.irs_generator is not None
        assert importer.ois_generator is not None
        assert importer.fra_generator is not None
        assert importer.basis_generator is not None
        assert importer.ccs_generator is not None
        assert importer.capfloor_generator is not None
        assert importer.swaption_generator is not None

    def test_parse_date(self):
        """Test date parsing"""
        importer = BatchTradeImporter()

        # Default format
        parsed = importer._parse_date("2024-01-15", "%Y-%m-%d")
        assert parsed == date(2024, 1, 15)

        # Different format
        parsed = importer._parse_date("01/15/2024", "%m/%d/%Y")
        assert parsed == date(2024, 1, 15)

    def test_parse_float(self):
        """Test float parsing"""
        importer = BatchTradeImporter()

        assert importer._parse_float("100000") == 100000.0
        assert importer._parse_float("100,000") == 100000.0
        assert importer._parse_float("0.05") == 0.05
        assert importer._parse_float("  0.05  ") == 0.05

    def test_parse_bool(self):
        """Test boolean parsing"""
        importer = BatchTradeImporter()

        assert importer._parse_bool("true") is True
        assert importer._parse_bool("TRUE") is True
        assert importer._parse_bool("yes") is True
        assert importer._parse_bool("1") is True
        assert importer._parse_bool("false") is False
        assert importer._parse_bool("no") is False
        assert importer._parse_bool("0") is False

    def test_import_irs_from_csv(self, tmp_path):
        """Test importing IRS trades from CSV"""
        # Create sample CSV
        csv_file = tmp_path / "irs_trades.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'product_type', 'currency', 'trade_date', 'maturity_years',
                'notional', 'fixed_rate', 'counterparty_id', 'is_payer'
            ])
            writer.writerow([
                'IRS', 'USD', '2024-01-15', '5.0',
                '100000000', '0.05', 'CP-001', 'true'
            ])
            writer.writerow([
                'IRS', 'EUR', '2024-01-15', '10.0',
                '50000000', '0.035', 'CP-002', 'false'
            ])

        # Import trades
        importer = BatchTradeImporter()
        summary = importer.import_from_csv(csv_file)

        # Verify results
        assert summary.total_rows == 2
        assert summary.successful >= 1
        assert summary.failed == 0

        # Check first trade
        result = summary.results[0]
        assert result.status in [ImportStatus.SUCCESS, ImportStatus.WARNING]
        assert result.trade_result is not None
        assert result.trade_result.trade.currency == "USD"
        assert result.trade_result.trade.notional == 100_000_000

    def test_import_mixed_products_from_csv(self, tmp_path):
        """Test importing mixed product types from CSV"""
        # Create sample CSV with different products
        csv_file = tmp_path / "mixed_trades.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            # Header with all possible columns
            writer.writerow([
                'product_type', 'currency', 'trade_date', 'maturity_years',
                'notional', 'fixed_rate', 'counterparty_id', 'is_payer',
                'fra_tenor', 'basis_spread', 'strike', 'volatility',
                'option_maturity_years', 'swap_maturity_years'
            ])
            # IRS
            writer.writerow([
                'IRS', 'USD', '2024-01-15', '5.0',
                '100000000', '0.05', 'CP-001', 'true',
                '', '', '', '', '', ''
            ])
            # FRA
            writer.writerow([
                'FRA', 'USD', '2024-01-15', '',
                '10000000', '0.045', 'CP-002', 'true',
                '3x6', '', '', '', '', ''
            ])
            # CAP
            writer.writerow([
                'CAP', 'USD', '2024-01-15', '5.0',
                '100000000', '', 'CP-003', '',
                '', '', '0.05', '0.20', '', ''
            ])

        # Import trades
        importer = BatchTradeImporter()
        summary = importer.import_from_csv(csv_file)

        # Verify results
        assert summary.total_rows == 3
        assert summary.successful >= 2  # At least 2 should succeed

    def test_import_with_errors(self, tmp_path):
        """Test importing with some invalid rows"""
        csv_file = tmp_path / "trades_with_errors.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'product_type', 'currency', 'trade_date', 'maturity_years',
                'notional', 'fixed_rate', 'counterparty_id'
            ])
            # Valid trade
            writer.writerow([
                'IRS', 'USD', '2024-01-15', '5.0',
                '100000000', '0.05', 'CP-001'
            ])
            # Invalid product type
            writer.writerow([
                'INVALID', 'USD', '2024-01-15', '5.0',
                '100000000', '0.05', 'CP-002'
            ])
            # Missing product type
            writer.writerow([
                '', 'USD', '2024-01-15', '5.0',
                '100000000', '0.05', 'CP-003'
            ])

        # Import trades
        importer = BatchTradeImporter()
        summary = importer.import_from_csv(csv_file)

        # Verify results
        assert summary.total_rows == 3
        assert summary.successful >= 1
        assert summary.failed >= 2

        # Check that failures have error messages
        failed_results = [r for r in summary.results if r.status == ImportStatus.FAILED]
        assert len(failed_results) >= 2
        for result in failed_results:
            assert result.error_message is not None

    def test_convenience_function(self, tmp_path):
        """Test convenience function"""
        csv_file = tmp_path / "trades.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'product_type', 'currency', 'trade_date', 'maturity_years',
                'notional', 'fixed_rate', 'counterparty_id'
            ])
            writer.writerow([
                'OIS', 'USD', '2024-01-15', '5.0',
                '100000000', '0.05', 'CP-001'
            ])

        # Import using convenience function
        summary = import_trades_from_csv(csv_file)

        assert summary.total_rows == 1
        assert summary.successful >= 1

    def test_file_not_found(self):
        """Test handling of missing file"""
        importer = BatchTradeImporter()

        with pytest.raises(FileNotFoundError):
            importer.import_from_csv("nonexistent_file.csv")

    def test_import_fra_with_tenor(self, tmp_path):
        """Test importing FRA with tenor notation"""
        csv_file = tmp_path / "fra_trades.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'product_type', 'currency', 'trade_date', 'notional',
                'fixed_rate', 'counterparty_id', 'fra_tenor'
            ])
            writer.writerow([
                'FRA', 'USD', '2024-01-15', '10000000',
                '0.045', 'CP-001', '3x6'
            ])
            writer.writerow([
                'FRA', 'EUR', '2024-01-15', '5000000',
                '0.035', 'CP-002', '6x12'
            ])

        importer = BatchTradeImporter()
        summary = importer.import_from_csv(csv_file)

        assert summary.total_rows == 2
        assert summary.successful >= 1

    def test_import_cap_and_floor(self, tmp_path):
        """Test importing Caps and Floors"""
        csv_file = tmp_path / "capfloor_trades.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'product_type', 'currency', 'trade_date', 'maturity_years',
                'notional', 'strike', 'counterparty_id', 'volatility'
            ])
            writer.writerow([
                'CAP', 'USD', '2024-01-15', '5.0',
                '100000000', '0.05', 'CP-001', '0.20'
            ])
            writer.writerow([
                'FLOOR', 'EUR', '2024-01-15', '5.0',
                '50000000', '0.02', 'CP-002', '0.15'
            ])

        importer = BatchTradeImporter()
        summary = importer.import_from_csv(csv_file)

        assert summary.total_rows == 2
        assert summary.successful >= 1

    def test_import_swaption(self, tmp_path):
        """Test importing Swaptions"""
        csv_file = tmp_path / "swaption_trades.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'product_type', 'currency', 'trade_date', 'option_maturity_years',
                'swap_maturity_years', 'notional', 'strike', 'counterparty_id',
                'is_payer', 'volatility'
            ])
            writer.writerow([
                'SWAPTION', 'USD', '2024-01-15', '1.0',
                '5.0', '100000000', '0.05', 'CP-001',
                'true', '0.20'
            ])

        importer = BatchTradeImporter()
        summary = importer.import_from_csv(csv_file)

        assert summary.total_rows == 1
        assert summary.successful >= 1

    def test_import_with_metadata(self, tmp_path):
        """Test importing with optional metadata fields"""
        csv_file = tmp_path / "trades_with_metadata.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'product_type', 'currency', 'trade_date', 'maturity_years',
                'notional', 'fixed_rate', 'counterparty_id',
                'trade_number', 'book_id', 'desk_id', 'trader_id'
            ])
            writer.writerow([
                'IRS', 'USD', '2024-01-15', '5.0',
                '100000000', '0.05', 'CP-001',
                'IRS-2024-001', 'BOOK-001', 'DESK-RATES', 'TRADER-123'
            ])

        importer = BatchTradeImporter()
        summary = importer.import_from_csv(csv_file)

        assert summary.total_rows == 1
        assert summary.successful >= 1

        # Check metadata was populated
        result = summary.results[0]
        if result.status == ImportStatus.SUCCESS:
            trade = result.trade_result.trade
            assert trade.trade_number == 'IRS-2024-001'
            assert trade.book_id == 'BOOK-001'
            assert trade.desk_id == 'DESK-RATES'
            assert trade.trader_id == 'TRADER-123'

    def test_summary_statistics(self, tmp_path):
        """Test summary statistics calculation"""
        csv_file = tmp_path / "mixed_results.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'product_type', 'currency', 'trade_date', 'maturity_years',
                'notional', 'fixed_rate', 'counterparty_id'
            ])
            # Success
            writer.writerow([
                'IRS', 'USD', '2024-01-15', '5.0',
                '100000000', '0.05', 'CP-001'
            ])
            # Success
            writer.writerow([
                'OIS', 'EUR', '2024-01-15', '5.0',
                '50000000', '0.035', 'CP-002'
            ])
            # Failure (invalid product type)
            writer.writerow([
                'INVALID', 'USD', '2024-01-15', '5.0',
                '100000000', '0.05', 'CP-003'
            ])

        importer = BatchTradeImporter()
        summary = importer.import_from_csv(csv_file)

        assert summary.total_rows == 3
        assert summary.successful >= 2
        assert summary.failed >= 1
        assert summary.successful + summary.failed + summary.warnings == summary.total_rows


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
