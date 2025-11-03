"""Tests for market conventions including IMM dates."""

from datetime import date

import pytest

from neutryx.market.conventions import (
    get_imm_code,
    get_imm_date,
    get_imm_dates_between,
    get_next_imm_date,
    is_imm_month,
    parse_imm_code,
)


class TestIMMMonths:
    """Tests for IMM month identification."""

    def test_imm_months(self):
        """Test that March, June, September, December are IMM months."""
        assert is_imm_month(3) is True
        assert is_imm_month(6) is True
        assert is_imm_month(9) is True
        assert is_imm_month(12) is True

    def test_non_imm_months(self):
        """Test that other months are not IMM months."""
        for month in [1, 2, 4, 5, 7, 8, 10, 11]:
            assert is_imm_month(month) is False

    def test_invalid_months(self):
        """Test invalid month numbers."""
        assert is_imm_month(0) is False
        assert is_imm_month(13) is False
        assert is_imm_month(-1) is False


class TestIMMDateCalculation:
    """Tests for IMM date calculation (3rd Wednesday)."""

    def test_march_2025(self):
        """Test March 2025 IMM date."""
        imm_date = get_imm_date(2025, 3)
        assert imm_date == date(2025, 3, 19)
        assert imm_date.weekday() == 2  # Wednesday

    def test_june_2025(self):
        """Test June 2025 IMM date."""
        imm_date = get_imm_date(2025, 6)
        assert imm_date == date(2025, 6, 18)
        assert imm_date.weekday() == 2  # Wednesday

    def test_september_2025(self):
        """Test September 2025 IMM date."""
        imm_date = get_imm_date(2025, 9)
        assert imm_date == date(2025, 9, 17)
        assert imm_date.weekday() == 2  # Wednesday

    def test_december_2025(self):
        """Test December 2025 IMM date."""
        imm_date = get_imm_date(2025, 12)
        assert imm_date == date(2025, 12, 17)
        assert imm_date.weekday() == 2  # Wednesday

    def test_all_2025_imm_dates(self):
        """Test all 2025 IMM dates are correct."""
        expected = {
            3: date(2025, 3, 19),
            6: date(2025, 6, 18),
            9: date(2025, 9, 17),
            12: date(2025, 12, 17),
        }
        for month, expected_date in expected.items():
            imm_date = get_imm_date(2025, month)
            assert imm_date == expected_date
            assert imm_date.weekday() == 2  # Wednesday

    def test_invalid_imm_month(self):
        """Test that non-IMM months raise ValueError."""
        with pytest.raises(ValueError, match="Month 1 is not an IMM month"):
            get_imm_date(2025, 1)


class TestIMMCodes:
    """Tests for IMM code generation and parsing."""

    def test_imm_code_2025(self):
        """Test IMM codes for 2025."""
        assert get_imm_code(date(2025, 3, 19)) == "H5"
        assert get_imm_code(date(2025, 6, 18)) == "M5"
        assert get_imm_code(date(2025, 9, 17)) == "U5"
        assert get_imm_code(date(2025, 12, 17)) == "Z5"

    def test_imm_code_2026(self):
        """Test IMM codes for 2026."""
        assert get_imm_code(date(2026, 3, 18)) == "H6"
        assert get_imm_code(date(2026, 6, 17)) == "M6"
        assert get_imm_code(date(2026, 9, 16)) == "U6"
        assert get_imm_code(date(2026, 12, 16)) == "Z6"

    def test_imm_code_invalid_month(self):
        """Test that non-IMM month raises ValueError."""
        with pytest.raises(ValueError, match="Date 2025-01-15 is not an IMM date"):
            get_imm_code(date(2025, 1, 15))

    def test_parse_imm_code_2025(self):
        """Test parsing IMM codes with explicit reference year."""
        assert parse_imm_code("H5", ref_year=2025) == date(2025, 3, 19)
        assert parse_imm_code("M5", ref_year=2025) == date(2025, 6, 18)
        assert parse_imm_code("U5", ref_year=2025) == date(2025, 9, 17)
        assert parse_imm_code("Z5", ref_year=2025) == date(2025, 12, 17)

    def test_parse_imm_code_decade_rollover(self):
        """Test parsing codes across decade boundary."""
        assert parse_imm_code("Z9", ref_year=2029) == date(2029, 12, 19)
        assert parse_imm_code("H0", ref_year=2030) == date(2030, 3, 20)

    def test_parse_imm_code_default_ref_year(self):
        """Test parsing with default reference year (current year)."""
        # Just check it doesn't raise an error
        result = parse_imm_code("H5")
        assert isinstance(result, date)
        assert result.month == 3
        assert result.weekday() == 2

    def test_parse_invalid_code_format(self):
        """Test that invalid code formats raise ValueError."""
        with pytest.raises(ValueError, match="Invalid IMM code format"):
            parse_imm_code("XX")
        with pytest.raises(ValueError, match="Invalid IMM code format"):
            parse_imm_code("H")
        with pytest.raises(ValueError, match="Invalid IMM code format"):
            parse_imm_code("123")

    def test_parse_invalid_month_code(self):
        """Test that invalid month codes raise ValueError."""
        with pytest.raises(ValueError, match="Invalid IMM month code"):
            parse_imm_code("X5")

    def test_roundtrip_conversion(self):
        """Test that date → code → date roundtrip works."""
        original_date = date(2025, 6, 18)
        code = get_imm_code(original_date)
        parsed_date = parse_imm_code(code, ref_year=2025)
        assert parsed_date == original_date


class TestNextIMMDate:
    """Tests for finding next IMM date."""

    def test_next_imm_from_january(self):
        """Test finding next IMM date from January."""
        ref_date = date(2025, 1, 15)
        next_imm = get_next_imm_date(ref_date)
        assert next_imm == date(2025, 3, 19)

    def test_next_imm_from_march_before_imm(self):
        """Test finding next IMM when before March IMM."""
        ref_date = date(2025, 3, 1)
        next_imm = get_next_imm_date(ref_date)
        assert next_imm == date(2025, 3, 19)

    def test_next_imm_from_march_on_imm(self):
        """Test that on IMM date, next is June IMM."""
        ref_date = date(2025, 3, 19)
        next_imm = get_next_imm_date(ref_date)
        assert next_imm == date(2025, 6, 18)

    def test_next_imm_from_march_after_imm(self):
        """Test finding next IMM when after March IMM."""
        ref_date = date(2025, 3, 20)
        next_imm = get_next_imm_date(ref_date)
        assert next_imm == date(2025, 6, 18)

    def test_next_imm_year_rollover(self):
        """Test next IMM date crosses year boundary."""
        ref_date = date(2025, 12, 20)
        next_imm = get_next_imm_date(ref_date)
        assert next_imm == date(2026, 3, 18)


class TestIMMDatesBetween:
    """Tests for getting all IMM dates in a range."""

    def test_imm_dates_full_year(self):
        """Test getting all IMM dates in a full year."""
        start = date(2025, 1, 1)
        end = date(2025, 12, 31)
        imm_dates = get_imm_dates_between(start, end)

        assert len(imm_dates) == 4
        assert imm_dates[0] == date(2025, 3, 19)
        assert imm_dates[1] == date(2025, 6, 18)
        assert imm_dates[2] == date(2025, 9, 17)
        assert imm_dates[3] == date(2025, 12, 17)

    def test_imm_dates_partial_year(self):
        """Test getting IMM dates for partial year."""
        start = date(2025, 5, 1)
        end = date(2025, 10, 31)
        imm_dates = get_imm_dates_between(start, end)

        assert len(imm_dates) == 2
        assert imm_dates[0] == date(2025, 6, 18)
        assert imm_dates[1] == date(2025, 9, 17)

    def test_imm_dates_cross_year(self):
        """Test getting IMM dates across year boundary."""
        start = date(2025, 11, 1)
        end = date(2026, 4, 30)
        imm_dates = get_imm_dates_between(start, end)

        assert len(imm_dates) == 2
        assert imm_dates[0] == date(2025, 12, 17)
        assert imm_dates[1] == date(2026, 3, 18)

    def test_imm_dates_empty_range(self):
        """Test that no IMM dates in range returns empty list."""
        start = date(2025, 1, 1)
        end = date(2025, 2, 28)
        imm_dates = get_imm_dates_between(start, end)

        assert len(imm_dates) == 0

    def test_imm_dates_includes_start(self):
        """Test that IMM date on start date is included."""
        start = date(2025, 3, 19)
        end = date(2025, 6, 30)
        imm_dates = get_imm_dates_between(start, end)

        assert len(imm_dates) == 2
        assert imm_dates[0] == date(2025, 3, 19)
        assert imm_dates[1] == date(2025, 6, 18)

    def test_imm_dates_excludes_end(self):
        """Test that IMM date on end date is excluded."""
        start = date(2025, 1, 1)
        end = date(2025, 6, 18)
        imm_dates = get_imm_dates_between(start, end)

        assert len(imm_dates) == 1
        assert imm_dates[0] == date(2025, 3, 19)

    def test_imm_dates_multiple_years(self):
        """Test getting IMM dates across multiple years."""
        start = date(2025, 1, 1)
        end = date(2027, 12, 31)
        imm_dates = get_imm_dates_between(start, end)

        # 4 IMM dates per year * 3 years = 12 dates
        assert len(imm_dates) == 12
        assert imm_dates[0] == date(2025, 3, 19)
        assert imm_dates[-1] == date(2027, 12, 15)


class TestIMMDatesIntegration:
    """Integration tests for IMM date functionality."""

    def test_futures_strip_schedule(self):
        """Test creating a complete futures strip schedule."""
        start = date(2025, 1, 1)
        imm_dates = get_imm_dates_between(start, date(2026, 12, 31))

        # Should have 8 quarterly contracts (2 years)
        assert len(imm_dates) == 8

        # Verify codes and dates
        codes = [get_imm_code(d) for d in imm_dates]
        assert codes == ["H5", "M5", "U5", "Z5", "H6", "M6", "U6", "Z6"]

        # Verify all are Wednesdays
        for imm_date in imm_dates:
            assert imm_date.weekday() == 2

    def test_code_parsing_consistency(self):
        """Test that all generated codes can be parsed back."""
        start = date(2025, 1, 1)
        imm_dates = get_imm_dates_between(start, date(2025, 12, 31))

        for original_date in imm_dates:
            code = get_imm_code(original_date)
            parsed_date = parse_imm_code(code, ref_year=2025)
            assert parsed_date == original_date
