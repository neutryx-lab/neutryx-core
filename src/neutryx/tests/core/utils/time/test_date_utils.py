"""Tests for date utility functions."""

import datetime
import pytest

from neutryx.core.utils.time.date_utils import (
    is_weekend,
    is_leap_year,
    days_in_month,
    days_in_year,
    add_months,
    add_years,
    date_diff_days,
    date_to_serial,
    serial_to_date,
    end_of_month,
    is_end_of_month,
    third_wednesday,
    imm_date,
    easter_date,
)


class TestWeekendAndLeapYear:
    """Tests for weekend and leap year detection."""

    def test_is_weekend(self):
        """Test weekend detection."""
        # Saturday
        assert is_weekend(datetime.date(2024, 1, 6))
        # Sunday
        assert is_weekend(datetime.date(2024, 1, 7))
        # Monday
        assert not is_weekend(datetime.date(2024, 1, 8))
        # Friday
        assert not is_weekend(datetime.date(2024, 1, 5))

    def test_is_leap_year(self):
        """Test leap year detection."""
        assert is_leap_year(2024)  # Divisible by 4
        assert is_leap_year(2000)  # Divisible by 400
        assert not is_leap_year(2023)  # Not divisible by 4
        assert not is_leap_year(1900)  # Divisible by 100 but not 400


class TestDaysInPeriod:
    """Tests for days in month/year calculations."""

    def test_days_in_month(self):
        """Test days in month calculation."""
        assert days_in_month(2024, 1) == 31
        assert days_in_month(2024, 2) == 29  # Leap year
        assert days_in_month(2023, 2) == 28  # Non-leap year
        assert days_in_month(2024, 4) == 30
        assert days_in_month(2024, 12) == 31

    def test_days_in_month_invalid(self):
        """Test invalid month raises error."""
        with pytest.raises(ValueError):
            days_in_month(2024, 13)

    def test_days_in_year(self):
        """Test days in year calculation."""
        assert days_in_year(2024) == 366  # Leap year
        assert days_in_year(2023) == 365  # Non-leap year


class TestDateArithmetic:
    """Tests for date arithmetic operations."""

    def test_add_months(self):
        """Test adding months to dates."""
        # Normal case
        assert add_months(datetime.date(2024, 1, 15), 1) == datetime.date(2024, 2, 15)
        assert add_months(datetime.date(2024, 1, 15), 12) == datetime.date(2025, 1, 15)

        # Month-end adjustment
        assert add_months(datetime.date(2024, 1, 31), 1) == datetime.date(2024, 2, 29)
        assert add_months(datetime.date(2024, 3, 31), 1) == datetime.date(2024, 4, 30)

        # Negative months
        assert add_months(datetime.date(2024, 3, 15), -1) == datetime.date(2024, 2, 15)
        assert add_months(datetime.date(2024, 3, 31), -1) == datetime.date(2024, 2, 29)

    def test_add_years(self):
        """Test adding years to dates."""
        # Normal case
        assert add_years(datetime.date(2024, 6, 15), 1) == datetime.date(2025, 6, 15)
        assert add_years(datetime.date(2024, 6, 15), -1) == datetime.date(2023, 6, 15)

        # Leap year adjustment
        assert add_years(datetime.date(2024, 2, 29), 1) == datetime.date(2025, 2, 28)
        assert add_years(datetime.date(2024, 2, 29), 4) == datetime.date(2028, 2, 29)

    def test_date_diff_days(self):
        """Test calculating day difference."""
        date1 = datetime.date(2024, 1, 1)
        date2 = datetime.date(2024, 1, 31)
        assert date_diff_days(date1, date2) == 30
        assert date_diff_days(date2, date1) == -30

        # Cross-year
        date1 = datetime.date(2023, 12, 31)
        date2 = datetime.date(2024, 1, 1)
        assert date_diff_days(date1, date2) == 1


class TestSerialConversion:
    """Tests for serial number conversion."""

    def test_date_to_serial(self):
        """Test converting date to serial."""
        base = datetime.date(1899, 12, 30)
        date = datetime.date(1900, 1, 1)
        assert date_to_serial(date) == 2

        date = datetime.date(2024, 1, 1)
        serial = date_to_serial(date)
        assert serial > 0

    def test_serial_to_date(self):
        """Test converting serial to date."""
        base = datetime.date(1899, 12, 30)
        assert serial_to_date(2) == datetime.date(1900, 1, 1)

    def test_serial_round_trip(self):
        """Test round-trip conversion."""
        original = datetime.date(2024, 6, 15)
        serial = date_to_serial(original)
        converted = serial_to_date(serial)
        assert converted == original


class TestMonthEnd:
    """Tests for month-end operations."""

    def test_end_of_month(self):
        """Test getting end of month."""
        assert end_of_month(datetime.date(2024, 1, 15)) == datetime.date(2024, 1, 31)
        assert end_of_month(datetime.date(2024, 2, 1)) == datetime.date(2024, 2, 29)
        assert end_of_month(datetime.date(2023, 2, 15)) == datetime.date(2023, 2, 28)

    def test_is_end_of_month(self):
        """Test checking if date is end of month."""
        assert is_end_of_month(datetime.date(2024, 1, 31))
        assert is_end_of_month(datetime.date(2024, 2, 29))
        assert not is_end_of_month(datetime.date(2024, 1, 15))
        assert not is_end_of_month(datetime.date(2024, 2, 28))


class TestSpecialDates:
    """Tests for special date calculations."""

    def test_third_wednesday(self):
        """Test calculating third Wednesday."""
        # January 2024
        wed = third_wednesday(2024, 1)
        assert wed.weekday() == 2  # Wednesday
        assert 15 <= wed.day <= 21

    def test_imm_date(self):
        """Test calculating IMM dates."""
        # Q1 2024 - March
        imm = imm_date(2024, 1)
        assert imm.month == 3
        assert imm.weekday() == 2

        # Q2 2024 - June
        imm = imm_date(2024, 2)
        assert imm.month == 6

        # Q3 2024 - September
        imm = imm_date(2024, 3)
        assert imm.month == 9

        # Q4 2024 - December
        imm = imm_date(2024, 4)
        assert imm.month == 12

    def test_imm_date_invalid_quarter(self):
        """Test invalid quarter raises error."""
        with pytest.raises(ValueError):
            imm_date(2024, 5)

    def test_easter_date(self):
        """Test Easter date calculation."""
        # Known Easter dates
        assert easter_date(2024) == datetime.date(2024, 3, 31)
        assert easter_date(2025) == datetime.date(2025, 4, 20)
        assert easter_date(2023) == datetime.date(2023, 4, 9)

        # Easter should always be in March or April
        for year in range(2020, 2030):
            easter = easter_date(year)
            assert easter.month in (3, 4)
