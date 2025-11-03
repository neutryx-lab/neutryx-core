"""Tests for day count conventions."""

import datetime
import pytest

from neutryx.core.utils.time.day_count import (
    Actual360,
    Actual365Fixed,
    ActualActual,
    ActualActualISDA,
    ActualActualICMA,
    Thirty360,
    ThirtyE360,
    Business252,
)


class TestActual360:
    """Tests for Actual/360 day count convention."""

    def test_day_count(self):
        """Test day count calculation."""
        dc = Actual360()
        date1 = datetime.date(2024, 1, 1)
        date2 = datetime.date(2024, 7, 1)
        days = dc.day_count(date1, date2)
        assert days == 182

    def test_year_fraction(self):
        """Test year fraction calculation."""
        dc = Actual360()
        date1 = datetime.date(2024, 1, 1)
        date2 = datetime.date(2024, 7, 1)
        yf = dc.year_fraction(date1, date2)
        assert yf == pytest.approx(182 / 360.0)

    def test_full_year(self):
        """Test year fraction for one year."""
        dc = Actual360()
        date1 = datetime.date(2024, 1, 1)
        date2 = datetime.date(2025, 1, 1)
        yf = dc.year_fraction(date1, date2)
        # 2024 is leap year: 366 days
        assert yf == pytest.approx(366 / 360.0)


class TestActual365Fixed:
    """Tests for Actual/365 Fixed day count convention."""

    def test_year_fraction(self):
        """Test year fraction calculation."""
        dc = Actual365Fixed()
        date1 = datetime.date(2024, 1, 1)
        date2 = datetime.date(2024, 7, 1)
        yf = dc.year_fraction(date1, date2)
        assert yf == pytest.approx(182 / 365.0)

    def test_full_year(self):
        """Test year fraction for one year."""
        dc = Actual365Fixed()
        date1 = datetime.date(2024, 1, 1)
        date2 = datetime.date(2025, 1, 1)
        yf = dc.year_fraction(date1, date2)
        # Fixed 365 denominator even in leap year
        assert yf == pytest.approx(366 / 365.0)


class TestActualActual:
    """Tests for Actual/Actual day count convention."""

    def test_same_year(self):
        """Test year fraction within same year."""
        dc = ActualActual()
        date1 = datetime.date(2024, 1, 1)
        date2 = datetime.date(2024, 7, 1)
        yf = dc.year_fraction(date1, date2)
        # 2024 is leap year
        assert yf == pytest.approx(182 / 366.0)

    def test_cross_year(self):
        """Test year fraction crossing years."""
        dc = ActualActual()
        date1 = datetime.date(2023, 7, 1)
        date2 = datetime.date(2024, 7, 1)
        yf = dc.year_fraction(date1, date2)
        # Should be close to 1.0
        assert yf == pytest.approx(1.0, rel=0.01)

    def test_full_year_leap(self):
        """Test year fraction for full leap year."""
        dc = ActualActual()
        date1 = datetime.date(2024, 1, 1)
        date2 = datetime.date(2025, 1, 1)
        yf = dc.year_fraction(date1, date2)
        assert yf == pytest.approx(1.0)

    def test_full_year_non_leap(self):
        """Test year fraction for full non-leap year."""
        dc = ActualActual()
        date1 = datetime.date(2023, 1, 1)
        date2 = datetime.date(2024, 1, 1)
        yf = dc.year_fraction(date1, date2)
        assert yf == pytest.approx(1.0)


class TestActualActualISDA:
    """Tests for Actual/Actual ISDA day count convention."""

    def test_same_year(self):
        """Test year fraction within same year."""
        dc = ActualActualISDA()
        date1 = datetime.date(2024, 1, 1)
        date2 = datetime.date(2024, 7, 1)
        yf = dc.year_fraction(date1, date2)
        # 2024 is leap year
        assert yf == pytest.approx(182 / 366.0)

    def test_cross_year_leap_to_normal(self):
        """Test year fraction from leap to normal year."""
        dc = ActualActualISDA()
        date1 = datetime.date(2024, 7, 1)
        date2 = datetime.date(2025, 7, 1)
        yf = dc.year_fraction(date1, date2)
        # Split calculation: 184 days in 2024 (leap) + 181 days in 2025 (normal)
        expected = 184 / 366.0 + 181 / 365.0
        assert yf == pytest.approx(expected)


class TestActualActualICMA:
    """Tests for Actual/Actual ICMA day count convention."""

    def test_with_reference_dates(self):
        """Test year fraction with reference dates."""
        dc = ActualActualICMA(frequency=2)  # Semi-annual
        date1 = datetime.date(2024, 1, 1)
        date2 = datetime.date(2024, 4, 1)
        ref_start = datetime.date(2024, 1, 1)
        ref_end = datetime.date(2024, 7, 1)

        yf = dc.year_fraction(date1, date2, ref_start, ref_end)
        days = 91  # Jan 1 to Apr 1 in leap year
        days_in_period = 182  # Jan 1 to Jul 1
        expected = days / (days_in_period * 2)
        assert yf == pytest.approx(expected)

    def test_missing_reference_dates(self):
        """Test that missing reference dates raises error."""
        dc = ActualActualICMA()
        date1 = datetime.date(2024, 1, 1)
        date2 = datetime.date(2024, 7, 1)

        with pytest.raises(ValueError):
            dc.year_fraction(date1, date2)


class TestThirty360:
    """Tests for 30/360 day count convention."""

    def test_regular_period(self):
        """Test year fraction for regular period."""
        dc = Thirty360()
        date1 = datetime.date(2024, 1, 1)
        date2 = datetime.date(2024, 7, 1)
        yf = dc.year_fraction(date1, date2)
        # 30/360: 6 months * 30 days = 180 days
        assert yf == pytest.approx(180 / 360.0)

    def test_day_count_same_month(self):
        """Test day count within same month."""
        dc = Thirty360()
        date1 = datetime.date(2024, 1, 1)
        date2 = datetime.date(2024, 1, 31)
        days = dc.day_count(date1, date2)
        # 30/360: 31st adjusted to 30th
        assert days == 30

    def test_day_count_31st_adjustments(self):
        """Test 31st day adjustments."""
        dc = Thirty360()
        # Both dates on 31st
        date1 = datetime.date(2024, 1, 31)
        date2 = datetime.date(2024, 3, 31)
        days = dc.day_count(date1, date2)
        # Jan 31 -> 30, Mar 31 -> 30
        # 360 * 0 + 30 * 2 + 0 = 60
        assert days == 60

    def test_full_year(self):
        """Test year fraction for one year."""
        dc = Thirty360()
        date1 = datetime.date(2024, 1, 1)
        date2 = datetime.date(2025, 1, 1)
        yf = dc.year_fraction(date1, date2)
        assert yf == pytest.approx(1.0)


class TestThirtyE360:
    """Tests for 30E/360 day count convention."""

    def test_regular_period(self):
        """Test year fraction for regular period."""
        dc = ThirtyE360()
        date1 = datetime.date(2024, 1, 1)
        date2 = datetime.date(2024, 7, 1)
        yf = dc.year_fraction(date1, date2)
        assert yf == pytest.approx(180 / 360.0)

    def test_day_count_31st_adjustments(self):
        """Test 31st day adjustments (European style)."""
        dc = ThirtyE360()
        # Both dates on 31st
        date1 = datetime.date(2024, 1, 31)
        date2 = datetime.date(2024, 3, 31)
        days = dc.day_count(date1, date2)
        # Jan 31 -> 30, Mar 31 -> 30 (both adjusted)
        assert days == 60

    def test_comparison_with_30_360(self):
        """Test difference between 30E/360 and 30/360."""
        date1 = datetime.date(2024, 1, 30)
        date2 = datetime.date(2024, 2, 31)  # Invalid date, but for testing

        # For valid dates, should be similar but can differ on edge cases
        date1 = datetime.date(2024, 1, 15)
        date2 = datetime.date(2024, 3, 31)

        dc_us = Thirty360()
        dc_eu = ThirtyE360()

        days_us = dc_us.day_count(date1, date2)
        days_eu = dc_eu.day_count(date1, date2)

        # Both should give same result for this case
        assert days_us == days_eu


class TestBusiness252:
    """Tests for Business/252 day count convention."""

    def test_weekdays_only(self):
        """Test counting business days (weekdays)."""
        dc = Business252()
        # Monday to Friday (same week)
        date1 = datetime.date(2024, 1, 1)  # Monday
        date2 = datetime.date(2024, 1, 6)  # Saturday
        days = dc.day_count(date1, date2)
        # Mon, Tue, Wed, Thu, Fri = 5 days
        assert days == 5

    def test_year_fraction(self):
        """Test year fraction calculation."""
        dc = Business252()
        date1 = datetime.date(2024, 1, 1)
        date2 = datetime.date(2024, 1, 6)
        yf = dc.year_fraction(date1, date2)
        days = dc.day_count(date1, date2)
        assert yf == pytest.approx(days / 252.0)

    def test_excluding_weekends(self):
        """Test that weekends are excluded."""
        dc = Business252()
        # Friday to Monday
        date1 = datetime.date(2024, 1, 5)  # Friday
        date2 = datetime.date(2024, 1, 8)  # Monday
        days = dc.day_count(date1, date2)
        # Friday is included, Sat/Sun excluded, Monday not included (exclusive)
        assert days == 1  # Only Friday


class TestDayCountComparison:
    """Tests comparing different day count conventions."""

    def test_same_dates_different_conventions(self):
        """Test that different conventions give different results."""
        date1 = datetime.date(2024, 1, 1)
        date2 = datetime.date(2024, 7, 1)

        act360 = Actual360().year_fraction(date1, date2)
        act365 = Actual365Fixed().year_fraction(date1, date2)
        actact = ActualActual().year_fraction(date1, date2)
        thirty360 = Thirty360().year_fraction(date1, date2)

        # All should be different
        assert act360 != act365
        assert act360 != actact
        assert act360 != thirty360

        # But all should be around 0.5 (half year)
        assert 0.49 < act360 < 0.51
        assert 0.49 < act365 < 0.51
        assert 0.49 < actact < 0.51
        assert 0.49 < thirty360 < 0.51
