"""Tests for business day conventions."""

import datetime
import pytest

from neutryx.core.dates.business_day import (
    Unadjusted,
    Following,
    Preceding,
    ModifiedFollowing,
    ModifiedPreceding,
    Nearest,
)
from neutryx.core.dates.calendar import NullCalendar, USCalendar


class TestUnadjusted:
    """Tests for unadjusted convention."""

    def test_no_adjustment(self):
        """Test that dates are not adjusted."""
        conv = Unadjusted()
        cal = NullCalendar()

        # Saturday (non-business day)
        date = datetime.date(2024, 1, 6)
        adjusted = conv.adjust(date, cal)
        assert adjusted == date

        # Monday (business day)
        date = datetime.date(2024, 1, 1)
        adjusted = conv.adjust(date, cal)
        assert adjusted == date


class TestFollowing:
    """Tests for following business day convention."""

    def test_business_day_unchanged(self):
        """Test that business days are not adjusted."""
        conv = Following()
        cal = NullCalendar()

        # Monday
        date = datetime.date(2024, 1, 1)
        adjusted = conv.adjust(date, cal)
        assert adjusted == date

    def test_saturday_to_monday(self):
        """Test Saturday adjusted to Monday."""
        conv = Following()
        cal = NullCalendar()

        # Saturday
        date = datetime.date(2024, 1, 6)
        adjusted = conv.adjust(date, cal)
        # Should move to Monday
        assert adjusted == datetime.date(2024, 1, 8)

    def test_sunday_to_monday(self):
        """Test Sunday adjusted to Monday."""
        conv = Following()
        cal = NullCalendar()

        # Sunday
        date = datetime.date(2024, 1, 7)
        adjusted = conv.adjust(date, cal)
        # Should move to Monday
        assert adjusted == datetime.date(2024, 1, 8)

    def test_with_holidays(self):
        """Test following convention with holidays."""
        conv = Following()
        cal = USCalendar()

        # New Year's Day 2024 (Monday)
        date = datetime.date(2024, 1, 1)
        adjusted = conv.adjust(date, cal)
        # Should move to Tuesday
        assert adjusted == datetime.date(2024, 1, 2)


class TestPreceding:
    """Tests for preceding business day convention."""

    def test_business_day_unchanged(self):
        """Test that business days are not adjusted."""
        conv = Preceding()
        cal = NullCalendar()

        # Monday
        date = datetime.date(2024, 1, 1)
        adjusted = conv.adjust(date, cal)
        assert adjusted == date

    def test_saturday_to_friday(self):
        """Test Saturday adjusted to Friday."""
        conv = Preceding()
        cal = NullCalendar()

        # Saturday
        date = datetime.date(2024, 1, 6)
        adjusted = conv.adjust(date, cal)
        # Should move to Friday
        assert adjusted == datetime.date(2024, 1, 5)

    def test_sunday_to_friday(self):
        """Test Sunday adjusted to Friday."""
        conv = Preceding()
        cal = NullCalendar()

        # Sunday
        date = datetime.date(2024, 1, 7)
        adjusted = conv.adjust(date, cal)
        # Should move to Friday
        assert adjusted == datetime.date(2024, 1, 5)


class TestModifiedFollowing:
    """Tests for modified following convention."""

    def test_business_day_unchanged(self):
        """Test that business days are not adjusted."""
        conv = ModifiedFollowing()
        cal = NullCalendar()

        # Monday
        date = datetime.date(2024, 1, 1)
        adjusted = conv.adjust(date, cal)
        assert adjusted == date

    def test_saturday_same_month(self):
        """Test Saturday adjusted forward in same month."""
        conv = ModifiedFollowing()
        cal = NullCalendar()

        # Saturday, January 6
        date = datetime.date(2024, 1, 6)
        adjusted = conv.adjust(date, cal)
        # Should move to Monday, January 8 (same month)
        assert adjusted == datetime.date(2024, 1, 8)

    def test_month_end_saturday(self):
        """Test month-end Saturday adjusted backward."""
        conv = ModifiedFollowing()
        cal = NullCalendar()

        # Saturday, March 30, 2024
        date = datetime.date(2024, 3, 30)
        adjusted = conv.adjust(date, cal)
        # Following would be April 1, but that crosses month
        # So should move backward to Friday, March 29
        assert adjusted == datetime.date(2024, 3, 29)

    def test_month_end_sunday(self):
        """Test month-end Sunday adjusted backward."""
        conv = ModifiedFollowing()
        cal = NullCalendar()

        # Sunday, December 31, 2023
        date = datetime.date(2023, 12, 31)
        adjusted = conv.adjust(date, cal)
        # Following would be January 1, 2024, but that crosses month
        # So should move backward to Friday, December 29
        assert adjusted == datetime.date(2023, 12, 29)


class TestModifiedPreceding:
    """Tests for modified preceding convention."""

    def test_business_day_unchanged(self):
        """Test that business days are not adjusted."""
        conv = ModifiedPreceding()
        cal = NullCalendar()

        # Monday
        date = datetime.date(2024, 1, 8)
        adjusted = conv.adjust(date, cal)
        assert adjusted == date

    def test_saturday_same_month(self):
        """Test Saturday adjusted backward in same month."""
        conv = ModifiedPreceding()
        cal = NullCalendar()

        # Saturday, January 6
        date = datetime.date(2024, 1, 6)
        adjusted = conv.adjust(date, cal)
        # Should move to Friday, January 5 (same month)
        assert adjusted == datetime.date(2024, 1, 5)

    def test_month_start_sunday(self):
        """Test month-start Sunday adjusted forward."""
        conv = ModifiedPreceding()
        cal = NullCalendar()

        # Sunday, September 1, 2024
        date = datetime.date(2024, 9, 1)
        adjusted = conv.adjust(date, cal)
        # Preceding would be August 30, but that crosses month
        # So should move forward to Monday, September 2
        assert adjusted == datetime.date(2024, 9, 2)


class TestNearest:
    """Tests for nearest business day convention."""

    def test_business_day_unchanged(self):
        """Test that business days are not adjusted."""
        conv = Nearest()
        cal = NullCalendar()

        # Monday
        date = datetime.date(2024, 1, 1)
        adjusted = conv.adjust(date, cal)
        assert adjusted == date

    def test_saturday_to_friday(self):
        """Test Saturday adjusted to Friday (nearer)."""
        conv = Nearest()
        cal = NullCalendar()

        # Saturday, January 6
        date = datetime.date(2024, 1, 6)
        adjusted = conv.adjust(date, cal)
        # Friday is 1 day back, Monday is 2 days forward
        # So should move to Friday
        assert adjusted == datetime.date(2024, 1, 5)

    def test_sunday_to_monday(self):
        """Test Sunday adjusted to Monday (equidistant, choose forward)."""
        conv = Nearest()
        cal = NullCalendar()

        # Sunday, January 7
        date = datetime.date(2024, 1, 7)
        adjusted = conv.adjust(date, cal)
        # Friday is 2 days back, Monday is 1 day forward
        # So should move to Monday
        assert adjusted == datetime.date(2024, 1, 8)


class TestConventionComparison:
    """Tests comparing different conventions on same dates."""

    def test_all_conventions_on_saturday(self):
        """Test all conventions on a Saturday."""
        cal = NullCalendar()
        date = datetime.date(2024, 1, 6)  # Saturday

        unadj = Unadjusted().adjust(date, cal)
        following = Following().adjust(date, cal)
        preceding = Preceding().adjust(date, cal)
        mod_following = ModifiedFollowing().adjust(date, cal)
        nearest = Nearest().adjust(date, cal)

        # Unadjusted stays same
        assert unadj == date

        # Following goes to Monday
        assert following == datetime.date(2024, 1, 8)

        # Preceding goes to Friday
        assert preceding == datetime.date(2024, 1, 5)

        # Modified following goes to Monday (same month)
        assert mod_following == datetime.date(2024, 1, 8)

        # Nearest goes to Friday (closer)
        assert nearest == datetime.date(2024, 1, 5)

    def test_month_end_conventions(self):
        """Test conventions on month-end dates."""
        cal = NullCalendar()
        # Saturday, March 30, 2024 (last weekend of month)
        date = datetime.date(2024, 3, 30)

        following = Following().adjust(date, cal)
        mod_following = ModifiedFollowing().adjust(date, cal)

        # Following crosses to next month
        assert following == datetime.date(2024, 4, 1)

        # Modified following stays in March
        assert mod_following == datetime.date(2024, 3, 29)


class TestWithHolidayCalendar:
    """Tests business day conventions with holiday calendars."""

    def test_following_over_holiday(self):
        """Test following convention over a holiday."""
        conv = Following()
        cal = USCalendar()

        # New Year's Eve 2023 (Sunday)
        # -> Jan 1, 2024 (Monday, holiday)
        # -> Jan 2, 2024 (Tuesday, business day)
        date = datetime.date(2023, 12, 31)
        adjusted = conv.adjust(date, cal)
        assert adjusted == datetime.date(2024, 1, 2)

    def test_modified_following_with_holidays(self):
        """Test modified following with holidays."""
        conv = ModifiedFollowing()
        cal = USCalendar()

        # Independence Day 2024 (Thursday, July 4)
        date = datetime.date(2024, 7, 4)
        adjusted = conv.adjust(date, cal)
        # Should move to Friday, July 5
        assert adjusted == datetime.date(2024, 7, 5)
