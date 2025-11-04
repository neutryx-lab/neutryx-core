"""Tests for holiday calendars."""

import datetime
import pytest

from neutryx.core.dates.calendar import (
    NullCalendar,
    TargetCalendar,
    USCalendar,
    UKCalendar,
    JPCalendar,
    JointCalendar,
)


class TestNullCalendar:
    """Tests for null calendar (no holidays)."""

    def test_weekday_is_business_day(self):
        """Test that weekdays are business days."""
        cal = NullCalendar()
        # Monday
        assert cal.is_business_day(datetime.date(2024, 1, 1))
        # Friday
        assert cal.is_business_day(datetime.date(2024, 1, 5))

    def test_weekend_not_business_day(self):
        """Test that weekends are not business days."""
        cal = NullCalendar()
        # Saturday
        assert not cal.is_business_day(datetime.date(2024, 1, 6))
        # Sunday
        assert not cal.is_business_day(datetime.date(2024, 1, 7))

    def test_is_holiday(self):
        """Test holiday detection."""
        cal = NullCalendar()
        assert cal.is_holiday(datetime.date(2024, 1, 6))  # Saturday
        assert not cal.is_holiday(datetime.date(2024, 1, 5))  # Friday


class TestTargetCalendar:
    """Tests for TARGET calendar."""

    def test_new_years_day(self):
        """Test New Year's Day holiday."""
        cal = TargetCalendar()
        assert not cal.is_business_day(datetime.date(2024, 1, 1))

    def test_labour_day(self):
        """Test Labour Day (May 1) holiday."""
        cal = TargetCalendar()
        # May 1, 2024 is Wednesday
        assert not cal.is_business_day(datetime.date(2024, 5, 1))

    def test_christmas(self):
        """Test Christmas holidays."""
        cal = TargetCalendar()
        # December 25
        assert not cal.is_business_day(datetime.date(2024, 12, 25))
        # December 26 (Boxing Day)
        assert not cal.is_business_day(datetime.date(2024, 12, 26))

    def test_good_friday(self):
        """Test Good Friday holiday."""
        cal = TargetCalendar()
        # Good Friday 2024: March 29
        assert not cal.is_business_day(datetime.date(2024, 3, 29))

    def test_easter_monday(self):
        """Test Easter Monday holiday."""
        cal = TargetCalendar()
        # Easter Monday 2024: April 1
        assert not cal.is_business_day(datetime.date(2024, 4, 1))

    def test_regular_business_day(self):
        """Test regular business day."""
        cal = TargetCalendar()
        # Random Tuesday
        assert cal.is_business_day(datetime.date(2024, 6, 18))


class TestUSCalendar:
    """Tests for US calendar."""

    def test_new_years_day(self):
        """Test New Year's Day holiday."""
        cal = USCalendar()
        # January 1, 2024 is Monday
        assert not cal.is_business_day(datetime.date(2024, 1, 1))

    def test_independence_day(self):
        """Test Independence Day holiday."""
        cal = USCalendar()
        # July 4, 2024 is Thursday
        assert not cal.is_business_day(datetime.date(2024, 7, 4))

    def test_independence_day_observed(self):
        """Test Independence Day observed on different day."""
        cal = USCalendar()
        # July 4, 2020 is Saturday, observed on Friday July 3
        assert not cal.is_business_day(datetime.date(2020, 7, 3))
        # July 4, 2021 is Sunday, observed on Monday July 5
        assert not cal.is_business_day(datetime.date(2021, 7, 5))

    def test_christmas(self):
        """Test Christmas holiday."""
        cal = USCalendar()
        # December 25, 2024 is Wednesday
        assert not cal.is_business_day(datetime.date(2024, 12, 25))

    def test_thanksgiving(self):
        """Test Thanksgiving holiday."""
        cal = USCalendar()
        # Thanksgiving 2024: 4th Thursday of November (Nov 28)
        assert not cal.is_business_day(datetime.date(2024, 11, 28))

    def test_mlk_day(self):
        """Test Martin Luther King Jr. Day."""
        cal = USCalendar()
        # MLK Day 2024: 3rd Monday in January (Jan 15)
        assert not cal.is_business_day(datetime.date(2024, 1, 15))

    def test_good_friday(self):
        """Test Good Friday holiday."""
        cal = USCalendar()
        # Good Friday 2024: March 29
        assert not cal.is_business_day(datetime.date(2024, 3, 29))


class TestUKCalendar:
    """Tests for UK calendar."""

    def test_new_years_day(self):
        """Test New Year's Day holiday."""
        cal = UKCalendar()
        # January 1, 2024 is Monday
        assert not cal.is_business_day(datetime.date(2024, 1, 1))

    def test_easter_monday(self):
        """Test Easter Monday holiday."""
        cal = UKCalendar()
        # Easter Monday 2024: April 1
        assert not cal.is_business_day(datetime.date(2024, 4, 1))

    def test_early_may_bank_holiday(self):
        """Test Early May Bank Holiday."""
        cal = UKCalendar()
        # First Monday in May 2024: May 6
        assert not cal.is_business_day(datetime.date(2024, 5, 6))

    def test_spring_bank_holiday(self):
        """Test Spring Bank Holiday."""
        cal = UKCalendar()
        # Last Monday in May 2024: May 27
        assert not cal.is_business_day(datetime.date(2024, 5, 27))

    def test_summer_bank_holiday(self):
        """Test Summer Bank Holiday."""
        cal = UKCalendar()
        # Last Monday in August 2024: Aug 26
        assert not cal.is_business_day(datetime.date(2024, 8, 26))

    def test_christmas_and_boxing_day(self):
        """Test Christmas and Boxing Day."""
        cal = UKCalendar()
        # December 25, 2024 is Wednesday
        assert not cal.is_business_day(datetime.date(2024, 12, 25))
        # December 26, 2024 is Thursday
        assert not cal.is_business_day(datetime.date(2024, 12, 26))


class TestJPCalendar:
    """Tests for Japanese calendar."""

    def test_new_years_day(self):
        """Test New Year's Day holiday."""
        cal = JPCalendar()
        assert not cal.is_business_day(datetime.date(2024, 1, 1))

    def test_national_foundation_day(self):
        """Test National Foundation Day."""
        cal = JPCalendar()
        # February 11
        assert not cal.is_business_day(datetime.date(2024, 2, 11))

    def test_golden_week(self):
        """Test Golden Week holidays."""
        cal = JPCalendar()
        # Showa Day (April 29)
        assert not cal.is_business_day(datetime.date(2024, 4, 29))
        # Constitution Day (May 3)
        assert not cal.is_business_day(datetime.date(2024, 5, 3))
        # Greenery Day (May 4)
        assert not cal.is_business_day(datetime.date(2024, 5, 4))
        # Children's Day (May 5)
        assert not cal.is_business_day(datetime.date(2024, 5, 5))

    def test_culture_day(self):
        """Test Culture Day."""
        cal = JPCalendar()
        # November 3
        assert not cal.is_business_day(datetime.date(2024, 11, 3))

    def test_coming_of_age_day(self):
        """Test Coming of Age Day."""
        cal = JPCalendar()
        # 2nd Monday in January 2024: Jan 8
        assert not cal.is_business_day(datetime.date(2024, 1, 8))


class TestJointCalendar:
    """Tests for joint calendar."""

    def test_us_and_uk_joint(self):
        """Test joint US and UK calendar."""
        us = USCalendar()
        uk = UKCalendar()
        joint = JointCalendar(us, uk)

        # US Independence Day (not UK holiday)
        us_only = datetime.date(2024, 7, 4)
        assert not us.is_business_day(us_only)
        assert uk.is_business_day(us_only)
        assert not joint.is_business_day(us_only)  # Holiday in either = holiday in joint

        # UK Summer Bank Holiday (not US holiday) - Aug 26, 2024
        uk_only = datetime.date(2024, 8, 26)
        assert us.is_business_day(uk_only)
        assert not uk.is_business_day(uk_only)
        assert not joint.is_business_day(uk_only)

        # Good Friday (both)
        both = datetime.date(2024, 3, 29)
        assert not us.is_business_day(both)
        assert not uk.is_business_day(both)
        assert not joint.is_business_day(both)

        # Regular business day
        regular = datetime.date(2024, 6, 18)
        assert us.is_business_day(regular)
        assert uk.is_business_day(regular)
        assert joint.is_business_day(regular)

    def test_empty_joint_calendar(self):
        """Test that empty joint calendar raises error."""
        with pytest.raises(ValueError):
            JointCalendar()


class TestCalendarAdvance:
    """Tests for advancing dates by business days."""

    def test_advance_forward(self):
        """Test advancing forward by business days."""
        cal = NullCalendar()
        # Friday
        date = datetime.date(2024, 1, 5)
        # Advance 1 business day -> Monday
        advanced = cal.advance(date, 1)
        assert advanced == datetime.date(2024, 1, 8)

    def test_advance_backward(self):
        """Test advancing backward by business days."""
        cal = NullCalendar()
        # Monday
        date = datetime.date(2024, 1, 8)
        # Advance -1 business day -> Friday
        advanced = cal.advance(date, -1)
        assert advanced == datetime.date(2024, 1, 5)

    def test_advance_zero(self):
        """Test advancing by zero days."""
        cal = NullCalendar()
        date = datetime.date(2024, 1, 5)
        advanced = cal.advance(date, 0)
        assert advanced == date

    def test_advance_over_weekend(self):
        """Test advancing over weekend."""
        cal = NullCalendar()
        # Thursday
        date = datetime.date(2024, 1, 4)
        # Advance 3 business days -> Tuesday (skip weekend)
        advanced = cal.advance(date, 3)
        assert advanced == datetime.date(2024, 1, 9)


class TestBusinessDaysBetween:
    """Tests for counting business days between dates."""

    def test_same_week(self):
        """Test counting business days within same week."""
        cal = NullCalendar()
        # Monday to Friday
        date1 = datetime.date(2024, 1, 1)
        date2 = datetime.date(2024, 1, 6)
        count = cal.business_days_between(date1, date2)
        # Mon, Tue, Wed, Thu, Fri = 5 days
        assert count == 5

    def test_over_weekend(self):
        """Test counting business days over weekend."""
        cal = NullCalendar()
        # Friday to next Monday
        date1 = datetime.date(2024, 1, 5)
        date2 = datetime.date(2024, 1, 8)
        count = cal.business_days_between(date1, date2)
        # Only Friday (Sat/Sun excluded, Mon not included)
        assert count == 1

    def test_with_holidays(self):
        """Test counting business days with holidays."""
        cal = USCalendar()
        # Include New Year's Day
        date1 = datetime.date(2024, 1, 1)  # Monday, New Year's
        date2 = datetime.date(2024, 1, 5)  # Friday
        count = cal.business_days_between(date1, date2)
        # Tue, Wed, Thu = 3 days (Mon is holiday)
        assert count == 3

    def test_same_date(self):
        """Test counting between same dates."""
        cal = NullCalendar()
        date = datetime.date(2024, 1, 1)
        count = cal.business_days_between(date, date)
        assert count == 0

    def test_reverse_order(self):
        """Test counting with reverse order dates."""
        cal = NullCalendar()
        date1 = datetime.date(2024, 1, 5)
        date2 = datetime.date(2024, 1, 1)
        count = cal.business_days_between(date1, date2)
        assert count == 0
