"""Tests for schedule generation."""

import datetime
import pytest

from neutryx.core.utils.time.schedule import (
    Frequency,
    generate_schedule,
    DateGeneration,
)
from neutryx.core.utils.time.calendar import NullCalendar, TargetCalendar
from neutryx.core.utils.time.business_day import ModifiedFollowing, Following
from neutryx.core.utils.time.day_count import Actual360, Thirty360


class TestFrequency:
    """Tests for Frequency enum."""

    def test_times_per_year(self):
        """Test times per year property."""
        assert Frequency.ANNUAL.times_per_year == 1
        assert Frequency.SEMI_ANNUAL.times_per_year == 2
        assert Frequency.QUARTERLY.times_per_year == 4
        assert Frequency.MONTHLY.times_per_year == 12

    def test_months_property(self):
        """Test months between payments."""
        assert Frequency.ANNUAL.months == 12
        assert Frequency.SEMI_ANNUAL.months == 6
        assert Frequency.QUARTERLY.months == 3
        assert Frequency.MONTHLY.months == 1


class TestGenerateScheduleBasic:
    """Basic tests for schedule generation."""

    def test_annual_schedule(self):
        """Test generating annual payment schedule."""
        schedule = generate_schedule(
            effective_date=datetime.date(2024, 1, 1),
            termination_date=datetime.date(2026, 1, 1),
            frequency=Frequency.ANNUAL,
        )

        assert len(schedule) == 2
        assert schedule.effective_date == datetime.date(2024, 1, 1)
        assert schedule.termination_date == datetime.date(2026, 1, 1)

        # Check payment dates
        dates = schedule.dates()
        assert len(dates) == 2
        assert dates[0] == datetime.date(2025, 1, 1)
        assert dates[1] == datetime.date(2026, 1, 1)

    def test_semi_annual_schedule(self):
        """Test generating semi-annual payment schedule."""
        schedule = generate_schedule(
            effective_date=datetime.date(2024, 1, 1),
            termination_date=datetime.date(2025, 1, 1),
            frequency=Frequency.SEMI_ANNUAL,
        )

        assert len(schedule) == 2

        # Check payment dates (July 1 and January 1)
        dates = schedule.dates()
        assert dates[0] == datetime.date(2024, 7, 1)
        assert dates[1] == datetime.date(2025, 1, 1)

    def test_quarterly_schedule(self):
        """Test generating quarterly payment schedule."""
        schedule = generate_schedule(
            effective_date=datetime.date(2024, 1, 1),
            termination_date=datetime.date(2025, 1, 1),
            frequency=Frequency.QUARTERLY,
        )

        assert len(schedule) == 4

        # Check payment dates
        dates = schedule.dates()
        assert dates[0] == datetime.date(2024, 4, 1)
        assert dates[1] == datetime.date(2024, 7, 1)
        assert dates[2] == datetime.date(2024, 10, 1)
        assert dates[3] == datetime.date(2025, 1, 1)

    def test_monthly_schedule(self):
        """Test generating monthly payment schedule."""
        schedule = generate_schedule(
            effective_date=datetime.date(2024, 1, 1),
            termination_date=datetime.date(2024, 4, 1),
            frequency=Frequency.MONTHLY,
        )

        assert len(schedule) == 3
        dates = schedule.dates()
        assert dates[0] == datetime.date(2024, 2, 1)
        assert dates[1] == datetime.date(2024, 3, 1)
        assert dates[2] == datetime.date(2024, 4, 1)

    def test_zero_coupon_schedule(self):
        """Test generating zero coupon schedule."""
        schedule = generate_schedule(
            effective_date=datetime.date(2024, 1, 1),
            termination_date=datetime.date(2025, 1, 1),
            frequency=Frequency.ZERO,
        )

        assert len(schedule) == 1
        assert schedule.periods[0].start_date == datetime.date(2024, 1, 1)
        assert schedule.periods[0].end_date == datetime.date(2025, 1, 1)
        assert schedule.periods[0].is_regular


class TestScheduleWithBusinessDayConventions:
    """Tests for schedules with business day conventions."""

    def test_following_convention(self):
        """Test schedule with following convention."""
        schedule = generate_schedule(
            effective_date=datetime.date(2024, 1, 1),
            termination_date=datetime.date(2024, 7, 6),  # Saturday
            frequency=Frequency.SEMI_ANNUAL,
            calendar=NullCalendar(),
            convention=Following(),
        )

        # Termination date falls on Saturday, should be adjusted to Monday
        assert schedule.periods[-1].payment_date == datetime.date(2024, 7, 8)

    def test_modified_following_convention(self):
        """Test schedule with modified following convention."""
        schedule = generate_schedule(
            effective_date=datetime.date(2024, 1, 1),
            termination_date=datetime.date(2024, 6, 30),  # Sunday
            frequency=Frequency.SEMI_ANNUAL,
            calendar=NullCalendar(),
            convention=ModifiedFollowing(),
        )

        # June 30 is Sunday, modified following should adjust backward
        # since following would cross into July
        assert schedule.periods[-1].payment_date == datetime.date(2024, 6, 28)


class TestScheduleWithDayCount:
    """Tests for schedules with different day count conventions."""

    def test_actual360_day_count(self):
        """Test schedule with Actual/360 day count."""
        schedule = generate_schedule(
            effective_date=datetime.date(2024, 1, 1),
            termination_date=datetime.date(2024, 7, 1),
            frequency=Frequency.SEMI_ANNUAL,
            day_count=Actual360(),
        )

        # Check year fractions
        year_fractions = schedule.year_fractions()
        assert len(year_fractions) == 1

        # First period: Jan 1 to Jul 1 (182 days in leap year)
        expected_yf = 182 / 360.0
        assert year_fractions[0] == pytest.approx(expected_yf)

    def test_thirty360_day_count(self):
        """Test schedule with 30/360 day count."""
        schedule = generate_schedule(
            effective_date=datetime.date(2024, 1, 1),
            termination_date=datetime.date(2024, 7, 1),
            frequency=Frequency.SEMI_ANNUAL,
            day_count=Thirty360(),
        )

        year_fractions = schedule.year_fractions()
        # 30/360: exactly 6 months = 180/360 = 0.5
        expected_yf = 180 / 360.0
        assert year_fractions[0] == pytest.approx(expected_yf)


class TestScheduleWithStubs:
    """Tests for schedules with stub periods."""

    def test_short_front_stub(self):
        """Test schedule with short front stub."""
        schedule = generate_schedule(
            effective_date=datetime.date(2024, 1, 15),
            termination_date=datetime.date(2025, 1, 1),
            frequency=Frequency.SEMI_ANNUAL,
            first_date=datetime.date(2024, 7, 1),
        )

        # First period should be short (Jan 15 to Jul 1)
        assert not schedule.periods[0].is_regular
        assert schedule.periods[0].start_date == datetime.date(2024, 1, 15)
        assert schedule.periods[0].end_date == datetime.date(2024, 7, 1)

        # Subsequent periods should be regular
        assert schedule.periods[1].is_regular

    def test_short_back_stub(self):
        """Test schedule with short back stub."""
        schedule = generate_schedule(
            effective_date=datetime.date(2024, 1, 1),
            termination_date=datetime.date(2024, 11, 15),
            frequency=Frequency.SEMI_ANNUAL,
            next_to_last_date=datetime.date(2024, 7, 1),
        )

        # Last period should be short (Jul 1 to Nov 15)
        assert not schedule.periods[-1].is_regular
        assert schedule.periods[-1].start_date == datetime.date(2024, 7, 1)
        assert schedule.periods[-1].end_date == datetime.date(2024, 11, 15)


class TestScheduleDateGeneration:
    """Tests for different date generation rules."""

    def test_backward_generation(self):
        """Test backward date generation."""
        schedule = generate_schedule(
            effective_date=datetime.date(2024, 1, 1),
            termination_date=datetime.date(2025, 1, 1),
            frequency=Frequency.QUARTERLY,
            date_generation=DateGeneration.BACKWARD,
        )

        # Dates should be generated backward from termination
        dates = schedule.dates()
        assert dates[-1] == datetime.date(2025, 1, 1)
        assert dates[-2] == datetime.date(2024, 10, 1)
        assert dates[-3] == datetime.date(2024, 7, 1)
        assert dates[-4] == datetime.date(2024, 4, 1)

    def test_forward_generation(self):
        """Test forward date generation."""
        schedule = generate_schedule(
            effective_date=datetime.date(2024, 1, 1),
            termination_date=datetime.date(2025, 1, 1),
            frequency=Frequency.QUARTERLY,
            date_generation=DateGeneration.FORWARD,
        )

        # Dates should be generated forward from effective
        dates = schedule.dates()
        assert dates[0] == datetime.date(2024, 4, 1)
        assert dates[1] == datetime.date(2024, 7, 1)
        assert dates[2] == datetime.date(2024, 10, 1)
        assert dates[3] == datetime.date(2025, 1, 1)

    def test_zero_generation(self):
        """Test zero date generation (single payment)."""
        schedule = generate_schedule(
            effective_date=datetime.date(2024, 1, 1),
            termination_date=datetime.date(2025, 1, 1),
            frequency=Frequency.ANNUAL,
            date_generation=DateGeneration.ZERO,
        )

        assert len(schedule) == 1
        assert schedule.periods[0].end_date == datetime.date(2025, 1, 1)


class TestScheduleWithHolidays:
    """Tests for schedules with holiday calendars."""

    def test_target_calendar(self):
        """Test schedule with TARGET calendar."""
        schedule = generate_schedule(
            effective_date=datetime.date(2023, 12, 1),
            termination_date=datetime.date(2024, 1, 1),  # New Year's Day
            frequency=Frequency.MONTHLY,
            calendar=TargetCalendar(),
            convention=Following(),
        )

        # January 1 is a holiday, should be adjusted to January 2
        assert schedule.periods[-1].payment_date == datetime.date(2024, 1, 2)


class TestScheduleValidation:
    """Tests for schedule input validation."""

    def test_invalid_date_order(self):
        """Test that invalid date order raises error."""
        with pytest.raises(ValueError):
            generate_schedule(
                effective_date=datetime.date(2025, 1, 1),
                termination_date=datetime.date(2024, 1, 1),  # Before effective
                frequency=Frequency.ANNUAL,
            )

    def test_same_effective_and_termination(self):
        """Test that same dates raise error."""
        with pytest.raises(ValueError):
            generate_schedule(
                effective_date=datetime.date(2024, 1, 1),
                termination_date=datetime.date(2024, 1, 1),
                frequency=Frequency.ANNUAL,
            )


class TestScheduleProperties:
    """Tests for schedule properties and methods."""

    def test_len(self):
        """Test schedule length."""
        schedule = generate_schedule(
            effective_date=datetime.date(2024, 1, 1),
            termination_date=datetime.date(2025, 1, 1),
            frequency=Frequency.QUARTERLY,
        )

        assert len(schedule) == 4
        assert len(schedule.periods) == 4

    def test_getitem(self):
        """Test schedule indexing."""
        schedule = generate_schedule(
            effective_date=datetime.date(2024, 1, 1),
            termination_date=datetime.date(2025, 1, 1),
            frequency=Frequency.QUARTERLY,
        )

        first_period = schedule[0]
        assert first_period.end_date == datetime.date(2024, 4, 1)

        last_period = schedule[-1]
        assert last_period.end_date == datetime.date(2025, 1, 1)

    def test_dates_method(self):
        """Test dates() method."""
        schedule = generate_schedule(
            effective_date=datetime.date(2024, 1, 1),
            termination_date=datetime.date(2024, 7, 1),
            frequency=Frequency.QUARTERLY,
        )

        dates = schedule.dates()
        assert len(dates) == 2
        assert all(isinstance(d, datetime.date) for d in dates)

    def test_year_fractions_method(self):
        """Test year_fractions() method."""
        schedule = generate_schedule(
            effective_date=datetime.date(2024, 1, 1),
            termination_date=datetime.date(2024, 7, 1),
            frequency=Frequency.QUARTERLY,
        )

        yfs = schedule.year_fractions()
        assert len(yfs) == 2
        assert all(isinstance(yf, float) for yf in yfs)
        assert all(yf > 0 for yf in yfs)


class TestRealWorldSchedules:
    """Tests with real-world schedule scenarios."""

    def test_swap_schedule(self):
        """Test typical interest rate swap schedule."""
        # 5-year swap, semi-annual payments
        schedule = generate_schedule(
            effective_date=datetime.date(2024, 1, 15),
            termination_date=datetime.date(2029, 1, 15),
            frequency=Frequency.SEMI_ANNUAL,
            calendar=TargetCalendar(),
            convention=ModifiedFollowing(),
            day_count=Actual360(),
        )

        # Should have 10 periods (2 per year × 5 years)
        assert len(schedule) == 10

        # All periods should be regular
        assert all(p.is_regular for p in schedule.periods)

        # Year fractions should sum to approximately 5.0
        total_yf = sum(schedule.year_fractions())
        assert total_yf == pytest.approx(5.0, rel=0.01)

    def test_bond_schedule(self):
        """Test typical bond coupon schedule."""
        # 10-year bond, annual coupons, 30/360 day count
        schedule = generate_schedule(
            effective_date=datetime.date(2024, 3, 15),
            termination_date=datetime.date(2034, 3, 15),
            frequency=Frequency.ANNUAL,
            calendar=TargetCalendar(),
            convention=Following(),
            day_count=Thirty360(),
        )

        # Should have 10 periods
        assert len(schedule) == 10

        # Each year fraction should be exactly 1.0 with 30/360
        for yf in schedule.year_fractions():
            assert yf == pytest.approx(1.0)

    def test_short_term_loan(self):
        """Test short-term loan schedule."""
        # 6-month loan, monthly payments
        schedule = generate_schedule(
            effective_date=datetime.date(2024, 1, 1),
            termination_date=datetime.date(2024, 7, 1),
            frequency=Frequency.MONTHLY,
            calendar=NullCalendar(),
            convention=ModifiedFollowing(),
            day_count=Actual365Fixed(),
        )

        # Should have 6 periods
        assert len(schedule) == 6

        # Sum of year fractions should be approximately 0.5
        total_yf = sum(schedule.year_fractions())
        assert total_yf == pytest.approx(0.5, rel=0.01)
