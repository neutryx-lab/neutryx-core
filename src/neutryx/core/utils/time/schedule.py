"""Schedule generation for cash flow dates.

This module provides functionality for generating payment schedules used in
interest rate swaps, bonds, and other fixed income instruments.
"""

import datetime
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from neutryx.core.utils.time.business_day import (
    BusinessDayConvention,
    MODIFIED_FOLLOWING,
)
from neutryx.core.utils.time.calendar import Calendar, NullCalendar
from neutryx.core.utils.time.date_utils import add_months, DateLike
from neutryx.core.utils.time.day_count import DayCountConvention, ACT_360


class Frequency(Enum):
    """Payment frequency enumeration."""

    ANNUAL = 1
    SEMI_ANNUAL = 2
    QUARTERLY = 4
    MONTHLY = 12
    WEEKLY = 52
    DAILY = 365
    ZERO = 0  # No intermediate payments (zero coupon)

    @property
    def months(self) -> int:
        """Get the number of months between payments."""
        if self == Frequency.ANNUAL:
            return 12
        elif self == Frequency.SEMI_ANNUAL:
            return 6
        elif self == Frequency.QUARTERLY:
            return 3
        elif self == Frequency.MONTHLY:
            return 1
        elif self == Frequency.ZERO:
            return 0
        else:
            raise ValueError(f"Cannot convert {self} to months")

    @property
    def times_per_year(self) -> int:
        """Get the number of payments per year."""
        return self.value


@dataclass
class SchedulePeriod:
    """Represents a single period in a payment schedule.

    Attributes:
        start_date: Unadjusted start date of the period
        end_date: Unadjusted end date of the period
        payment_date: Adjusted payment date (business day)
        accrual_start: Adjusted accrual start date
        accrual_end: Adjusted accrual end date
        year_fraction: Year fraction for the period (using day count convention)
        is_regular: True if this is a regular period, False for stub periods
    """

    start_date: datetime.date
    end_date: datetime.date
    payment_date: datetime.date
    accrual_start: datetime.date
    accrual_end: datetime.date
    year_fraction: float
    is_regular: bool = True


@dataclass
class Schedule:
    """Payment schedule for a financial instrument.

    Attributes:
        periods: List of schedule periods
        effective_date: Start date of the schedule
        termination_date: End date of the schedule
        frequency: Payment frequency
        calendar: Holiday calendar used
        convention: Business day convention used
        day_count: Day count convention used
    """

    periods: List[SchedulePeriod]
    effective_date: datetime.date
    termination_date: datetime.date
    frequency: Frequency
    calendar: Calendar
    convention: BusinessDayConvention
    day_count: DayCountConvention

    def __len__(self) -> int:
        """Get the number of periods in the schedule."""
        return len(self.periods)

    def __getitem__(self, index: int) -> SchedulePeriod:
        """Get a period by index."""
        return self.periods[index]

    def dates(self) -> List[datetime.date]:
        """Get all payment dates in the schedule."""
        return [period.payment_date for period in self.periods]

    def year_fractions(self) -> List[float]:
        """Get all year fractions in the schedule."""
        return [period.year_fraction for period in self.periods]


class StubType(Enum):
    """Type of stub period for irregular schedules."""

    NONE = "none"
    SHORT_FRONT = "short_front"
    SHORT_BACK = "short_back"
    LONG_FRONT = "long_front"
    LONG_BACK = "long_back"


class DateGeneration(Enum):
    """Date generation rule for schedules."""

    BACKWARD = "backward"  # Generate dates backward from termination date
    FORWARD = "forward"    # Generate dates forward from effective date
    ZERO = "zero"          # Only effective and termination dates
    THIRD_WEDNESDAY = "third_wednesday"  # Third Wednesday of month (for IMM dates)
    TWENTIETH = "twentieth"  # 20th of month
    TWENTIETH_IMM = "twentieth_imm"  # 20th of IMM months (Mar, Jun, Sep, Dec)


def generate_schedule(
    effective_date: DateLike,
    termination_date: DateLike,
    frequency: Frequency,
    calendar: Optional[Calendar] = None,
    convention: Optional[BusinessDayConvention] = None,
    termination_convention: Optional[BusinessDayConvention] = None,
    day_count: Optional[DayCountConvention] = None,
    end_of_month: bool = False,
    first_date: Optional[DateLike] = None,
    next_to_last_date: Optional[DateLike] = None,
    date_generation: DateGeneration = DateGeneration.BACKWARD,
) -> Schedule:
    """Generate a payment schedule.

    Args:
        effective_date: Start date of the schedule
        termination_date: End date of the schedule
        frequency: Payment frequency
        calendar: Holiday calendar (default: NullCalendar)
        convention: Business day convention (default: ModifiedFollowing)
        termination_convention: Business day convention for final date (default: same as convention)
        day_count: Day count convention (default: Actual360)
        end_of_month: If True, ensure dates fall on month end
        first_date: First regular payment date (creates front stub if different from natural date)
        next_to_last_date: Next-to-last payment date (creates back stub)
        date_generation: Rule for generating dates

    Returns:
        Generated payment schedule

    Raises:
        ValueError: If inputs are invalid
    """
    # Set defaults
    if calendar is None:
        calendar = NullCalendar()
    if convention is None:
        convention = MODIFIED_FOLLOWING
    if termination_convention is None:
        termination_convention = convention
    if day_count is None:
        day_count = ACT_360

    # Convert to dates
    if isinstance(effective_date, datetime.datetime):
        effective_date = effective_date.date()
    if isinstance(termination_date, datetime.datetime):
        termination_date = termination_date.date()
    if first_date is not None and isinstance(first_date, datetime.datetime):
        first_date = first_date.date()
    if next_to_last_date is not None and isinstance(next_to_last_date, datetime.datetime):
        next_to_last_date = next_to_last_date.date()

    # Validate inputs
    if effective_date >= termination_date:
        raise ValueError("Effective date must be before termination date")

    # Handle zero coupon case
    if frequency == Frequency.ZERO or date_generation == DateGeneration.ZERO:
        payment_date = termination_convention.adjust(termination_date, calendar)
        year_frac = day_count.year_fraction(effective_date, termination_date)

        period = SchedulePeriod(
            start_date=effective_date,
            end_date=termination_date,
            payment_date=payment_date,
            accrual_start=effective_date,
            accrual_end=termination_date,
            year_fraction=year_frac,
            is_regular=True,
        )

        return Schedule(
            periods=[period],
            effective_date=effective_date,
            termination_date=termination_date,
            frequency=frequency,
            calendar=calendar,
            convention=convention,
            day_count=day_count,
        )

    # Generate unadjusted dates
    if date_generation == DateGeneration.BACKWARD:
        unadjusted_dates = _generate_backward(
            effective_date,
            termination_date,
            frequency,
            first_date,
            next_to_last_date,
            end_of_month,
        )
    elif date_generation == DateGeneration.FORWARD:
        unadjusted_dates = _generate_forward(
            effective_date,
            termination_date,
            frequency,
            first_date,
            next_to_last_date,
            end_of_month,
        )
    else:
        raise ValueError(f"Unsupported date generation rule: {date_generation}")

    # Create periods
    periods = []
    for i in range(len(unadjusted_dates) - 1):
        start = unadjusted_dates[i]
        end = unadjusted_dates[i + 1]

        # Apply business day conventions
        if i == len(unadjusted_dates) - 2:
            # Last period - use termination convention
            payment_date = termination_convention.adjust(end, calendar)
        else:
            payment_date = convention.adjust(end, calendar)

        accrual_start = convention.adjust(start, calendar) if i > 0 else start
        accrual_end = payment_date

        # Calculate year fraction
        year_frac = day_count.year_fraction(
            accrual_start,
            accrual_end,
            ref_start=start,
            ref_end=end,
        )

        # Determine if regular period
        is_regular = True
        if i == 0 and first_date is not None:
            is_regular = False
        elif i == len(unadjusted_dates) - 2 and next_to_last_date is not None:
            is_regular = False

        period = SchedulePeriod(
            start_date=start,
            end_date=end,
            payment_date=payment_date,
            accrual_start=accrual_start,
            accrual_end=accrual_end,
            year_fraction=year_frac,
            is_regular=is_regular,
        )
        periods.append(period)

    return Schedule(
        periods=periods,
        effective_date=effective_date,
        termination_date=termination_date,
        frequency=frequency,
        calendar=calendar,
        convention=convention,
        day_count=day_count,
    )


def _generate_backward(
    effective_date: datetime.date,
    termination_date: datetime.date,
    frequency: Frequency,
    first_date: Optional[datetime.date],
    next_to_last_date: Optional[datetime.date],
    end_of_month: bool,
) -> List[datetime.date]:
    """Generate unadjusted dates backward from termination date."""
    dates = [termination_date]
    current_date = termination_date

    # Handle next-to-last date (back stub)
    if next_to_last_date is not None:
        dates.insert(0, next_to_last_date)
        current_date = next_to_last_date

    # Generate regular periods
    months_step = frequency.months
    while True:
        next_date = add_months(current_date, -months_step)

        # Check if we've reached or passed the effective date
        if first_date is not None:
            if next_date <= first_date:
                if first_date not in dates:
                    dates.insert(0, first_date)
                break
        else:
            if next_date <= effective_date:
                break

        dates.insert(0, next_date)
        current_date = next_date

    # Add effective date
    if effective_date not in dates:
        dates.insert(0, effective_date)

    return dates


def _generate_forward(
    effective_date: datetime.date,
    termination_date: datetime.date,
    frequency: Frequency,
    first_date: Optional[datetime.date],
    next_to_last_date: Optional[datetime.date],
    end_of_month: bool,
) -> List[datetime.date]:
    """Generate unadjusted dates forward from effective date."""
    dates = [effective_date]
    current_date = effective_date

    # Handle first date (front stub)
    if first_date is not None:
        dates.append(first_date)
        current_date = first_date

    # Generate regular periods
    months_step = frequency.months
    while True:
        next_date = add_months(current_date, months_step)

        # Check if we've reached or passed the termination date
        if next_to_last_date is not None:
            if next_date >= next_to_last_date:
                if next_to_last_date not in dates:
                    dates.append(next_to_last_date)
                break
        else:
            if next_date >= termination_date:
                break

        dates.append(next_date)
        current_date = next_date

    # Add termination date
    if termination_date not in dates:
        dates.append(termination_date)

    return dates
