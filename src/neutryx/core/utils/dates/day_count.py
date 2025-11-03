"""Day count conventions for financial calculations.

Day count conventions determine how interest accrues between two dates.
Different conventions are used in different markets and for different instruments.
"""

import datetime
from abc import ABC, abstractmethod
from typing import Union

from neutryx.core.utils.dates.date_utils import (
    date_diff_days,
    days_in_year,
    is_leap_year,
    DateLike,
)


class DayCountConvention(ABC):
    """Abstract base class for day count conventions."""

    @abstractmethod
    def year_fraction(self, date1: DateLike, date2: DateLike, ref_start: DateLike = None, ref_end: DateLike = None) -> float:
        """Calculate the year fraction between two dates.

        Args:
            date1: Start date
            date2: End date
            ref_start: Reference period start (used by some conventions)
            ref_end: Reference period end (used by some conventions)

        Returns:
            Year fraction according to the convention
        """
        pass

    @abstractmethod
    def day_count(self, date1: DateLike, date2: DateLike) -> int:
        """Calculate the day count between two dates.

        Args:
            date1: Start date
            date2: End date

        Returns:
            Day count according to the convention
        """
        pass

    def __repr__(self) -> str:
        return self.__class__.__name__


class Actual360(DayCountConvention):
    """Actual/360 day count convention.

    Year fraction = Actual days / 360

    Commonly used for money market instruments and some floating rate notes.
    """

    def year_fraction(self, date1: DateLike, date2: DateLike, ref_start: DateLike = None, ref_end: DateLike = None) -> float:
        """Calculate year fraction using Actual/360 convention."""
        days = self.day_count(date1, date2)
        return days / 360.0

    def day_count(self, date1: DateLike, date2: DateLike) -> int:
        """Calculate actual day count."""
        return date_diff_days(date1, date2)


class Actual365Fixed(DayCountConvention):
    """Actual/365 Fixed day count convention.

    Year fraction = Actual days / 365

    Commonly used for GBP and some other markets.
    """

    def year_fraction(self, date1: DateLike, date2: DateLike, ref_start: DateLike = None, ref_end: DateLike = None) -> float:
        """Calculate year fraction using Actual/365 Fixed convention."""
        days = self.day_count(date1, date2)
        return days / 365.0

    def day_count(self, date1: DateLike, date2: DateLike) -> int:
        """Calculate actual day count."""
        return date_diff_days(date1, date2)


class ActualActual(DayCountConvention):
    """Actual/Actual day count convention.

    This is a simplified version that uses the actual days in the year.
    For more precise ISDA or ICMA versions, use ActualActualISDA or ActualActualICMA.

    Year fraction = Actual days / Actual days in year
    """

    def year_fraction(self, date1: DateLike, date2: DateLike, ref_start: DateLike = None, ref_end: DateLike = None) -> float:
        """Calculate year fraction using Actual/Actual convention."""
        if isinstance(date1, datetime.datetime):
            date1 = date1.date()
        if isinstance(date2, datetime.datetime):
            date2 = date2.date()

        days = date_diff_days(date1, date2)

        # Simple approach: use the average days in year across the period
        if date1.year == date2.year:
            return days / days_in_year(date1.year)
        else:
            # For multi-year periods, calculate weighted average
            total_fraction = 0.0
            current_date = date1

            while current_date.year < date2.year:
                year_end = datetime.date(current_date.year, 12, 31)
                year_days = date_diff_days(current_date, year_end) + 1
                total_fraction += year_days / days_in_year(current_date.year)
                current_date = datetime.date(current_date.year + 1, 1, 1)

            # Add the final partial year
            if current_date < date2:
                year_days = date_diff_days(current_date, date2)
                total_fraction += year_days / days_in_year(date2.year)

            return total_fraction

    def day_count(self, date1: DateLike, date2: DateLike) -> int:
        """Calculate actual day count."""
        return date_diff_days(date1, date2)


class ActualActualISDA(DayCountConvention):
    """Actual/Actual (ISDA) day count convention.

    This convention splits the period into portions falling in leap years
    and non-leap years, then calculates the year fraction separately for each.

    Commonly used for USD swaps and bonds.
    """

    def year_fraction(self, date1: DateLike, date2: DateLike, ref_start: DateLike = None, ref_end: DateLike = None) -> float:
        """Calculate year fraction using Actual/Actual ISDA convention."""
        if isinstance(date1, datetime.datetime):
            date1 = date1.date()
        if isinstance(date2, datetime.datetime):
            date2 = date2.date()

        if date1 >= date2:
            return 0.0

        # Split the period by year
        total_fraction = 0.0
        current_date = date1

        while current_date < date2:
            # Find the end of the current year or date2, whichever comes first
            year_end = datetime.date(current_date.year, 12, 31)
            period_end = min(year_end, date2)

            # Calculate days in this portion
            days = date_diff_days(current_date, period_end)
            year_basis = days_in_year(current_date.year)

            total_fraction += days / year_basis

            # Move to next year
            current_date = period_end
            if current_date <= year_end and current_date < date2:
                current_date = datetime.date(current_date.year + 1, 1, 1)

        return total_fraction

    def day_count(self, date1: DateLike, date2: DateLike) -> int:
        """Calculate actual day count."""
        return date_diff_days(date1, date2)


class ActualActualICMA(DayCountConvention):
    """Actual/Actual (ICMA) day count convention.

    This convention requires reference dates (the coupon period start and end).
    Year fraction = Actual days / (Days in reference period × Frequency)

    Commonly used for government bonds.
    """

    def __init__(self, frequency: int = 2):
        """Initialize ActualActualICMA.

        Args:
            frequency: Number of coupon payments per year (default: 2 for semi-annual)
        """
        self.frequency = frequency

    def year_fraction(self, date1: DateLike, date2: DateLike, ref_start: DateLike = None, ref_end: DateLike = None) -> float:
        """Calculate year fraction using Actual/Actual ICMA convention.

        Args:
            date1: Start date
            date2: End date
            ref_start: Reference period start date (required)
            ref_end: Reference period end date (required)

        Returns:
            Year fraction

        Raises:
            ValueError: If ref_start or ref_end is not provided
        """
        if ref_start is None or ref_end is None:
            raise ValueError("ActualActualICMA requires ref_start and ref_end dates")

        if isinstance(date1, datetime.datetime):
            date1 = date1.date()
        if isinstance(date2, datetime.datetime):
            date2 = date2.date()
        if isinstance(ref_start, datetime.datetime):
            ref_start = ref_start.date()
        if isinstance(ref_end, datetime.datetime):
            ref_end = ref_end.date()

        days = date_diff_days(date1, date2)
        days_in_period = date_diff_days(ref_start, ref_end)

        return days / (days_in_period * self.frequency)

    def day_count(self, date1: DateLike, date2: DateLike) -> int:
        """Calculate actual day count."""
        return date_diff_days(date1, date2)


class Thirty360(DayCountConvention):
    """30/360 (US) day count convention.

    Assumes 30 days in each month and 360 days in a year.
    Also known as 30/360 Bond Basis.

    Commonly used for US corporate and municipal bonds.
    """

    def year_fraction(self, date1: DateLike, date2: DateLike, ref_start: DateLike = None, ref_end: DateLike = None) -> float:
        """Calculate year fraction using 30/360 US convention."""
        days = self.day_count(date1, date2)
        return days / 360.0

    def day_count(self, date1: DateLike, date2: DateLike) -> int:
        """Calculate day count using 30/360 US convention."""
        if isinstance(date1, datetime.datetime):
            date1 = date1.date()
        if isinstance(date2, datetime.datetime):
            date2 = date2.date()

        y1, m1, d1 = date1.year, date1.month, date1.day
        y2, m2, d2 = date2.year, date2.month, date2.day

        # 30/360 US adjustments
        if d1 == 31:
            d1 = 30
        if d2 == 31 and d1 >= 30:
            d2 = 30

        return 360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)


class ThirtyE360(DayCountConvention):
    """30E/360 (Eurobond) day count convention.

    Assumes 30 days in each month and 360 days in a year.
    Similar to 30/360 but with European adjustments.

    Commonly used for Eurobonds.
    """

    def year_fraction(self, date1: DateLike, date2: DateLike, ref_start: DateLike = None, ref_end: DateLike = None) -> float:
        """Calculate year fraction using 30E/360 convention."""
        days = self.day_count(date1, date2)
        return days / 360.0

    def day_count(self, date1: DateLike, date2: DateLike) -> int:
        """Calculate day count using 30E/360 convention."""
        if isinstance(date1, datetime.datetime):
            date1 = date1.date()
        if isinstance(date2, datetime.datetime):
            date2 = date2.date()

        y1, m1, d1 = date1.year, date1.month, date1.day
        y2, m2, d2 = date2.year, date2.month, date2.day

        # 30E/360 adjustments
        if d1 == 31:
            d1 = 30
        if d2 == 31:
            d2 = 30

        return 360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)


class Business252(DayCountConvention):
    """Business/252 day count convention.

    Year fraction = Business days / 252

    Commonly used in Brazilian markets.
    """

    def __init__(self, calendar=None):
        """Initialize Business252.

        Args:
            calendar: Holiday calendar to use (if None, only weekends are excluded)
        """
        self.calendar = calendar

    def year_fraction(self, date1: DateLike, date2: DateLike, ref_start: DateLike = None, ref_end: DateLike = None) -> float:
        """Calculate year fraction using Business/252 convention."""
        days = self.day_count(date1, date2)
        return days / 252.0

    def day_count(self, date1: DateLike, date2: DateLike) -> int:
        """Calculate business day count."""
        if isinstance(date1, datetime.datetime):
            date1 = date1.date()
        if isinstance(date2, datetime.datetime):
            date2 = date2.date()

        if self.calendar is None:
            # Count weekdays only
            business_days = 0
            current_date = date1
            while current_date < date2:
                if current_date.weekday() < 5:  # Monday = 0, Friday = 4
                    business_days += 1
                current_date += datetime.timedelta(days=1)
            return business_days
        else:
            # Use calendar to count business days
            return self.calendar.business_days_between(date1, date2)


# Convenience aliases
ACT_360 = Actual360()
ACT_365 = Actual365Fixed()
ACT_ACT = ActualActual()
ACT_ACT_ISDA = ActualActualISDA()
THIRTY_360 = Thirty360()
THIRTY_E360 = ThirtyE360()
