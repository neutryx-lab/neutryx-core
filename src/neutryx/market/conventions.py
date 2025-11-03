"""
Day count conventions and business calendar utilities.

This module provides standard financial market conventions for:
- Day count basis (ACT/360, ACT/365, 30/360, etc.)
- Business day calendars and holiday rules
- Date adjustment conventions (FOLLOWING, MODIFIED_FOLLOWING, etc.)
"""

from __future__ import annotations

from datetime import date, timedelta
from enum import Enum
from typing import Optional, Set


class DayCountConvention(Enum):
    """
    Standard day count conventions used in financial markets.

    References:
        - ISDA definitions
        - Bloomberg convention codes
    """

    # Actual day count conventions
    ACT_360 = "ACT/360"         # Actual/360 (money market)
    ACT_365 = "ACT/365"         # Actual/365 (Fixed)
    ACT_365L = "ACT/365L"       # Actual/365 (Leap year aware)
    ACT_ACT = "ACT/ACT"         # Actual/Actual (ISDA)
    ACT_ACT_ISDA = "ACT/ACT_ISDA"  # Same as ACT/ACT
    ACT_ACT_ICMA = "ACT/ACT_ICMA"  # Actual/Actual (ICMA)

    # 30/360 conventions
    THIRTY_360 = "30/360"        # 30/360 (Bond basis, US)
    THIRTY_360_US = "30/360_US"  # Same as 30/360
    THIRTY_E_360 = "30E/360"     # European 30/360
    THIRTY_E_360_ISDA = "30E/360_ISDA"  # European 30/360 (ISDA)

    # Other
    BUS_252 = "BUS/252"          # Business days / 252


class BusinessDayConvention(Enum):
    """Date adjustment conventions for rolling onto business days."""

    FOLLOWING = "FOLLOWING"              # Next business day
    MODIFIED_FOLLOWING = "MODFOLLOWING"  # Following, unless crosses month
    PRECEDING = "PRECEDING"              # Previous business day
    MODIFIED_PRECEDING = "MODPRECEDING"  # Preceding, unless crosses month
    UNADJUSTED = "UNADJUSTED"           # No adjustment
    END_OF_MONTH = "EOM"                # End of month adjustment


def year_fraction(
    start_date: date,
    end_date: date,
    convention: DayCountConvention = DayCountConvention.ACT_365
) -> float:
    """
    Calculate year fraction between two dates using specified day count convention.

    Args:
        start_date: Start date
        end_date: End date
        convention: Day count convention to use

    Returns:
        Year fraction as float

    Example:
        >>> from datetime import date
        >>> start = date(2024, 1, 1)
        >>> end = date(2024, 7, 1)
        >>> yf = year_fraction(start, end, DayCountConvention.ACT_365)
        >>> abs(yf - 0.4986) < 0.001  # ~181/365
        True
    """
    if end_date < start_date:
        raise ValueError("end_date must be >= start_date")

    if convention in (DayCountConvention.ACT_365, DayCountConvention.ACT_365L):
        days = (end_date - start_date).days
        if convention == DayCountConvention.ACT_365:
            return days / 365.0
        else:  # ACT_365L
            # Leap year aware: use 366 if leap year in period
            year_start = start_date.year
            year_end = end_date.year
            if _is_leap_year(year_start) or (year_start != year_end and _is_leap_year(year_end)):
                return days / 366.0
            return days / 365.0

    elif convention == DayCountConvention.ACT_360:
        days = (end_date - start_date).days
        return days / 360.0

    elif convention in (DayCountConvention.ACT_ACT, DayCountConvention.ACT_ACT_ISDA):
        # Actual/Actual (ISDA): separate by year
        total_fraction = 0.0
        current = start_date

        while current < end_date:
            year_end = date(current.year, 12, 31)
            period_end = min(end_date, year_end)

            days_in_period = (period_end - current).days
            days_in_year = 366 if _is_leap_year(current.year) else 365

            total_fraction += days_in_period / days_in_year

            current = period_end
            if current < end_date:
                current += timedelta(days=1)  # Move to next year

        return total_fraction

    elif convention == DayCountConvention.ACT_ACT_ICMA:
        # Actual/Actual (ICMA) - simplified version
        # For full implementation, would need payment frequency
        days = (end_date - start_date).days
        return days / 365.25

    elif convention in (DayCountConvention.THIRTY_360, DayCountConvention.THIRTY_360_US):
        return _thirty_360_us(start_date, end_date)

    elif convention == DayCountConvention.THIRTY_E_360:
        return _thirty_e_360(start_date, end_date)

    elif convention == DayCountConvention.THIRTY_E_360_ISDA:
        return _thirty_e_360_isda(start_date, end_date)

    elif convention == DayCountConvention.BUS_252:
        # Business days (simplified - assumes no holidays)
        # For production, would use actual business calendar
        days = (end_date - start_date).days
        # Approximate: weekdays only
        bus_days = _count_business_days(start_date, end_date)
        return bus_days / 252.0

    else:
        raise ValueError(f"Unsupported day count convention: {convention}")


def _is_leap_year(year: int) -> bool:
    """Check if year is a leap year."""
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def _thirty_360_us(start_date: date, end_date: date) -> float:
    """
    30/360 (US) day count calculation.

    Convention:
    - Treats all months as having 30 days
    - Adjusts day 31 to 30
    - Special handling for Feb end dates
    """
    y1, m1, d1 = start_date.year, start_date.month, start_date.day
    y2, m2, d2 = end_date.year, end_date.month, end_date.day

    # Adjust day 31
    if d1 == 31:
        d1 = 30
    if d2 == 31 and d1 >= 30:
        d2 = 30

    return ((y2 - y1) * 360 + (m2 - m1) * 30 + (d2 - d1)) / 360.0


def _thirty_e_360(start_date: date, end_date: date) -> float:
    """
    30E/360 (European) day count calculation.

    Convention:
    - Day 31 is always adjusted to 30
    """
    y1, m1, d1 = start_date.year, start_date.month, start_date.day
    y2, m2, d2 = end_date.year, end_date.month, end_date.day

    # Adjust day 31
    if d1 == 31:
        d1 = 30
    if d2 == 31:
        d2 = 30

    return ((y2 - y1) * 360 + (m2 - m1) * 30 + (d2 - d1)) / 360.0


def _thirty_e_360_isda(start_date: date, end_date: date) -> float:
    """
    30E/360 (ISDA) day count calculation.

    Convention:
    - Similar to 30E/360 but with special Feb end handling
    """
    y1, m1, d1 = start_date.year, start_date.month, start_date.day
    y2, m2, d2 = end_date.year, end_date.month, end_date.day

    # Check for February end
    if m1 == 2 and _is_last_day_of_month(start_date):
        d1 = 30
    if m2 == 2 and _is_last_day_of_month(end_date):
        d2 = 30

    # Adjust day 31
    if d1 == 31:
        d1 = 30
    if d2 == 31:
        d2 = 30

    return ((y2 - y1) * 360 + (m2 - m1) * 30 + (d2 - d1)) / 360.0


def _is_last_day_of_month(d: date) -> bool:
    """Check if date is the last day of the month."""
    next_day = d + timedelta(days=1)
    return next_day.month != d.month


def _count_business_days(start_date: date, end_date: date) -> int:
    """
    Count business days (Mon-Fri) between dates, excluding holidays.

    Simplified version without holiday calendar.
    """
    count = 0
    current = start_date

    while current < end_date:
        # Monday = 0, Sunday = 6
        if current.weekday() < 5:  # Mon-Fri
            count += 1
        current += timedelta(days=1)

    return count


class BusinessCalendar:
    """
    Business day calendar with holiday support.

    Attributes:
        name: Calendar name (e.g., "USD", "EUR", "GBP")
        holidays: Set of holiday dates
        weekend_days: Set of weekend day numbers (0=Monday, 6=Sunday)
    """

    def __init__(
        self,
        name: str,
        holidays: Optional[Set[date]] = None,
        weekend_days: Optional[Set[int]] = None
    ):
        """
        Initialize business calendar.

        Args:
            name: Calendar name
            holidays: Set of holiday dates (default: empty)
            weekend_days: Weekend day numbers (default: {5, 6} = Sat, Sun)
        """
        self.name = name
        self.holidays = holidays or set()
        self.weekend_days = weekend_days or {5, 6}  # Saturday, Sunday

    def is_business_day(self, d: date) -> bool:
        """Check if date is a business day."""
        return (
            d.weekday() not in self.weekend_days
            and d not in self.holidays
        )

    def is_holiday(self, d: date) -> bool:
        """Check if date is a holiday or weekend."""
        return not self.is_business_day(d)

    def add_holiday(self, d: date) -> None:
        """Add a holiday to the calendar."""
        self.holidays.add(d)

    def adjust(
        self,
        d: date,
        convention: BusinessDayConvention = BusinessDayConvention.FOLLOWING
    ) -> date:
        """
        Adjust date to business day using specified convention.

        Args:
            d: Date to adjust
            convention: Adjustment convention

        Returns:
            Adjusted date

        Example:
            >>> cal = BusinessCalendar("USD")
            >>> # If Jan 1, 2024 is a holiday/weekend
            >>> adjusted = cal.adjust(date(2024, 1, 1))
        """
        if convention == BusinessDayConvention.UNADJUSTED:
            return d

        if convention == BusinessDayConvention.FOLLOWING:
            return self._following(d)

        if convention == BusinessDayConvention.MODIFIED_FOLLOWING:
            adjusted = self._following(d)
            # If crosses month boundary, use preceding instead
            if adjusted.month != d.month:
                return self._preceding(d)
            return adjusted

        if convention == BusinessDayConvention.PRECEDING:
            return self._preceding(d)

        if convention == BusinessDayConvention.MODIFIED_PRECEDING:
            adjusted = self._preceding(d)
            # If crosses month boundary, use following instead
            if adjusted.month != d.month:
                return self._following(d)
            return adjusted

        if convention == BusinessDayConvention.END_OF_MONTH:
            # Go to end of month, then adjust backward
            last_day = self._get_last_day_of_month(d)
            return self._preceding(last_day)

        raise ValueError(f"Unsupported convention: {convention}")

    def _following(self, d: date) -> date:
        """Find next business day on or after d."""
        current = d
        while not self.is_business_day(current):
            current += timedelta(days=1)
        return current

    def _preceding(self, d: date) -> date:
        """Find previous business day on or before d."""
        current = d
        while not self.is_business_day(current):
            current -= timedelta(days=1)
        return current

    def _get_last_day_of_month(self, d: date) -> date:
        """Get last day of the month for given date."""
        # Go to first day of next month, then back one day
        if d.month == 12:
            next_month = date(d.year + 1, 1, 1)
        else:
            next_month = date(d.year, d.month + 1, 1)
        return next_month - timedelta(days=1)

    def business_days_between(
        self,
        start_date: date,
        end_date: date,
        include_end: bool = False
    ) -> int:
        """
        Count business days between two dates.

        Args:
            start_date: Start date (included)
            end_date: End date
            include_end: Whether to include end_date in count

        Returns:
            Number of business days
        """
        count = 0
        current = start_date

        while current < end_date:
            if self.is_business_day(current):
                count += 1
            current += timedelta(days=1)

        if include_end and self.is_business_day(end_date):
            count += 1

        return count

    def __repr__(self) -> str:
        return f"BusinessCalendar(name='{self.name}', holidays={len(self.holidays)})"


# Standard calendar instances
CALENDAR_USD = BusinessCalendar("USD", weekend_days={5, 6})
CALENDAR_EUR = BusinessCalendar("EUR", weekend_days={5, 6})
CALENDAR_GBP = BusinessCalendar("GBP", weekend_days={5, 6})
CALENDAR_JPY = BusinessCalendar("JPY", weekend_days={5, 6})


def get_calendar(name: str) -> BusinessCalendar:
    """
    Get standard calendar by name.

    Args:
        name: Calendar name (e.g., "USD", "EUR")

    Returns:
        Business calendar instance

    Note:
        For production use, would load actual holiday calendars from data source.
    """
    calendars = {
        "USD": CALENDAR_USD,
        "EUR": CALENDAR_EUR,
        "GBP": CALENDAR_GBP,
        "JPY": CALENDAR_JPY,
    }

    if name not in calendars:
        # Return default calendar with no holidays
        return BusinessCalendar(name)

    return calendars[name]
