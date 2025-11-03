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


# ==============================================================================
# IMM (International Monetary Market) Dates
# ==============================================================================


def is_imm_month(month: int) -> bool:
    """
    Check if a month is an IMM month (March, June, September, December).

    Args:
        month: Month number (1-12)

    Returns:
        True if IMM month, False otherwise

    Example:
        >>> is_imm_month(3)   # March
        True
        >>> is_imm_month(4)   # April
        False
    """
    return month in (3, 6, 9, 12)


def get_imm_date(year: int, month: int) -> date:
    """
    Get the IMM date for a given year and month.

    IMM dates are the third Wednesday of March, June, September, and December.
    These are standard settlement dates for exchange-traded derivatives.

    Args:
        year: Year
        month: Month (must be 3, 6, 9, or 12)

    Returns:
        IMM date as datetime.date

    Raises:
        ValueError: If month is not an IMM month

    Example:
        >>> # Get March 2024 IMM date (should be March 20, 2024 - 3rd Wednesday)
        >>> imm = get_imm_date(2024, 3)
        >>> imm.day >= 15 and imm.day <= 21  # 3rd week
        True
        >>> imm.weekday() == 2  # Wednesday
        True
    """
    if not is_imm_month(month):
        raise ValueError(f"Month {month} is not an IMM month (3, 6, 9, 12)")

    # Start from first day of month
    first_day = date(year, month, 1)

    # Find first Wednesday (weekday 2 = Wednesday)
    days_until_wednesday = (2 - first_day.weekday()) % 7
    first_wednesday = first_day + timedelta(days=days_until_wednesday)

    # Third Wednesday is 2 weeks after first Wednesday
    third_wednesday = first_wednesday + timedelta(weeks=2)

    return third_wednesday


def get_next_imm_date(ref_date: date, num_imm: int = 1) -> date:
    """
    Get the next IMM date(s) after a reference date.

    Args:
        ref_date: Reference date
        num_imm: Number of IMM cycles forward (default 1)

    Returns:
        The n-th next IMM date

    Example:
        >>> from datetime import date
        >>> ref = date(2024, 1, 15)
        >>> next_imm = get_next_imm_date(ref)  # Next IMM after Jan 15, 2024
        >>> next_imm.month  # Should be March
        3
        >>> next_imm.weekday()  # Should be Wednesday
        2
    """
    if num_imm < 1:
        raise ValueError("num_imm must be at least 1")

    # IMM months cycle
    imm_months = [3, 6, 9, 12]

    # Start from reference month
    current_month = ref_date.month
    current_year = ref_date.year

    count = 0
    while count < num_imm:
        # Find next IMM month
        next_imm_months = [m for m in imm_months if m > current_month]
        if next_imm_months:
            current_month = next_imm_months[0]
        else:
            # Wrap to next year
            current_month = imm_months[0]
            current_year += 1

        # Get the IMM date
        imm_date = get_imm_date(current_year, current_month)

        # Check if it's actually after ref_date
        if imm_date > ref_date:
            count += 1
            if count == num_imm:
                return imm_date

    # Should never reach here
    raise RuntimeError("Failed to find IMM date")


def get_imm_code(imm_date: date) -> str:
    """
    Get the standard IMM contract code for a date.

    IMM codes are in format: [Month Code][Year Digit]
    - Month codes: H=Mar, M=Jun, U=Sep, Z=Dec
    - Year digit: Last digit of year

    Args:
        imm_date: IMM date

    Returns:
        IMM code string (e.g., "H4" for March 2024)

    Raises:
        ValueError: If date is not in an IMM month

    Example:
        >>> from datetime import date
        >>> imm = date(2024, 3, 20)  # March 2024
        >>> get_imm_code(imm)
        'H4'
        >>> imm = date(2025, 12, 17)  # December 2025
        >>> get_imm_code(imm)
        'Z5'
    """
    if not is_imm_month(imm_date.month):
        raise ValueError(f"Date {imm_date} is not in an IMM month")

    # Month code mapping
    month_codes = {
        3: 'H',   # March
        6: 'M',   # June
        9: 'U',   # September
        12: 'Z',  # December
    }

    month_code = month_codes[imm_date.month]
    year_digit = str(imm_date.year)[-1]  # Last digit of year

    return f"{month_code}{year_digit}"


def parse_imm_code(imm_code: str, ref_year: Optional[int] = None) -> date:
    """
    Parse an IMM contract code to get the IMM date.

    Args:
        imm_code: IMM code (e.g., "H4", "Z5")
        ref_year: Reference year for decade disambiguation (optional)
                  If not provided, assumes current decade

    Returns:
        IMM date

    Raises:
        ValueError: If code is invalid

    Example:
        >>> parse_imm_code("H4", ref_year=2024)
        datetime.date(2024, 3, 20)
        >>> parse_imm_code("M5", ref_year=2025)
        datetime.date(2025, 6, 18)
    """
    if len(imm_code) != 2:
        raise ValueError(f"IMM code must be 2 characters, got: {imm_code}")

    month_code = imm_code[0].upper()
    year_digit = imm_code[1]

    # Month code mapping
    code_to_month = {
        'H': 3,   # March
        'M': 6,   # June
        'U': 9,   # September
        'Z': 12,  # December
    }

    if month_code not in code_to_month:
        raise ValueError(f"Invalid month code: {month_code}")

    month = code_to_month[month_code]

    # Determine year
    if ref_year is None:
        from datetime import datetime
        ref_year = datetime.now().year

    decade_start = (ref_year // 10) * 10
    year = decade_start + int(year_digit)

    # If year is more than 5 years in the past, assume next decade
    if year < ref_year - 5:
        year += 10

    return get_imm_date(year, month)


def get_imm_dates_between(start_date: date, end_date: date) -> list[date]:
    """
    Get all IMM dates between two dates (inclusive).

    Args:
        start_date: Start date
        end_date: End date

    Returns:
        List of IMM dates

    Example:
        >>> from datetime import date
        >>> start = date(2024, 1, 1)
        >>> end = date(2024, 12, 31)
        >>> imm_dates = get_imm_dates_between(start, end)
        >>> len(imm_dates)  # Should be 4 (Mar, Jun, Sep, Dec)
        4
        >>> [d.month for d in imm_dates]
        [3, 6, 9, 12]
    """
    if end_date < start_date:
        raise ValueError("end_date must be >= start_date")

    imm_dates = []
    current_date = start_date

    # Find first IMM date at or after start_date
    first_imm = get_next_imm_date(current_date - timedelta(days=1), num_imm=1)

    current_imm = first_imm
    while current_imm <= end_date:
        imm_dates.append(current_imm)
        # Get next IMM date
        current_imm = get_next_imm_date(current_imm, num_imm=1)

    return imm_dates
