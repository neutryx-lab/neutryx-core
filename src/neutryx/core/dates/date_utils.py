"""Date utility functions for financial calculations."""

import datetime
from typing import Union

DateLike = Union[datetime.date, datetime.datetime]


def is_weekend(date: DateLike) -> bool:
    """Check if a date falls on a weekend (Saturday or Sunday).

    Args:
        date: Date to check

    Returns:
        True if the date is a Saturday or Sunday, False otherwise
    """
    return date.weekday() >= 5


def is_leap_year(year: int) -> bool:
    """Check if a year is a leap year.

    A year is a leap year if:
    - It is divisible by 4 AND
    - Either not divisible by 100 OR divisible by 400

    Args:
        year: Year to check

    Returns:
        True if the year is a leap year, False otherwise
    """
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def days_in_month(year: int, month: int) -> int:
    """Get the number of days in a specific month.

    Args:
        year: Year
        month: Month (1-12)

    Returns:
        Number of days in the month
    """
    if month in (1, 3, 5, 7, 8, 10, 12):
        return 31
    elif month in (4, 6, 9, 11):
        return 30
    elif month == 2:
        return 29 if is_leap_year(year) else 28
    else:
        raise ValueError(f"Invalid month: {month}")


def days_in_year(year: int) -> int:
    """Get the number of days in a year.

    Args:
        year: Year

    Returns:
        366 for leap years, 365 otherwise
    """
    return 366 if is_leap_year(year) else 365


def add_months(date: datetime.date, months: int) -> datetime.date:
    """Add a number of months to a date.

    If the resulting date would be invalid (e.g., Jan 31 + 1 month),
    the date is adjusted to the last valid day of the month.

    Args:
        date: Starting date
        months: Number of months to add (can be negative)

    Returns:
        New date after adding months
    """
    # Calculate new month and year
    total_months = date.month + months
    year = date.year + (total_months - 1) // 12
    month = ((total_months - 1) % 12) + 1

    # Adjust day if necessary
    max_day = days_in_month(year, month)
    day = min(date.day, max_day)

    return datetime.date(year, month, day)


def add_years(date: datetime.date, years: int) -> datetime.date:
    """Add a number of years to a date.

    If the date is Feb 29 and the target year is not a leap year,
    the date is adjusted to Feb 28.

    Args:
        date: Starting date
        years: Number of years to add (can be negative)

    Returns:
        New date after adding years
    """
    year = date.year + years
    month = date.month
    day = date.day

    # Handle Feb 29 in non-leap years
    if month == 2 and day == 29 and not is_leap_year(year):
        day = 28

    return datetime.date(year, month, day)


def date_diff_days(date1: DateLike, date2: DateLike) -> int:
    """Calculate the number of days between two dates.

    Args:
        date1: First date
        date2: Second date

    Returns:
        Number of days from date1 to date2 (positive if date2 > date1)
    """
    if isinstance(date1, datetime.datetime):
        date1 = date1.date()
    if isinstance(date2, datetime.datetime):
        date2 = date2.date()

    return (date2 - date1).days


def date_to_serial(date: DateLike, base_date: datetime.date = datetime.date(1899, 12, 30)) -> int:
    """Convert a date to a serial number (Excel-style).

    Args:
        date: Date to convert
        base_date: Base date for serial numbering (default: Dec 30, 1899)

    Returns:
        Serial number representing the date
    """
    if isinstance(date, datetime.datetime):
        date = date.date()

    return (date - base_date).days


def serial_to_date(serial: int, base_date: datetime.date = datetime.date(1899, 12, 30)) -> datetime.date:
    """Convert a serial number to a date (Excel-style).

    Args:
        serial: Serial number
        base_date: Base date for serial numbering (default: Dec 30, 1899)

    Returns:
        Date corresponding to the serial number
    """
    return base_date + datetime.timedelta(days=serial)


def year_fraction_simple(date1: DateLike, date2: DateLike) -> float:
    """Calculate a simple year fraction between two dates.

    This is a simple approximation: days / 365.25
    For accurate calculations, use a DayCountConvention.

    Args:
        date1: Start date
        date2: End date

    Returns:
        Year fraction between the dates
    """
    days = date_diff_days(date1, date2)
    return days / 365.25


def end_of_month(date: DateLike) -> datetime.date:
    """Get the last day of the month for a given date.

    Args:
        date: Input date

    Returns:
        Last day of the month
    """
    if isinstance(date, datetime.datetime):
        date = date.date()

    last_day = days_in_month(date.year, date.month)
    return datetime.date(date.year, date.month, last_day)


def is_end_of_month(date: DateLike) -> bool:
    """Check if a date is the last day of the month.

    Args:
        date: Date to check

    Returns:
        True if the date is the last day of the month
    """
    if isinstance(date, datetime.datetime):
        date = date.date()

    return date.day == days_in_month(date.year, date.month)


def third_wednesday(year: int, month: int) -> datetime.date:
    """Get the third Wednesday of a given month.

    This is useful for calculating option expiration dates.

    Args:
        year: Year
        month: Month (1-12)

    Returns:
        Date of the third Wednesday
    """
    # First day of the month
    first_day = datetime.date(year, month, 1)

    # Find the first Wednesday
    days_until_wednesday = (2 - first_day.weekday()) % 7
    first_wednesday = first_day + datetime.timedelta(days=days_until_wednesday)

    # Add 14 days to get the third Wednesday
    third_wednesday = first_wednesday + datetime.timedelta(days=14)

    return third_wednesday


def imm_date(year: int, quarter: int) -> datetime.date:
    """Get the IMM (International Monetary Market) date for a given quarter.

    IMM dates are the third Wednesday of March, June, September, and December.

    Args:
        year: Year
        quarter: Quarter (1-4)

    Returns:
        IMM date for the specified quarter

    Raises:
        ValueError: If quarter is not 1-4
    """
    if quarter not in (1, 2, 3, 4):
        raise ValueError(f"Quarter must be 1-4, got {quarter}")

    month = quarter * 3  # 3, 6, 9, 12
    return third_wednesday(year, month)


def easter_date(year: int) -> datetime.date:
    """Calculate Easter Sunday for a given year using Meeus/Jones/Butcher algorithm.

    This is useful for calculating holidays that depend on Easter.

    Args:
        year: Year

    Returns:
        Date of Easter Sunday
    """
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1

    return datetime.date(year, month, day)
