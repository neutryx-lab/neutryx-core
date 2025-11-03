"""Holiday calendars for financial markets.

This module provides holiday calendar implementations for various markets,
used to determine business days for trade settlement and schedule generation.
"""

import datetime
from abc import ABC, abstractmethod
from typing import Set, List, Union

from neutryx.core.utils.time.date_utils import (
    is_weekend,
    easter_date,
    DateLike,
    add_months,
)


class Calendar(ABC):
    """Abstract base class for holiday calendars."""

    @abstractmethod
    def is_business_day(self, date: DateLike) -> bool:
        """Check if a date is a business day.

        Args:
            date: Date to check

        Returns:
            True if the date is a business day, False otherwise
        """
        pass

    def is_holiday(self, date: DateLike) -> bool:
        """Check if a date is a holiday.

        Args:
            date: Date to check

        Returns:
            True if the date is a holiday, False otherwise
        """
        return not self.is_business_day(date)

    def adjust(self, date: DateLike, convention) -> datetime.date:
        """Adjust a date according to a business day convention.

        Args:
            date: Date to adjust
            convention: Business day convention to apply

        Returns:
            Adjusted date
        """
        return convention.adjust(date, self)

    def advance(self, date: DateLike, days: int) -> datetime.date:
        """Advance a date by a number of business days.

        Args:
            date: Starting date
            days: Number of business days to advance (can be negative)

        Returns:
            New date after advancing
        """
        if isinstance(date, datetime.datetime):
            date = date.date()

        if days == 0:
            return date

        step = 1 if days > 0 else -1
        remaining = abs(days)
        current = date

        while remaining > 0:
            current += datetime.timedelta(days=step)
            if self.is_business_day(current):
                remaining -= 1

        return current

    def business_days_between(self, date1: DateLike, date2: DateLike) -> int:
        """Count business days between two dates.

        Args:
            date1: Start date (inclusive)
            date2: End date (exclusive)

        Returns:
            Number of business days
        """
        if isinstance(date1, datetime.datetime):
            date1 = date1.date()
        if isinstance(date2, datetime.datetime):
            date2 = date2.date()

        if date1 >= date2:
            return 0

        count = 0
        current = date1
        while current < date2:
            if self.is_business_day(current):
                count += 1
            current += datetime.timedelta(days=1)

        return count


class NullCalendar(Calendar):
    """Calendar with no holidays (all days except weekends are business days)."""

    def is_business_day(self, date: DateLike) -> bool:
        """Check if date is a business day (not weekend)."""
        return not is_weekend(date)


class HolidayCalendar(Calendar):
    """Base class for calendars with specific holiday rules."""

    def __init__(self):
        """Initialize the holiday calendar."""
        self._holiday_cache: Set[datetime.date] = set()
        self._cache_years: Set[int] = set()

    @abstractmethod
    def _generate_holidays(self, year: int) -> List[datetime.date]:
        """Generate holidays for a specific year.

        Args:
            year: Year to generate holidays for

        Returns:
            List of holiday dates
        """
        pass

    def _ensure_year_cached(self, year: int):
        """Ensure holidays for a year are cached.

        Args:
            year: Year to cache
        """
        if year not in self._cache_years:
            holidays = self._generate_holidays(year)
            self._holiday_cache.update(holidays)
            self._cache_years.add(year)

    def is_business_day(self, date: DateLike) -> bool:
        """Check if date is a business day."""
        if isinstance(date, datetime.datetime):
            date = date.date()

        if is_weekend(date):
            return False

        self._ensure_year_cached(date.year)
        return date not in self._holiday_cache


class TargetCalendar(HolidayCalendar):
    """TARGET (Trans-European Automated Real-time Gross Settlement Express Transfer) calendar.

    This is the settlement calendar for the Euro zone.
    """

    def _generate_holidays(self, year: int) -> List[datetime.date]:
        """Generate TARGET holidays for a year."""
        holidays = [
            datetime.date(year, 1, 1),   # New Year's Day
            datetime.date(year, 5, 1),   # Labour Day
            datetime.date(year, 12, 25), # Christmas Day
            datetime.date(year, 12, 26), # Boxing Day
        ]

        # Easter-based holidays
        easter = easter_date(year)
        holidays.append(easter - datetime.timedelta(days=2))  # Good Friday
        holidays.append(easter + datetime.timedelta(days=1))  # Easter Monday

        # If December 31 is a business day and the next year's January 1 is not a weekend,
        # December 31 is also a holiday (Year-end closing)
        dec_31 = datetime.date(year, 12, 31)
        if dec_31.weekday() < 5:  # Not a weekend
            holidays.append(dec_31)

        return holidays


class USCalendar(HolidayCalendar):
    """US settlement calendar (NYSE/Federal Reserve)."""

    def _generate_holidays(self, year: int) -> List[datetime.date]:
        """Generate US holidays for a year."""
        holidays = []

        # New Year's Day (observed)
        new_years = datetime.date(year, 1, 1)
        if new_years.weekday() == 6:  # Sunday
            holidays.append(datetime.date(year, 1, 2))
        elif new_years.weekday() == 5:  # Saturday
            # Not observed on Friday before in financial markets
            pass
        else:
            holidays.append(new_years)

        # Martin Luther King Jr. Day (3rd Monday in January)
        jan_1 = datetime.date(year, 1, 1)
        days_to_monday = (7 - jan_1.weekday()) % 7
        first_monday = jan_1 + datetime.timedelta(days=days_to_monday)
        mlk_day = first_monday + datetime.timedelta(days=14)  # 3rd Monday
        holidays.append(mlk_day)

        # Presidents' Day (3rd Monday in February)
        feb_1 = datetime.date(year, 2, 1)
        days_to_monday = (7 - feb_1.weekday()) % 7
        first_monday = feb_1 + datetime.timedelta(days=days_to_monday)
        presidents_day = first_monday + datetime.timedelta(days=14)  # 3rd Monday
        holidays.append(presidents_day)

        # Good Friday
        easter = easter_date(year)
        holidays.append(easter - datetime.timedelta(days=2))

        # Memorial Day (last Monday in May)
        may_31 = datetime.date(year, 5, 31)
        days_back_to_monday = (may_31.weekday() - 0) % 7
        memorial_day = may_31 - datetime.timedelta(days=days_back_to_monday)
        holidays.append(memorial_day)

        # Juneteenth (June 19, observed)
        juneteenth = datetime.date(year, 6, 19)
        if year >= 2021:  # Federal holiday since 2021
            if juneteenth.weekday() == 6:  # Sunday
                holidays.append(datetime.date(year, 6, 20))
            elif juneteenth.weekday() == 5:  # Saturday
                holidays.append(datetime.date(year, 6, 18))
            else:
                holidays.append(juneteenth)

        # Independence Day (July 4, observed)
        independence = datetime.date(year, 7, 4)
        if independence.weekday() == 6:  # Sunday
            holidays.append(datetime.date(year, 7, 5))
        elif independence.weekday() == 5:  # Saturday
            holidays.append(datetime.date(year, 7, 3))
        else:
            holidays.append(independence)

        # Labor Day (1st Monday in September)
        sep_1 = datetime.date(year, 9, 1)
        days_to_monday = (7 - sep_1.weekday()) % 7
        labor_day = sep_1 + datetime.timedelta(days=days_to_monday)
        holidays.append(labor_day)

        # Thanksgiving (4th Thursday in November)
        nov_1 = datetime.date(year, 11, 1)
        days_to_thursday = (3 - nov_1.weekday()) % 7
        first_thursday = nov_1 + datetime.timedelta(days=days_to_thursday)
        thanksgiving = first_thursday + datetime.timedelta(days=21)  # 4th Thursday
        holidays.append(thanksgiving)

        # Christmas (December 25, observed)
        christmas = datetime.date(year, 12, 25)
        if christmas.weekday() == 6:  # Sunday
            holidays.append(datetime.date(year, 12, 26))
        elif christmas.weekday() == 5:  # Saturday
            holidays.append(datetime.date(year, 12, 24))
        else:
            holidays.append(christmas)

        return holidays


class UKCalendar(HolidayCalendar):
    """UK settlement calendar (London Stock Exchange)."""

    def _generate_holidays(self, year: int) -> List[datetime.date]:
        """Generate UK holidays for a year."""
        holidays = []

        # New Year's Day (observed)
        new_years = datetime.date(year, 1, 1)
        if new_years.weekday() == 6:  # Sunday
            holidays.append(datetime.date(year, 1, 2))
        elif new_years.weekday() == 5:  # Saturday
            holidays.append(datetime.date(year, 1, 3))
        else:
            holidays.append(new_years)

        # Easter-based holidays
        easter = easter_date(year)
        holidays.append(easter - datetime.timedelta(days=2))  # Good Friday
        holidays.append(easter + datetime.timedelta(days=1))  # Easter Monday

        # Early May Bank Holiday (1st Monday in May)
        may_1 = datetime.date(year, 5, 1)
        days_to_monday = (7 - may_1.weekday()) % 7
        early_may = may_1 + datetime.timedelta(days=days_to_monday)
        holidays.append(early_may)

        # Spring Bank Holiday (last Monday in May)
        may_31 = datetime.date(year, 5, 31)
        days_back_to_monday = (may_31.weekday() - 0) % 7
        spring_bank = may_31 - datetime.timedelta(days=days_back_to_monday)
        holidays.append(spring_bank)

        # Summer Bank Holiday (last Monday in August)
        aug_31 = datetime.date(year, 8, 31)
        days_back_to_monday = (aug_31.weekday() - 0) % 7
        summer_bank = aug_31 - datetime.timedelta(days=days_back_to_monday)
        holidays.append(summer_bank)

        # Christmas Day (observed)
        christmas = datetime.date(year, 12, 25)
        if christmas.weekday() == 6:  # Sunday
            holidays.append(datetime.date(year, 12, 27))
        elif christmas.weekday() == 5:  # Saturday
            holidays.append(datetime.date(year, 12, 27))
        else:
            holidays.append(christmas)

        # Boxing Day (observed)
        boxing = datetime.date(year, 12, 26)
        if boxing.weekday() == 6:  # Sunday
            holidays.append(datetime.date(year, 12, 28))
        elif boxing.weekday() == 5:  # Saturday
            holidays.append(datetime.date(year, 12, 28))
        else:
            holidays.append(boxing)

        return holidays


class JPCalendar(HolidayCalendar):
    """Japanese settlement calendar (Tokyo Stock Exchange)."""

    def _generate_holidays(self, year: int) -> List[datetime.date]:
        """Generate Japanese holidays for a year."""
        holidays = [
            datetime.date(year, 1, 1),   # New Year's Day
            datetime.date(year, 2, 11),  # National Foundation Day
            datetime.date(year, 2, 23),  # Emperor's Birthday (from 2020)
            datetime.date(year, 4, 29),  # Showa Day
            datetime.date(year, 5, 3),   # Constitution Day
            datetime.date(year, 5, 4),   # Greenery Day
            datetime.date(year, 5, 5),   # Children's Day
            datetime.date(year, 8, 11),  # Mountain Day (from 2016)
            datetime.date(year, 11, 3),  # Culture Day
            datetime.date(year, 11, 23), # Labor Thanksgiving Day
        ]

        # Coming of Age Day (2nd Monday in January)
        jan_1 = datetime.date(year, 1, 1)
        days_to_monday = (7 - jan_1.weekday()) % 7
        first_monday = jan_1 + datetime.timedelta(days=days_to_monday)
        coming_of_age = first_monday + datetime.timedelta(days=7)  # 2nd Monday
        holidays.append(coming_of_age)

        # Marine Day (3rd Monday in July, subject to Olympic changes)
        jul_1 = datetime.date(year, 7, 1)
        days_to_monday = (7 - jul_1.weekday()) % 7
        first_monday = jul_1 + datetime.timedelta(days=days_to_monday)
        marine_day = first_monday + datetime.timedelta(days=14)  # 3rd Monday
        holidays.append(marine_day)

        # Respect for the Aged Day (3rd Monday in September)
        sep_1 = datetime.date(year, 9, 1)
        days_to_monday = (7 - sep_1.weekday()) % 7
        first_monday = sep_1 + datetime.timedelta(days=days_to_monday)
        aged_day = first_monday + datetime.timedelta(days=14)  # 3rd Monday
        holidays.append(aged_day)

        # Health and Sports Day (2nd Monday in October)
        oct_1 = datetime.date(year, 10, 1)
        days_to_monday = (7 - oct_1.weekday()) % 7
        first_monday = oct_1 + datetime.timedelta(days=days_to_monday)
        sports_day = first_monday + datetime.timedelta(days=7)  # 2nd Monday
        holidays.append(sports_day)

        # Vernal Equinox (approximate - typically March 20 or 21)
        vernal = datetime.date(year, 3, 20)
        holidays.append(vernal)

        # Autumnal Equinox (approximate - typically September 22 or 23)
        autumnal = datetime.date(year, 9, 23)
        holidays.append(autumnal)

        # Substitute holidays (if a holiday falls on Sunday, Monday becomes a holiday)
        observed_holidays = []
        for holiday in holidays:
            if holiday.weekday() == 6:  # Sunday
                next_day = holiday + datetime.timedelta(days=1)
                observed_holidays.append(next_day)

        holidays.extend(observed_holidays)

        return holidays


class JointCalendar(Calendar):
    """Calendar that combines multiple calendars.

    A day is a business day only if it's a business day in all constituent calendars.
    """

    def __init__(self, *calendars: Calendar):
        """Initialize joint calendar.

        Args:
            calendars: One or more calendars to combine
        """
        if not calendars:
            raise ValueError("At least one calendar must be provided")
        self.calendars = calendars

    def is_business_day(self, date: DateLike) -> bool:
        """Check if date is a business day in all calendars."""
        return all(cal.is_business_day(date) for cal in self.calendars)


# Convenience instances
NULL_CALENDAR = NullCalendar()
TARGET = TargetCalendar()
US = USCalendar()
UK = UKCalendar()
JP = JPCalendar()
