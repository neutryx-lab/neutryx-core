"""Business day conventions for date adjustments.

Business day conventions determine how dates that fall on non-business days
(weekends or holidays) are adjusted to valid business days.
"""

import datetime
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from neutryx.core.utils.dates.date_utils import DateLike

if TYPE_CHECKING:
    from neutryx.core.utils.dates.calendar import Calendar


class BusinessDayConvention(ABC):
    """Abstract base class for business day conventions."""

    @abstractmethod
    def adjust(self, date: DateLike, calendar: "Calendar") -> datetime.date:
        """Adjust a date according to the convention.

        Args:
            date: Date to adjust
            calendar: Holiday calendar to use

        Returns:
            Adjusted date that is a valid business day
        """
        pass

    def __repr__(self) -> str:
        return self.__class__.__name__


class Unadjusted(BusinessDayConvention):
    """Unadjusted convention - no adjustment is made.

    The date is returned as-is, even if it falls on a non-business day.
    """

    def adjust(self, date: DateLike, calendar: "Calendar") -> datetime.date:
        """Return the date without adjustment."""
        if isinstance(date, datetime.datetime):
            return date.date()
        return date


class Following(BusinessDayConvention):
    """Following business day convention.

    If the date falls on a non-business day, move forward to the next business day.
    """

    def adjust(self, date: DateLike, calendar: "Calendar") -> datetime.date:
        """Adjust forward to the next business day."""
        if isinstance(date, datetime.datetime):
            date = date.date()

        while not calendar.is_business_day(date):
            date += datetime.timedelta(days=1)

        return date


class Preceding(BusinessDayConvention):
    """Preceding business day convention.

    If the date falls on a non-business day, move backward to the previous business day.
    """

    def adjust(self, date: DateLike, calendar: "Calendar") -> datetime.date:
        """Adjust backward to the previous business day."""
        if isinstance(date, datetime.datetime):
            date = date.date()

        while not calendar.is_business_day(date):
            date -= datetime.timedelta(days=1)

        return date


class ModifiedFollowing(BusinessDayConvention):
    """Modified following business day convention.

    If the date falls on a non-business day, move forward to the next business day,
    unless that would cross into the next month, in which case move backward to
    the previous business day.
    """

    def adjust(self, date: DateLike, calendar: "Calendar") -> datetime.date:
        """Adjust using modified following convention."""
        if isinstance(date, datetime.datetime):
            date = date.date()

        original_month = date.month
        adjusted = date

        # Move forward to next business day
        while not calendar.is_business_day(adjusted):
            adjusted += datetime.timedelta(days=1)

        # If we crossed into next month, go backward instead
        if adjusted.month != original_month:
            adjusted = date
            while not calendar.is_business_day(adjusted):
                adjusted -= datetime.timedelta(days=1)

        return adjusted


class ModifiedPreceding(BusinessDayConvention):
    """Modified preceding business day convention.

    If the date falls on a non-business day, move backward to the previous business day,
    unless that would cross into the previous month, in which case move forward to
    the next business day.
    """

    def adjust(self, date: DateLike, calendar: "Calendar") -> datetime.date:
        """Adjust using modified preceding convention."""
        if isinstance(date, datetime.datetime):
            date = date.date()

        original_month = date.month
        adjusted = date

        # Move backward to previous business day
        while not calendar.is_business_day(adjusted):
            adjusted -= datetime.timedelta(days=1)

        # If we crossed into previous month, go forward instead
        if adjusted.month != original_month:
            adjusted = date
            while not calendar.is_business_day(adjusted):
                adjusted += datetime.timedelta(days=1)

        return adjusted


class HalfMonthModifiedFollowing(BusinessDayConvention):
    """Half-month modified following convention.

    Similar to Modified Following, but the month boundary check is done at the 15th:
    - For dates 1-15: adjust forward unless it crosses the 15th
    - For dates 16-end: adjust forward unless it crosses month end
    """

    def adjust(self, date: DateLike, calendar: "Calendar") -> datetime.date:
        """Adjust using half-month modified following convention."""
        if isinstance(date, datetime.datetime):
            date = date.date()

        original_day = date.day
        adjusted = date

        # Move forward to next business day
        while not calendar.is_business_day(adjusted):
            adjusted += datetime.timedelta(days=1)

        # Check if we crossed the half-month boundary
        if original_day <= 15 and adjusted.day > 15:
            # Crossed mid-month boundary, go backward
            adjusted = date
            while not calendar.is_business_day(adjusted):
                adjusted -= datetime.timedelta(days=1)
        elif original_day > 15 and adjusted.month != date.month:
            # Crossed month boundary, go backward
            adjusted = date
            while not calendar.is_business_day(adjusted):
                adjusted -= datetime.timedelta(days=1)

        return adjusted


class Nearest(BusinessDayConvention):
    """Nearest business day convention.

    If the date falls on a non-business day, move to the nearest business day.
    If equidistant, move forward.
    """

    def adjust(self, date: DateLike, calendar: "Calendar") -> datetime.date:
        """Adjust to the nearest business day."""
        if isinstance(date, datetime.datetime):
            date = date.date()

        if calendar.is_business_day(date):
            return date

        forward = date
        backward = date
        forward_days = 0
        backward_days = 0

        # Find next business day forward
        while not calendar.is_business_day(forward):
            forward += datetime.timedelta(days=1)
            forward_days += 1

        # Find next business day backward
        while not calendar.is_business_day(backward):
            backward -= datetime.timedelta(days=1)
            backward_days += 1

        # Return the nearest (forward if equal)
        if forward_days <= backward_days:
            return forward
        else:
            return backward


# Convenience instances
UNADJUSTED = Unadjusted()
FOLLOWING = Following()
PRECEDING = Preceding()
MODIFIED_FOLLOWING = ModifiedFollowing()
MODIFIED_PRECEDING = ModifiedPreceding()
NEAREST = Nearest()
