"""
Trade Factory - Core Engine for Convention-Based Trade Generation

This module provides the central TradeFactory class that generates trades
conforming to market conventions while allowing individual overrides. It
coordinates between convention profiles, validation, schedule generation,
and product instantiation.

Key Features:
- Automatic convention application from registry
- Individual field override support
- Schedule generation with market conventions
- Trade and Product simultaneous generation
- Validation warning collection
- Support for all input interfaces (Python API, REST, FpML, CSV)
"""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional, Dict, Any, Tuple, List
from uuid import uuid4

from neutryx.portfolio.contracts.trade import Trade, ProductType, TradeStatus, SettlementType
from neutryx.core.dates.schedule import (
    generate_schedule,
    Schedule,
    Frequency,
    DateGeneration,
)
from neutryx.core.dates.calendar import Calendar, US, TARGET, UK, JP
from neutryx.core.dates.business_day import (
    BusinessDayConvention,
    MODIFIED_FOLLOWING,
    FOLLOWING,
)
from neutryx.core.dates.day_count import DayCountConvention
from neutryx.market.convention_profiles import (
    ConventionProfile,
    ProductTypeConvention,
    get_convention_profile,
)
from neutryx.portfolio.trade_generation.validation import (
    ConventionValidator,
    ValidationResult,
    ValidationWarning,
    ValidationSeverity,
)


@dataclass
class TradeGenerationRequest:
    """
    Request object for trade generation

    Attributes:
        currency: Currency code (e.g., "USD")
        product_type: Product type
        trade_date: Trade execution date
        tenor: Tenor string (e.g., "5Y", "10Y") or explicit maturity_date
        maturity_date: Explicit maturity date (alternative to tenor)
        notional: Notional amount
        counterparty_id: Counterparty identifier

        # Product-specific parameters
        fixed_rate: Fixed rate (for IRS, OIS)
        spread: Spread over floating index
        swap_type: "PAYER" or "RECEIVER"

        # Convention overrides (optional)
        fixed_leg_frequency: Override fixed leg frequency
        floating_leg_frequency: Override floating leg frequency
        fixed_leg_day_count: Override fixed leg day count
        floating_leg_day_count: Override floating leg day count
        spot_lag: Override spot lag
        calendars: Override calendars

        # Trade metadata
        trade_number: External trade number
        book_id: Book identifier
        desk_id: Desk identifier
        trader_id: Trader identifier
    """
    currency: str
    product_type: ProductTypeConvention
    trade_date: date
    notional: float
    counterparty_id: str

    # Tenor or maturity
    tenor: Optional[str] = None
    maturity_date: Optional[date] = None

    # Product parameters
    fixed_rate: Optional[float] = None
    spread: float = 0.0
    swap_type: str = "PAYER"  # PAYER or RECEIVER

    # Convention overrides
    fixed_leg_frequency: Optional[Frequency] = None
    floating_leg_frequency: Optional[Frequency] = None
    fixed_leg_day_count: Optional[DayCountConvention] = None
    floating_leg_day_count: Optional[DayCountConvention] = None
    fixed_leg_business_day_convention: Optional[BusinessDayConvention] = None
    floating_leg_business_day_convention: Optional[BusinessDayConvention] = None
    spot_lag: Optional[int] = None
    calendars: Optional[List[str]] = None
    end_of_month: Optional[bool] = None

    # Trade metadata
    trade_number: Optional[str] = None
    external_id: Optional[str] = None
    book_id: Optional[str] = None
    desk_id: Optional[str] = None
    trader_id: Optional[str] = None
    status: TradeStatus = TradeStatus.ACTIVE


@dataclass
class TradeGenerationResult:
    """
    Result of trade generation

    Attributes:
        trade: Generated Trade object
        product: Generated Product object (stored in trade.product_details)
        schedules: Generated payment schedules (if applicable)
        validation_result: Validation warnings
        convention_profile: Convention profile used
    """
    trade: Trade
    product: Any  # Product type varies
    schedules: Optional[Dict[str, Schedule]] = None
    validation_result: Optional[ValidationResult] = None
    convention_profile: Optional[ConventionProfile] = None

    def has_warnings(self) -> bool:
        """Check if generation produced warnings"""
        return self.validation_result and self.validation_result.has_warnings()

    def get_warnings(self) -> List[ValidationWarning]:
        """Get list of warnings"""
        if self.validation_result:
            return self.validation_result.warnings
        return []


class TradeFactory:
    """
    Core factory for generating convention-based trades

    This factory is the central component called by all input interfaces
    (Python API, REST API, FpML, CSV). It applies market conventions,
    handles overrides, generates schedules, and creates Trade + Product pairs.
    """

    def __init__(self, validator: Optional[ConventionValidator] = None):
        """
        Initialize trade factory

        Args:
            validator: Optional custom validator (defaults to standard validator)
        """
        self.validator = validator or ConventionValidator()
        self._calendar_cache: Dict[str, Calendar] = {
            "USD": US,
            "TARGET": TARGET,
            "EUR": TARGET,
            "GBP": UK,
            "JPY": JP,
        }

    def _get_calendar(self, calendar_name: str) -> Calendar:
        """Get calendar by name"""
        if calendar_name in self._calendar_cache:
            return self._calendar_cache[calendar_name]
        # Default to US calendar if not found
        return US

    def _parse_tenor(self, tenor: str) -> int:
        """
        Parse tenor string to number of years

        Args:
            tenor: Tenor string (e.g., "5Y", "10Y", "18M")

        Returns:
            Number of years (fractional for months)
        """
        tenor = tenor.upper().strip()
        if tenor.endswith("Y"):
            return int(tenor[:-1])
        elif tenor.endswith("M"):
            return int(tenor[:-1]) / 12
        elif tenor.endswith("W"):
            return int(tenor[:-1]) / 52
        else:
            raise ValueError(f"Invalid tenor format: {tenor}. Use format like '5Y', '18M', '26W'")

    def _calculate_maturity(
        self,
        trade_date: date,
        effective_date: date,
        tenor: Optional[str] = None,
        maturity_date: Optional[date] = None,
    ) -> date:
        """Calculate maturity date from tenor or use explicit date"""
        if maturity_date:
            return maturity_date

        if not tenor:
            raise ValueError("Either tenor or maturity_date must be provided")

        years = self._parse_tenor(tenor)
        # Calculate approximate maturity (will be adjusted by schedule generation)
        days = int(years * 365.25)
        return effective_date + timedelta(days=days)

    def _calculate_effective_date(
        self,
        trade_date: date,
        spot_lag: int,
        calendar: Calendar,
    ) -> date:
        """Calculate effective date from trade date and spot lag"""
        effective = trade_date
        business_days = 0

        while business_days < spot_lag:
            effective += timedelta(days=1)
            if calendar.is_business_day(effective):
                business_days += 1

        return effective

    def _merge_conventions_with_overrides(
        self,
        profile: ConventionProfile,
        request: TradeGenerationRequest,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Merge convention profile with user overrides

        Returns:
            Tuple of (fixed_leg_params, floating_leg_params, other_params)
        """
        # Fixed leg parameters
        fixed_leg = {}
        if profile.fixed_leg:
            fixed_leg = {
                "frequency": request.fixed_leg_frequency or profile.fixed_leg.frequency,
                "day_count": request.fixed_leg_day_count or profile.fixed_leg.day_count,
                "business_day_convention": request.fixed_leg_business_day_convention or profile.fixed_leg.business_day_convention,
                "payment_lag": profile.fixed_leg.payment_lag,
            }

        # Floating leg parameters
        floating_leg = {}
        if profile.floating_leg:
            floating_leg = {
                "frequency": request.floating_leg_frequency or profile.floating_leg.frequency,
                "day_count": request.floating_leg_day_count or profile.floating_leg.day_count,
                "business_day_convention": request.floating_leg_business_day_convention or profile.floating_leg.business_day_convention,
                "payment_lag": profile.floating_leg.payment_lag,
                "rate_index": profile.floating_leg.rate_index,
                "spread": request.spread,
                "compounding": profile.floating_leg.compounding,
            }

        # Other parameters
        other_params = {
            "spot_lag": request.spot_lag if request.spot_lag is not None else profile.spot_lag,
            "calendars": request.calendars or profile.calendars,
            "end_of_month": request.end_of_month if request.end_of_month is not None else profile.end_of_month,
            "date_generation": profile.date_generation,
        }

        return fixed_leg, floating_leg, other_params

    def _generate_schedules(
        self,
        effective_date: date,
        maturity_date: date,
        fixed_leg: Dict[str, Any],
        floating_leg: Dict[str, Any],
        other_params: Dict[str, Any],
    ) -> Dict[str, Schedule]:
        """Generate payment schedules for fixed and floating legs"""
        schedules = {}

        # Get calendar
        calendar_names = other_params.get("calendars", ["USD"])
        calendar = self._get_calendar(calendar_names[0])

        # Generate fixed leg schedule
        if fixed_leg:
            schedules["fixed"] = generate_schedule(
                effective_date=effective_date,
                termination_date=maturity_date,
                frequency=fixed_leg["frequency"],
                calendar=calendar,
                convention=fixed_leg["business_day_convention"],
                day_count=fixed_leg["day_count"],
                end_of_month=other_params.get("end_of_month", False),
                date_generation=DateGeneration.BACKWARD,
            )

        # Generate floating leg schedule
        if floating_leg:
            schedules["floating"] = generate_schedule(
                effective_date=effective_date,
                termination_date=maturity_date,
                frequency=floating_leg["frequency"],
                calendar=calendar,
                convention=floating_leg["business_day_convention"],
                day_count=floating_leg["day_count"],
                end_of_month=other_params.get("end_of_month", False),
                date_generation=DateGeneration.BACKWARD,
            )

        return schedules

    def _create_trade_object(
        self,
        request: TradeGenerationRequest,
        effective_date: date,
        maturity_date: date,
        profile: ConventionProfile,
        product_details: Dict[str, Any],
    ) -> Trade:
        """Create Trade object with all metadata"""
        # Map ProductTypeConvention to ProductType enum
        product_type_map = {
            ProductTypeConvention.INTEREST_RATE_SWAP: ProductType.INTEREST_RATE_SWAP,
            ProductTypeConvention.OVERNIGHT_INDEX_SWAP: ProductType.INTEREST_RATE_SWAP,
            ProductTypeConvention.CROSS_CURRENCY_SWAP: ProductType.INTEREST_RATE_SWAP,
            ProductTypeConvention.BASIS_SWAP: ProductType.INTEREST_RATE_SWAP,
            ProductTypeConvention.SWAPTION: ProductType.SWAPTION,
            ProductTypeConvention.FX_OPTION: ProductType.FX_OPTION,
            ProductTypeConvention.EQUITY_OPTION: ProductType.EQUITY_OPTION,
            ProductTypeConvention.CDS: ProductType.CREDIT_DEFAULT_SWAP,
        }

        product_type = product_type_map.get(
            request.product_type,
            ProductType.OTHER
        )

        trade = Trade(
            id=str(uuid4()),
            trade_number=request.trade_number,
            external_id=request.external_id,
            counterparty_id=request.counterparty_id,
            book_id=request.book_id,
            desk_id=request.desk_id,
            trader_id=request.trader_id,
            product_type=product_type,
            trade_date=request.trade_date,
            effective_date=effective_date,
            maturity_date=maturity_date,
            status=request.status,
            notional=request.notional,
            currency=request.currency,
            settlement_type=SettlementType.CASH,
            product_details=product_details,
            convention_profile_id=profile.get_profile_id(),
            generated_from_convention=True,
        )

        return trade

    def _validate_request(
        self,
        request: TradeGenerationRequest,
        fixed_leg: Dict[str, Any],
        floating_leg: Dict[str, Any],
        other_params: Dict[str, Any],
    ) -> ValidationResult:
        """Validate trade request against conventions"""
        return self.validator.validate_trade_parameters(
            currency=request.currency,
            product_type=request.product_type,
            fixed_leg_params=fixed_leg if fixed_leg else None,
            floating_leg_params=floating_leg if floating_leg else None,
            other_params=other_params,
        )

    def generate_trade(
        self,
        request: TradeGenerationRequest,
    ) -> TradeGenerationResult:
        """
        Generate a trade from a request

        This is the main entry point for trade generation. It:
        1. Looks up convention profile
        2. Merges conventions with overrides
        3. Validates parameters
        4. Calculates dates
        5. Generates schedules
        6. Creates Trade and Product objects

        Args:
            request: Trade generation request

        Returns:
            TradeGenerationResult with trade, product, schedules, and warnings

        Example:
            >>> factory = TradeFactory()
            >>> request = TradeGenerationRequest(
            ...     currency="USD",
            ...     product_type=ProductTypeConvention.INTEREST_RATE_SWAP,
            ...     trade_date=date(2024, 1, 15),
            ...     tenor="5Y",
            ...     notional=10_000_000,
            ...     fixed_rate=0.045,
            ...     counterparty_id="CP-001",
            ... )
            >>> result = factory.generate_trade(request)
            >>> print(result.trade)
            >>> print(result.has_warnings())
        """
        # Get convention profile
        profile = get_convention_profile(request.currency, request.product_type)
        if profile is None:
            raise ValueError(
                f"No convention profile found for {request.currency} "
                f"{request.product_type.value}"
            )

        # Merge conventions with overrides
        fixed_leg, floating_leg, other_params = self._merge_conventions_with_overrides(
            profile, request
        )

        # Validate parameters
        validation_result = self._validate_request(
            request, fixed_leg, floating_leg, other_params
        )

        # Calculate dates
        calendar = self._get_calendar(other_params["calendars"][0])
        effective_date = self._calculate_effective_date(
            request.trade_date,
            other_params["spot_lag"],
            calendar,
        )
        maturity_date = self._calculate_maturity(
            request.trade_date,
            effective_date,
            request.tenor,
            request.maturity_date,
        )

        # Generate schedules
        schedules = self._generate_schedules(
            effective_date,
            maturity_date,
            fixed_leg,
            floating_leg,
            other_params,
        )

        # Create product details
        product_details = {
            "product_type_convention": request.product_type.value,
            "convention_profile_id": profile.get_profile_id(),
            "fixed_leg": fixed_leg,
            "floating_leg": floating_leg,
            "other_params": other_params,
            "fixed_rate": request.fixed_rate,
            "spread": request.spread,
            "swap_type": request.swap_type,
        }

        # Create Trade object
        trade = self._create_trade_object(
            request,
            effective_date,
            maturity_date,
            profile,
            product_details,
        )

        # Return result
        return TradeGenerationResult(
            trade=trade,
            product=product_details,  # Placeholder - will be replaced by actual Product objects
            schedules=schedules,
            validation_result=validation_result,
            convention_profile=profile,
        )
