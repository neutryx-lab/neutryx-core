"""
Convention Profile System for Market-Standard Trade Generation

This module provides a comprehensive registry of market conventions for
generating trades that conform to standard market practices. Each profile
defines the default conventions for a specific currency and product type,
while allowing for individual field overrides.

Key Features:
- Market-standard conventions for all major currencies
- Product-specific convention profiles (IRS, OIS, CCS, Basis, FRA, etc.)
- Override mechanism for non-standard trades
- Warning generation for convention violations
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List
from datetime import date

from neutryx.core.dates.schedule import Frequency
from neutryx.core.dates.day_count import (
    DayCountConvention,
    ACT_360,
    ACT_365,
    ACT_ACT,
    THIRTY_360,
)
from neutryx.core.dates.business_day import (
    BusinessDayConvention,
    MODIFIED_FOLLOWING,
    FOLLOWING,
)
from neutryx.market.rate_indices import RateIndex, get_rate_index, get_rfr_index


class ProductTypeConvention(Enum):
    """Product types with convention support"""
    INTEREST_RATE_SWAP = "IRS"
    OVERNIGHT_INDEX_SWAP = "OIS"
    CROSS_CURRENCY_SWAP = "CCS"
    BASIS_SWAP = "BASIS"
    FORWARD_RATE_AGREEMENT = "FRA"
    SWAPTION = "SWAPTION"
    CAP = "CAP"
    FLOOR = "FLOOR"
    FX_FORWARD = "FX_FWD"
    FX_OPTION = "FX_OPT"
    EQUITY_OPTION = "EQ_OPT"
    CDS = "CDS"


@dataclass
class LegConvention:
    """
    Convention specification for a single leg (fixed or floating)

    Attributes:
        frequency: Payment frequency (e.g., SEMI_ANNUAL, QUARTERLY)
        day_count: Day count convention (e.g., THIRTY_360, ACT_360)
        business_day_convention: Business day adjustment rule
        payment_lag: Days between period end and payment (default 0)
        rate_index: Reference rate index for floating legs (optional)
        spread: Default spread over index (default 0.0)
        compounding: Compounding method for RFR legs (optional)
    """
    frequency: Frequency
    day_count: DayCountConvention
    business_day_convention: BusinessDayConvention
    payment_lag: int = 0
    rate_index: Optional[RateIndex] = None
    spread: float = 0.0
    compounding: Optional[str] = None  # "COMPOUND", "AVERAGING", "FLAT"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        result = {
            "frequency": self.frequency.value if isinstance(self.frequency, Enum) else self.frequency,
            "day_count": str(self.day_count),
            "business_day_convention": str(self.business_day_convention),
            "payment_lag": self.payment_lag,
            "spread": self.spread,
        }
        if self.rate_index:
            result["rate_index"] = self.rate_index.name
        if self.compounding:
            result["compounding"] = self.compounding
        return result


@dataclass
class ConventionProfile:
    """
    Complete convention profile for a specific currency and product type

    Attributes:
        currency: Currency code (e.g., "USD", "EUR")
        product_type: Product type (e.g., IRS, OIS)
        fixed_leg: Convention for fixed leg (if applicable)
        floating_leg: Convention for floating leg (if applicable)
        calendars: List of holiday calendar names
        spot_lag: Days from trade date to effective date
        end_of_month: Apply end-of-month rule for date generation
        date_generation: Date generation method ("BACKWARD" or "FORWARD")
        description: Human-readable description
    """
    currency: str
    product_type: ProductTypeConvention
    fixed_leg: Optional[LegConvention] = None
    floating_leg: Optional[LegConvention] = None
    calendars: List[str] = field(default_factory=list)
    spot_lag: int = 2
    end_of_month: bool = False
    date_generation: str = "BACKWARD"
    description: str = ""

    def get_profile_id(self) -> str:
        """Generate unique profile identifier"""
        return f"{self.currency}_{self.product_type.value}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        result = {
            "currency": self.currency,
            "product_type": self.product_type.value,
            "calendars": self.calendars,
            "spot_lag": self.spot_lag,
            "end_of_month": self.end_of_month,
            "date_generation": self.date_generation,
            "description": self.description,
        }
        if self.fixed_leg:
            result["fixed_leg"] = self.fixed_leg.to_dict()
        if self.floating_leg:
            result["floating_leg"] = self.floating_leg.to_dict()
        return result


class ConventionProfileRegistry:
    """
    Registry of all market convention profiles

    This class maintains a comprehensive registry of market-standard conventions
    for all supported currencies and product types. It provides lookup methods
    and validation capabilities.
    """

    def __init__(self):
        self._profiles: Dict[str, ConventionProfile] = {}
        self._initialize_standard_profiles()

    def _initialize_standard_profiles(self):
        """Initialize all standard market convention profiles"""
        # USD Interest Rate Swaps
        self.register_profile(ConventionProfile(
            currency="USD",
            product_type=ProductTypeConvention.INTEREST_RATE_SWAP,
            fixed_leg=LegConvention(
                frequency=Frequency.SEMI_ANNUAL,
                day_count=THIRTY_360,
                business_day_convention=MODIFIED_FOLLOWING,
            ),
            floating_leg=LegConvention(
                frequency=Frequency.QUARTERLY,
                day_count=ACT_360,
                business_day_convention=MODIFIED_FOLLOWING,
                rate_index=get_rfr_index("USD"),  # SOFR
                compounding="COMPOUND",
            ),
            calendars=["USD"],
            spot_lag=2,
            description="USD Standard Interest Rate Swap (Fixed vs SOFR)",
        ))

        # USD Overnight Index Swaps
        self.register_profile(ConventionProfile(
            currency="USD",
            product_type=ProductTypeConvention.OVERNIGHT_INDEX_SWAP,
            fixed_leg=LegConvention(
                frequency=Frequency.ANNUAL,
                day_count=ACT_360,
                business_day_convention=MODIFIED_FOLLOWING,
            ),
            floating_leg=LegConvention(
                frequency=Frequency.ANNUAL,
                day_count=ACT_360,
                business_day_convention=MODIFIED_FOLLOWING,
                rate_index=get_rfr_index("USD"),  # SOFR
                compounding="COMPOUND",
            ),
            calendars=["USD"],
            spot_lag=2,
            description="USD Overnight Index Swap (SOFR)",
        ))

        # EUR Interest Rate Swaps
        self.register_profile(ConventionProfile(
            currency="EUR",
            product_type=ProductTypeConvention.INTEREST_RATE_SWAP,
            fixed_leg=LegConvention(
                frequency=Frequency.ANNUAL,
                day_count=THIRTY_360,
                business_day_convention=MODIFIED_FOLLOWING,
            ),
            floating_leg=LegConvention(
                frequency=Frequency.SEMI_ANNUAL,
                day_count=ACT_360,
                business_day_convention=MODIFIED_FOLLOWING,
                rate_index=get_rate_index("EURIBOR-6M"),
                compounding="FLAT",
            ),
            calendars=["TARGET"],
            spot_lag=2,
            description="EUR Standard Interest Rate Swap (Fixed vs EURIBOR-6M)",
        ))

        # EUR Overnight Index Swaps
        self.register_profile(ConventionProfile(
            currency="EUR",
            product_type=ProductTypeConvention.OVERNIGHT_INDEX_SWAP,
            fixed_leg=LegConvention(
                frequency=Frequency.ANNUAL,
                day_count=ACT_360,
                business_day_convention=MODIFIED_FOLLOWING,
            ),
            floating_leg=LegConvention(
                frequency=Frequency.ANNUAL,
                day_count=ACT_360,
                business_day_convention=MODIFIED_FOLLOWING,
                rate_index=get_rfr_index("EUR"),  # ESTR
                compounding="COMPOUND",
            ),
            calendars=["TARGET"],
            spot_lag=2,
            description="EUR Overnight Index Swap (ESTR)",
        ))

        # GBP Interest Rate Swaps
        self.register_profile(ConventionProfile(
            currency="GBP",
            product_type=ProductTypeConvention.INTEREST_RATE_SWAP,
            fixed_leg=LegConvention(
                frequency=Frequency.SEMI_ANNUAL,
                day_count=ACT_365,
                business_day_convention=MODIFIED_FOLLOWING,
            ),
            floating_leg=LegConvention(
                frequency=Frequency.SEMI_ANNUAL,
                day_count=ACT_365,
                business_day_convention=MODIFIED_FOLLOWING,
                rate_index=get_rfr_index("GBP"),  # SONIA
                compounding="COMPOUND",
            ),
            calendars=["GBP"],
            spot_lag=0,  # GBP typically T+0
            description="GBP Standard Interest Rate Swap (Fixed vs SONIA)",
        ))

        # GBP Overnight Index Swaps
        self.register_profile(ConventionProfile(
            currency="GBP",
            product_type=ProductTypeConvention.OVERNIGHT_INDEX_SWAP,
            fixed_leg=LegConvention(
                frequency=Frequency.ANNUAL,
                day_count=ACT_365,
                business_day_convention=MODIFIED_FOLLOWING,
            ),
            floating_leg=LegConvention(
                frequency=Frequency.ANNUAL,
                day_count=ACT_365,
                business_day_convention=MODIFIED_FOLLOWING,
                rate_index=get_rfr_index("GBP"),  # SONIA
                compounding="COMPOUND",
            ),
            calendars=["GBP"],
            spot_lag=0,
            description="GBP Overnight Index Swap (SONIA)",
        ))

        # JPY Interest Rate Swaps
        self.register_profile(ConventionProfile(
            currency="JPY",
            product_type=ProductTypeConvention.INTEREST_RATE_SWAP,
            fixed_leg=LegConvention(
                frequency=Frequency.SEMI_ANNUAL,
                day_count=ACT_365,
                business_day_convention=MODIFIED_FOLLOWING,
            ),
            floating_leg=LegConvention(
                frequency=Frequency.SEMI_ANNUAL,
                day_count=ACT_365,
                business_day_convention=MODIFIED_FOLLOWING,
                rate_index=get_rfr_index("JPY"),  # TONAR
                compounding="COMPOUND",
            ),
            calendars=["JPY"],
            spot_lag=2,
            description="JPY Standard Interest Rate Swap (Fixed vs TONAR)",
        ))

        # JPY Overnight Index Swaps
        self.register_profile(ConventionProfile(
            currency="JPY",
            product_type=ProductTypeConvention.OVERNIGHT_INDEX_SWAP,
            fixed_leg=LegConvention(
                frequency=Frequency.ANNUAL,
                day_count=ACT_365,
                business_day_convention=MODIFIED_FOLLOWING,
            ),
            floating_leg=LegConvention(
                frequency=Frequency.ANNUAL,
                day_count=ACT_365,
                business_day_convention=MODIFIED_FOLLOWING,
                rate_index=get_rfr_index("JPY"),  # TONAR
                compounding="COMPOUND",
            ),
            calendars=["JPY"],
            spot_lag=2,
            description="JPY Overnight Index Swap (TONAR)",
        ))

        # CHF Interest Rate Swaps
        self.register_profile(ConventionProfile(
            currency="CHF",
            product_type=ProductTypeConvention.INTEREST_RATE_SWAP,
            fixed_leg=LegConvention(
                frequency=Frequency.ANNUAL,
                day_count=THIRTY_360,
                business_day_convention=MODIFIED_FOLLOWING,
            ),
            floating_leg=LegConvention(
                frequency=Frequency.SEMI_ANNUAL,
                day_count=ACT_360,
                business_day_convention=MODIFIED_FOLLOWING,
                rate_index=get_rfr_index("CHF"),  # SARON
                compounding="COMPOUND",
            ),
            calendars=["CHF"],
            spot_lag=2,
            description="CHF Standard Interest Rate Swap (Fixed vs SARON)",
        ))

        # CHF Overnight Index Swaps
        self.register_profile(ConventionProfile(
            currency="CHF",
            product_type=ProductTypeConvention.OVERNIGHT_INDEX_SWAP,
            fixed_leg=LegConvention(
                frequency=Frequency.ANNUAL,
                day_count=ACT_360,
                business_day_convention=MODIFIED_FOLLOWING,
            ),
            floating_leg=LegConvention(
                frequency=Frequency.ANNUAL,
                day_count=ACT_360,
                business_day_convention=MODIFIED_FOLLOWING,
                rate_index=get_rfr_index("CHF"),  # SARON
                compounding="COMPOUND",
            ),
            calendars=["CHF"],
            spot_lag=2,
            description="CHF Overnight Index Swap (SARON)",
        ))

        # Add FRA conventions for major currencies
        for currency in ["USD", "EUR", "GBP", "JPY", "CHF"]:
            calendar = "TARGET" if currency == "EUR" else currency
            day_count = ACT_360 if currency in ["USD", "EUR", "CHF"] else ACT_365

            self.register_profile(ConventionProfile(
                currency=currency,
                product_type=ProductTypeConvention.FORWARD_RATE_AGREEMENT,
                floating_leg=LegConvention(
                    frequency=Frequency.ZERO,  # Single period
                    day_count=day_count,
                    business_day_convention=MODIFIED_FOLLOWING,
                    rate_index=get_rfr_index(currency),
                ),
                calendars=[calendar],
                spot_lag=2 if currency != "GBP" else 0,
                description=f"{currency} Forward Rate Agreement",
            ))

        # USD Basis Swap (3M SOFR vs 1M SOFR typical)
        self.register_profile(ConventionProfile(
            currency="USD",
            product_type=ProductTypeConvention.BASIS_SWAP,
            fixed_leg=LegConvention(  # First floating leg (3M)
                frequency=Frequency.QUARTERLY,
                day_count=ACT_360,
                business_day_convention=MODIFIED_FOLLOWING,
                rate_index=get_rfr_index("USD"),  # SOFR
                compounding="COMPOUND",
            ),
            floating_leg=LegConvention(  # Second floating leg (1M) with spread
                frequency=Frequency.MONTHLY,
                day_count=ACT_360,
                business_day_convention=MODIFIED_FOLLOWING,
                rate_index=get_rfr_index("USD"),  # SOFR
                compounding="COMPOUND",
            ),
            calendars=["USD"],
            spot_lag=2,
            description="USD Basis Swap (3M SOFR vs 1M SOFR)",
        ))

        # EUR Basis Swap (3M EURIBOR vs 6M EURIBOR typical)
        self.register_profile(ConventionProfile(
            currency="EUR",
            product_type=ProductTypeConvention.BASIS_SWAP,
            fixed_leg=LegConvention(  # First floating leg (3M)
                frequency=Frequency.QUARTERLY,
                day_count=ACT_360,
                business_day_convention=MODIFIED_FOLLOWING,
                rate_index=get_rate_index("EURIBOR-3M"),
            ),
            floating_leg=LegConvention(  # Second floating leg (6M) with spread
                frequency=Frequency.SEMI_ANNUAL,
                day_count=ACT_360,
                business_day_convention=MODIFIED_FOLLOWING,
                rate_index=get_rate_index("EURIBOR-6M"),
            ),
            calendars=["TARGET"],
            spot_lag=2,
            description="EUR Basis Swap (3M EURIBOR vs 6M EURIBOR)",
        ))

        # GBP Basis Swap (3M SONIA vs 6M SONIA typical)
        self.register_profile(ConventionProfile(
            currency="GBP",
            product_type=ProductTypeConvention.BASIS_SWAP,
            fixed_leg=LegConvention(  # First floating leg (3M)
                frequency=Frequency.QUARTERLY,
                day_count=ACT_365,
                business_day_convention=MODIFIED_FOLLOWING,
                rate_index=get_rfr_index("GBP"),  # SONIA
                compounding="COMPOUND",
            ),
            floating_leg=LegConvention(  # Second floating leg (6M) with spread
                frequency=Frequency.SEMI_ANNUAL,
                day_count=ACT_365,
                business_day_convention=MODIFIED_FOLLOWING,
                rate_index=get_rfr_index("GBP"),  # SONIA
                compounding="COMPOUND",
            ),
            calendars=["GBP"],
            spot_lag=0,
            description="GBP Basis Swap (3M SONIA vs 6M SONIA)",
        ))

        # USD/EUR Cross-Currency Swap
        self.register_profile(ConventionProfile(
            currency="USDEUR",  # Currency pair notation
            product_type=ProductTypeConvention.CROSS_CURRENCY_SWAP,
            fixed_leg=LegConvention(  # USD leg (domestic)
                frequency=Frequency.SEMI_ANNUAL,
                day_count=THIRTY_360,
                business_day_convention=MODIFIED_FOLLOWING,
            ),
            floating_leg=LegConvention(  # EUR leg (foreign)
                frequency=Frequency.ANNUAL,
                day_count=THIRTY_360,
                business_day_convention=MODIFIED_FOLLOWING,
            ),
            calendars=["USD", "TARGET"],
            spot_lag=2,
            description="USD/EUR Cross-Currency Swap",
        ))

        # USD/JPY Cross-Currency Swap
        self.register_profile(ConventionProfile(
            currency="USDJPY",  # Currency pair notation
            product_type=ProductTypeConvention.CROSS_CURRENCY_SWAP,
            fixed_leg=LegConvention(  # USD leg (domestic)
                frequency=Frequency.SEMI_ANNUAL,
                day_count=THIRTY_360,
                business_day_convention=MODIFIED_FOLLOWING,
            ),
            floating_leg=LegConvention(  # JPY leg (foreign)
                frequency=Frequency.SEMI_ANNUAL,
                day_count=ACT_365,
                business_day_convention=MODIFIED_FOLLOWING,
            ),
            calendars=["USD", "JPY"],
            spot_lag=2,
            description="USD/JPY Cross-Currency Swap",
        ))

        # EUR/GBP Cross-Currency Swap
        self.register_profile(ConventionProfile(
            currency="EURGBP",  # Currency pair notation
            product_type=ProductTypeConvention.CROSS_CURRENCY_SWAP,
            fixed_leg=LegConvention(  # EUR leg (domestic)
                frequency=Frequency.ANNUAL,
                day_count=THIRTY_360,
                business_day_convention=MODIFIED_FOLLOWING,
            ),
            floating_leg=LegConvention(  # GBP leg (foreign)
                frequency=Frequency.SEMI_ANNUAL,
                day_count=ACT_365,
                business_day_convention=MODIFIED_FOLLOWING,
            ),
            calendars=["TARGET", "GBP"],
            spot_lag=2,
            description="EUR/GBP Cross-Currency Swap",
        ))

        # Add Cap/Floor conventions for major currencies
        for currency in ["USD", "EUR", "GBP", "JPY", "CHF"]:
            calendar = "TARGET" if currency == "EUR" else currency
            day_count = ACT_360 if currency in ["USD", "EUR", "CHF"] else ACT_365

            # Cap conventions
            self.register_profile(ConventionProfile(
                currency=currency,
                product_type=ProductTypeConvention.CAP,
                floating_leg=LegConvention(
                    frequency=Frequency.QUARTERLY,  # Standard quarterly resets
                    day_count=day_count,
                    business_day_convention=MODIFIED_FOLLOWING,
                    rate_index=get_rfr_index(currency),
                ),
                calendars=[calendar],
                spot_lag=2 if currency != "GBP" else 0,
                description=f"{currency} Interest Rate Cap",
            ))

            # Floor conventions
            self.register_profile(ConventionProfile(
                currency=currency,
                product_type=ProductTypeConvention.FLOOR,
                floating_leg=LegConvention(
                    frequency=Frequency.QUARTERLY,  # Standard quarterly resets
                    day_count=day_count,
                    business_day_convention=MODIFIED_FOLLOWING,
                    rate_index=get_rfr_index(currency),
                ),
                calendars=[calendar],
                spot_lag=2 if currency != "GBP" else 0,
                description=f"{currency} Interest Rate Floor",
            ))

        # Add Swaption conventions for major currencies
        for currency in ["USD", "EUR", "GBP", "JPY", "CHF"]:
            calendar = "TARGET" if currency == "EUR" else currency
            day_count = ACT_360 if currency in ["USD", "EUR", "CHF"] else ACT_365

            self.register_profile(ConventionProfile(
                currency=currency,
                product_type=ProductTypeConvention.SWAPTION,
                fixed_leg=LegConvention(  # Underlying swap fixed leg
                    frequency=Frequency.SEMI_ANNUAL if currency != "EUR" else Frequency.ANNUAL,
                    day_count=THIRTY_360,
                    business_day_convention=MODIFIED_FOLLOWING,
                ),
                floating_leg=LegConvention(  # Underlying swap floating leg
                    frequency=Frequency.QUARTERLY if currency != "EUR" else Frequency.SEMI_ANNUAL,
                    day_count=day_count,
                    business_day_convention=MODIFIED_FOLLOWING,
                    rate_index=get_rfr_index(currency),
                ),
                calendars=[calendar],
                spot_lag=2 if currency != "GBP" else 0,
                description=f"{currency} Swaption",
            ))

    def register_profile(self, profile: ConventionProfile):
        """Register a new convention profile"""
        profile_id = profile.get_profile_id()
        self._profiles[profile_id] = profile

    def get_profile(
        self,
        currency: str,
        product_type: ProductTypeConvention
    ) -> Optional[ConventionProfile]:
        """
        Retrieve a convention profile

        Args:
            currency: Currency code (e.g., "USD")
            product_type: Product type enum

        Returns:
            ConventionProfile if found, None otherwise
        """
        profile_id = f"{currency}_{product_type.value}"
        return self._profiles.get(profile_id)

    def has_profile(
        self,
        currency: str,
        product_type: ProductTypeConvention
    ) -> bool:
        """Check if a profile exists"""
        profile_id = f"{currency}_{product_type.value}"
        return profile_id in self._profiles

    def list_currencies(self) -> List[str]:
        """List all currencies with registered profiles"""
        currencies = set()
        for profile in self._profiles.values():
            currencies.add(profile.currency)
        return sorted(list(currencies))

    def list_product_types(self, currency: Optional[str] = None) -> List[ProductTypeConvention]:
        """
        List all product types with registered profiles

        Args:
            currency: Optional currency filter

        Returns:
            List of product types
        """
        product_types = set()
        for profile in self._profiles.values():
            if currency is None or profile.currency == currency:
                product_types.add(profile.product_type)
        return sorted(list(product_types), key=lambda x: x.value)

    def get_all_profiles(self) -> Dict[str, ConventionProfile]:
        """Get all registered profiles"""
        return self._profiles.copy()


# Global singleton registry
_GLOBAL_REGISTRY: Optional[ConventionProfileRegistry] = None


def get_convention_registry() -> ConventionProfileRegistry:
    """Get the global convention profile registry (singleton)"""
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        _GLOBAL_REGISTRY = ConventionProfileRegistry()
    return _GLOBAL_REGISTRY


def get_convention_profile(
    currency: str,
    product_type: ProductTypeConvention
) -> Optional[ConventionProfile]:
    """
    Convenience function to get a convention profile from the global registry

    Args:
        currency: Currency code (e.g., "USD")
        product_type: Product type enum

    Returns:
        ConventionProfile if found, None otherwise

    Example:
        >>> profile = get_convention_profile("USD", ProductTypeConvention.INTEREST_RATE_SWAP)
        >>> print(profile.fixed_leg.frequency)
        Frequency.SEMI_ANNUAL
    """
    registry = get_convention_registry()
    return registry.get_profile(currency, product_type)
