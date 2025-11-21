"""
Forward Rate Agreement (FRA) Generator

High-level API for generating Forward Rate Agreement trades using market conventions.
FRAs are forward contracts on interest rates, typically cash-settled at the
beginning of the interest period.

Common FRA notation:
- "3x6" = 3 months to settlement, 3-month interest period (total 6 months)
- "6x12" = 6 months to settlement, 6-month interest period (total 12 months)
"""

from datetime import date, timedelta
from typing import Optional, Tuple

import jax.numpy as jnp

from neutryx.portfolio.contracts.trade import Trade
from neutryx.portfolio.trade_generation.factory import (
    TradeFactory,
    TradeGenerationRequest,
    TradeGenerationResult,
)
from neutryx.market.convention_profiles import ProductTypeConvention
from neutryx.products.linear_rates.fra import (
    ForwardRateAgreement,
    SettlementType,
)
from neutryx.core.dates.schedule import Frequency
from neutryx.core.dates.day_count import (
    DayCountConvention,
    Actual360,
    Actual365Fixed,
)


class FRAGenerator:
    """
    Generator for Forward Rate Agreement trades

    This class provides a high-level interface for creating FRA trades
    that conform to market conventions. It wraps the TradeFactory and
    provides FRA-specific convenience methods.
    """

    def __init__(self, factory: Optional[TradeFactory] = None):
        """
        Initialize FRA generator

        Args:
            factory: Optional TradeFactory instance (creates default if not provided)
        """
        self.factory = factory or TradeFactory()

    def _parse_fra_tenor(self, fra_tenor: str) -> Tuple[int, int]:
        """
        Parse FRA tenor notation (e.g., "3x6", "6x12")

        Args:
            fra_tenor: FRA tenor string (e.g., "3x6", "3x9")

        Returns:
            Tuple of (settlement_months, maturity_months)

        Example:
            "3x6" -> (3, 6) = 3 months to settlement, total 6 months
        """
        if 'x' not in fra_tenor.lower():
            raise ValueError(
                f"Invalid FRA tenor format: '{fra_tenor}'. "
                "Expected format like '3x6' or '6x12'"
            )

        parts = fra_tenor.lower().split('x')
        if len(parts) != 2:
            raise ValueError(f"Invalid FRA tenor format: '{fra_tenor}'")

        try:
            settlement_months = int(parts[0])
            maturity_months = int(parts[1])
        except ValueError:
            raise ValueError(f"Invalid FRA tenor format: '{fra_tenor}'. Expected integers.")

        if maturity_months <= settlement_months:
            raise ValueError(
                f"Invalid FRA tenor: maturity ({maturity_months}M) must be "
                f"greater than settlement ({settlement_months}M)"
            )

        return settlement_months, maturity_months

    def _calculate_fra_dates(
        self,
        trade_date: date,
        settlement_months: int,
        period_months: int,
        spot_lag: int,
    ) -> Tuple[date, date, date]:
        """
        Calculate FRA dates

        Args:
            trade_date: Trade execution date
            settlement_months: Months from effective date to settlement
            period_months: Length of interest period in months
            spot_lag: Spot lag in business days

        Returns:
            Tuple of (effective_date, settlement_date, maturity_date)
        """
        # Effective date (T+spot_lag)
        effective_date = trade_date + timedelta(days=spot_lag)

        # Settlement date (effective + settlement_months)
        settlement_date = effective_date + timedelta(days=settlement_months * 30)

        # Maturity date (settlement + period_months)
        maturity_date = settlement_date + timedelta(days=period_months * 30)

        return effective_date, settlement_date, maturity_date

    def _create_fra_product(
        self,
        notional: float,
        fixed_rate: float,
        settlement_years: float,
        period_years: float,
        is_payer: bool,
    ) -> ForwardRateAgreement:
        """Create ForwardRateAgreement product instance"""
        return ForwardRateAgreement(
            T=settlement_years,
            notional=notional,
            fixed_rate=fixed_rate,
            period_length=period_years,
            settlement_type=SettlementType.ADVANCE,  # Market standard
            is_payer=is_payer,
        )

    def generate(
        self,
        currency: str,
        trade_date: date,
        fra_tenor: str,  # e.g., "3x6", "6x12"
        notional: float,
        fixed_rate: float,
        counterparty_id: str,
        is_payer: bool = True,  # True = pay fixed (long FRA)
        # Trade metadata
        trade_number: Optional[str] = None,
        book_id: Optional[str] = None,
        desk_id: Optional[str] = None,
        trader_id: Optional[str] = None,
    ) -> TradeGenerationResult:
        """
        Generate a Forward Rate Agreement trade

        Args:
            currency: Currency code (e.g., "USD", "EUR")
            trade_date: Trade execution date
            fra_tenor: FRA tenor notation (e.g., "3x6" = 3mo to settlement, 3mo period)
            notional: Notional amount
            fixed_rate: Fixed rate (e.g., 0.045 for 4.5%)
            counterparty_id: Counterparty identifier
            is_payer: True if paying fixed (long FRA), False if receiving (short FRA)

            # Trade metadata (optional)
            trade_number: External trade number
            book_id: Book identifier
            desk_id: Desk identifier
            trader_id: Trader identifier

        Returns:
            TradeGenerationResult containing Trade, Product, and Warnings

        Example:
            >>> generator = FRAGenerator()
            >>> result = generator.generate(
            ...     currency="USD",
            ...     trade_date=date(2024, 1, 15),
            ...     fra_tenor="3x6",  # 3mo to settlement, 3mo period
            ...     notional=10_000_000,
            ...     fixed_rate=0.045,
            ...     counterparty_id="CP-001",
            ...     is_payer=True,
            ... )
            >>> print(result.trade)
            >>> print(result.has_warnings())
        """
        # Parse FRA tenor
        settlement_months, maturity_months = self._parse_fra_tenor(fra_tenor)
        period_months = maturity_months - settlement_months

        # Get convention profile for FRA
        from neutryx.market.convention_profiles import get_convention_profile
        profile = get_convention_profile(currency, ProductTypeConvention.FORWARD_RATE_AGREEMENT)

        if profile is None:
            raise ValueError(f"No FRA convention profile found for {currency}")

        # Calculate dates
        effective_date, settlement_date, maturity_date = self._calculate_fra_dates(
            trade_date,
            settlement_months,
            period_months,
            profile.spot_lag,
        )

        # Convert to years
        settlement_years = settlement_months / 12.0
        period_years = period_months / 12.0

        # Create FRA product
        fra_product = self._create_fra_product(
            notional=notional,
            fixed_rate=fixed_rate,
            settlement_years=settlement_years,
            period_years=period_years,
            is_payer=is_payer,
        )

        # Create Trade using simplified approach (FRA doesn't need full factory)
        from neutryx.portfolio.contracts.trade import (
            Trade,
            ProductType,
            TradeStatus,
            SettlementType as TradeSettlementType,
        )
        from uuid import uuid4

        trade = Trade(
            id=str(uuid4()),
            trade_number=trade_number,
            counterparty_id=counterparty_id,
            book_id=book_id,
            desk_id=desk_id,
            trader_id=trader_id,
            product_type=ProductType.FORWARD,
            trade_date=trade_date,
            effective_date=effective_date,
            maturity_date=maturity_date,
            status=TradeStatus.ACTIVE,
            notional=notional,
            currency=currency,
            settlement_type=TradeSettlementType.CASH,
            product_details={
                "fra_tenor": fra_tenor,
                "settlement_date": settlement_date.isoformat(),
                "fixed_rate": fixed_rate,
                "is_payer": is_payer,
                "settlement_months": settlement_months,
                "period_months": period_months,
                "fra_product": {
                    "T": fra_product.T,
                    "notional": fra_product.notional,
                    "fixed_rate": fra_product.fixed_rate,
                    "period_length": fra_product.period_length,
                    "is_payer": fra_product.is_payer,
                    "settlement_type": fra_product.settlement_type.value,
                },
            },
            convention_profile_id=profile.get_profile_id(),
            generated_from_convention=True,
        )

        # Create validation result (FRA conventions are simpler)
        from neutryx.portfolio.trade_generation.validation import ValidationResult
        validation_result = ValidationResult(convention_profile=profile)

        # Create result
        result = TradeGenerationResult(
            trade=trade,
            product=fra_product,
            schedules=None,  # FRAs don't have payment schedules
            validation_result=validation_result,
            convention_profile=profile,
        )

        return result


def generate_fra_trade(
    currency: str,
    trade_date: date,
    fra_tenor: str,
    notional: float,
    fixed_rate: float,
    counterparty_id: str,
    is_payer: bool = True,
    **kwargs
) -> Tuple[Trade, ForwardRateAgreement, TradeGenerationResult]:
    """
    Convenience function to generate a FRA trade

    This is a simplified interface that returns the Trade, Product, and full result.

    Args:
        currency: Currency code
        trade_date: Trade execution date
        fra_tenor: FRA tenor (e.g., "3x6")
        notional: Notional amount
        fixed_rate: Fixed rate
        counterparty_id: Counterparty ID
        is_payer: True if paying fixed (long FRA)
        **kwargs: Additional parameters (see FRAGenerator.generate)

    Returns:
        Tuple of (Trade, ForwardRateAgreement, TradeGenerationResult)

    Example:
        >>> trade, product, result = generate_fra_trade(
        ...     "USD",
        ...     date(2024, 1, 15),
        ...     "3x6",
        ...     10_000_000,
        ...     0.045,
        ...     "CP-001",
        ... )
        >>> print(f"Trade ID: {trade.id}")
        >>> print(f"Settlement in {product.T:.2f} years")
        >>> print(f"Interest period: {product.period_length:.2f} years")
    """
    generator = FRAGenerator()
    result = generator.generate(
        currency=currency,
        trade_date=trade_date,
        fra_tenor=fra_tenor,
        notional=notional,
        fixed_rate=fixed_rate,
        counterparty_id=counterparty_id,
        is_payer=is_payer,
        **kwargs
    )
    return result.trade, result.product, result
