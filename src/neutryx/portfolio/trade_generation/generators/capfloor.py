"""
Interest Rate Cap and Floor Generator

High-level API for generating Interest Rate Cap and Floor trades using market conventions.
- Cap: Portfolio of caplets protecting against rising interest rates
- Floor: Portfolio of floorlets protecting against falling interest rates
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
from neutryx.products.linear_rates.caps_floors import (
    InterestRateCapFloorCollar,
    CapFloorType,
)
from neutryx.core.dates.schedule import Frequency


class CapFloorGenerator:
    """
    Generator for Interest Rate Cap and Floor trades

    This class provides a high-level interface for creating cap and floor
    trades that conform to market conventions.
    """

    def __init__(self, factory: Optional[TradeFactory] = None):
        """
        Initialize Cap/Floor generator

        Args:
            factory: Optional TradeFactory instance (creates default if not provided)
        """
        self.factory = factory or TradeFactory()

    def _frequency_to_payments_per_year(self, frequency: Frequency) -> int:
        """
        Convert Frequency to payment frequency (payments per year)

        Args:
            frequency: Frequency enum value

        Returns:
            Payment frequency as integer (payments per year)
        """
        mapping = {
            Frequency.ANNUAL: 1,
            Frequency.SEMI_ANNUAL: 2,
            Frequency.QUARTERLY: 4,
            Frequency.MONTHLY: 12,
        }

        return mapping.get(frequency, 4)  # Default to quarterly

    def _create_capfloor_product(
        self,
        notional: float,
        strike: float,
        maturity_years: float,
        cap_floor_type: CapFloorType,
        payment_frequency: int,
        volatility: float,
    ) -> InterestRateCapFloorCollar:
        """Create InterestRateCapFloorCollar product instance"""
        return InterestRateCapFloorCollar(
            T=maturity_years,
            notional=notional,
            strike=strike,
            cap_floor_type=cap_floor_type,
            payment_frequency=payment_frequency,
            volatility=volatility,
        )

    def generate(
        self,
        currency: str,
        trade_date: date,
        maturity_years: float,
        notional: float,
        strike: float,
        counterparty_id: str,
        is_cap: bool = True,  # True for Cap, False for Floor
        volatility: float = 0.20,  # Default 20% vol
        # Trade metadata
        trade_number: Optional[str] = None,
        book_id: Optional[str] = None,
        desk_id: Optional[str] = None,
        trader_id: Optional[str] = None,
    ) -> TradeGenerationResult:
        """
        Generate an Interest Rate Cap or Floor trade

        Args:
            currency: Currency code (e.g., "USD", "EUR")
            trade_date: Trade execution date
            maturity_years: Maturity in years
            notional: Notional amount
            strike: Strike rate (cap rate or floor rate, e.g., 0.05 for 5%)
            counterparty_id: Counterparty identifier
            is_cap: True for Cap (protection against rising rates), False for Floor
            volatility: Black volatility (e.g., 0.20 for 20%)

            # Trade metadata (optional)
            trade_number: External trade number
            book_id: Book identifier
            desk_id: Desk identifier
            trader_id: Trader identifier

        Returns:
            TradeGenerationResult containing Trade, Product, and Warnings

        Example:
            >>> generator = CapFloorGenerator()
            >>> result = generator.generate(
            ...     currency="USD",
            ...     trade_date=date(2024, 1, 15),
            ...     maturity_years=5.0,
            ...     notional=100_000_000,
            ...     strike=0.05,  # 5% cap rate
            ...     counterparty_id="CP-001",
            ...     is_cap=True,
            ...     volatility=0.20,  # 20% vol
            ... )
            >>> print(result.trade)
        """
        # Determine product type
        product_type_conv = ProductTypeConvention.CAP if is_cap else ProductTypeConvention.FLOOR
        cap_floor_type = CapFloorType.CAP if is_cap else CapFloorType.FLOOR

        # Get convention profile
        from neutryx.market.convention_profiles import get_convention_profile
        profile = get_convention_profile(currency, product_type_conv)

        if profile is None:
            product_name = "Cap" if is_cap else "Floor"
            raise ValueError(f"No {product_name} convention profile found for {currency}")

        # Get payment frequency from convention
        payment_frequency = self._frequency_to_payments_per_year(
            profile.floating_leg.frequency if profile.floating_leg else Frequency.QUARTERLY
        )

        # Calculate dates
        effective_date = trade_date + timedelta(days=profile.spot_lag)
        maturity_date = effective_date + timedelta(days=int(maturity_years * 365))

        # Create Cap/Floor product
        capfloor_product = self._create_capfloor_product(
            notional=notional,
            strike=strike,
            maturity_years=maturity_years,
            cap_floor_type=cap_floor_type,
            payment_frequency=payment_frequency,
            volatility=volatility,
        )

        # Create Trade
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
            product_type=ProductType.SWAPTION,  # Caps/Floors are option-like
            trade_date=trade_date,
            effective_date=effective_date,
            maturity_date=maturity_date,
            status=TradeStatus.ACTIVE,
            notional=notional,
            currency=currency,
            settlement_type=TradeSettlementType.CASH,
            product_details={
                "product_subtype": "CAP" if is_cap else "FLOOR",
                "strike": strike,
                "volatility": volatility,
                "payment_frequency": payment_frequency,
                "maturity_years": maturity_years,
                "capfloor_product": {
                    "T": capfloor_product.T,
                    "notional": capfloor_product.notional,
                    "strike": capfloor_product.strike,
                    "cap_floor_type": capfloor_product.cap_floor_type.value,
                    "payment_frequency": capfloor_product.payment_frequency,
                    "volatility": capfloor_product.volatility,
                },
            },
            convention_profile_id=profile.get_profile_id(),
            generated_from_convention=True,
        )

        # Create validation result
        from neutryx.portfolio.trade_generation.validation import ValidationResult
        validation_result = ValidationResult(convention_profile=profile)

        # Create result
        result = TradeGenerationResult(
            trade=trade,
            product=capfloor_product,
            schedules=None,  # Could add schedule generation in future
            validation_result=validation_result,
            convention_profile=profile,
        )

        return result

    def generate_cap(
        self,
        currency: str,
        trade_date: date,
        maturity_years: float,
        notional: float,
        strike: float,
        counterparty_id: str,
        volatility: float = 0.20,
        **kwargs
    ) -> TradeGenerationResult:
        """
        Convenience method to generate a Cap (protection against rising rates)

        Args:
            currency: Currency code
            trade_date: Trade execution date
            maturity_years: Maturity in years
            notional: Notional amount
            strike: Cap strike rate
            counterparty_id: Counterparty ID
            volatility: Black volatility
            **kwargs: Additional parameters

        Returns:
            TradeGenerationResult
        """
        return self.generate(
            currency=currency,
            trade_date=trade_date,
            maturity_years=maturity_years,
            notional=notional,
            strike=strike,
            counterparty_id=counterparty_id,
            is_cap=True,
            volatility=volatility,
            **kwargs
        )

    def generate_floor(
        self,
        currency: str,
        trade_date: date,
        maturity_years: float,
        notional: float,
        strike: float,
        counterparty_id: str,
        volatility: float = 0.20,
        **kwargs
    ) -> TradeGenerationResult:
        """
        Convenience method to generate a Floor (protection against falling rates)

        Args:
            currency: Currency code
            trade_date: Trade execution date
            maturity_years: Maturity in years
            notional: Notional amount
            strike: Floor strike rate
            counterparty_id: Counterparty ID
            volatility: Black volatility
            **kwargs: Additional parameters

        Returns:
            TradeGenerationResult
        """
        return self.generate(
            currency=currency,
            trade_date=trade_date,
            maturity_years=maturity_years,
            notional=notional,
            strike=strike,
            counterparty_id=counterparty_id,
            is_cap=False,
            volatility=volatility,
            **kwargs
        )


def generate_cap_trade(
    currency: str,
    trade_date: date,
    maturity_years: float,
    notional: float,
    strike: float,
    counterparty_id: str,
    volatility: float = 0.20,
    **kwargs
) -> Tuple[Trade, InterestRateCapFloorCollar, TradeGenerationResult]:
    """
    Convenience function to generate a Cap trade

    Args:
        currency: Currency code
        trade_date: Trade execution date
        maturity_years: Maturity in years
        notional: Notional amount
        strike: Cap strike rate
        counterparty_id: Counterparty ID
        volatility: Black volatility
        **kwargs: Additional parameters

    Returns:
        Tuple of (Trade, InterestRateCapFloorCollar, TradeGenerationResult)

    Example:
        >>> trade, product, result = generate_cap_trade(
        ...     "USD",
        ...     date(2024, 1, 15),
        ...     5.0,  # 5 years
        ...     100_000_000,
        ...     0.05,  # 5% cap
        ...     "CP-001",
        ...     volatility=0.20,
        ... )
        >>> print(f"Cap strike: {product.strike * 100:.2f}%")
    """
    generator = CapFloorGenerator()
    result = generator.generate_cap(
        currency=currency,
        trade_date=trade_date,
        maturity_years=maturity_years,
        notional=notional,
        strike=strike,
        counterparty_id=counterparty_id,
        volatility=volatility,
        **kwargs
    )
    return result.trade, result.product, result


def generate_floor_trade(
    currency: str,
    trade_date: date,
    maturity_years: float,
    notional: float,
    strike: float,
    counterparty_id: str,
    volatility: float = 0.20,
    **kwargs
) -> Tuple[Trade, InterestRateCapFloorCollar, TradeGenerationResult]:
    """
    Convenience function to generate a Floor trade

    Args:
        currency: Currency code
        trade_date: Trade execution date
        maturity_years: Maturity in years
        notional: Notional amount
        strike: Floor strike rate
        counterparty_id: Counterparty ID
        volatility: Black volatility
        **kwargs: Additional parameters

    Returns:
        Tuple of (Trade, InterestRateCapFloorCollar, TradeGenerationResult)

    Example:
        >>> trade, product, result = generate_floor_trade(
        ...     "USD",
        ...     date(2024, 1, 15),
        ...     5.0,  # 5 years
        ...     100_000_000,
        ...     0.02,  # 2% floor
        ...     "CP-001",
        ...     volatility=0.20,
        ... )
        >>> print(f"Floor strike: {product.strike * 100:.2f}%")
    """
    generator = CapFloorGenerator()
    result = generator.generate_floor(
        currency=currency,
        trade_date=trade_date,
        maturity_years=maturity_years,
        notional=notional,
        strike=strike,
        counterparty_id=counterparty_id,
        volatility=volatility,
        **kwargs
    )
    return result.trade, result.product, result
