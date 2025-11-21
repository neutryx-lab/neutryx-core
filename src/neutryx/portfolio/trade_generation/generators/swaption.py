"""
Swaption Generator

High-level API for generating Swaption trades using market conventions.
A swaption is an option to enter into an interest rate swap:
- Payer swaption: Right to pay fixed, receive floating
- Receiver swaption: Right to receive fixed, pay floating
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
from neutryx.products.swaptions import EuropeanSwaption, SwaptionType
from neutryx.core.dates.schedule import Frequency


class SwaptionGenerator:
    """
    Generator for Swaption trades

    This class provides a high-level interface for creating swaption trades
    that conform to market conventions. Swaptions are options on interest rate swaps.
    """

    def __init__(self, factory: Optional[TradeFactory] = None):
        """
        Initialize Swaption generator

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

        return mapping.get(frequency, 2)  # Default to semi-annual

    def _calculate_annuity(
        self,
        swap_maturity_years: float,
        payment_frequency: int,
        discount_rate: float = 0.03,
    ) -> float:
        """
        Calculate approximate swap annuity

        Args:
            swap_maturity_years: Maturity of underlying swap in years
            payment_frequency: Payments per year
            discount_rate: Discount rate for PV calculation

        Returns:
            Annuity factor (PV of 1 per period)
        """
        n_payments = int(swap_maturity_years * payment_frequency)
        period_length = 1.0 / payment_frequency

        annuity = 0.0
        for i in range(1, n_payments + 1):
            payment_time = i * period_length
            discount_factor = jnp.exp(-discount_rate * payment_time)
            annuity += float(discount_factor * period_length)

        return annuity

    def _create_swaption_product(
        self,
        notional: float,
        strike: float,
        option_maturity_years: float,
        swap_maturity_years: float,
        is_payer: bool,
        annuity: float,
    ) -> EuropeanSwaption:
        """Create EuropeanSwaption product instance"""
        return EuropeanSwaption(
            T=option_maturity_years,
            strike=strike,
            annuity=annuity,
            notional=notional,
            swaption_type=SwaptionType.PAYER if is_payer else SwaptionType.RECEIVER,
        )

    def generate(
        self,
        currency: str,
        trade_date: date,
        option_maturity_years: float,  # Time to swaption expiry
        swap_maturity_years: float,  # Tenor of underlying swap
        notional: float,
        strike: float,  # Fixed rate of underlying swap
        counterparty_id: str,
        is_payer: bool = True,  # True for payer swaption
        volatility: float = 0.20,  # Default 20% vol for pricing
        # Trade metadata
        trade_number: Optional[str] = None,
        book_id: Optional[str] = None,
        desk_id: Optional[str] = None,
        trader_id: Optional[str] = None,
    ) -> TradeGenerationResult:
        """
        Generate a Swaption trade

        Args:
            currency: Currency code (e.g., "USD", "EUR")
            trade_date: Trade execution date
            option_maturity_years: Time to swaption expiry in years
            swap_maturity_years: Tenor of the underlying swap in years
            notional: Notional amount
            strike: Fixed rate of the underlying swap (e.g., 0.05 for 5%)
            counterparty_id: Counterparty identifier
            is_payer: True for payer swaption (pay fixed), False for receiver
            volatility: Black volatility for pricing (e.g., 0.20 for 20%)

            # Trade metadata (optional)
            trade_number: External trade number
            book_id: Book identifier
            desk_id: Desk identifier
            trader_id: Trader identifier

        Returns:
            TradeGenerationResult containing Trade, Product, and Warnings

        Example:
            >>> generator = SwaptionGenerator()
            >>> result = generator.generate(
            ...     currency="USD",
            ...     trade_date=date(2024, 1, 15),
            ...     option_maturity_years=1.0,  # 1Y option
            ...     swap_maturity_years=5.0,    # Into a 5Y swap
            ...     notional=100_000_000,
            ...     strike=0.05,  # 5% swap rate
            ...     counterparty_id="CP-001",
            ...     is_payer=True,
            ... )
            >>> print(result.trade)
        """
        # Get convention profile for Swaption
        from neutryx.market.convention_profiles import get_convention_profile
        profile = get_convention_profile(currency, ProductTypeConvention.SWAPTION)

        if profile is None:
            raise ValueError(f"No Swaption convention profile found for {currency}")

        # Get payment frequency from underlying swap convention
        payment_frequency = self._frequency_to_payments_per_year(
            profile.fixed_leg.frequency if profile.fixed_leg else Frequency.SEMI_ANNUAL
        )

        # Calculate annuity
        annuity = self._calculate_annuity(
            swap_maturity_years,
            payment_frequency,
            discount_rate=strike,  # Use strike as approximation
        )

        # Calculate dates
        effective_date = trade_date + timedelta(days=profile.spot_lag)
        option_expiry_date = effective_date + timedelta(days=int(option_maturity_years * 365))
        swap_maturity_date = option_expiry_date + timedelta(days=int(swap_maturity_years * 365))

        # Create Swaption product
        swaption_product = self._create_swaption_product(
            notional=notional,
            strike=strike,
            option_maturity_years=option_maturity_years,
            swap_maturity_years=swap_maturity_years,
            is_payer=is_payer,
            annuity=annuity,
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
            product_type=ProductType.SWAPTION,
            trade_date=trade_date,
            effective_date=effective_date,
            maturity_date=swap_maturity_date,  # Final maturity of underlying swap
            status=TradeStatus.ACTIVE,
            notional=notional,
            currency=currency,
            settlement_type=TradeSettlementType.CASH,
            product_details={
                "product_subtype": "SWAPTION",
                "swaption_type": "PAYER" if is_payer else "RECEIVER",
                "option_expiry_date": option_expiry_date.isoformat(),
                "option_maturity_years": option_maturity_years,
                "swap_maturity_years": swap_maturity_years,
                "strike": strike,
                "volatility": volatility,
                "payment_frequency": payment_frequency,
                "annuity": annuity,
                "swaption_product": {
                    "T": swaption_product.T,
                    "notional": swaption_product.notional,
                    "strike": swaption_product.strike,
                    "annuity": swaption_product.annuity,
                    "swaption_type": swaption_product.swaption_type.value,
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
            product=swaption_product,
            schedules=None,  # Could add schedule generation in future
            validation_result=validation_result,
            convention_profile=profile,
        )

        return result


def generate_swaption_trade(
    currency: str,
    trade_date: date,
    option_maturity_years: float,
    swap_maturity_years: float,
    notional: float,
    strike: float,
    counterparty_id: str,
    is_payer: bool = True,
    volatility: float = 0.20,
    **kwargs
) -> Tuple[Trade, EuropeanSwaption, TradeGenerationResult]:
    """
    Convenience function to generate a Swaption trade

    Args:
        currency: Currency code
        trade_date: Trade execution date
        option_maturity_years: Time to swaption expiry
        swap_maturity_years: Tenor of underlying swap
        notional: Notional amount
        strike: Fixed rate of underlying swap
        counterparty_id: Counterparty ID
        is_payer: True for payer swaption
        volatility: Black volatility
        **kwargs: Additional parameters

    Returns:
        Tuple of (Trade, EuropeanSwaption, TradeGenerationResult)

    Example:
        >>> trade, product, result = generate_swaption_trade(
        ...     "USD",
        ...     date(2024, 1, 15),
        ...     1.0,  # 1Y option
        ...     5.0,  # Into 5Y swap
        ...     100_000_000,
        ...     0.05,  # 5% strike
        ...     "CP-001",
        ... )
        >>> print(f"Swaption type: {product.swaption_type.value}")
        >>> print(f"Strike: {product.strike * 100:.2f}%")
    """
    generator = SwaptionGenerator()
    result = generator.generate(
        currency=currency,
        trade_date=trade_date,
        option_maturity_years=option_maturity_years,
        swap_maturity_years=swap_maturity_years,
        notional=notional,
        strike=strike,
        counterparty_id=counterparty_id,
        is_payer=is_payer,
        volatility=volatility,
        **kwargs
    )
    return result.trade, result.product, result
