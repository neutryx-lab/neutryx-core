"""
Basis Swap Generator

High-level API for generating Basis Swap trades using market conventions.
Basis swaps exchange floating cash flows based on two different reference rates:
- Tenor basis: Same currency, different tenors (e.g., 3M SOFR vs 1M SOFR)
- The basis spread is applied to the second floating leg
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
from neutryx.products.linear_rates.swaps import BasisSwap, Tenor
from neutryx.core.dates.schedule import Frequency


class BasisSwapGenerator:
    """
    Generator for Basis Swap trades

    This class provides a high-level interface for creating basis swap trades
    that conform to market conventions. Basis swaps exchange floating rate
    payments based on two different reference rates or tenors.
    """

    def __init__(self, factory: Optional[TradeFactory] = None):
        """
        Initialize Basis Swap generator

        Args:
            factory: Optional TradeFactory instance (creates default if not provided)
        """
        self.factory = factory or TradeFactory()

    def _frequency_to_tenor(self, frequency: Frequency) -> Tenor:
        """
        Map Frequency enum to Tenor enum

        Args:
            frequency: Frequency enum value

        Returns:
            Corresponding Tenor enum value
        """
        mapping = {
            Frequency.MONTHLY: Tenor.ONE_MONTH,
            Frequency.QUARTERLY: Tenor.THREE_MONTH,
            Frequency.SEMI_ANNUAL: Tenor.SIX_MONTH,
            Frequency.ANNUAL: Tenor.TWELVE_MONTH,
        }

        if frequency not in mapping:
            raise ValueError(
                f"Frequency {frequency} cannot be mapped to Tenor. "
                f"Valid frequencies: {list(mapping.keys())}"
            )

        return mapping[frequency]

    def _tenor_to_payment_frequency(self, tenor: Tenor) -> int:
        """
        Convert Tenor to payment frequency (payments per year)

        Args:
            tenor: Tenor enum value

        Returns:
            Payment frequency as integer (payments per year)
        """
        mapping = {
            Tenor.ONE_MONTH: 12,
            Tenor.THREE_MONTH: 4,
            Tenor.SIX_MONTH: 2,
            Tenor.TWELVE_MONTH: 1,
        }

        return mapping.get(tenor, 4)  # Default to quarterly

    def _create_basis_swap_product(
        self,
        notional: float,
        maturity_years: float,
        tenor_1: Tenor,
        tenor_2: Tenor,
        basis_spread: float,
        payment_frequency: int,
    ) -> BasisSwap:
        """Create BasisSwap product instance"""
        return BasisSwap(
            T=maturity_years,
            notional=notional,
            tenor_1=tenor_1,
            tenor_2=tenor_2,
            basis_spread=basis_spread,
            payment_frequency=payment_frequency,
        )

    def generate(
        self,
        currency: str,
        trade_date: date,
        maturity_years: float,
        notional: float,
        basis_spread: float,
        counterparty_id: str,
        tenor_1: Optional[Tenor] = None,
        tenor_2: Optional[Tenor] = None,
        # Trade metadata
        trade_number: Optional[str] = None,
        book_id: Optional[str] = None,
        desk_id: Optional[str] = None,
        trader_id: Optional[str] = None,
    ) -> TradeGenerationResult:
        """
        Generate a Basis Swap trade

        Args:
            currency: Currency code (e.g., "USD", "EUR")
            trade_date: Trade execution date
            maturity_years: Maturity in years
            notional: Notional amount
            basis_spread: Basis spread applied to second leg (e.g., 0.0025 for 25bp)
            counterparty_id: Counterparty identifier
            tenor_1: Optional first leg tenor (uses convention if not provided)
            tenor_2: Optional second leg tenor (uses convention if not provided)

            # Trade metadata (optional)
            trade_number: External trade number
            book_id: Book identifier
            desk_id: Desk identifier
            trader_id: Trader identifier

        Returns:
            TradeGenerationResult containing Trade, Product, and Warnings

        Example:
            >>> generator = BasisSwapGenerator()
            >>> result = generator.generate(
            ...     currency="USD",
            ...     trade_date=date(2024, 1, 15),
            ...     maturity_years=5.0,
            ...     notional=50_000_000,
            ...     basis_spread=0.0025,  # 25 basis points
            ...     counterparty_id="CP-001",
            ... )
            >>> print(result.trade)
            >>> print(result.has_warnings())
        """
        # Get convention profile for Basis Swap
        from neutryx.market.convention_profiles import get_convention_profile
        profile = get_convention_profile(currency, ProductTypeConvention.BASIS_SWAP)

        if profile is None:
            raise ValueError(f"No Basis Swap convention profile found for {currency}")

        # Determine tenors from conventions if not provided
        if tenor_1 is None:
            if profile.fixed_leg is None:
                raise ValueError("Convention profile missing first leg specification")
            tenor_1 = self._frequency_to_tenor(profile.fixed_leg.frequency)

        if tenor_2 is None:
            if profile.floating_leg is None:
                raise ValueError("Convention profile missing second leg specification")
            tenor_2 = self._frequency_to_tenor(profile.floating_leg.frequency)

        # Payment frequency (use the higher frequency leg)
        payment_frequency = max(
            self._tenor_to_payment_frequency(tenor_1),
            self._tenor_to_payment_frequency(tenor_2),
        )

        # Calculate dates
        effective_date = trade_date + timedelta(days=profile.spot_lag)
        maturity_date = effective_date + timedelta(days=int(maturity_years * 365))

        # Create Basis Swap product
        basis_swap_product = self._create_basis_swap_product(
            notional=notional,
            maturity_years=maturity_years,
            tenor_1=tenor_1,
            tenor_2=tenor_2,
            basis_spread=basis_spread,
            payment_frequency=payment_frequency,
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
            product_type=ProductType.INTEREST_RATE_SWAP,
            trade_date=trade_date,
            effective_date=effective_date,
            maturity_date=maturity_date,
            status=TradeStatus.ACTIVE,
            notional=notional,
            currency=currency,
            settlement_type=TradeSettlementType.CASH,
            product_details={
                "product_subtype": "BASIS_SWAP",
                "tenor_1": tenor_1.value,
                "tenor_2": tenor_2.value,
                "basis_spread": basis_spread,
                "payment_frequency": payment_frequency,
                "maturity_years": maturity_years,
                "basis_swap_product": {
                    "T": basis_swap_product.T,
                    "notional": basis_swap_product.notional,
                    "tenor_1": basis_swap_product.tenor_1.value,
                    "tenor_2": basis_swap_product.tenor_2.value,
                    "basis_spread": basis_swap_product.basis_spread,
                    "payment_frequency": basis_swap_product.payment_frequency,
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
            product=basis_swap_product,
            schedules=None,  # Could add schedule generation in future
            validation_result=validation_result,
            convention_profile=profile,
        )

        return result


def generate_basis_swap_trade(
    currency: str,
    trade_date: date,
    maturity_years: float,
    notional: float,
    basis_spread: float,
    counterparty_id: str,
    tenor_1: Optional[Tenor] = None,
    tenor_2: Optional[Tenor] = None,
    **kwargs
) -> Tuple[Trade, BasisSwap, TradeGenerationResult]:
    """
    Convenience function to generate a Basis Swap trade

    This is a simplified interface that returns the Trade, Product, and full result.

    Args:
        currency: Currency code
        trade_date: Trade execution date
        maturity_years: Maturity in years
        notional: Notional amount
        basis_spread: Basis spread on second leg
        counterparty_id: Counterparty ID
        tenor_1: Optional first leg tenor
        tenor_2: Optional second leg tenor
        **kwargs: Additional parameters (see BasisSwapGenerator.generate)

    Returns:
        Tuple of (Trade, BasisSwap, TradeGenerationResult)

    Example:
        >>> trade, product, result = generate_basis_swap_trade(
        ...     "USD",
        ...     date(2024, 1, 15),
        ...     5.0,  # 5 years
        ...     50_000_000,
        ...     0.0025,  # 25bp spread
        ...     "CP-001",
        ... )
        >>> print(f"Trade ID: {trade.id}")
        >>> print(f"Tenor 1: {product.tenor_1.value}, Tenor 2: {product.tenor_2.value}")
        >>> print(f"Basis spread: {product.basis_spread * 10000:.1f} bp")
    """
    generator = BasisSwapGenerator()
    result = generator.generate(
        currency=currency,
        trade_date=trade_date,
        maturity_years=maturity_years,
        notional=notional,
        basis_spread=basis_spread,
        counterparty_id=counterparty_id,
        tenor_1=tenor_1,
        tenor_2=tenor_2,
        **kwargs
    )
    return result.trade, result.product, result
