"""
Market data environment for unified market data management.

The MarketDataEnvironment is the central container for all market data used in pricing,
risk analysis, and calibration. It provides a unified interface for:

- Discount curves (by currency)
- Dividend/borrow curves (by underlier)
- Volatility surfaces (by underlier/currency pair)
- Credit curves (by issuer/seniority)
- FX spots and forward curves
- Correlation data

Design principles:
- Immutable: All modifications return new environments
- JAX-compatible: Can be registered as pytree for differentiation
- Type-safe: Comprehensive type hints throughout
- Explicit: No hidden state or side effects
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date
from typing import Any, Dict, Optional, Tuple

import jax.numpy as jnp
from jax import Array

from .base import CreditCurve, Curve, DiscountCurve, Surface, VolatilitySurface


@dataclass(frozen=True)
class MarketDataEnvironment:
    """
    Immutable container for all market data.

    Attributes:
        reference_date: Market data as-of date
        discount_curves: Discount curves by currency code (e.g., "USD", "EUR")
        dividend_curves: Dividend/borrow curves by underlier (e.g., "SPX", "AAPL")
        forward_curves: Forward rate curves by currency and tenor (e.g., ("USD", "3M"))
        vol_surfaces: Volatility surfaces by underlier (e.g., "SPX")
        credit_curves: Credit/hazard rate curves by (issuer, seniority)
        fx_spots: FX spot rates as (from_ccy, to_ccy) -> rate
        fx_forward_curves: FX forward curves by currency pair
        fx_vol_surfaces: FX volatility surfaces by currency pair
        correlations: Correlation matrix for multi-asset pricing
        metadata: Additional metadata (e.g., source, timestamp)

    Example:
        >>> from datetime import date
        >>> from neutryx.market.curves import FlatCurve
        >>> env = MarketDataEnvironment(
        ...     reference_date=date(2024, 1, 1),
        ...     discount_curves={"USD": FlatCurve(0.05)},
        ... )
        >>> df = env.get_discount_factor("USD", 1.0)
    """

    reference_date: date
    discount_curves: Dict[str, DiscountCurve] = field(default_factory=dict)
    dividend_curves: Dict[str, Curve] = field(default_factory=dict)
    forward_curves: Dict[Tuple[str, str], Curve] = field(default_factory=dict)
    vol_surfaces: Dict[str, VolatilitySurface] = field(default_factory=dict)
    credit_curves: Dict[Tuple[str, str], CreditCurve] = field(default_factory=dict)
    fx_spots: Dict[Tuple[str, str], float] = field(default_factory=dict)
    fx_forward_curves: Dict[Tuple[str, str], Curve] = field(default_factory=dict)
    fx_vol_surfaces: Dict[Tuple[str, str], VolatilitySurface] = field(default_factory=dict)
    correlations: Optional[Array] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Convenience accessors for discount curves

    def get_discount_factor(
        self,
        currency: str,
        t: float | Array
    ) -> float | Array:
        """
        Get discount factor for a currency at time t.

        Args:
            currency: Currency code (e.g., "USD", "EUR")
            t: Time in years from reference date

        Returns:
            Discount factor DF(0, t)

        Raises:
            KeyError: If currency not found in discount_curves
        """
        curve = self.discount_curves.get(currency)
        if curve is None:
            raise KeyError(f"No discount curve found for currency: {currency}")
        return curve.df(t)

    def get_zero_rate(
        self,
        currency: str,
        t: float | Array
    ) -> float | Array:
        """
        Get continuously-compounded zero rate for a currency at time t.

        Args:
            currency: Currency code
            t: Time in years from reference date

        Returns:
            Zero rate r(t)

        Raises:
            KeyError: If currency not found in discount_curves
        """
        curve = self.discount_curves.get(currency)
        if curve is None:
            raise KeyError(f"No discount curve found for currency: {currency}")
        return curve.zero_rate(t)

    def get_forward_rate(
        self,
        currency: str,
        t1: float | Array,
        t2: float | Array
    ) -> float | Array:
        """
        Get forward rate between t1 and t2 for a currency.

        Args:
            currency: Currency code
            t1: Start time in years
            t2: End time in years

        Returns:
            Forward rate f(t1, t2)

        Raises:
            KeyError: If currency not found in discount_curves
        """
        curve = self.discount_curves.get(currency)
        if curve is None:
            raise KeyError(f"No discount curve found for currency: {currency}")
        return curve.forward_rate(t1, t2)

    # Dividend curve accessors

    def get_dividend_yield(
        self,
        underlier: str,
        t: float | Array
    ) -> float | Array:
        """
        Get dividend yield for an underlier at time t.

        Args:
            underlier: Underlier identifier (e.g., "SPX", "AAPL")
            t: Time in years from reference date

        Returns:
            Dividend yield q(t). Returns 0.0 if no curve specified.
        """
        curve = self.dividend_curves.get(underlier)
        if curve is None:
            # Default to zero dividend if not specified
            return 0.0 if isinstance(t, float) else jnp.zeros_like(t)
        return curve.value(t)

    # Volatility surface accessors

    def get_implied_vol(
        self,
        underlier: str,
        expiry: float | Array,
        strike: float | Array
    ) -> float | Array:
        """
        Get implied volatility for an underlier at (expiry, strike).

        Args:
            underlier: Underlier identifier
            expiry: Time to expiry in years
            strike: Strike price

        Returns:
            Implied volatility (annualized)

        Raises:
            KeyError: If underlier not found in vol_surfaces
        """
        surface = self.vol_surfaces.get(underlier)
        if surface is None:
            raise KeyError(f"No volatility surface found for underlier: {underlier}")
        return surface.implied_vol(expiry, strike)

    # FX accessors

    def get_fx_spot(
        self,
        from_ccy: str,
        to_ccy: str
    ) -> float:
        """
        Get FX spot rate from one currency to another.

        Args:
            from_ccy: Source currency (e.g., "EUR")
            to_ccy: Target currency (e.g., "USD")

        Returns:
            Spot rate (units of to_ccy per unit of from_ccy)

        Raises:
            KeyError: If currency pair not found

        Note:
            Automatically handles triangulation through base currency if direct
            pair not available (future enhancement).
        """
        if from_ccy == to_ccy:
            return 1.0

        # Try direct lookup
        pair = (from_ccy, to_ccy)
        if pair in self.fx_spots:
            return self.fx_spots[pair]

        # Try inverse
        inverse_pair = (to_ccy, from_ccy)
        if inverse_pair in self.fx_spots:
            return 1.0 / self.fx_spots[inverse_pair]

        raise KeyError(f"No FX spot found for pair: {from_ccy}/{to_ccy}")

    def get_fx_forward(
        self,
        from_ccy: str,
        to_ccy: str,
        t: float | Array
    ) -> float | Array:
        """
        Get FX forward rate at time t.

        Args:
            from_ccy: Source currency
            to_ccy: Target currency
            t: Forward time in years

        Returns:
            Forward FX rate F(0, t)

        Note:
            If no explicit FX forward curve exists, computes from
            spot * DF(to_ccy) / DF(from_ccy) (covered interest parity).
        """
        if from_ccy == to_ccy:
            return 1.0 if isinstance(t, float) else jnp.ones_like(t)

        pair = (from_ccy, to_ccy)

        # Try explicit forward curve
        if pair in self.fx_forward_curves:
            return self.fx_forward_curves[pair].value(t)

        # Try inverse forward curve
        inverse_pair = (to_ccy, from_ccy)
        if inverse_pair in self.fx_forward_curves:
            return 1.0 / self.fx_forward_curves[inverse_pair].value(t)

        # Fall back to covered interest parity: F = S * DF_to / DF_from
        try:
            spot = self.get_fx_spot(from_ccy, to_ccy)
            df_from = self.get_discount_factor(from_ccy, t)
            df_to = self.get_discount_factor(to_ccy, t)
            return spot * df_to / df_from
        except KeyError as e:
            raise KeyError(
                f"Cannot compute FX forward for {from_ccy}/{to_ccy}: {e}"
            )

    # Credit curve accessors

    def get_survival_probability(
        self,
        issuer: str,
        seniority: str,
        t: float | Array
    ) -> float | Array:
        """
        Get survival probability for an issuer at time t.

        Args:
            issuer: Issuer identifier (e.g., "CORP_A")
            seniority: Seniority level (e.g., "SENIOR", "SUBORDINATED")
            t: Time in years

        Returns:
            Survival probability P(tau > t)

        Raises:
            KeyError: If credit curve not found
        """
        key = (issuer, seniority)
        curve = self.credit_curves.get(key)
        if curve is None:
            raise KeyError(
                f"No credit curve found for issuer: {issuer}, seniority: {seniority}"
            )
        return curve.survival_probability(t)

    def with_credit_curve(
        self,
        issuer: str,
        seniority: str,
        curve: CreditCurve,
    ) -> "MarketDataEnvironment":
        """Return new environment with updated credit curve."""

        if not isinstance(curve, CreditCurve):
            raise TypeError(
                "curve must implement the CreditCurve protocol and provide "
                "survival_probability/default_probability methods"
            )

        new_curves = dict(self.credit_curves)
        new_curves[(issuer, seniority)] = curve
        return replace(self, credit_curves=new_curves)

    # Mutation methods (return new environment)

    def with_discount_curve(
        self,
        currency: str,
        curve: DiscountCurve
    ) -> MarketDataEnvironment:
        """
        Return new environment with updated discount curve.

        Args:
            currency: Currency code
            curve: New discount curve

        Returns:
            New MarketDataEnvironment with updated curve
        """
        new_curves = dict(self.discount_curves)
        new_curves[currency] = curve
        return replace(self, discount_curves=new_curves)

    def with_vol_surface(
        self,
        underlier: str,
        surface: VolatilitySurface
    ) -> MarketDataEnvironment:
        """
        Return new environment with updated volatility surface.

        Args:
            underlier: Underlier identifier
            surface: New volatility surface

        Returns:
            New MarketDataEnvironment with updated surface
        """
        new_surfaces = dict(self.vol_surfaces)
        new_surfaces[underlier] = surface
        return replace(self, vol_surfaces=new_surfaces)

    def with_dividend_curve(
        self,
        underlier: str,
        curve: Curve
    ) -> MarketDataEnvironment:
        """
        Return new environment with updated dividend curve.

        Args:
            underlier: Underlier identifier
            curve: New dividend curve

        Returns:
            New MarketDataEnvironment with updated curve
        """
        new_curves = dict(self.dividend_curves)
        new_curves[underlier] = curve
        return replace(self, dividend_curves=new_curves)

    def with_fx_spot(
        self,
        from_ccy: str,
        to_ccy: str,
        rate: float
    ) -> MarketDataEnvironment:
        """
        Return new environment with updated FX spot rate.

        Args:
            from_ccy: Source currency
            to_ccy: Target currency
            rate: Spot rate

        Returns:
            New MarketDataEnvironment with updated FX spot
        """
        new_spots = dict(self.fx_spots)
        new_spots[(from_ccy, to_ccy)] = rate
        return replace(self, fx_spots=new_spots)

    def with_fx_forward_curve(
        self,
        from_ccy: str,
        to_ccy: str,
        curve: Curve
    ) -> MarketDataEnvironment:
        """
        Return new environment with updated FX forward curve.

        Args:
            from_ccy: Source currency
            to_ccy: Target currency
            curve: New forward curve for the FX pair

        Returns:
            New MarketDataEnvironment with updated FX forward curve
        """

        pair = (from_ccy, to_ccy)
        new_curves = dict(self.fx_forward_curves)
        new_curves[pair] = curve
        return replace(self, fx_forward_curves=new_curves)

    def with_fx_vol_surface(
        self,
        from_ccy: str,
        to_ccy: str,
        surface: VolatilitySurface
    ) -> MarketDataEnvironment:
        """
        Return new environment with updated FX volatility surface.

        Args:
            from_ccy: Source currency
            to_ccy: Target currency
            surface: New FX volatility surface for the pair

        Returns:
            New MarketDataEnvironment with updated FX volatility surface
        """

        pair = (from_ccy, to_ccy)
        new_surfaces = dict(self.fx_vol_surfaces)
        new_surfaces[pair] = surface
        return replace(self, fx_vol_surfaces=new_surfaces)

    def with_metadata(
        self,
        key: str,
        value: Any
    ) -> MarketDataEnvironment:
        """
        Return new environment with updated metadata.

        Args:
            key: Metadata key
            value: Metadata value

        Returns:
            New MarketDataEnvironment with updated metadata
        """
        new_metadata = dict(self.metadata)
        new_metadata[key] = value
        return replace(self, metadata=new_metadata)

    # Utility methods

    def list_currencies(self) -> list[str]:
        """Return list of all currencies with discount curves."""
        return list(self.discount_curves.keys())

    def list_underliers(self) -> list[str]:
        """Return list of all underliers with volatility surfaces."""
        return list(self.vol_surfaces.keys())

    def list_fx_pairs(self) -> list[Tuple[str, str]]:
        """Return list of all FX pairs with spot rates."""
        return list(self.fx_spots.keys())

    def __repr__(self) -> str:
        """String representation showing available market data."""
        lines = [
            f"MarketDataEnvironment(ref_date={self.reference_date})",
            f"  Discount curves: {len(self.discount_curves)} ({', '.join(self.list_currencies())})",
            f"  Dividend curves: {len(self.dividend_curves)}",
            f"  Vol surfaces: {len(self.vol_surfaces)} ({', '.join(self.list_underliers())})",
            f"  FX spots: {len(self.fx_spots)}",
            f"  Credit curves: {len(self.credit_curves)}",
        ]
        return "\n".join(lines)
