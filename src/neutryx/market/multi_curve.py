"""
Multi-curve bootstrapping framework for modern derivatives pricing.

Post-2008 financial crisis, the market moved to a multi-curve framework where:
- Discount curves (OIS/RFR) are used for discounting cash flows
- Projection curves (LIBOR/tenor curves) are used for projecting floating rates
- Cross-currency basis adjusts foreign currency discount curves

This module provides tools to build and manage multiple related curves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from neutryx.market.curves import (
    BootstrappedCurve,
    CurrencyBasisSwap,
    Deposit,
    FRA,
    Future,
    FixedRateSwap,
    MarketInstrument,
    OIS,
    TenorBasisSwap,
)


@dataclass
class CurveDefinition:
    """
    Definition of a single curve to be bootstrapped.

    Attributes:
        name: Unique identifier for the curve (e.g., "USD-OIS", "EUR-3M")
        currency: Currency code (e.g., "USD", "EUR")
        curve_type: Type of curve ("discount" or "projection")
        tenor: Tenor for projection curves (e.g., "3M", "6M"), None for discount
        instruments: List of market instruments to bootstrap from
        depends_on: Names of curves that must be built before this one
    """

    name: str
    currency: str
    curve_type: str  # "discount" or "projection"
    tenor: Optional[str] = None
    instruments: List[MarketInstrument] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)


@dataclass
class MultiCurveEnvironment:
    """
    Container for multiple related curves.

    Attributes:
        curves: Dictionary mapping curve names to BootstrappedCurve objects
        fx_spots: Dictionary of FX spot rates (currency_pair -> rate)
    """

    curves: Dict[str, BootstrappedCurve] = field(default_factory=dict)
    fx_spots: Dict[Tuple[str, str], float] = field(default_factory=dict)

    def get_discount_curve(self, currency: str) -> Optional[BootstrappedCurve]:
        """Get the discount curve for a currency."""
        curve_name = f"{currency}-OIS"
        return self.curves.get(curve_name)

    def get_projection_curve(self, currency: str, tenor: str) -> Optional[BootstrappedCurve]:
        """Get the projection curve for a currency and tenor."""
        curve_name = f"{currency}-{tenor}"
        return self.curves.get(curve_name)

    def get_fx_rate(self, from_ccy: str, to_ccy: str) -> Optional[float]:
        """Get FX spot rate from one currency to another."""
        if from_ccy == to_ccy:
            return 1.0

        # Try direct quote
        if (from_ccy, to_ccy) in self.fx_spots:
            return self.fx_spots[(from_ccy, to_ccy)]

        # Try inverse quote
        if (to_ccy, from_ccy) in self.fx_spots:
            return 1.0 / self.fx_spots[(to_ccy, from_ccy)]

        return None


class MultiCurveBuilder:
    """
    Builder for constructing multiple related curves in correct dependency order.

    Example:
        >>> builder = MultiCurveBuilder()
        >>>
        >>> # Define USD OIS discount curve
        >>> builder.add_curve_definition(
        ...     CurveDefinition(
        ...         name="USD-OIS",
        ...         currency="USD",
        ...         curve_type="discount",
        ...         instruments=[
        ...             Deposit(maturity=0.25, rate=0.053),
        ...             OIS(fixed_rate=0.055, payment_times=[0.5, 1.0], accrual_factors=[0.5, 0.5])
        ...         ]
        ...     )
        ... )
        >>>
        >>> # Define USD 3M projection curve
        >>> builder.add_curve_definition(
        ...     CurveDefinition(
        ...         name="USD-3M",
        ...         currency="USD",
        ...         curve_type="projection",
        ...         tenor="3M",
        ...         instruments=[...],  # FRAs, Futures, Tenor basis swaps
        ...         depends_on=["USD-OIS"]  # Needs discount curve for PV
        ...     )
        ... )
        >>>
        >>> # Build all curves
        >>> env = builder.build()
    """

    def __init__(self):
        self.curve_definitions: Dict[str, CurveDefinition] = {}
        self.fx_spots: Dict[Tuple[str, str], float] = {}

    def add_curve_definition(self, definition: CurveDefinition) -> None:
        """Add a curve definition to the builder."""
        self.curve_definitions[definition.name] = definition

    def add_fx_spot(self, from_ccy: str, to_ccy: str, rate: float) -> None:
        """Add an FX spot rate."""
        self.fx_spots[(from_ccy, to_ccy)] = rate

    def build(self) -> MultiCurveEnvironment:
        """
        Build all curves in dependency order.

        Returns:
            MultiCurveEnvironment containing all bootstrapped curves.

        Raises:
            ValueError: If there are circular dependencies or missing dependencies.
        """
        env = MultiCurveEnvironment(fx_spots=self.fx_spots)

        # Topological sort to determine build order
        build_order = self._resolve_dependencies()

        # Build each curve in order
        for curve_name in build_order:
            definition = self.curve_definitions[curve_name]
            curve = self._build_single_curve(definition, env)
            env.curves[curve_name] = curve

        return env

    def _resolve_dependencies(self) -> List[str]:
        """
        Resolve curve dependencies using topological sort.

        Returns:
            List of curve names in dependency order.

        Raises:
            ValueError: If there are circular dependencies.
        """
        # Build dependency graph
        in_degree = {name: 0 for name in self.curve_definitions}
        adjacency = {name: [] for name in self.curve_definitions}

        for name, definition in self.curve_definitions.items():
            for dep in definition.depends_on:
                if dep not in self.curve_definitions:
                    raise ValueError(f"Curve {name} depends on undefined curve {dep}")
                adjacency[dep].append(name)
                in_degree[name] += 1

        # Kahn's algorithm for topological sort
        queue = [name for name, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            current = queue.pop(0)
            result.append(current)

            for neighbor in adjacency[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != len(self.curve_definitions):
            raise ValueError("Circular dependency detected in curve definitions")

        return result

    def _build_single_curve(
        self, definition: CurveDefinition, env: MultiCurveEnvironment
    ) -> BootstrappedCurve:
        """
        Build a single curve, injecting dependencies where needed.

        Args:
            definition: The curve definition to build
            env: The environment containing already-built curves

        Returns:
            The bootstrapped curve
        """
        # For instruments that need reference curves, inject them
        instruments = []
        for instrument in definition.instruments:
            if isinstance(instrument, TenorBasisSwap):
                # Tenor basis needs a discount curve
                if definition.currency in env.curves:
                    discount_curve = env.get_discount_curve(definition.currency)
                    instrument.discount_curve = discount_curve
            elif isinstance(instrument, CurrencyBasisSwap):
                # Currency basis needs domestic discount curve
                domestic_curve = env.get_discount_curve(instrument.domestic_currency)
                if domestic_curve:
                    instrument.domestic_discount_curve = domestic_curve

            instruments.append(instrument)

        # Bootstrap the curve
        curve = BootstrappedCurve(instruments)
        return curve


def build_simple_multi_curve(
    currency: str,
    ois_instruments: List[MarketInstrument],
    libor_3m_instruments: List[MarketInstrument] = None,
) -> MultiCurveEnvironment:
    """
    Convenience function to build a simple OIS discount + 3M projection curve.

    Args:
        currency: Currency code (e.g., "USD")
        ois_instruments: Instruments for OIS discount curve
        libor_3m_instruments: Instruments for 3M projection curve (optional)

    Returns:
        MultiCurveEnvironment with discount and projection curves

    Example:
        >>> env = build_simple_multi_curve(
        ...     currency="USD",
        ...     ois_instruments=[
        ...         Deposit(maturity=0.25, rate=0.053),
        ...         OIS(fixed_rate=0.055, payment_times=[1.0], accrual_factors=[1.0])
        ...     ],
        ...     libor_3m_instruments=[
        ...         FRA(start=0.25, end=0.5, rate=0.056),
        ...         Future(start=0.5, end=0.75, price=94.3)
        ...     ]
        ... )
    """
    builder = MultiCurveBuilder()

    # Add OIS discount curve
    builder.add_curve_definition(
        CurveDefinition(
            name=f"{currency}-OIS",
            currency=currency,
            curve_type="discount",
            instruments=ois_instruments,
        )
    )

    # Add 3M projection curve if provided
    if libor_3m_instruments:
        builder.add_curve_definition(
            CurveDefinition(
                name=f"{currency}-3M",
                currency=currency,
                curve_type="projection",
                tenor="3M",
                instruments=libor_3m_instruments,
                depends_on=[f"{currency}-OIS"],
            )
        )

    return builder.build()
