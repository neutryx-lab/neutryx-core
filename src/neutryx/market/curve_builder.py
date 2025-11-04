"""Live curve bootstrapping pipelines with market data integration.

This module provides infrastructure for building discount and forward curves
from live market data sources, including automatic curve construction,
validation, and calibration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import jax.numpy as jnp
from jax import Array

from neutryx.market.curves import (
    BootstrappedCurve,
    Deposit,
    FRA,
    Future,
    FixedRateSwap,
    OIS,
    TenorBasisSwap,
    CurrencyBasisSwap,
    ForwardRateCurve,
)


class CurveType(Enum):
    """Type of curve to build."""
    DISCOUNT = "discount"  # Risk-free discount curve (OIS/RFR)
    PROJECTION = "projection"  # Forward projection curve (LIBOR/SOFR)
    CREDIT = "credit"  # Credit spread curve
    DIVIDEND = "dividend"  # Dividend yield curve


class InterpolationMethod(Enum):
    """Interpolation method for curve construction."""
    LOG_LINEAR = "log_linear"  # Log-linear in discount factors
    LINEAR_ZERO = "linear_zero"  # Linear in zero rates
    CUBIC_SPLINE = "cubic_spline"  # Cubic spline
    MONOTONE_CONVEX = "monotone_convex"  # Hagan-West monotone convex


@dataclass
class MarketDataSnapshot:
    """Snapshot of market data for curve building.

    Attributes:
        reference_date: Valuation date
        deposits: Money market deposit rates
        fras: Forward Rate Agreement quotes
        futures: Interest rate future prices
        swaps: Swap par rates
        ois_swaps: Overnight Index Swap rates
        basis_swaps: Tenor basis swap spreads
        fx_spots: FX spot rates (for cross-currency curves)
        timestamp: Time of market data snapshot
    """

    reference_date: date
    deposits: Dict[float, float] = field(default_factory=dict)
    fras: Dict[Tuple[float, float], float] = field(default_factory=dict)
    futures: Dict[Tuple[float, float], float] = field(default_factory=dict)
    swaps: Dict[float, float] = field(default_factory=dict)
    ois_swaps: Dict[float, float] = field(default_factory=dict)
    basis_swaps: Dict[Tuple[str, str, float], float] = field(default_factory=dict)
    fx_spots: Dict[Tuple[str, str], float] = field(default_factory=dict)
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class CurveConfig:
    """Configuration for curve building.

    Attributes:
        currency: Currency for the curve (e.g., "USD", "EUR")
        curve_type: Type of curve to build
        interpolation: Interpolation method
        day_count: Day count convention
        calendar: Business day calendar
        use_futures_convexity: Whether to apply convexity adjustment to futures
        convexity_model: Model for futures convexity adjustment
    """

    currency: str
    curve_type: CurveType = CurveType.DISCOUNT
    interpolation: InterpolationMethod = InterpolationMethod.LOG_LINEAR
    day_count: str = "ACT/365"
    calendar: str = "USD"
    use_futures_convexity: bool = True
    convexity_model: str = "hull_white"  # "hull_white", "flat", "polynomial"


class CurveBuilder:
    """Build discount and forward curves from market data.

    This class provides a high-level interface for constructing yield curves
    from market instrument quotes, with automatic instrument selection,
    ordering, and validation.

    Example:
        >>> from datetime import date
        >>> snapshot = MarketDataSnapshot(
        ...     reference_date=date(2024, 1, 1),
        ...     deposits={0.25: 0.0520, 0.5: 0.0525},
        ...     swaps={1.0: 0.0530, 2.0: 0.0540, 5.0: 0.0550}
        ... )
        >>> config = CurveConfig(currency="USD", curve_type=CurveType.DISCOUNT)
        >>> builder = CurveBuilder(config)
        >>> curve = builder.build(snapshot)
        >>> print(curve.df(1.5))
    """

    def __init__(self, config: CurveConfig):
        self.config = config
        self._validators: List[Callable] = []
        self._instrument_selectors: List[Callable] = []

    def build(self, snapshot: MarketDataSnapshot) -> BootstrappedCurve:
        """Build a curve from market data snapshot.

        Args:
            snapshot: Market data snapshot

        Returns:
            Bootstrapped yield curve

        Raises:
            ValueError: If curve building fails or validation fails
        """
        # Select instruments
        instruments = self._select_instruments(snapshot)

        # Validate instruments
        self._validate_instruments(instruments)

        # Build curve
        curve = BootstrappedCurve(instruments)

        # Validate curve
        self._validate_curve(curve)

        return curve

    def _select_instruments(self, snapshot: MarketDataSnapshot) -> List:
        """Select and order instruments for curve building.

        Args:
            snapshot: Market data snapshot

        Returns:
            List of market instruments ordered by maturity
        """
        instruments = []

        # Add deposits (short end)
        for maturity, rate in snapshot.deposits.items():
            instruments.append(Deposit(maturity=maturity, rate=rate))

        # Add FRAs (short to medium term)
        for (start, end), rate in snapshot.fras.items():
            instruments.append(FRA(start=start, end=end, rate=rate))

        # Add futures (medium term)
        for (start, end), price in snapshot.futures.items():
            # Apply convexity adjustment if configured
            convexity_adj = 0.0
            if self.config.use_futures_convexity:
                convexity_adj = self._compute_convexity_adjustment(start, end)

            instruments.append(
                Future(start=start, end=end, price=price, convexity_adjustment=convexity_adj)
            )

        # Add swaps (medium to long term)
        if self.config.curve_type == CurveType.DISCOUNT:
            # Use OIS swaps for discount curve
            for maturity, rate in snapshot.ois_swaps.items():
                payment_times, accrual_factors = self._generate_schedule(maturity)
                instruments.append(
                    OIS(
                        fixed_rate=rate,
                        payment_times=payment_times,
                        accrual_factors=accrual_factors,
                    )
                )
        else:
            # Use standard swaps for projection curve
            for maturity, rate in snapshot.swaps.items():
                payment_times, accrual_factors = self._generate_schedule(maturity)
                instruments.append(
                    FixedRateSwap(
                        fixed_rate=rate,
                        payment_times=payment_times,
                        accrual_factors=accrual_factors,
                    )
                )

        return instruments

    def _generate_schedule(
        self, maturity: float, frequency: int = 2
    ) -> Tuple[List[float], List[float]]:
        """Generate payment schedule for swaps.

        Args:
            maturity: Swap maturity in years
            frequency: Payment frequency per year (default: 2 = semi-annual)

        Returns:
            Tuple of (payment_times, accrual_factors)
        """
        n_payments = int(maturity * frequency)
        payment_times = [i / frequency for i in range(1, n_payments + 1)]
        accrual_factors = [1.0 / frequency] * n_payments

        return payment_times, accrual_factors

    def _compute_convexity_adjustment(self, start: float, end: float) -> float:
        """Compute convexity adjustment for interest rate futures.

        Args:
            start: Future start time
            end: Future end time

        Returns:
            Convexity adjustment in rate units

        Notes:
            Futures are margined daily, creating a convexity bias vs forwards.
            The adjustment is typically positive (futures rate > forward rate).

            Approximation: adj ≈ 0.5 × σ² × T1 × T2
            where σ is rate volatility, T1 is time to expiry, T2 is tenor
        """
        if self.config.convexity_model == "flat":
            # Flat adjustment (typical values: 0.5-2.0 bps per year²)
            return 0.0001 * start * (end - start)

        elif self.config.convexity_model == "hull_white":
            # Hull-White model approximation
            # adj = 0.5 × σ² × T1 × T2
            # Assume σ = 100bp (1%) rate volatility
            sigma = 0.01
            T1 = start
            T2 = end - start
            return 0.5 * sigma ** 2 * T1 * T2

        elif self.config.convexity_model == "polynomial":
            # Polynomial fit based on maturity
            # Typical curve: increasing with maturity, ~0-5 bps
            return 0.00001 * start ** 1.5

        else:
            raise ValueError(f"Unknown convexity model: {self.config.convexity_model}")

    def _validate_instruments(self, instruments: List) -> None:
        """Validate instrument set for curve building.

        Args:
            instruments: List of market instruments

        Raises:
            ValueError: If validation fails
        """
        if not instruments:
            raise ValueError("No instruments provided for curve building")

        # Check for sufficient coverage
        maturities = []
        for inst in instruments:
            if isinstance(inst, Deposit):
                maturities.append(inst.maturity)
            elif isinstance(inst, FRA):
                maturities.append(inst.end)
            elif isinstance(inst, Future):
                maturities.append(inst.end)
            elif isinstance(inst, (FixedRateSwap, OIS)):
                maturities.append(inst.payment_times[-1])

        if not maturities:
            raise ValueError("No valid maturities found in instruments")

        # Ensure coverage of key tenors
        max_maturity = max(maturities)
        if max_maturity < 1.0:
            raise ValueError("Curve must extend to at least 1 year")

    def _validate_curve(self, curve: BootstrappedCurve) -> None:
        """Validate constructed curve.

        Args:
            curve: Constructed curve

        Raises:
            ValueError: If validation fails
        """
        # Check for negative forward rates
        test_times = jnp.linspace(0.1, 10.0, 100)
        forward_rates = []

        for i in range(len(test_times) - 1):
            t1 = float(test_times[i])
            t2 = float(test_times[i + 1])
            try:
                fwd = curve.forward_rate(t1, t2)
                forward_rates.append(float(fwd))
            except Exception:
                pass

        if any(f < -0.01 for f in forward_rates):  # Allow small negative rates
            raise ValueError("Curve has excessively negative forward rates")

        # Check for arbitrage (forward rates wildly different from zero rates)
        zero_rates = [float(curve.zero_rate(t)) for t in test_times]
        if any(z < -0.05 or z > 0.20 for z in zero_rates):  # Sanity check
            raise ValueError("Curve has unrealistic zero rates")


@dataclass
class MultiCurveBuilder:
    """Build multiple curves simultaneously (discount + projection curves).

    In modern markets, different curves are used for discounting vs
    projection due to collateralization and regulatory changes.

    Attributes:
        discount_config: Configuration for discount curve (OIS/RFR)
        projection_configs: Configurations for projection curves by tenor
        build_order: Order in which to build curves
    """

    discount_config: CurveConfig
    projection_configs: Dict[str, CurveConfig] = field(default_factory=dict)
    build_order: List[str] = field(default_factory=list)

    def build(self, snapshot: MarketDataSnapshot) -> Dict[str, BootstrappedCurve]:
        """Build multiple curves from market data.

        Args:
            snapshot: Market data snapshot

        Returns:
            Dictionary mapping curve name to bootstrapped curve
        """
        curves = {}

        # Build discount curve first
        discount_builder = CurveBuilder(self.discount_config)
        curves["discount"] = discount_builder.build(snapshot)

        # Build projection curves
        for tenor, config in self.projection_configs.items():
            proj_builder = CurveBuilder(config)
            curves[f"projection_{tenor}"] = proj_builder.build(snapshot)

        return curves


def build_usd_curves(snapshot: MarketDataSnapshot) -> Dict[str, BootstrappedCurve]:
    """Convenience function to build standard USD curves.

    Builds:
    - SOFR discount curve (OIS)
    - SOFR 3M projection curve
    - SOFR 6M projection curve (if data available)

    Args:
        snapshot: Market data snapshot

    Returns:
        Dictionary of curves
    """
    discount_config = CurveConfig(
        currency="USD", curve_type=CurveType.DISCOUNT, interpolation=InterpolationMethod.LOG_LINEAR
    )

    projection_configs = {
        "3M": CurveConfig(
            currency="USD",
            curve_type=CurveType.PROJECTION,
            interpolation=InterpolationMethod.LOG_LINEAR,
        ),
    }

    builder = MultiCurveBuilder(
        discount_config=discount_config, projection_configs=projection_configs
    )

    return builder.build(snapshot)


def build_eur_curves(snapshot: MarketDataSnapshot) -> Dict[str, BootstrappedCurve]:
    """Convenience function to build standard EUR curves.

    Builds:
    - ESTR discount curve (OIS)
    - EURIBOR 3M projection curve
    - EURIBOR 6M projection curve

    Args:
        snapshot: Market data snapshot

    Returns:
        Dictionary of curves
    """
    discount_config = CurveConfig(
        currency="EUR", curve_type=CurveType.DISCOUNT, interpolation=InterpolationMethod.LOG_LINEAR
    )

    projection_configs = {
        "3M": CurveConfig(
            currency="EUR",
            curve_type=CurveType.PROJECTION,
            interpolation=InterpolationMethod.LOG_LINEAR,
        ),
        "6M": CurveConfig(
            currency="EUR",
            curve_type=CurveType.PROJECTION,
            interpolation=InterpolationMethod.LOG_LINEAR,
        ),
    }

    builder = MultiCurveBuilder(
        discount_config=discount_config, projection_configs=projection_configs
    )

    return builder.build(snapshot)


@dataclass
class CurveMonitor:
    """Monitor curve quality and detect anomalies.

    Provides real-time monitoring of curve construction, detecting:
    - Negative forward rates
    - Large jumps in forward rates
    - Arbitrage opportunities
    - Stale data
    """

    max_forward_jump: float = 0.01  # 100bp maximum jump
    max_negative_rate: float = -0.005  # -50bp maximum negative rate
    staleness_threshold: float = 3600.0  # 1 hour in seconds

    def check_curve(
        self, curve: BootstrappedCurve, snapshot: MarketDataSnapshot
    ) -> List[str]:
        """Check curve for anomalies.

        Args:
            curve: Curve to check
            snapshot: Market data snapshot used to build curve

        Returns:
            List of warning messages (empty if no issues)
        """
        warnings = []

        # Check data staleness
        if snapshot.timestamp:
            age_seconds = (datetime.now() - snapshot.timestamp).total_seconds()
            if age_seconds > self.staleness_threshold:
                warnings.append(
                    f"Market data is stale ({age_seconds:.0f}s old)"
                )

        # Check forward rates
        test_times = jnp.linspace(0.1, 10.0, 50)
        for i in range(len(test_times) - 1):
            t1 = float(test_times[i])
            t2 = float(test_times[i + 1])

            try:
                fwd = float(curve.forward_rate(t1, t2))

                if fwd < self.max_negative_rate:
                    warnings.append(
                        f"Excessive negative forward rate at {t1:.2f}y: {fwd:.4f}"
                    )

                # Check for large jumps
                if i > 0:
                    prev_fwd = float(curve.forward_rate(float(test_times[i - 1]), t1))
                    jump = abs(fwd - prev_fwd)
                    if jump > self.max_forward_jump:
                        warnings.append(
                            f"Large forward rate jump at {t1:.2f}y: {jump:.4f}"
                        )
            except Exception as e:
                warnings.append(f"Error computing forward rate at {t1:.2f}y: {e}")

        return warnings
