"""Non-Modellable Risk Factors (NMRF) identification and treatment.

This module implements Basel III/FRTB requirements for identifying and treating
non-modellable risk factors (NMRFs) that don't pass the modellability test.

Per Basel III FRTB:
- Risk factors must have at least 24 real price observations over 12 months
- Maximum gap between observations: 1 month
- NMRFs must use stressed scenarios instead of internal models
- Stress period: Most severe 12-month period for the risk factor
- Capital: Expected Shortfall at 97.5% confidence on stressed scenarios

References:
- Basel III FRTB MAR21: Modellability requirements
- Basel III FRTB MAR22: Stress scenario risk measure
"""

from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

import jax.numpy as jnp
from jax import Array

from .expected_shortfall import calculate_expected_shortfall


class ModellabilityStatus(Enum):
    """Modellability status for a risk factor."""

    MODELLABLE = "modellable"  # Passes modellability test
    NON_MODELLABLE = "non_modellable"  # Fails modellability test
    INSUFFICIENT_DATA = "insufficient_data"  # Not enough historical data


@dataclass
class ObservationGap:
    """Gap between consecutive observations."""

    start_date: date
    end_date: date
    days: int

    @property
    def exceeds_threshold(self) -> bool:
        """Check if gap exceeds 1-month threshold (~30 days)."""
        return self.days > 30


@dataclass
class ModellabilityTestResult:
    """Results of modellability testing for a risk factor."""

    risk_factor: str
    status: ModellabilityStatus
    observation_count: int
    observation_period_days: int
    max_gap_days: int
    gaps_exceeding_threshold: List[ObservationGap]
    real_price_observations: int
    reason: str

    @property
    def is_modellable(self) -> bool:
        """Check if risk factor is modellable."""
        return self.status == ModellabilityStatus.MODELLABLE


@dataclass
class StressPeriod:
    """Identified stress period for NMRF calibration."""

    start_date: date
    end_date: date
    stressed_returns: Array
    stressed_var: float
    stressed_es: float
    severity_score: float  # Higher = more severe

    @property
    def duration_days(self) -> int:
        """Duration of stress period in days."""
        return (self.end_date - self.start_date).days


@dataclass
class NMRFCapital:
    """Capital calculation results for NMRFs."""

    risk_factor: str
    stress_period: StressPeriod
    stressed_es: float
    capital_charge: float
    liquidity_horizon_days: int
    scaling_factor: float
    diagnostics: Dict[str, float]


class ModellabilityTester:
    """Tests whether risk factors pass modellability requirements."""

    def __init__(
        self,
        min_observations: int = 24,
        observation_period_days: int = 365,
        max_gap_days: int = 30,
        require_real_prices: bool = True
    ):
        """Initialize modellability tester.

        Args:
            min_observations: Minimum number of observations required (Basel: 24)
            observation_period_days: Period over which observations needed (Basel: 365)
            max_gap_days: Maximum allowed gap between observations (Basel: ~30)
            require_real_prices: Whether to require real (not proxy) prices
        """
        self.min_observations = min_observations
        self.observation_period_days = observation_period_days
        self.max_gap_days = max_gap_days
        self.require_real_prices = require_real_prices

    def test_risk_factor(
        self,
        risk_factor: str,
        observation_dates: List[date],
        real_price_observations: Optional[int] = None
    ) -> ModellabilityTestResult:
        """Test if a risk factor passes modellability requirements.

        Args:
            risk_factor: Name/identifier of risk factor
            observation_dates: Dates of available observations (sorted)
            real_price_observations: Number of real (vs proxy) price observations

        Returns:
            ModellabilityTestResult with detailed test results
        """
        if not observation_dates:
            return ModellabilityTestResult(
                risk_factor=risk_factor,
                status=ModellabilityStatus.INSUFFICIENT_DATA,
                observation_count=0,
                observation_period_days=0,
                max_gap_days=0,
                gaps_exceeding_threshold=[],
                real_price_observations=0,
                reason="No observations available"
            )

        # Sort dates
        sorted_dates = sorted(observation_dates)
        observation_count = len(sorted_dates)

        # Check observation period
        period_start = sorted_dates[0]
        period_end = sorted_dates[-1]
        period_days = (period_end - period_start).days

        # Calculate gaps between consecutive observations
        gaps = []
        max_gap_days = 0

        for i in range(len(sorted_dates) - 1):
            gap_days = (sorted_dates[i + 1] - sorted_dates[i]).days
            if gap_days > 0:  # Skip same-day observations
                gap = ObservationGap(
                    start_date=sorted_dates[i],
                    end_date=sorted_dates[i + 1],
                    days=gap_days
                )
                gaps.append(gap)
                max_gap_days = max(max_gap_days, gap_days)

        # Find gaps exceeding threshold
        large_gaps = [g for g in gaps if g.exceeds_threshold]

        # Count real price observations
        if real_price_observations is None:
            real_price_observations = observation_count

        # Determine modellability status
        reasons = []

        # Check: Sufficient observations
        if observation_count < self.min_observations:
            reasons.append(
                f"Insufficient observations: {observation_count} < {self.min_observations}"
            )

        # Check: Sufficient observation period
        if period_days < self.observation_period_days:
            reasons.append(
                f"Insufficient observation period: {period_days} days < {self.observation_period_days} days"
            )

        # Check: No excessive gaps
        if large_gaps:
            reasons.append(
                f"Excessive gaps: {len(large_gaps)} gap(s) > {self.max_gap_days} days"
            )

        # Check: Real prices
        if self.require_real_prices and real_price_observations < self.min_observations:
            reasons.append(
                f"Insufficient real prices: {real_price_observations} < {self.min_observations}"
            )

        # Determine final status
        if not reasons:
            status = ModellabilityStatus.MODELLABLE
            reason = "Passes all modellability requirements"
        else:
            status = ModellabilityStatus.NON_MODELLABLE
            reason = "; ".join(reasons)

        return ModellabilityTestResult(
            risk_factor=risk_factor,
            status=status,
            observation_count=observation_count,
            observation_period_days=period_days,
            max_gap_days=max_gap_days,
            gaps_exceeding_threshold=large_gaps,
            real_price_observations=real_price_observations,
            reason=reason
        )


class StressScenarioCalibrator:
    """Calibrates stress scenarios for NMRFs."""

    def __init__(
        self,
        stress_period_days: int = 365,
        confidence_level: float = 0.975
    ):
        """Initialize stress scenario calibrator.

        Args:
            stress_period_days: Length of stress period (Basel: 12 months = 365 days)
            confidence_level: Confidence level for ES (Basel: 97.5%)
        """
        self.stress_period_days = stress_period_days
        self.confidence_level = confidence_level

    def identify_stress_period(
        self,
        returns: Array,
        dates: List[date]
    ) -> StressPeriod:
        """Identify the most severe stress period in historical data.

        Uses rolling window to find the 12-month period with highest loss/volatility.

        Args:
            returns: Historical returns for the risk factor
            dates: Dates corresponding to returns

        Returns:
            StressPeriod with identified stress period and calibrated scenarios
        """
        if len(returns) < self.stress_period_days:
            raise ValueError(
                f"Insufficient data: {len(returns)} observations < "
                f"{self.stress_period_days} required for stress period"
            )

        # Convert to numpy for windowing
        returns_np = jnp.array(returns)

        # Rolling window approach to find most severe period
        best_stress_period = None
        max_severity = -jnp.inf

        # Try each possible stress period window
        window_size = min(len(returns_np), self.stress_period_days)

        for i in range(len(returns_np) - window_size + 1):
            window_returns = returns_np[i:i + window_size]
            window_start = dates[i]
            window_end = dates[min(i + window_size - 1, len(dates) - 1)]

            # Calculate severity metrics for this window
            # Severity = combination of negative mean return and high volatility
            mean_return = float(jnp.mean(window_returns))
            volatility = float(jnp.std(window_returns))
            es, var, _ = calculate_expected_shortfall(
                window_returns,
                confidence_level=self.confidence_level
            )

            # Severity score: emphasize both losses and tail risk
            # Higher score = more severe
            severity = -mean_return + 2.0 * volatility + abs(es)

            if severity > max_severity:
                max_severity = severity
                best_stress_period = StressPeriod(
                    start_date=window_start,
                    end_date=window_end,
                    stressed_returns=window_returns,
                    stressed_var=var,
                    stressed_es=es,
                    severity_score=severity
                )

        if best_stress_period is None:
            raise ValueError("Failed to identify stress period")

        return best_stress_period

    def calibrate_stress_scenarios(
        self,
        stress_period: StressPeriod,
        num_scenarios: int = 1000
    ) -> Array:
        """Calibrate stress scenarios from identified stress period.

        Uses bootstrap resampling from stress period returns to generate scenarios.

        Args:
            stress_period: Identified stress period
            num_scenarios: Number of scenarios to generate

        Returns:
            Array of stress scenarios
        """
        import jax.random as jrandom

        # Bootstrap resample from stress period
        key = jrandom.PRNGKey(42)
        n_observations = len(stress_period.stressed_returns)

        # Sample with replacement
        indices = jrandom.randint(
            key,
            shape=(num_scenarios,),
            minval=0,
            maxval=n_observations
        )

        stress_scenarios = stress_period.stressed_returns[indices]

        return stress_scenarios


class NMRFCapitalCalculator:
    """Calculates regulatory capital for non-modellable risk factors."""

    def __init__(
        self,
        confidence_level: float = 0.975,
        liquidity_horizon_days: int = 10
    ):
        """Initialize NMRF capital calculator.

        Args:
            confidence_level: Confidence level for ES (Basel: 97.5%)
            liquidity_horizon_days: Liquidity horizon for risk factor
        """
        self.confidence_level = confidence_level
        self.liquidity_horizon_days = liquidity_horizon_days

    def calculate_capital(
        self,
        risk_factor: str,
        stress_period: StressPeriod,
        stress_scenarios: Array,
        position_size: float = 1.0
    ) -> NMRFCapital:
        """Calculate regulatory capital for a NMRF.

        Args:
            risk_factor: Name of risk factor
            stress_period: Identified stress period
            stress_scenarios: Calibrated stress scenarios
            position_size: Size of position in risk factor

        Returns:
            NMRFCapital with capital charge and diagnostics
        """
        # Calculate ES on stress scenarios
        stressed_es, stressed_var, diagnostics = calculate_expected_shortfall(
            stress_scenarios,
            confidence_level=self.confidence_level
        )

        # Apply liquidity horizon scaling (square root of time)
        # Basel: scale from 10-day to actual liquidity horizon
        scaling_factor = jnp.sqrt(self.liquidity_horizon_days / 10.0)
        scaled_es = abs(stressed_es) * float(scaling_factor)

        # Capital charge = scaled ES * position size
        capital_charge = scaled_es * abs(position_size)

        # Additional diagnostics
        scenario_diagnostics = {
            'num_scenarios': len(stress_scenarios),
            'stressed_es_unscaled': float(abs(stressed_es)),
            'stressed_var': float(abs(stressed_var)),
            'scaling_factor': float(scaling_factor),
            'stress_period_days': stress_period.duration_days,
            'position_size': position_size,
            'mean_scenario_loss': float(jnp.mean(stress_scenarios)),
            'max_scenario_loss': float(jnp.min(stress_scenarios)),  # Most negative
            'scenario_volatility': float(jnp.std(stress_scenarios))
        }

        return NMRFCapital(
            risk_factor=risk_factor,
            stress_period=stress_period,
            stressed_es=stressed_es,
            capital_charge=capital_charge,
            liquidity_horizon_days=self.liquidity_horizon_days,
            scaling_factor=float(scaling_factor),
            diagnostics=scenario_diagnostics
        )


def identify_nmrfs(
    risk_factors: Dict[str, List[date]],
    min_observations: int = 24,
    observation_period_days: int = 365,
    max_gap_days: int = 30
) -> Tuple[List[str], List[str], Dict[str, ModellabilityTestResult]]:
    """Identify modellable and non-modellable risk factors.

    Args:
        risk_factors: Dict mapping risk factor names to observation dates
        min_observations: Minimum observations for modellability
        observation_period_days: Required observation period
        max_gap_days: Maximum gap between observations

    Returns:
        Tuple of (modellable_rfs, non_modellable_rfs, test_results)
    """
    tester = ModellabilityTester(
        min_observations=min_observations,
        observation_period_days=observation_period_days,
        max_gap_days=max_gap_days
    )

    modellable = []
    non_modellable = []
    test_results = {}

    for rf_name, obs_dates in risk_factors.items():
        result = tester.test_risk_factor(rf_name, obs_dates)
        test_results[rf_name] = result

        if result.is_modellable:
            modellable.append(rf_name)
        else:
            non_modellable.append(rf_name)

    return modellable, non_modellable, test_results


def calculate_nmrf_capital_total(
    nmrf_capitals: List[NMRFCapital],
    correlation_factor: float = 1.0
) -> Dict[str, float]:
    """Calculate total NMRF capital across multiple risk factors.

    Args:
        nmrf_capitals: List of NMRF capital calculations
        correlation_factor: Correlation factor for diversification (1.0 = perfect correlation)

    Returns:
        Dictionary with total capital and breakdown
    """
    if not nmrf_capitals:
        return {
            'total_capital': 0.0,
            'undiversified_sum': 0.0,
            'correlation_factor': correlation_factor,
            'num_nmrfs': 0
        }

    # Sum individual capitals
    undiversified_sum = sum(cap.capital_charge for cap in nmrf_capitals)

    # Apply correlation factor for diversification
    # correlation_factor = 1.0 means no diversification (perfect correlation)
    # correlation_factor < 1.0 allows diversification benefit
    total_capital = undiversified_sum * correlation_factor

    return {
        'total_capital': total_capital,
        'undiversified_sum': undiversified_sum,
        'correlation_factor': correlation_factor,
        'num_nmrfs': len(nmrf_capitals),
        'individual_capitals': {
            cap.risk_factor: cap.capital_charge
            for cap in nmrf_capitals
        }
    }
