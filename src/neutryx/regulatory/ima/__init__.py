"""Internal Models Approach (IMA) for market risk under Basel III/FRTB.

This module implements the Basel III/FRTB Internal Models Approach including:
- Expected Shortfall (ES) at 97.5% confidence level
- P&L Attribution (PLA) testing for model validation
- Backtesting with traffic light approach
- Non-Modellable Risk Factors (NMRF) identification and treatment

Key Components:
--------------
- expected_shortfall: ES calculation with liquidity horizon adjustments
- pla_test: P&L attribution testing (Spearman correlation, KS test)
- backtesting: VaR and ES backtesting with traffic light zones
- nmrf: NMRF identification, stress scenario calibration, capital calculation

References:
-----------
- Basel III FRTB MAR11: Internal models approach
- Basel III FRTB MAR12: Expected shortfall risk measure
- Basel III FRTB MAR21: Modellability requirements
- Basel III FRTB MAR22: Stress scenario risk measure
- Basel III FRTB MAR33: Model validation standards

Example:
--------
    >>> from neutryx.regulatory.ima import (
    ...     calculate_expected_shortfall,
    ...     calculate_pla_metrics,
    ...     backtest_var,
    ...     identify_nmrfs
    ... )
    >>>
    >>> # Calculate ES
    >>> es, var, diagnostics = calculate_expected_shortfall(
    ...     pnl_scenarios,
    ...     confidence_level=0.975
    ... )
    >>>
    >>> # Test P&L attribution
    >>> pla_metrics = calculate_pla_metrics(
    ...     hypothetical_pnl,
    ...     risk_theoretical_pnl
    ... )
    >>>
    >>> # Backtest VaR
    >>> backtest_result = backtest_var(actual_pnl, var_forecasts)
    >>>
    >>> # Identify NMRFs
    >>> modellable, non_modellable, results = identify_nmrfs(risk_factors)
"""

from .backtesting import (
    BacktestException,
    BacktestResult,
    TrafficLightZone,
    backtest_expected_shortfall,
    backtest_var,
    calculate_traffic_light_zone,
    rolling_backtest,
)
from .expected_shortfall import (
    ESResult,
    LiquidityHorizon,
    calculate_expected_shortfall,
    calculate_stressed_es,
    get_liquidity_horizon,
)
from .nmrf import (
    ModellabilityStatus,
    ModellabilityTestResult,
    ModellabilityTester,
    NMRFCapital,
    NMRFCapitalCalculator,
    ObservationGap,
    StressPeriod,
    StressScenarioCalibrator,
    calculate_nmrf_capital_total,
    identify_nmrfs,
)
from .pla_test import (
    PLAMetrics,
    PLATestResult,
    calculate_pla_metrics,
    diagnose_pla_failures,
)

__all__ = [
    # Expected Shortfall
    "calculate_expected_shortfall",
    "calculate_stressed_es",
    "get_liquidity_horizon",
    "ESResult",
    "LiquidityHorizon",
    # P&L Attribution
    "calculate_pla_metrics",
    "diagnose_pla_failures",
    "PLAMetrics",
    "PLATestResult",
    # Backtesting
    "backtest_var",
    "backtest_expected_shortfall",
    "rolling_backtest",
    "calculate_traffic_light_zone",
    "BacktestResult",
    "BacktestException",
    "TrafficLightZone",
    # NMRF
    "identify_nmrfs",
    "calculate_nmrf_capital_total",
    "ModellabilityTester",
    "StressScenarioCalibrator",
    "NMRFCapitalCalculator",
    "ModellabilityTestResult",
    "ModellabilityStatus",
    "ObservationGap",
    "StressPeriod",
    "NMRFCapital",
]
