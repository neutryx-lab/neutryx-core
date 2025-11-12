"""Research and backtesting tools for strategy development.

This module provides comprehensive tools for:
- Strategy backtesting with realistic execution simulation
- Walk-forward analysis and optimization
- Transaction cost modeling (spread, slippage, market impact)
- Performance attribution and risk decomposition
"""

from .backtest import (
    BacktestEngine,
    BacktestResult,
    BacktestConfig,
    Strategy,
    Position,
    Trade,
    ExecutionModel,
)
from .transaction_costs import (
    TransactionCostModel,
    SpreadModel,
    SlippageModel,
    MarketImpactModel,
    TotalCostModel,
)
from .performance import (
    PerformanceMetrics,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_calmar_ratio,
)
from .attribution import (
    PerformanceAttribution,
    AttributionResult,
    risk_decomposition,
)
from .walk_forward import (
    WalkForwardAnalysis,
    WalkForwardResult,
    OptimizationConfig,
)
from .multiobjective import (
    RankedSolution,
    pareto_front_to_dataframe,
    rank_pareto_solutions,
    select_preferred_solution,
    plot_pareto_front,
    integrate_model_selection,
)

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "BacktestConfig",
    "Strategy",
    "Position",
    "Trade",
    "ExecutionModel",
    "TransactionCostModel",
    "SpreadModel",
    "SlippageModel",
    "MarketImpactModel",
    "TotalCostModel",
    "PerformanceMetrics",
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
    "calculate_max_drawdown",
    "calculate_calmar_ratio",
    "PerformanceAttribution",
    "AttributionResult",
    "risk_decomposition",
    "WalkForwardAnalysis",
    "WalkForwardResult",
    "OptimizationConfig",
    "RankedSolution",
    "pareto_front_to_dataframe",
    "rank_pareto_solutions",
    "select_preferred_solution",
    "plot_pareto_front",
    "integrate_model_selection",
]
