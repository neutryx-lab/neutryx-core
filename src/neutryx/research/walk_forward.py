"""Walk-forward analysis and optimization for strategy validation.

Walk-forward analysis is a method to validate trading strategies by repeatedly:
1. Optimizing parameters on an in-sample (training) period
2. Testing the strategy on an out-of-sample (testing) period
3. Rolling the window forward and repeating

This helps detect overfitting and provides more realistic performance estimates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

from .backtest import BacktestEngine, BacktestConfig, BacktestResult, Strategy
from .performance import PerformanceMetrics, calculate_comprehensive_metrics


@dataclass
class OptimizationConfig:
    """Configuration for strategy optimization."""

    # Parameter bounds (parameter_name -> (min, max))
    parameter_bounds: Dict[str, Tuple[float, float]]

    # Optimization objective ('sharpe', 'sortino', 'calmar', 'return')
    objective: str = "sharpe"

    # Optimization method ('grid', 'random', 'differential_evolution', 'bayesian')
    method: str = "differential_evolution"

    # Method-specific parameters
    max_iterations: int = 100
    population_size: int = 15
    convergence_tolerance: float = 1e-3

    # Constraints
    max_drawdown_constraint: Optional[float] = None  # Maximum acceptable drawdown
    min_trades_constraint: Optional[int] = None  # Minimum number of trades

    def validate_parameters(self, params: Dict[str, float]) -> bool:
        """Check if parameters satisfy constraints.

        Args:
            params: Parameter dictionary

        Returns:
            True if valid, False otherwise
        """
        for param, value in params.items():
            if param in self.parameter_bounds:
                min_val, max_val = self.parameter_bounds[param]
                if not (min_val <= value <= max_val):
                    return False
        return True


@dataclass
class WalkForwardResult:
    """Results from walk-forward analysis."""

    # Overall results
    in_sample_results: List[BacktestResult]
    out_of_sample_results: List[BacktestResult]
    optimal_parameters: List[Dict[str, float]]

    # Aggregate metrics
    is_performance: PerformanceMetrics
    oos_performance: PerformanceMetrics

    # Efficiency ratio (OOS / IS performance)
    efficiency_ratio: float

    # Robustness metrics
    parameter_stability: Dict[str, float]  # Std dev of optimal parameters
    performance_degradation: float  # (IS - OOS) / IS

    # Combined equity curve
    combined_equity_curve: pd.Series
    is_equity_curve: pd.Series
    oos_equity_curve: pd.Series

    def summary(self) -> Dict:
        """Get summary of walk-forward results."""
        return {
            "Number of Windows": len(self.out_of_sample_results),
            "IS Sharpe": f"{self.is_performance.sharpe_ratio:.2f}",
            "OOS Sharpe": f"{self.oos_performance.sharpe_ratio:.2f}",
            "Efficiency Ratio": f"{self.efficiency_ratio:.2%}",
            "Performance Degradation": f"{self.performance_degradation:.2%}",
            "IS Annualized Return": f"{self.is_performance.annualized_return:.2%}",
            "OOS Annualized Return": f"{self.oos_performance.annualized_return:.2%}",
            "IS Max Drawdown": f"{self.is_performance.max_drawdown:.2%}",
            "OOS Max Drawdown": f"{self.oos_performance.max_drawdown:.2%}",
        }


class WalkForwardAnalysis:
    """Walk-forward analysis engine."""

    def __init__(
        self,
        strategy_factory: Callable[[Dict[str, float]], Strategy],
        market_data: pd.DataFrame,
        optimization_config: OptimizationConfig,
        backtest_config: Optional[BacktestConfig] = None,
    ):
        """Initialize walk-forward analyzer.

        Args:
            strategy_factory: Function that creates strategy from parameters
            market_data: Market data for backtesting
            optimization_config: Optimization configuration
            backtest_config: Backtest configuration
        """
        self.strategy_factory = strategy_factory
        self.market_data = market_data
        self.optimization_config = optimization_config
        self.backtest_config = backtest_config or BacktestConfig()

    def run(
        self,
        in_sample_periods: int = 252,  # 1 year
        out_of_sample_periods: int = 63,  # 3 months
        step_size: int = 63,  # Roll forward 3 months
        anchored: bool = False,  # Use anchored or rolling window
    ) -> WalkForwardResult:
        """Run walk-forward analysis.

        Args:
            in_sample_periods: Number of periods for in-sample optimization
            out_of_sample_periods: Number of periods for out-of-sample testing
            step_size: Number of periods to roll forward
            anchored: If True, use anchored window (expanding). If False, rolling window.

        Returns:
            WalkForwardResult with complete analysis
        """
        is_results = []
        oos_results = []
        optimal_params_list = []

        # Get timestamps
        timestamps = self.market_data.index

        # Calculate window positions
        start_idx = 0
        end_idx = in_sample_periods + out_of_sample_periods

        while end_idx <= len(timestamps):
            # Define in-sample and out-of-sample periods
            if anchored:
                is_start = 0
            else:
                is_start = start_idx

            is_end = start_idx + in_sample_periods
            oos_start = is_end
            oos_end = is_end + out_of_sample_periods

            is_data = self.market_data.iloc[is_start:is_end]
            oos_data = self.market_data.iloc[oos_start:oos_end]

            print(f"Window: IS {timestamps[is_start]} to {timestamps[is_end-1]}, "
                  f"OOS {timestamps[oos_start]} to {timestamps[oos_end-1]}")

            # Optimize on in-sample data
            optimal_params = self._optimize(is_data)
            optimal_params_list.append(optimal_params)

            # Run backtest on in-sample with optimal parameters
            is_strategy = self.strategy_factory(optimal_params)
            is_engine = BacktestEngine(
                strategy=is_strategy,
                market_data=is_data,
                config=self.backtest_config,
            )
            is_result = is_engine.run()
            is_results.append(is_result)

            # Run backtest on out-of-sample with optimal parameters
            oos_strategy = self.strategy_factory(optimal_params)
            oos_engine = BacktestEngine(
                strategy=oos_strategy,
                market_data=oos_data,
                config=self.backtest_config,
            )
            oos_result = oos_engine.run()
            oos_results.append(oos_result)

            # Roll forward
            start_idx += step_size
            end_idx = start_idx + in_sample_periods + out_of_sample_periods

        # Calculate aggregate results
        return self._aggregate_results(
            is_results,
            oos_results,
            optimal_params_list,
        )

    def _optimize(self, data: pd.DataFrame) -> Dict[str, float]:
        """Optimize strategy parameters on given data.

        Args:
            data: Market data for optimization

        Returns:
            Dictionary of optimal parameters
        """
        # Define objective function
        def objective(params_array):
            # Convert array to parameter dictionary
            params = {
                name: params_array[i]
                for i, name in enumerate(self.optimization_config.parameter_bounds.keys())
            }

            # Check constraints
            if not self.optimization_config.validate_parameters(params):
                return 1e10  # Return large penalty

            # Run backtest
            strategy = self.strategy_factory(params)
            engine = BacktestEngine(
                strategy=strategy,
                market_data=data,
                config=self.backtest_config,
            )

            try:
                result = engine.run()

                # Check additional constraints
                if self.optimization_config.max_drawdown_constraint:
                    if abs(result.max_drawdown) > self.optimization_config.max_drawdown_constraint:
                        return 1e10

                if self.optimization_config.min_trades_constraint:
                    if result.num_trades < self.optimization_config.min_trades_constraint:
                        return 1e10

                # Return negative of objective (for minimization)
                if self.optimization_config.objective == "sharpe":
                    return -result.sharpe_ratio
                elif self.optimization_config.objective == "sortino":
                    return -result.sortino_ratio
                elif self.optimization_config.objective == "calmar":
                    return -result.calmar_ratio
                elif self.optimization_config.objective == "return":
                    return -result.annualized_return
                else:
                    return -result.sharpe_ratio

            except Exception as e:
                print(f"Optimization error: {e}")
                return 1e10

        # Get bounds
        bounds = list(self.optimization_config.parameter_bounds.values())

        # Optimize based on method
        if self.optimization_config.method == "differential_evolution":
            result = differential_evolution(
                objective,
                bounds,
                maxiter=self.optimization_config.max_iterations,
                popsize=self.optimization_config.population_size,
                tol=self.optimization_config.convergence_tolerance,
                seed=42,
            )
            optimal_params_array = result.x

        elif self.optimization_config.method == "grid":
            # Simple grid search
            best_params = None
            best_value = float('inf')

            # Create grid
            grid_points = 5  # Points per parameter
            param_grids = []

            for name, (min_val, max_val) in self.optimization_config.parameter_bounds.items():
                param_grids.append(np.linspace(min_val, max_val, grid_points))

            # Search grid
            import itertools
            for params_tuple in itertools.product(*param_grids):
                value = objective(np.array(params_tuple))
                if value < best_value:
                    best_value = value
                    best_params = params_tuple

            optimal_params_array = np.array(best_params)

        else:
            # Default to differential evolution
            result = differential_evolution(
                objective,
                bounds,
                maxiter=self.optimization_config.max_iterations,
            )
            optimal_params_array = result.x

        # Convert back to dictionary
        optimal_params = {
            name: optimal_params_array[i]
            for i, name in enumerate(self.optimization_config.parameter_bounds.keys())
        }

        return optimal_params

    def _aggregate_results(
        self,
        is_results: List[BacktestResult],
        oos_results: List[BacktestResult],
        optimal_params: List[Dict[str, float]],
    ) -> WalkForwardResult:
        """Aggregate results from all windows.

        Args:
            is_results: In-sample backtest results
            oos_results: Out-of-sample backtest results
            optimal_params: Optimal parameters for each window

        Returns:
            WalkForwardResult
        """
        # Concatenate equity curves
        is_equity = pd.concat([r.equity_curve for r in is_results])
        oos_equity = pd.concat([r.equity_curve for r in oos_results])

        # Calculate returns
        is_returns = is_equity.pct_change().dropna()
        oos_returns = oos_equity.pct_change().dropna()

        # Calculate comprehensive metrics
        is_metrics = calculate_comprehensive_metrics(is_returns)
        oos_metrics = calculate_comprehensive_metrics(oos_returns)

        # Efficiency ratio
        if is_metrics.sharpe_ratio != 0:
            efficiency_ratio = oos_metrics.sharpe_ratio / is_metrics.sharpe_ratio
        else:
            efficiency_ratio = 0.0

        # Performance degradation
        if is_metrics.annualized_return != 0:
            degradation = (
                (is_metrics.annualized_return - oos_metrics.annualized_return)
                / abs(is_metrics.annualized_return)
            )
        else:
            degradation = 0.0

        # Parameter stability (std dev of parameters across windows)
        param_stability = {}
        if optimal_params:
            for param_name in optimal_params[0].keys():
                param_values = [p[param_name] for p in optimal_params]
                param_stability[param_name] = float(np.std(param_values))

        # Combined equity curve (concatenate OOS results)
        combined_equity = oos_equity

        return WalkForwardResult(
            in_sample_results=is_results,
            out_of_sample_results=oos_results,
            optimal_parameters=optimal_params,
            is_performance=is_metrics,
            oos_performance=oos_metrics,
            efficiency_ratio=efficiency_ratio,
            parameter_stability=param_stability,
            performance_degradation=degradation,
            combined_equity_curve=combined_equity,
            is_equity_curve=is_equity,
            oos_equity_curve=oos_equity,
        )


def monte_carlo_permutation_test(
    walk_forward_result: WalkForwardResult,
    num_simulations: int = 1000,
) -> Dict[str, float]:
    """Perform Monte Carlo permutation test on walk-forward results.

    Randomly shuffle the out-of-sample returns to test if the strategy's
    performance is significantly different from random.

    Args:
        walk_forward_result: Walk-forward analysis result
        num_simulations: Number of Monte Carlo simulations

    Returns:
        Dictionary with p-values and statistics
    """
    oos_returns = walk_forward_result.oos_equity_curve.pct_change().dropna()
    actual_sharpe = walk_forward_result.oos_performance.sharpe_ratio

    # Run simulations
    simulated_sharpes = []

    for _ in range(num_simulations):
        # Shuffle returns
        shuffled_returns = oos_returns.sample(frac=1.0).values

        # Calculate Sharpe
        if np.std(shuffled_returns) > 0:
            simulated_sharpe = (
                np.sqrt(252) * np.mean(shuffled_returns) / np.std(shuffled_returns)
            )
            simulated_sharpes.append(simulated_sharpe)

    # Calculate p-value
    simulated_sharpes = np.array(simulated_sharpes)
    p_value = np.mean(simulated_sharpes >= actual_sharpe)

    return {
        "actual_sharpe": actual_sharpe,
        "mean_simulated_sharpe": float(np.mean(simulated_sharpes)),
        "p_value": float(p_value),
        "percentile": float(np.percentile(simulated_sharpes, 95)),
    }


__all__ = [
    "WalkForwardAnalysis",
    "WalkForwardResult",
    "OptimizationConfig",
    "monte_carlo_permutation_test",
]
