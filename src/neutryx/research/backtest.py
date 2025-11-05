"""Strategy backtesting engine with realistic execution simulation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Callable

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import Array


class OrderType(str, Enum):
    """Order types for execution."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(str, Enum):
    """Order side."""

    BUY = "buy"
    SELL = "sell"


@dataclass
class Trade:
    """Represents a trade execution."""

    timestamp: datetime
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    commission: float = 0.0
    slippage: float = 0.0
    market_impact: float = 0.0

    @property
    def total_cost(self) -> float:
        """Total transaction cost."""
        return self.commission + abs(self.slippage) + abs(self.market_impact)

    @property
    def effective_price(self) -> float:
        """Effective execution price including costs."""
        if self.side == OrderSide.BUY:
            return self.price + self.slippage + self.market_impact
        else:
            return self.price - self.slippage - self.market_impact

    @property
    def notional(self) -> float:
        """Notional value of trade."""
        return abs(self.quantity * self.price)


@dataclass
class Position:
    """Represents a position in a security."""

    symbol: str
    quantity: float = 0.0
    avg_price: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0

    def update(self, trade: Trade):
        """Update position with new trade."""
        if trade.side == OrderSide.BUY:
            # Buying - increase position
            total_cost = self.quantity * self.avg_price + trade.quantity * trade.effective_price
            self.quantity += trade.quantity
            self.avg_price = total_cost / self.quantity if self.quantity != 0 else 0.0
        else:
            # Selling - decrease position
            realized = trade.quantity * (trade.effective_price - self.avg_price)
            self.realized_pnl += realized
            self.quantity -= trade.quantity

            # Handle position reversal
            if self.quantity < 0:
                self.avg_price = trade.effective_price
            elif self.quantity == 0:
                self.avg_price = 0.0

    def mark_to_market(self, current_price: float):
        """Mark position to market."""
        if self.quantity != 0:
            self.unrealized_pnl = self.quantity * (current_price - self.avg_price)
        else:
            self.unrealized_pnl = 0.0

    @property
    def total_pnl(self) -> float:
        """Total P&L (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl


class Strategy(ABC):
    """Abstract base class for trading strategies."""

    def __init__(self, name: str):
        """Initialize strategy.

        Args:
            name: Strategy name
        """
        self.name = name

    @abstractmethod
    def generate_signals(
        self,
        timestamp: datetime,
        market_data: pd.DataFrame,
        positions: Dict[str, Position],
        portfolio_value: float,
    ) -> List[Dict]:
        """Generate trading signals.

        Args:
            timestamp: Current timestamp
            market_data: Market data up to current timestamp
            positions: Current positions
            portfolio_value: Current portfolio value

        Returns:
            List of order dictionaries with keys:
                - symbol: str
                - side: OrderSide
                - quantity: float
                - order_type: OrderType
                - limit_price: Optional[float]
        """
        pass

    def on_trade(self, trade: Trade):
        """Called when a trade is executed.

        Args:
            trade: Executed trade
        """
        pass

    def on_bar(self, timestamp: datetime, bar: pd.Series):
        """Called on each bar/timestamp.

        Args:
            timestamp: Bar timestamp
            bar: Bar data
        """
        pass


class ExecutionModel(ABC):
    """Abstract base class for execution models."""

    @abstractmethod
    def execute_order(
        self,
        order: Dict,
        market_data: pd.Series,
        volume: float,
    ) -> Optional[Trade]:
        """Execute an order.

        Args:
            order: Order dictionary
            market_data: Current market data
            volume: Available volume

        Returns:
            Trade if executed, None otherwise
        """
        pass


class SimpleExecutionModel(ExecutionModel):
    """Simple execution model with immediate fills."""

    def __init__(
        self,
        slippage_bps: float = 5.0,
        commission_bps: float = 1.0,
    ):
        """Initialize execution model.

        Args:
            slippage_bps: Slippage in basis points
            commission_bps: Commission in basis points
        """
        self.slippage_bps = slippage_bps
        self.commission_bps = commission_bps

    def execute_order(
        self,
        order: Dict,
        market_data: pd.Series,
        volume: float,
    ) -> Optional[Trade]:
        """Execute order at current price with slippage."""
        price = market_data.get("close", market_data.get("price"))
        if price is None or np.isnan(price):
            return None

        # Calculate costs
        slippage = price * (self.slippage_bps / 10000.0)
        commission = price * order["quantity"] * (self.commission_bps / 10000.0)

        trade = Trade(
            timestamp=market_data.name if hasattr(market_data, "name") else datetime.now(),
            symbol=order["symbol"],
            side=order["side"],
            quantity=order["quantity"],
            price=price,
            commission=commission,
            slippage=slippage if order["side"] == OrderSide.BUY else -slippage,
            market_impact=0.0,
        )

        return trade


@dataclass
class BacktestConfig:
    """Configuration for backtest."""

    initial_capital: float = 1_000_000.0
    commission_bps: float = 1.0
    slippage_bps: float = 5.0
    market_impact_bps: float = 2.0
    enable_short_selling: bool = True
    max_leverage: float = 1.0
    margin_requirement: float = 0.0
    risk_free_rate: float = 0.02


@dataclass
class BacktestResult:
    """Results from a backtest."""

    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    num_trades: int
    avg_trade_pnl: float
    total_commission: float
    total_slippage: float
    total_market_impact: float

    # Time series data
    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.DataFrame
    trades: pd.DataFrame

    # Detailed metrics
    daily_returns: pd.Series
    monthly_returns: pd.Series
    drawdown_series: pd.Series
    rolling_sharpe: pd.Series

    def summary(self) -> Dict:
        """Get summary statistics."""
        return {
            "strategy": self.strategy_name,
            "period": f"{self.start_date.date()} to {self.end_date.date()}",
            "total_return": f"{self.total_return:.2%}",
            "annualized_return": f"{self.annualized_return:.2%}",
            "sharpe_ratio": f"{self.sharpe_ratio:.2f}",
            "sortino_ratio": f"{self.sortino_ratio:.2f}",
            "max_drawdown": f"{self.max_drawdown:.2%}",
            "calmar_ratio": f"{self.calmar_ratio:.2f}",
            "win_rate": f"{self.win_rate:.2%}",
            "profit_factor": f"{self.profit_factor:.2f}",
            "num_trades": self.num_trades,
            "avg_trade_pnl": f"${self.avg_trade_pnl:,.2f}",
            "total_costs": f"${self.total_commission + self.total_slippage + self.total_market_impact:,.2f}",
        }


class BacktestEngine:
    """Backtesting engine for strategy simulation."""

    def __init__(
        self,
        strategy: Strategy,
        market_data: pd.DataFrame,
        config: Optional[BacktestConfig] = None,
        execution_model: Optional[ExecutionModel] = None,
    ):
        """Initialize backtest engine.

        Args:
            strategy: Trading strategy
            market_data: Historical market data (must have datetime index)
            config: Backtest configuration
            execution_model: Execution model for trade simulation
        """
        self.strategy = strategy
        self.market_data = market_data
        self.config = config or BacktestConfig()
        self.execution_model = execution_model or SimpleExecutionModel(
            slippage_bps=self.config.slippage_bps,
            commission_bps=self.config.commission_bps,
        )

        # State
        self.cash = self.config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.timestamps: List[datetime] = []

    def run(self) -> BacktestResult:
        """Run backtest.

        Returns:
            BacktestResult with performance metrics and analytics
        """
        # Initialize
        self.cash = self.config.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = [self.config.initial_capital]
        self.timestamps = []

        # Iterate through market data
        for timestamp, bar in self.market_data.iterrows():
            self.timestamps.append(timestamp)
            self.strategy.on_bar(timestamp, bar)

            # Get current portfolio value
            portfolio_value = self._calculate_portfolio_value(bar)

            # Generate signals
            signals = self.strategy.generate_signals(
                timestamp=timestamp,
                market_data=self.market_data.loc[:timestamp],
                positions=self.positions,
                portfolio_value=portfolio_value,
            )

            # Execute orders
            for order in signals:
                trade = self._execute_order(order, bar)
                if trade:
                    self.trades.append(trade)
                    self.strategy.on_trade(trade)
                    self._update_position(trade)

            # Mark to market
            self._mark_to_market(bar)

            # Record equity
            portfolio_value = self._calculate_portfolio_value(bar)
            self.equity_curve.append(portfolio_value)

        # Calculate results
        return self._calculate_results()

    def _execute_order(self, order: Dict, bar: pd.Series) -> Optional[Trade]:
        """Execute an order."""
        # Check if we have data for this symbol
        symbol = order["symbol"]

        # Get volume (if available)
        volume = bar.get("volume", float("inf"))

        # Execute through execution model
        trade = self.execution_model.execute_order(order, bar, volume)

        # Check capital constraints
        if trade and not self._check_capital_constraints(trade):
            return None

        # Update cash
        if trade:
            if trade.side == OrderSide.BUY:
                self.cash -= trade.quantity * trade.effective_price + trade.commission
            else:
                self.cash += trade.quantity * trade.effective_price - trade.commission

        return trade

    def _update_position(self, trade: Trade):
        """Update position with trade."""
        if trade.symbol not in self.positions:
            self.positions[trade.symbol] = Position(symbol=trade.symbol)

        self.positions[trade.symbol].update(trade)

    def _mark_to_market(self, bar: pd.Series):
        """Mark all positions to market."""
        for symbol, position in self.positions.items():
            price = bar.get("close", bar.get("price"))
            if price is not None and not np.isnan(price):
                position.mark_to_market(price)

    def _calculate_portfolio_value(self, bar: pd.Series) -> float:
        """Calculate current portfolio value."""
        positions_value = sum(
            pos.quantity * bar.get("close", bar.get("price", 0.0))
            for pos in self.positions.values()
        )
        return self.cash + positions_value

    def _check_capital_constraints(self, trade: Trade) -> bool:
        """Check if trade satisfies capital constraints."""
        required_cash = trade.quantity * trade.effective_price + trade.commission

        if trade.side == OrderSide.BUY:
            return self.cash >= required_cash
        else:
            # Check short selling constraints
            if not self.config.enable_short_selling:
                position = self.positions.get(trade.symbol)
                if not position or position.quantity < trade.quantity:
                    return False

        return True

    def _calculate_results(self) -> BacktestResult:
        """Calculate backtest results."""
        # Create equity curve series
        equity_series = pd.Series(
            self.equity_curve[1:], index=self.timestamps
        )

        # Calculate returns
        returns = equity_series.pct_change().dropna()
        daily_returns = returns.resample("D").sum()
        monthly_returns = returns.resample("ME").sum()  # ME = Month End

        # Calculate drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        # Calculate metrics
        total_return = (equity_series.iloc[-1] - self.config.initial_capital) / self.config.initial_capital
        years = (self.timestamps[-1] - self.timestamps[0]).days / 365.25
        annualized_return = (1 + total_return) ** (1 / years) - 1

        sharpe = self._calculate_sharpe(returns)
        sortino = self._calculate_sortino(returns)
        max_dd = abs(drawdown.min())
        calmar = annualized_return / max_dd if max_dd != 0 else 0.0

        # Trade statistics
        trades_df = pd.DataFrame([
            {
                "timestamp": t.timestamp,
                "symbol": t.symbol,
                "side": t.side.value,
                "quantity": t.quantity,
                "price": t.price,
                "pnl": 0.0,  # Would need to calculate based on position
                "commission": t.commission,
                "slippage": t.slippage,
                "market_impact": t.market_impact,
            }
            for t in self.trades
        ])

        win_rate = 0.0
        profit_factor = 0.0
        if len(self.trades) > 0:
            winning_trades = sum(1 for t in self.trades if getattr(t, "pnl", 0) > 0)
            win_rate = winning_trades / len(self.trades)

            gross_profit = sum(getattr(t, "pnl", 0) for t in self.trades if getattr(t, "pnl", 0) > 0)
            gross_loss = abs(sum(getattr(t, "pnl", 0) for t in self.trades if getattr(t, "pnl", 0) < 0))
            profit_factor = gross_profit / gross_loss if gross_loss != 0 else 0.0

        # Rolling Sharpe
        rolling_sharpe = returns.rolling(window=252).apply(
            lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() != 0 else 0.0
        )

        return BacktestResult(
            strategy_name=self.strategy.name,
            start_date=self.timestamps[0],
            end_date=self.timestamps[-1],
            initial_capital=self.config.initial_capital,
            final_capital=equity_series.iloc[-1],
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            calmar_ratio=calmar,
            win_rate=win_rate,
            profit_factor=profit_factor,
            num_trades=len(self.trades),
            avg_trade_pnl=sum(getattr(t, "pnl", 0) for t in self.trades) / len(self.trades) if self.trades else 0.0,
            total_commission=sum(t.commission for t in self.trades),
            total_slippage=sum(abs(t.slippage) for t in self.trades),
            total_market_impact=sum(abs(t.market_impact) for t in self.trades),
            equity_curve=equity_series,
            returns=returns,
            positions=pd.DataFrame(),  # Would populate with position history
            trades=trades_df,
            daily_returns=daily_returns,
            monthly_returns=monthly_returns,
            drawdown_series=drawdown,
            rolling_sharpe=rolling_sharpe,
        )

    def _calculate_sharpe(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0

        excess_returns = returns - self.config.risk_free_rate / 252
        return np.sqrt(252) * excess_returns.mean() / returns.std()

    def _calculate_sortino(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio."""
        if len(returns) == 0:
            return 0.0

        excess_returns = returns - self.config.risk_free_rate / 252
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0

        return np.sqrt(252) * excess_returns.mean() / downside_returns.std()


__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "BacktestConfig",
    "Strategy",
    "Position",
    "Trade",
    "OrderType",
    "OrderSide",
    "ExecutionModel",
    "SimpleExecutionModel",
]
