"""Tests for backtest engine."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

from neutryx.research.backtest import (
    BacktestEngine,
    BacktestConfig,
    BacktestResult,
    Strategy,
    Position,
    Trade,
    OrderSide,
    OrderType,
    SimpleExecutionModel,
)


class BuyAndHoldStrategy(Strategy):
    """Simple buy-and-hold strategy for testing."""

    def __init__(self, name: str = "BuyAndHold"):
        super().__init__(name)
        self.entered = False

    def generate_signals(
        self,
        timestamp: datetime,
        market_data: pd.DataFrame,
        positions: Dict[str, Position],
        portfolio_value: float,
    ) -> List[Dict]:
        """Buy on first bar, hold until end."""
        if not self.entered and len(market_data) > 0:
            self.entered = True
            return [{
                "symbol": "TEST",
                "side": OrderSide.BUY,
                "quantity": 100.0,
                "order_type": OrderType.MARKET,
            }]
        return []


class MomentumStrategy(Strategy):
    """Simple momentum strategy for testing."""

    def __init__(self, lookback: int = 20, name: str = "Momentum"):
        super().__init__(name)
        self.lookback = lookback

    def generate_signals(
        self,
        timestamp: datetime,
        market_data: pd.DataFrame,
        positions: Dict[str, Position],
        portfolio_value: float,
    ) -> List[Dict]:
        """Buy if price > moving average, sell otherwise."""
        if len(market_data) < self.lookback:
            return []

        current_price = market_data["close"].iloc[-1]
        ma = market_data["close"].iloc[-self.lookback:].mean()

        current_position = positions.get("TEST")
        current_qty = current_position.quantity if current_position else 0.0

        signals = []

        if current_price > ma and current_qty == 0:
            # Buy signal
            signals.append({
                "symbol": "TEST",
                "side": OrderSide.BUY,
                "quantity": 100.0,
                "order_type": OrderType.MARKET,
            })
        elif current_price < ma and current_qty > 0:
            # Sell signal
            signals.append({
                "symbol": "TEST",
                "side": OrderSide.SELL,
                "quantity": current_qty,
                "order_type": OrderType.MARKET,
            })

        return signals


@pytest.fixture
def sample_market_data():
    """Create sample market data."""
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=252, freq="D")

    # Generate trending price series
    returns = np.random.normal(0.0005, 0.02, 252)
    prices = 100 * np.exp(np.cumsum(returns))

    data = pd.DataFrame({
        "close": prices,
        "open": prices * 0.99,
        "high": prices * 1.01,
        "low": prices * 0.98,
        "volume": np.random.randint(1000, 10000, 252),
    }, index=dates)

    return data


class TestPosition:
    """Test Position class."""

    def test_position_creation(self):
        """Test creating a position."""
        pos = Position(symbol="TEST")
        assert pos.symbol == "TEST"
        assert pos.quantity == 0.0
        assert pos.avg_price == 0.0
        assert pos.realized_pnl == 0.0
        assert pos.unrealized_pnl == 0.0

    def test_position_buy(self):
        """Test buying into a position."""
        pos = Position(symbol="TEST")
        trade = Trade(
            timestamp=datetime.now(),
            symbol="TEST",
            side=OrderSide.BUY,
            quantity=100,
            price=50.0,
            commission=5.0,
        )

        pos.update(trade)

        assert pos.quantity == 100
        assert pos.avg_price == 50.0

    def test_position_sell(self):
        """Test selling from a position."""
        pos = Position(symbol="TEST", quantity=100, avg_price=50.0)
        trade = Trade(
            timestamp=datetime.now(),
            symbol="TEST",
            side=OrderSide.SELL,
            quantity=50,
            price=55.0,
            commission=2.5,
        )

        pos.update(trade)

        assert pos.quantity == 50
        assert pos.realized_pnl == 250.0  # 50 * (55 - 50)

    def test_position_mark_to_market(self):
        """Test marking position to market."""
        pos = Position(symbol="TEST", quantity=100, avg_price=50.0)
        pos.mark_to_market(60.0)

        assert pos.unrealized_pnl == 1000.0  # 100 * (60 - 50)


class TestTrade:
    """Test Trade class."""

    def test_trade_creation(self):
        """Test creating a trade."""
        trade = Trade(
            timestamp=datetime.now(),
            symbol="TEST",
            side=OrderSide.BUY,
            quantity=100,
            price=50.0,
            commission=5.0,
            slippage=0.10,
            market_impact=0.05,
        )

        assert trade.quantity == 100
        assert trade.price == 50.0
        assert abs(trade.total_cost - 5.15) < 1e-10  # commission + slippage + impact
        assert abs(trade.effective_price - 50.15) < 1e-10  # price + slippage + impact for buy

    def test_trade_notional(self):
        """Test trade notional calculation."""
        trade = Trade(
            timestamp=datetime.now(),
            symbol="TEST",
            side=OrderSide.BUY,
            quantity=100,
            price=50.0,
        )

        assert trade.notional == 5000.0


class TestSimpleExecutionModel:
    """Test SimpleExecutionModel."""

    def test_execute_market_order(self):
        """Test executing a market order."""
        model = SimpleExecutionModel(slippage_bps=5.0, commission_bps=1.0)

        order = {
            "symbol": "TEST",
            "side": OrderSide.BUY,
            "quantity": 100,
            "order_type": OrderType.MARKET,
        }

        market_data = pd.Series({
            "close": 50.0,
            "volume": 10000,
        })

        trade = model.execute_order(order, market_data, 10000)

        assert trade is not None
        assert trade.quantity == 100
        assert trade.price == 50.0
        assert trade.slippage > 0  # Should have slippage for buy
        assert trade.commission > 0


class TestBacktestEngine:
    """Test BacktestEngine."""

    def test_buy_and_hold_backtest(self, sample_market_data):
        """Test buy-and-hold strategy backtest."""
        strategy = BuyAndHoldStrategy()
        config = BacktestConfig(initial_capital=100000)

        engine = BacktestEngine(
            strategy=strategy,
            market_data=sample_market_data,
            config=config,
        )

        result = engine.run()

        assert isinstance(result, BacktestResult)
        assert result.num_trades == 1  # Single buy trade
        assert result.final_capital > 0
        assert len(result.equity_curve) == len(sample_market_data)

    def test_momentum_strategy(self, sample_market_data):
        """Test momentum strategy backtest."""
        strategy = MomentumStrategy(lookback=20)
        config = BacktestConfig(initial_capital=100000)

        engine = BacktestEngine(
            strategy=strategy,
            market_data=sample_market_data,
            config=config,
        )

        result = engine.run()

        assert isinstance(result, BacktestResult)
        assert result.num_trades > 0
        assert result.final_capital > 0

    def test_backtest_metrics(self, sample_market_data):
        """Test backtest result metrics."""
        strategy = BuyAndHoldStrategy()
        config = BacktestConfig(initial_capital=100000)

        engine = BacktestEngine(
            strategy=strategy,
            market_data=sample_market_data,
            config=config,
        )

        result = engine.run()

        # Check that all metrics are calculated
        assert isinstance(result.total_return, float)
        assert isinstance(result.annualized_return, float)
        assert isinstance(result.sharpe_ratio, float)
        assert isinstance(result.sortino_ratio, float)
        assert isinstance(result.max_drawdown, float)
        assert isinstance(result.calmar_ratio, float)

    def test_transaction_costs(self, sample_market_data):
        """Test that transaction costs are applied."""
        strategy = BuyAndHoldStrategy()
        config = BacktestConfig(
            initial_capital=100000,
            commission_bps=10.0,
            slippage_bps=10.0,
        )

        engine = BacktestEngine(
            strategy=strategy,
            market_data=sample_market_data,
            config=config,
        )

        result = engine.run()

        assert result.total_commission > 0
        assert result.total_slippage > 0

    def test_short_selling_disabled(self, sample_market_data):
        """Test that short selling can be disabled."""
        # Strategy that tries to short
        class ShortStrategy(Strategy):
            def __init__(self):
                super().__init__("Short")

            def generate_signals(self, timestamp, market_data, positions, portfolio_value):
                if len(market_data) == 10:
                    return [{
                        "symbol": "TEST",
                        "side": OrderSide.SELL,
                        "quantity": 100,
                        "order_type": OrderType.MARKET,
                    }]
                return []

        strategy = ShortStrategy()
        config = BacktestConfig(
            initial_capital=100000,
            enable_short_selling=False,
        )

        engine = BacktestEngine(
            strategy=strategy,
            market_data=sample_market_data,
            config=config,
        )

        result = engine.run()

        # Should have no trades since short selling is disabled
        assert result.num_trades == 0


class TestBacktestConfig:
    """Test BacktestConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = BacktestConfig()

        assert config.initial_capital == 1_000_000.0
        assert config.commission_bps == 1.0
        assert config.slippage_bps == 5.0
        assert config.enable_short_selling is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = BacktestConfig(
            initial_capital=500_000.0,
            commission_bps=2.0,
            slippage_bps=10.0,
            enable_short_selling=False,
        )

        assert config.initial_capital == 500_000.0
        assert config.commission_bps == 2.0
        assert config.slippage_bps == 10.0
        assert config.enable_short_selling is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
