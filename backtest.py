"""Backtesting framework for Polymarket strategies."""
import json
import structlog
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

from strategy import TradingStrategy, MarketData, Signal
from risk_manager import RiskManager, Trade

logger = structlog.get_logger()


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    num_trades: int
    avg_trade_return: float
    equity_curve: List[float]
    trades: List[Trade]


class Backtester:
    """
    Backtesting engine for Polymarket strategies.

    Simulates trading on historical or cached market data.
    """

    def __init__(
        self,
        strategy: TradingStrategy,
        risk_manager: RiskManager,
        initial_capital: float = 10000.0,
    ):
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.equity_curve = [initial_capital]
        self.trades: List[Trade] = []

    def run(
        self,
        historical_data: List[Dict[str, Any]],
        verbose: bool = False,
    ) -> BacktestResult:
        """
        Run backtest on historical market data.

        Args:
            historical_data: List of market snapshots with price history
            verbose: Print detailed progress

        Returns:
            BacktestResult with performance metrics
        """
        logger.info("Starting backtest", markets=len(historical_data))

        for i, market_snapshot in enumerate(historical_data):
            # Convert to MarketData
            market = self._create_market_data(market_snapshot)
            if not market:
                continue

            # Generate signal
            signal = self.strategy.analyze_market(market)
            if not signal:
                continue

            # Simulate trade execution
            self._simulate_trade(signal, market_snapshot)

            if verbose and i % 100 == 0:
                print(f"Processed {i}/{len(historical_data)} markets...")

        return self._calculate_results()

    def _create_market_data(self, data: Dict[str, Any]) -> Optional[MarketData]:
        """Create MarketData from historical snapshot."""
        try:
            return MarketData(
                token_id=data.get('condition_id', ''),
                market_slug=data.get('market_slug', ''),
                question=data.get('question', ''),
                description=data.get('description', ''),
                current_price=float(data.get('price', 0.5)),
                best_bid=float(data.get('best_bid', 0.49)),
                best_ask=float(data.get('best_ask', 0.51)),
                spread=float(data.get('spread', 0.02)),
                volume_24h=float(data.get('volume', 0)),
                liquidity=float(data.get('liquidity', 10000)),
                num_traders=int(data.get('num_traders', 0)),
                resolution_time=None,
                time_to_resolution_hours=None,
                is_active=data.get('active', True),
                is_closed=data.get('closed', False),
            )
        except Exception as e:
            logger.debug("Failed to create market data", error=str(e))
            return None

    def _simulate_trade(self, signal: Signal, market_data: Dict[str, Any]) -> None:
        """Simulate trade execution and outcome."""
        # Check risk limits
        can_trade, reason = self.risk_manager.can_trade(
            token_id=signal.token_id,
            size=signal.suggested_size,
            side=signal.side,
        )

        if not can_trade:
            return

        # Simulate outcome (in real backtest, you'd use actual resolution data)
        # For now, assume fair probability is correct on average
        realized_return = self._calculate_simulated_return(signal, market_data)

        trade = Trade(
            token_id=signal.token_id,
            side=signal.side,
            size=signal.suggested_size,
            price=signal.current_price,
            timestamp=datetime.now(timezone.utc),
            pnl=realized_return,
        )

        self.trades.append(trade)
        self.risk_manager.record_trade(
            token_id=signal.token_id,
            side=signal.side,
            size=signal.suggested_size,
            price=signal.current_price,
            pnl=realized_return,
        )

        self.capital += realized_return
        self.equity_curve.append(self.capital)

    def _calculate_simulated_return(
        self,
        signal: Signal,
        market_data: Dict[str, Any],
    ) -> float:
        """
        Calculate simulated trade return.

        In a real implementation, this would use actual market outcomes.
        Here we simulate based on the edge and some noise.
        """
        # If our fair probability is correct, expected value is edge
        edge = signal.edge_percentage / 100

        # Add noise to simulate variance
        noise = np.random.normal(0, 0.1)  # 10% std dev

        # Return is edge + noise, scaled by position size
        return signal.suggested_size * (edge + noise)

    def _calculate_results(self) -> BacktestResult:
        """Calculate backtest performance metrics."""
        if not self.trades:
            logger.warning("No trades executed in backtest")
            return BacktestResult(
                total_return=0,
                sharpe_ratio=0,
                max_drawdown=0,
                win_rate=0,
                profit_factor=0,
                num_trades=0,
                avg_trade_return=0,
                equity_curve=self.equity_curve,
                trades=[],
            )

        returns = [t.pnl for t in self.trades if t.pnl is not None]
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r <= 0]

        # Calculate metrics
        total_return = sum(returns)

        # Sharpe ratio (simplified)
        if len(returns) > 1:
            sharpe = np.mean(returns) / (np.std(returns) + 1e-6)
        else:
            sharpe = 0

        # Max drawdown
        peak = self.initial_capital
        max_dd = 0
        for equity in self.equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak
            max_dd = max(max_dd, dd)

        win_rate = len(wins) / len(returns) if returns else 0
        profit_factor = abs(sum(wins)) / abs(sum(losses)) if losses and sum(losses) != 0 else float('inf')

        return BacktestResult(
            total_return=total_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            profit_factor=profit_factor,
            num_trades=len(self.trades),
            avg_trade_return=np.mean(returns) if returns else 0,
            equity_curve=self.equity_curve,
            trades=self.trades,
        )

    def save_results(self, filepath: str) -> None:
        """Save backtest results to file."""
        results = self._calculate_results()

        output = {
            "total_return": results.total_return,
            "sharpe_ratio": results.sharpe_ratio,
            "max_drawdown": results.max_drawdown,
            "win_rate": results.win_rate,
            "profit_factor": results.profit_factor,
            "num_trades": results.num_trades,
            "avg_trade_return": results.avg_trade_return,
            "final_capital": self.capital,
            "initial_capital": self.initial_capital,
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)

        logger.info("Backtest results saved", filepath=filepath)


def run_sample_backtest():
    """Run a sample backtest with mock data."""
    print("\nRunning sample backtest...\n")

    # Create sample market data
    sample_data = []
    for i in range(100):
        sample_data.append({
            'condition_id': f'token_{i}',
            'market_slug': f'market_{i}',
            'question': f'Will event {i} happen?',
            'price': 0.3 + (i % 5) * 0.1,  # Varied prices
            'best_bid': 0.25 + (i % 5) * 0.1,
            'best_ask': 0.35 + (i % 5) * 0.1,
            'spread': 0.05,
            'volume': 50000 + i * 1000,
            'liquidity': 100000 + i * 5000,
            'num_traders': 100 + i,
            'active': True,
            'closed': False,
        })

    # Initialize components
    strategy = TradingStrategy(
        confidence_threshold=0.6,
        min_edge_percentage=2.0,
        max_position_size=100.0,
        kelly_fraction=0.25,
    )

    risk_manager = RiskManager(
        max_daily_loss=500.0,
        max_position_size=100.0,
        max_positions=10,
    )

    backtester = Backtester(
        strategy=strategy,
        risk_manager=risk_manager,
        initial_capital=10000.0,
    )

    # Run backtest
    results = backtester.run(sample_data, verbose=True)

    # Print results
    print("\n" + "=" * 50)
    print("BACKTEST RESULTS")
    print("=" * 50)
    print(f"Total Return: ${results.total_return:.2f}")
    print(f"Return %: {(results.total_return / 10000) * 100:.2f}%")
    print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {results.max_drawdown:.2%}")
    print(f"Win Rate: {results.win_rate:.1%}")
    print(f"Profit Factor: {results.profit_factor:.2f}")
    print(f"Number of Trades: {results.num_trades}")
    print(f"Avg Trade: ${results.avg_trade_return:.2f}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    run_sample_backtest()