"""Risk management for Polymarket trading bot."""
import structlog
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime, timezone, timedelta
from decimal import Decimal

logger = structlog.get_logger()


@dataclass
class Trade:
    """Record of executed trade."""
    token_id: str
    side: str
    size: float
    price: float
    timestamp: datetime
    pnl: Optional[float] = None


@dataclass
class RiskState:
    """Current risk state."""
    daily_pnl: float = 0.0
    total_trades_today: int = 0
    open_positions: Dict[str, float] = field(default_factory=dict)
    trades: List[Trade] = field(default_factory=list)
    last_reset: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class RiskManager:
    """Manages trading risk and position limits."""

    def __init__(
        self,
        max_daily_loss: float = 500.0,
        max_position_size: float = 100.0,
        max_positions: int = 10,
        max_trades_per_day: int = 50,
    ):
        self.max_daily_loss = max_daily_loss
        self.max_position_size = max_position_size
        self.max_positions = max_positions
        self.max_trades_per_day = max_trades_per_day

        self.state = RiskState()

    def can_trade(
        self,
        token_id: str,
        size: float,
        side: str,
        current_positions: Optional[Dict[str, float]] = None,
    ) -> tuple[bool, str]:
        """
        Check if a trade is allowed under risk rules.

        Returns:
            (allowed, reason)
        """
        self._check_daily_reset()

        # Check daily loss limit
        if self.state.daily_pnl <= -self.max_daily_loss:
            return False, f"Daily loss limit hit: ${abs(self.state.daily_pnl):.2f}"

        # Check trade size
        if size > self.max_position_size:
            return False, f"Position size ${size:.2f} exceeds max ${self.max_position_size}"

        # Check daily trade count
        if self.state.total_trades_today >= self.max_trades_per_day:
            return False, f"Daily trade limit reached: {self.max_trades_per_day}"

        # Check number of open positions
        positions = current_positions or self.state.open_positions
        if side == "BUY" and len(positions) >= self.max_positions:
            if token_id not in positions:
                return False, f"Max positions reached: {self.max_positions}"

        # Check if already have position in this market
        current_position = positions.get(token_id, 0)
        if side == "BUY" and current_position >= self.max_position_size:
            return False, f"Already at max position size for {token_id[:20]}"

        if side == "SELL" and current_position <= 0:
            return False, f"No position to sell for {token_id[:20]}"

        return True, "OK"

    def record_trade(
        self,
        token_id: str,
        side: str,
        size: float,
        price: float,
        pnl: Optional[float] = None,
    ) -> None:
        """Record a trade in risk tracking."""
        self._check_daily_reset()

        trade = Trade(
            token_id=token_id,
            side=side,
            size=size,
            price=price,
            timestamp=datetime.now(timezone.utc),
            pnl=pnl,
        )
        self.state.trades.append(trade)
        self.state.total_trades_today += 1

        # Update position tracking
        if side == "BUY":
            self.state.open_positions[token_id] = self.state.open_positions.get(token_id, 0) + size
        else:
            self.state.open_positions[token_id] = self.state.open_positions.get(token_id, 0) - size
            if self.state.open_positions[token_id] <= 0:
                del self.state.open_positions[token_id]

        if pnl is not None:
            self.state.daily_pnl += pnl

        logger.info(
            "Trade recorded",
            token_id=token_id[:20],
            side=side,
            size=size,
            daily_pnl=self.state.daily_pnl,
        )

    def update_position_value(self, token_id: str, value: float) -> None:
        """Update tracked position value."""
        if value <= 0.01:
            if token_id in self.state.open_positions:
                del self.state.open_positions[token_id]
        else:
            self.state.open_positions[token_id] = value

    def get_position_sizes(self) -> Dict[str, float]:
        """Get current position sizes."""
        return self.state.open_positions.copy()

    def get_daily_stats(self) -> Dict[str, any]:
        """Get daily trading statistics."""
        self._check_daily_reset()

        today_trades = [
            t for t in self.state.trades
            if t.timestamp >= self.state.last_reset
        ]

        return {
            "daily_pnl": self.state.daily_pnl,
            "trades_today": self.state.total_trades_today,
            "open_positions": len(self.state.open_positions),
            "open_exposure": sum(self.state.open_positions.values()),
            "remaining_trades": self.max_trades_per_day - self.state.total_trades_today,
            "trades": len(today_trades),
        }

    def _check_daily_reset(self) -> None:
        """Reset daily counters if it's a new day."""
        now = datetime.now(timezone.utc)
        if now.date() > self.state.last_reset.date():
            logger.info("Resetting daily risk counters", previous_date=self.state.last_reset.date())
            self.state.daily_pnl = 0.0
            self.state.total_trades_today = 0
            self.state.last_reset = now

    def get_risk_report(self) -> str:
        """Generate human-readable risk report."""
        stats = self.get_daily_stats()

        return f"""
Risk Report:
============
Daily P&L: ${stats['daily_pnl']:.2f}
Daily Loss Limit: ${self.max_daily_loss:.2f}
Trades Today: {stats['trades_today']} / {self.max_trades_per_day}
Open Positions: {stats['open_positions']}
Open Exposure: ${stats['open_exposure']:.2f}
"""


class PositionSizer:
    """Advanced position sizing algorithms."""

    def __init__(
        self,
        max_position: float = 100.0,
        risk_per_trade: float = 0.02,  # 2% of capital
    ):
        self.max_position = max_position
        self.risk_per_trade = risk_per_trade

    def size_fixed_dollar(self, capital: float, fraction: float = 0.1) -> float:
        """Fixed dollar amount sizing."""
        return min(capital * fraction, self.max_position)

    def size_kelly(
        self,
        win_prob: float,
        win_loss_ratio: float,
        capital: float,
        kelly_fraction: float = 0.25,
    ) -> float:
        """
        Kelly criterion sizing.

        f* = (p*b - q) / b
        """
        if win_loss_ratio <= 0 or win_prob <= 0:
            return 0.0

        loss_prob = 1 - win_prob
        kelly = (win_prob * win_loss_ratio - loss_prob) / win_loss_ratio

        # Fractional Kelly for safety
        position = capital * kelly * kelly_fraction

        return min(position, self.max_position)

    def size_volatility_adjusted(
        self,
        base_size: float,
        volatility: float,
        target_volatility: float = 0.02,
    ) -> float:
        """Adjust position size based on volatility."""
        if volatility <= 0:
            return base_size

        vol_adjustment = target_volatility / volatility
        return min(base_size * vol_adjustment, self.max_position)