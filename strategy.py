"""
Trading strategy for Polymarket prediction markets.

Analyses market data and generates buy/sell signals based on
edge detection, spread analysis, and confidence scoring.
"""
import structlog
from dataclasses import dataclass
from typing import Optional, List

logger = structlog.get_logger()


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class MarketData:
    """Normalised snapshot of a single Polymarket market."""

    token_id: str
    market_slug: str
    question: str
    description: str
    current_price: float          # mid-price or last-trade price (0-1)
    best_bid: float
    best_ask: float
    spread: float
    volume_24h: float             # USDC
    liquidity: float              # USDC
    num_traders: int
    resolution_time: Optional[str]       # ISO timestamp or None
    time_to_resolution_hours: Optional[float]
    is_active: bool
    is_closed: bool
    neg_risk: bool = False
    minimum_tick_size: float = 0.01
    condition_id: str = ""
    outcome: str = "Yes"           # "Yes" or "No"


@dataclass
class Signal:
    """A trading signal produced by the strategy."""

    token_id: str
    market_slug: str
    side: str                     # "BUY" or "SELL"
    confidence: float             # 0-1
    edge_percentage: float        # estimated edge in %
    current_price: float
    fair_probability: float       # our estimated fair price
    suggested_size: float         # USDC amount
    reasoning: str
    neg_risk: bool = False
    minimum_tick_size: float = 0.01
    condition_id: str = ""


# ── Strategy ──────────────────────────────────────────────────────────────────

class TradingStrategy:
    """
    Generates trading signals from market data.

    The default implementation uses a simple mean-reversion /
    mispricing heuristic.  Swap in your own `estimate_fair_probability`
    to use a model, LLM, or external data source.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.65,
        min_edge_percentage: float = 2.0,
        max_position_size: float = 100.0,
        kelly_fraction: float = 0.25,
        min_liquidity: float = 5_000.0,
        min_volume_24h: float = 1_000.0,
        max_spread: float = 0.10,
    ):
        self.confidence_threshold = confidence_threshold
        self.min_edge_percentage = min_edge_percentage
        self.max_position_size = max_position_size
        self.kelly_fraction = kelly_fraction
        self.min_liquidity = min_liquidity
        self.min_volume_24h = min_volume_24h
        self.max_spread = max_spread

    # ── Public API ────────────────────────────────────────────────────

    def analyze_market(self, market: MarketData) -> Optional[Signal]:
        """Analyse a single market and return a Signal (or None)."""

        # Gate: skip inactive, closed, or illiquid markets
        if not market.is_active or market.is_closed:
            return None
        if market.liquidity < self.min_liquidity:
            return None
        if market.volume_24h < self.min_volume_24h:
            return None
        if market.spread > self.max_spread:
            return None

        # Estimate fair probability
        fair_prob = self.estimate_fair_probability(market)
        if fair_prob is None:
            return None

        # Calculate edge
        edge = self._calculate_edge(market.current_price, fair_prob)
        if abs(edge) < self.min_edge_percentage:
            return None

        # Determine side
        if fair_prob > market.current_price:
            side = "BUY"
        elif fair_prob < market.current_price:
            side = "SELL"
        else:
            return None

        # Confidence heuristic
        confidence = self._calculate_confidence(market, edge, fair_prob)
        if confidence < self.confidence_threshold:
            return None

        # Position sizing (fractional Kelly)
        # For BUY: we pay `current_price`, expect to win with prob `fair_prob`
        # For SELL: we pay `1 - current_price`, expect to win with prob `1 - fair_prob`
        if side == "BUY":
            suggested_size = self._calculate_size(
                market_price=market.current_price,
                fair_prob=fair_prob,
            )
        else:
            suggested_size = self._calculate_size(
                market_price=1.0 - market.current_price,
                fair_prob=1.0 - fair_prob,
            )

        reasoning = self._build_reasoning(market, fair_prob, edge, side, confidence)

        return Signal(
            token_id=market.token_id,
            market_slug=market.market_slug,
            side=side,
            confidence=confidence,
            edge_percentage=edge,
            current_price=market.current_price,
            fair_probability=fair_prob,
            suggested_size=suggested_size,
            reasoning=reasoning,
            neg_risk=market.neg_risk,
            minimum_tick_size=market.minimum_tick_size,
            condition_id=market.condition_id,
        )

    def analyze_markets(self, markets: List[MarketData]) -> List[Signal]:
        """Analyse many markets and return signals sorted by edge."""
        signals: list[Signal] = []
        for m in markets:
            sig = self.analyze_market(m)
            if sig is not None:
                signals.append(sig)
        # Best edge first
        signals.sort(key=lambda s: abs(s.edge_percentage), reverse=True)
        return signals

    # ── Probability estimation ────────────────────────────────────────

    def estimate_fair_probability(self, market: MarketData) -> Optional[float]:
        """
        Estimate the fair probability for a market.

        Default: a simple heuristic that looks for mispriced extremes.
        Override this method to plug in a model / LLM / external data.

        Returns a probability in (0, 1) or None to skip.
        """
        price = market.current_price
        if price <= 0.01 or price >= 0.99:
            return None  # too extreme to trade safely

        # Simple heuristic: mean-revert prices that are far from 0.50
        # combined with volume-weighted confidence in current price.
        #
        # Markets with high volume & liquidity ➜ current price is fair ➜ no edge
        # Markets with lower activity ➜ possible mispricing

        # Volume-adjusted "staleness" factor (higher = more mispriced)
        vol_factor = 1.0
        if market.volume_24h > 0:
            vol_factor = max(0.5, min(2.0, self.min_volume_24h / market.volume_24h))

        # Slight mean-reversion pull toward 0.50
        mean_reversion_strength = 0.05 * vol_factor
        fair = price + mean_reversion_strength * (0.50 - price)

        # Clamp
        fair = max(0.02, min(0.98, fair))
        return fair

    # ── Private helpers ───────────────────────────────────────────────

    @staticmethod
    def _calculate_edge(current_price: float, fair_prob: float) -> float:
        """Edge as a percentage."""
        if current_price <= 0:
            return 0.0
        return ((fair_prob - current_price) / current_price) * 100

    def _calculate_confidence(
        self, market: MarketData, edge: float, fair_prob: float
    ) -> float:
        """Heuristic confidence score (0-1)."""
        score = 0.0

        # Edge magnitude (up to 0.35)
        score += min(abs(edge) / 20.0, 0.35)

        # Liquidity quality (up to 0.25)
        liq_ratio = min(market.liquidity / 100_000, 1.0)
        score += liq_ratio * 0.25

        # Tight spread is good (up to 0.20)
        spread_quality = max(0, 1.0 - market.spread / self.max_spread)
        score += spread_quality * 0.20

        # Volume signal (up to 0.20)
        vol_ratio = min(market.volume_24h / 50_000, 1.0)
        score += vol_ratio * 0.20

        return min(score, 1.0)

    def _calculate_size(self, market_price: float, fair_prob: float) -> float:
        """
        Fractional-Kelly position sizing in USDC.

        For a binary market token bought at `market_price` where we believe
        the true probability is `fair_prob`:
          - Cost per share: market_price
          - Win payout: 1.0  (net gain = 1 - market_price)
          - Loss: market_price
          - b = (1 - market_price) / market_price   (payout ratio)
          - Kelly f* = (p*b - q) / b
        """
        if market_price <= 0 or market_price >= 1 or fair_prob <= 0 or fair_prob >= 1:
            return 0.0

        b = (1.0 - market_price) / market_price   # payout ratio
        if b <= 0:
            return 0.0

        p = fair_prob
        q = 1.0 - fair_prob
        kelly = (p * b - q) / b
        kelly = max(kelly, 0.0)

        size = self.max_position_size * kelly * self.kelly_fraction

        # Apply a minimum size so small-Kelly signals still get a position
        if kelly > 0 and size < 5.0:
            size = min(5.0, self.max_position_size)

        return max(min(size, self.max_position_size), 0.0)

    @staticmethod
    def _build_reasoning(
        market: MarketData,
        fair_prob: float,
        edge: float,
        side: str,
        confidence: float,
    ) -> str:
        return (
            f"{side} signal on '{market.question}' | "
            f"market={market.current_price:.3f} fair={fair_prob:.3f} "
            f"edge={edge:+.2f}% conf={confidence:.0%} "
            f"vol24h=${market.volume_24h:,.0f} liq=${market.liquidity:,.0f}"
        )
