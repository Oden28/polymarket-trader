"""
Smart trading strategy for crypto/financial Polymarket markets.

Uses real-time crypto prices + options-implied volatility to
calculate mathematically rigorous fair probabilities, then
compares to Polymarket prices to find edge.
"""
import asyncio
import structlog
from typing import Optional, List

from strategy import TradingStrategy, MarketData, Signal
from oracle import (
    DataOracle,
    parse_question,
    ParsedQuestion,
    prob_above_price,
    prob_between_prices,
    prob_up_or_down,
)

logger = structlog.get_logger()


class SmartStrategy(TradingStrategy):
    """
    Strategy that overrides `estimate_fair_probability` with
    real market data instead of naive heuristics.

    For crypto price markets: uses Black-Scholes / log-normal model
    with live prices and Deribit implied volatility.

    For unrecognized markets: falls back to the parent heuristic.
    """

    def __init__(self, oracle: DataOracle, **kwargs):
        super().__init__(**kwargs)
        self.oracle = oracle
        # Cache for oracle lookups within a single scan cycle
        self._fair_cache: dict[str, Optional[float]] = {}

    async def analyze_markets_async(self, markets: List[MarketData]) -> List[Signal]:
        """Async version of analyze_markets that fetches oracle data."""
        # Pre-warm the price cache with a single batched request
        await self._prefetch_prices(markets)

        signals: list[Signal] = []
        for m in markets:
            sig = await self.analyze_market_async(m)
            if sig is not None:
                signals.append(sig)

        signals.sort(key=lambda s: abs(s.edge_percentage), reverse=True)
        return signals

    async def analyze_market_async(self, market: MarketData) -> Optional[Signal]:
        """Async analysis with oracle data."""
        # Apply the same gate filters as parent
        if not market.is_active or market.is_closed:
            return None
        if market.liquidity < self.min_liquidity:
            return None
        if market.volume_24h < self.min_volume_24h:
            return None
        if market.spread > self.max_spread:
            return None

        # Try oracle-based fair probability
        fair_prob = await self._estimate_fair_probability_async(market)
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

        # Confidence
        confidence = self._calculate_confidence(market, edge, fair_prob)
        if confidence < self.confidence_threshold:
            return None

        # Position sizing
        if side == "BUY":
            suggested_size = self._calculate_size(
                market_price=market.current_price, fair_prob=fair_prob
            )
        else:
            suggested_size = self._calculate_size(
                market_price=1.0 - market.current_price, fair_prob=1.0 - fair_prob
            )

        reasoning = self._build_smart_reasoning(market, fair_prob, edge, side, confidence)

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

    async def _estimate_fair_probability_async(self, market: MarketData) -> Optional[float]:
        """
        Estimate fair probability using real market data.

        Returns None if we can't price this market.
        """
        # Check cache
        cache_key = market.token_id
        if cache_key in self._fair_cache:
            return self._fair_cache[cache_key]

        # Parse the question
        parsed = parse_question(
            market.question,
            end_date=market.resolution_time,
        )

        fair = None
        if parsed and parsed.asset and parsed.market_type:
            fair = await self._price_crypto_market(parsed, market)

        # Cache result
        self._fair_cache[cache_key] = fair

        if fair is not None:
            logger.debug(
                "Oracle pricing",
                market=market.market_slug[:30],
                fair=f"{fair:.4f}",
                market_price=f"{market.current_price:.4f}",
                asset=parsed.asset if parsed else "?",
            )

        return fair

    async def _price_crypto_market(
        self, parsed: ParsedQuestion, market: MarketData
    ) -> Optional[float]:
        """Calculate fair probability for a crypto price market."""

        # Get current price
        spot = await self.oracle.get_crypto_price(parsed.asset)
        if spot is None:
            return None

        # Get implied volatility
        vol = await self.oracle.get_volatility(parsed.asset)

        # Calculate hours to expiry
        if parsed.expiry_date:
            hours = self.oracle.hours_until(parsed.expiry_date)
        else:
            hours = 24.0  # default 1 day

        # Skip markets that already expired or are about to
        # (under 2 hours = likely already settled, no real edge)
        if hours < 2.0:
            return None

        if parsed.market_type == "price_target":
            if parsed.direction == "above" and parsed.target_price:
                fair = prob_above_price(spot, parsed.target_price, hours, vol)

            elif parsed.direction == "below" and parsed.target_price:
                fair = 1.0 - prob_above_price(spot, parsed.target_price, hours, vol)

            elif parsed.direction == "between" and parsed.target_price and parsed.target_price_upper:
                fair = prob_between_prices(
                    spot, parsed.target_price, parsed.target_price_upper, hours, vol
                )

            elif parsed.direction == "reach" and parsed.target_price:
                # "Will BTC reach $X?" means price touches X at any point
                # Approximate with above-at-expiry (conservative)
                if parsed.target_price > spot:
                    fair = prob_above_price(spot, parsed.target_price, hours, vol)
                else:
                    fair = 1.0 - prob_above_price(spot, parsed.target_price, hours, vol)
                # Touching probability is higher than ending-above probability
                # Apply a multiplier (barrier option approximation)
                fair = min(fair * 1.5, 0.95)

            elif parsed.direction == "dip" and parsed.target_price:
                # "Will BTC dip to $X?" means price touches X from above
                if parsed.target_price < spot:
                    # Prob of touching a lower barrier ~ 2 * prob_below_at_expiry
                    p_below = 1.0 - prob_above_price(spot, parsed.target_price, hours, vol)
                    fair = min(p_below * 1.8, 0.95)
                else:
                    fair = 0.9  # already below target

            else:
                return None

        elif parsed.market_type == "up_down":
            p_up, p_down = prob_up_or_down(spot, hours, vol)
            if parsed.direction == "up":
                fair = p_up
            else:
                fair = p_down

        else:
            return None

        # Clamp to tradeable range
        fair = max(0.01, min(0.99, fair))
        return fair

    async def _prefetch_prices(self, markets: List[MarketData]):
        """Pre-fetch prices for all crypto assets in the market list."""
        assets = set()
        for m in markets:
            parsed = parse_question(m.question, m.resolution_time)
            if parsed and parsed.asset:
                assets.add(parsed.asset)

        # Fetch all in parallel
        tasks = [self.oracle.get_crypto_price(a) for a in assets]
        await asyncio.gather(*tasks, return_exceptions=True)

        # Also prefetch volatilities
        vol_tasks = [self.oracle.get_volatility(a) for a in assets]
        await asyncio.gather(*vol_tasks, return_exceptions=True)

    @staticmethod
    def _build_smart_reasoning(
        market: MarketData,
        fair_prob: float,
        edge: float,
        side: str,
        confidence: float,
    ) -> str:
        return (
            f"{side} signal on '{market.question}' | "
            f"Oracle fair={fair_prob:.4f} vs market={market.current_price:.4f} "
            f"edge={edge:+.2f}% conf={confidence:.0%} "
            f"vol24h=${market.volume_24h:,.0f} liq=${market.liquidity:,.0f}"
        )

    # Override parent method so non-async code path still works
    def estimate_fair_probability(self, market: MarketData) -> Optional[float]:
        """Sync fallback – returns cached result or uses parent heuristic."""
        cached = self._fair_cache.get(market.token_id)
        if cached is not None:
            return cached
        # Fall back to parent heuristic for non-crypto markets
        return super().estimate_fair_probability(market)
