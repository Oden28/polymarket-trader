"""
Polymarket trader – connects to the CLOB, discovers markets,
generates signals, and (optionally) executes orders.
"""
import asyncio
import httpx
import structlog
from typing import List, Optional, Dict, Any

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import (
    OrderArgs,
    MarketOrderArgs,
    OrderType,
    OpenOrderParams,
    BookParams,
)
from py_clob_client.order_builder.constants import BUY, SELL

from config import TradingConfig
from strategy import TradingStrategy, MarketData, Signal
from smart_strategy import SmartStrategy
from oracle import DataOracle
from risk_manager import RiskManager

logger = structlog.get_logger()

GAMMA_API = "https://gamma-api.polymarket.com"


class PolymarketTrader:
    """High-level trading orchestrator."""

    def __init__(self, cfg: TradingConfig):
        self.cfg = cfg

        # ── CLOB client ───────────────────────────────────────────────
        client_kwargs: Dict[str, Any] = {
            "host": cfg.poly_api_url,
            "key": cfg.poly_private_key,
            "chain_id": cfg.poly_chain_id,
        }
        if cfg.poly_signature_type:
            client_kwargs["signature_type"] = cfg.poly_signature_type
        if cfg.poly_funder_address:
            client_kwargs["funder"] = cfg.poly_funder_address

        self.client = ClobClient(**client_kwargs)

        # ── Oracle & strategy ─────────────────────────────────────────
        self.oracle = DataOracle()
        self.strategy = SmartStrategy(
            oracle=self.oracle,
            confidence_threshold=cfg.confidence_threshold,
            min_edge_percentage=cfg.min_edge_percentage,
            max_position_size=cfg.max_position_size_usdc,
            kelly_fraction=cfg.kelly_fraction,
            min_liquidity=cfg.min_liquidity,
            min_volume_24h=cfg.min_volume_24h,
            max_spread=cfg.max_spread,
        )
        self.risk = RiskManager(
            max_daily_loss=cfg.max_daily_loss_usdc,
            max_position_size=cfg.max_position_size_usdc,
            max_positions=cfg.max_positions,
            max_trades_per_day=cfg.max_trades_per_day,
        )

        self._running = False
        self._http = httpx.AsyncClient(timeout=30)

    # ── Lifecycle ─────────────────────────────────────────────────────

    async def start(self):
        """Authenticate and enter the main trading loop."""
        logger.info("Authenticating with Polymarket CLOB …")
        try:
            creds = self.client.create_or_derive_api_creds()
            self.client.set_api_creds(creds)
            logger.info("Authentication successful")
        except Exception as exc:
            logger.error("Authentication failed", error=str(exc))
            raise

        self._running = True
        logger.info(
            "Trading loop started",
            dry_run=self.cfg.dry_run,
            interval=self.cfg.update_interval_seconds,
        )

        while self._running:
            try:
                signals = await self.run_once()
                if signals and not self.cfg.dry_run:
                    await self._execute_signals(signals)
                elif signals:
                    logger.info(
                        "DRY RUN – skipping execution",
                        signal_count=len(signals),
                    )
            except Exception as exc:
                logger.error("Error in trading loop", error=str(exc))

            await asyncio.sleep(self.cfg.update_interval_seconds)

    async def stop(self):
        self._running = False
        await self._http.aclose()
        await self.oracle.close()
        logger.info("Trader stopped")

    # ── Core loop ─────────────────────────────────────────────────────

    async def run_once(self) -> List[Signal]:
        """Fetch markets → analyse → return signals."""
        markets = await self.fetch_markets()
        logger.info("Fetched markets", count=len(markets))

        enriched = await self._enrich_with_orderbook(markets)

        # Use async smart strategy if available
        if isinstance(self.strategy, SmartStrategy):
            signals = await self.strategy.analyze_markets_async(enriched)
        else:
            signals = self.strategy.analyze_markets(enriched)
        logger.info("Generated signals", count=len(signals))

        for sig in signals[:5]:
            logger.info(
                "Signal",
                market=sig.market_slug,
                side=sig.side,
                edge=f"{sig.edge_percentage:.2f}%",
                confidence=f"{sig.confidence:.0%}",
                size=f"${sig.suggested_size:.2f}",
            )
        return signals

    # ── Market discovery (Gamma API) ──────────────────────────────────

    async def fetch_markets(self) -> List[MarketData]:
        """Fetch active markets from the Gamma API and normalise."""
        markets: list[MarketData] = []
        offset = 0
        limit = 100
        max_pages = 5  # cap to avoid rate limits

        for _ in range(max_pages):
            url = (
                f"{self.cfg.gamma_api_url}/markets"
                f"?active=true&closed=false&limit={limit}&offset={offset}"
                f"&order=volume24hr&ascending=false"
            )
            try:
                resp = await self._http.get(url)
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                logger.warning("Gamma API request failed", error=str(exc))
                break

            # Gamma returns a list directly (not paginated wrapper)
            items = data if isinstance(data, list) else data.get("data", data)
            if not items:
                break

            for item in items:
                md = self._parse_gamma_market(item)
                if md is not None:
                    markets.append(md)

            if len(items) < limit:
                break
            offset += limit

        return markets

    def _parse_gamma_market(self, raw: dict) -> Optional[MarketData]:
        """Convert a Gamma API market dict into a MarketData."""
        try:
            clob_ids = raw.get("clobTokenIds") or raw.get("clob_token_ids")
            if not clob_ids or not isinstance(clob_ids, list):
                # Try string-encoded JSON list
                import json
                clob_ids = json.loads(raw.get("clobTokenIds", "[]"))
            if not clob_ids:
                return None

            # Use the YES token (first element)
            token_id = clob_ids[0]

            outcomes = raw.get("outcomes") or raw.get("outcomePrices") or "[]"
            if isinstance(outcomes, str):
                import json
                outcomes = json.loads(outcomes)

            # Parse price – prefer lastTradePrice, fall back to outcomePrices
            last_trade = raw.get("lastTradePrice")
            outcome_prices = raw.get("outcomePrices")
            if last_trade is not None:
                price = float(last_trade)
            elif isinstance(outcome_prices, list) and outcome_prices:
                price = float(outcome_prices[0])
            elif isinstance(outcome_prices, str):
                import json as _json
                try:
                    price = float(_json.loads(outcome_prices)[0])
                except (json.JSONDecodeError, IndexError):
                    price = 0.5
            else:
                price = 0.5
            if price <= 0 or price >= 1:
                price = 0.5

            best_bid = float(raw.get("bestBid", price - 0.01))
            best_ask = float(raw.get("bestAsk", price + 0.01))
            spread = max(best_ask - best_bid, 0.0)

            vol_24h = float(raw.get("volume24hr") or raw.get("volume_24hr") or 0)
            liquidity = float(raw.get("liquidity") or 0)
            num_traders = int(raw.get("numTraders") or raw.get("uniqueTraders") or 0)

            neg_risk = bool(raw.get("negRisk") or raw.get("neg_risk", False))
            min_tick = float(raw.get("minimumTickSize") or raw.get("minimum_tick_size") or 0.01)

            return MarketData(
                token_id=token_id,
                market_slug=raw.get("slug") or raw.get("market_slug", ""),
                question=raw.get("question", ""),
                description=raw.get("description", ""),
                current_price=price,
                best_bid=best_bid,
                best_ask=best_ask,
                spread=spread,
                volume_24h=vol_24h,
                liquidity=liquidity,
                num_traders=num_traders,
                resolution_time=raw.get("endDate") or raw.get("end_date_iso"),
                time_to_resolution_hours=None,
                is_active=raw.get("active", True),
                is_closed=raw.get("closed", False),
                neg_risk=neg_risk,
                minimum_tick_size=min_tick,
                condition_id=raw.get("conditionId") or raw.get("condition_id", ""),
                outcome="Yes",
            )
        except Exception as exc:
            logger.debug("Failed to parse market", error=str(exc))
            return None

    # ── Orderbook enrichment (CLOB public endpoints) ──────────────────

    async def _enrich_with_orderbook(
        self, markets: List[MarketData]
    ) -> List[MarketData]:
        """Fetch orderbook data from the CLOB for each market (batched)."""
        enriched: list[MarketData] = []
        batch_size = 10

        for i in range(0, len(markets), batch_size):
            batch = markets[i : i + batch_size]
            tasks = [self._fetch_orderbook(m) for m in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for market, result in zip(batch, results):
                if isinstance(result, Exception):
                    enriched.append(market)  # keep original data
                else:
                    enriched.append(result)

        return enriched

    async def _fetch_orderbook(self, market: MarketData) -> MarketData:
        """Fetch orderbook for a single token and update spread/bid/ask."""
        try:
            url = f"{self.cfg.poly_api_url}/book?token_id={market.token_id}"
            resp = await self._http.get(url)
            resp.raise_for_status()
            book = resp.json()

            bids = book.get("bids", [])
            asks = book.get("asks", [])

            # CLOB returns bids and asks sorted ascending by price.
            # Best bid = highest bid price, best ask = lowest ask price.
            if bids:
                best_bid = max(float(b["price"]) for b in bids)
                market.best_bid = best_bid
            if asks:
                best_ask = min(float(a["price"]) for a in asks)
                market.best_ask = best_ask

            if bids and asks:
                market.spread = max(market.best_ask - market.best_bid, 0.0)
                market.current_price = (market.best_bid + market.best_ask) / 2

            # Also use last_trade_price if available
            ltp = book.get("last_trade_price")
            if ltp is not None:
                market.current_price = float(ltp)

        except Exception as exc:
            logger.debug(
                "Orderbook fetch failed",
                token=market.token_id[:20],
                error=str(exc),
            )

        return market

    # ── Order execution ───────────────────────────────────────────────

    async def _execute_signals(self, signals: List[Signal]):
        """Execute the top signals through the CLOB."""
        for signal in signals:
            # Risk check
            can_trade, reason = self.risk.can_trade(
                token_id=signal.token_id,
                size=signal.suggested_size,
                side=signal.side,
            )
            if not can_trade:
                logger.info("Risk rejected", reason=reason, market=signal.market_slug)
                continue

            try:
                await self._place_order(signal)
            except Exception as exc:
                logger.error(
                    "Order failed",
                    market=signal.market_slug,
                    error=str(exc),
                )

    async def _place_order(self, signal: Signal):
        """Place a limit order via the CLOB client."""
        side_const = BUY if signal.side == "BUY" else SELL

        # Round price to tick size
        tick = signal.minimum_tick_size or 0.01
        price = round(signal.fair_probability / tick) * tick
        price = max(tick, min(price, 1.0 - tick))

        # Size in number of shares = USDC / price
        if price <= 0:
            return
        size = round(signal.suggested_size / price, 2)
        if size < 1:
            return

        logger.info(
            "Placing order",
            market=signal.market_slug,
            side=signal.side,
            price=price,
            size=size,
        )

        order_args = OrderArgs(
            token_id=signal.token_id,
            price=price,
            size=size,
            side=side_const,
        )

        # create_order and post_order are synchronous in py-clob-client
        signed_order = self.client.create_order(order_args)
        resp = self.client.post_order(signed_order, OrderType.GTC)

        logger.info("Order response", response=resp, market=signal.market_slug)

        # Record with risk manager
        self.risk.record_trade(
            token_id=signal.token_id,
            side=signal.side,
            size=signal.suggested_size,
            price=price,
        )

    # ── Portfolio helpers ─────────────────────────────────────────────

    def get_open_orders(self) -> list:
        """Return current open orders."""
        try:
            return self.client.get_orders(OpenOrderParams())
        except Exception as exc:
            logger.warning("Failed to fetch open orders", error=str(exc))
            return []

    def cancel_all_orders(self):
        """Cancel every open order."""
        try:
            self.client.cancel_all()
            logger.info("All orders cancelled")
        except Exception as exc:
            logger.error("Cancel all failed", error=str(exc))
