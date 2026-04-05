"""
External data oracle for Polymarket strategy.

Pulls real-time crypto prices, volatility, and financial data
to calculate fair probabilities for prediction markets.
"""
import re
import math
import asyncio
import structlog
import httpx
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple
from functools import lru_cache

logger = structlog.get_logger()

# ── Probability math ──────────────────────────────────────────────────────────

def prob_above_price(
    current_price: float,
    target_price: float,
    hours_to_expiry: float,
    annual_volatility: float,
    drift: float = 0.0,
) -> float:
    """
    Probability that a GBM asset ends above `target_price` at expiry.

    Uses log-normal model:
        P(S_T > K) = N(d2)
    where
        d2 = [ln(S/K) + (mu - sigma^2/2)*T] / (sigma * sqrt(T))

    This is the same model used by options traders (Black-Scholes).
    """
    if current_price <= 0 or target_price <= 0 or hours_to_expiry <= 0:
        return 0.5
    if annual_volatility <= 0:
        # No vol → deterministic: either above or below
        return 1.0 if current_price > target_price else 0.0

    T = hours_to_expiry / 8760  # hours → years
    sigma = annual_volatility
    mu = drift  # annualized drift (set to 0 for risk-neutral)

    d2 = (math.log(current_price / target_price) + (mu - 0.5 * sigma**2) * T) / (
        sigma * math.sqrt(T)
    )
    return _norm_cdf(d2)


def prob_between_prices(
    current_price: float,
    lower: float,
    upper: float,
    hours_to_expiry: float,
    annual_volatility: float,
    drift: float = 0.0,
) -> float:
    """Probability that asset price ends between lower and upper."""
    p_above_lower = prob_above_price(current_price, lower, hours_to_expiry, annual_volatility, drift)
    p_above_upper = prob_above_price(current_price, upper, hours_to_expiry, annual_volatility, drift)
    return max(p_above_lower - p_above_upper, 0.0)


def prob_up_or_down(
    current_price: float,
    hours_to_expiry: float,
    annual_volatility: float,
) -> Tuple[float, float]:
    """Probability of price being UP vs DOWN from current at expiry."""
    # "Up" means S_T > S_0, i.e. target = current_price
    p_up = prob_above_price(current_price, current_price, hours_to_expiry, annual_volatility)
    return p_up, 1 - p_up


def _norm_cdf(x: float) -> float:
    """Standard normal CDF (no scipy dependency)."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# ── Market question parser ────────────────────────────────────────────────────

@dataclass
class ParsedQuestion:
    """Structured info extracted from a Polymarket question."""
    asset: Optional[str] = None          # "bitcoin", "ethereum", etc.
    target_price: Optional[float] = None
    target_price_upper: Optional[float] = None  # for "between X and Y"
    direction: Optional[str] = None      # "above", "below", "between", "up", "down", "reach", "dip"
    expiry_date: Optional[str] = None    # ISO date string
    market_type: Optional[str] = None    # "price_target", "up_down", "reach", "fed_rate", etc.


# Map of asset keywords → CoinGecko IDs
ASSET_MAP = {
    "bitcoin": "bitcoin", "btc": "bitcoin",
    "ethereum": "ethereum", "eth": "ethereum",
    "solana": "solana", "sol": "solana",
    "xrp": "ripple", "ripple": "ripple",
    "dogecoin": "dogecoin", "doge": "dogecoin",
    "cardano": "cardano", "ada": "cardano",
    "polygon": "matic-network", "matic": "matic-network",
    "avalanche": "avalanche-2", "avax": "avalanche-2",
    "chainlink": "chainlink", "link": "chainlink",
    "litecoin": "litecoin", "ltc": "litecoin",
}

MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9,
    "oct": 10, "nov": 11, "dec": 12,
}


def parse_question(question: str, end_date: Optional[str] = None) -> Optional[ParsedQuestion]:
    """
    Parse a Polymarket question to extract trading-relevant structure.

    Returns None if the question isn't a crypto/financial price market.
    """
    q = question.lower().strip()
    result = ParsedQuestion()

    # Detect asset
    for keyword, gecko_id in ASSET_MAP.items():
        if keyword in q:
            result.asset = gecko_id
            break

    if not result.asset:
        return None  # not a crypto market we can price

    # Detect "above $X" pattern
    above_match = re.search(r"above\s+\$?([\d,]+(?:\.\d+)?)", q)
    below_match = re.search(r"below\s+\$?([\d,]+(?:\.\d+)?)", q)
    between_match = re.search(r"between\s+\$?([\d,]+(?:\.\d+)?)\s+and\s+\$?([\d,]+(?:\.\d+)?)", q)
    reach_match = re.search(r"reach\s+\$?([\d,]+(?:\.\d+)?)", q)
    dip_match = re.search(r"dip\s+to\s+\$?([\d,]+(?:\.\d+)?)", q)
    up_down_match = re.search(r"(up|down)\s+on", q)

    if between_match:
        result.target_price = float(between_match.group(1).replace(",", ""))
        result.target_price_upper = float(between_match.group(2).replace(",", ""))
        result.direction = "between"
        result.market_type = "price_target"
    elif above_match:
        result.target_price = float(above_match.group(1).replace(",", ""))
        result.direction = "above"
        result.market_type = "price_target"
    elif below_match:
        result.target_price = float(below_match.group(1).replace(",", ""))
        result.direction = "below"
        result.market_type = "price_target"
    elif reach_match:
        result.target_price = float(reach_match.group(1).replace(",", ""))
        result.direction = "reach"
        result.market_type = "price_target"
    elif dip_match:
        result.target_price = float(dip_match.group(1).replace(",", ""))
        result.direction = "dip"
        result.market_type = "price_target"
    elif up_down_match:
        result.direction = up_down_match.group(1)
        result.market_type = "up_down"
    else:
        return None  # can't parse the market type

    # Detect date
    if end_date:
        result.expiry_date = end_date[:10]  # ISO date portion
    else:
        # Try parsing "on April 5" or "in April" from question
        date_match = re.search(
            r"(?:on|by|in)\s+(january|february|march|april|may|june|july|august|"
            r"september|october|november|december|jan|feb|mar|apr|jun|jul|aug|"
            r"sep|oct|nov|dec)\s*(\d{1,2})?(?:\s*,?\s*(\d{4}))?", q
        )
        if date_match:
            month_str = date_match.group(1)
            day = int(date_match.group(2)) if date_match.group(2) else 28
            year = int(date_match.group(3)) if date_match.group(3) else datetime.now(timezone.utc).year
            month = MONTH_MAP.get(month_str, 1)
            result.expiry_date = f"{year}-{month:02d}-{day:02d}"

    return result


# ── Data fetcher ──────────────────────────────────────────────────────────────

class DataOracle:
    """Fetches real-time market data from free public APIs."""

    def __init__(self):
        self._http = httpx.AsyncClient(timeout=15)
        self._price_cache: Dict[str, Tuple[float, datetime]] = {}
        self._vol_cache: Dict[str, Tuple[float, datetime]] = {}
        self._cache_ttl = timedelta(seconds=30)

    async def close(self):
        await self._http.aclose()

    async def get_crypto_price(self, gecko_id: str) -> Optional[float]:
        """Get current USD price from CoinGecko (free, no key)."""
        # Check cache
        cached = self._price_cache.get(gecko_id)
        if cached and datetime.now(timezone.utc) - cached[1] < self._cache_ttl:
            return cached[0]

        try:
            resp = await self._http.get(
                f"https://api.coingecko.com/api/v3/simple/price",
                params={"ids": gecko_id, "vs_currencies": "usd"},
            )
            resp.raise_for_status()
            data = resp.json()
            price = data.get(gecko_id, {}).get("usd")
            if price is not None:
                self._price_cache[gecko_id] = (float(price), datetime.now(timezone.utc))
                return float(price)
        except Exception as exc:
            logger.debug("CoinGecko price fetch failed", asset=gecko_id, error=str(exc))

        # Fallback: Deribit index (BTC/ETH only)
        if gecko_id in ("bitcoin", "ethereum"):
            return await self._get_deribit_price(gecko_id)
        return None

    async def _get_deribit_price(self, gecko_id: str) -> Optional[float]:
        """Fallback price from Deribit."""
        index_name = "btc_usd" if gecko_id == "bitcoin" else "eth_usd"
        try:
            resp = await self._http.get(
                f"https://www.deribit.com/api/v2/public/get_index_price",
                params={"index_name": index_name},
            )
            resp.raise_for_status()
            price = resp.json().get("result", {}).get("index_price")
            if price:
                self._price_cache[gecko_id] = (float(price), datetime.now(timezone.utc))
                return float(price)
        except Exception as exc:
            logger.debug("Deribit price fetch failed", error=str(exc))
        return None

    async def get_btc_volatility(self) -> float:
        """Get BTC 30-day implied volatility from Deribit DVOL (annualized)."""
        cached = self._vol_cache.get("btc_dvol")
        if cached and datetime.now(timezone.utc) - cached[1] < timedelta(minutes=10):
            return cached[0]

        try:
            now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            start_ms = now_ms - 86400 * 1000  # last 24h
            resp = await self._http.get(
                "https://www.deribit.com/api/v2/public/get_volatility_index_data",
                params={
                    "currency": "BTC",
                    "resolution": "3600",
                    "start_timestamp": start_ms,
                    "end_timestamp": now_ms,
                },
            )
            resp.raise_for_status()
            data_points = resp.json().get("result", {}).get("data", [])
            if data_points:
                dvol = data_points[-1][4]  # close value
                vol = dvol / 100  # convert from percentage to decimal
                self._vol_cache["btc_dvol"] = (vol, datetime.now(timezone.utc))
                return vol
        except Exception as exc:
            logger.debug("Deribit DVOL fetch failed", error=str(exc))

        # Fallback: historical average
        return 0.50  # ~50% annualized is typical for BTC

    async def get_eth_volatility(self) -> float:
        """Get ETH 30-day implied volatility from Deribit DVOL."""
        cached = self._vol_cache.get("eth_dvol")
        if cached and datetime.now(timezone.utc) - cached[1] < timedelta(minutes=10):
            return cached[0]

        try:
            now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            start_ms = now_ms - 86400 * 1000
            resp = await self._http.get(
                "https://www.deribit.com/api/v2/public/get_volatility_index_data",
                params={
                    "currency": "ETH",
                    "resolution": "3600",
                    "start_timestamp": start_ms,
                    "end_timestamp": now_ms,
                },
            )
            resp.raise_for_status()
            data_points = resp.json().get("result", {}).get("data", [])
            if data_points:
                dvol = data_points[-1][4]
                vol = dvol / 100
                self._vol_cache["eth_dvol"] = (vol, datetime.now(timezone.utc))
                return vol
        except Exception as exc:
            logger.debug("Deribit ETH DVOL fetch failed", error=str(exc))
        return 0.60  # ETH typically ~60%

    async def get_volatility(self, gecko_id: str) -> float:
        """Get implied volatility for an asset (falls back to BTC DVOL scaled)."""
        if gecko_id == "bitcoin":
            return await self.get_btc_volatility()
        elif gecko_id == "ethereum":
            return await self.get_eth_volatility()
        else:
            # Altcoins: scale BTC vol by a multiplier
            btc_vol = await self.get_btc_volatility()
            ALT_VOL_MULTIPLIER = {
                "solana": 1.4, "ripple": 1.3, "dogecoin": 1.6,
                "cardano": 1.4, "avalanche-2": 1.5, "chainlink": 1.4,
                "matic-network": 1.4, "litecoin": 1.2,
            }
            mult = ALT_VOL_MULTIPLIER.get(gecko_id, 1.5)
            return btc_vol * mult

    def hours_until(self, date_str: str) -> float:
        """Hours from now until a date (end of day UTC)."""
        try:
            target = datetime.strptime(date_str, "%Y-%m-%d").replace(
                hour=23, minute=59, tzinfo=timezone.utc
            )
            now = datetime.now(timezone.utc)
            diff = (target - now).total_seconds() / 3600
            return max(diff, 0.01)  # at least a tiny fraction
        except:
            return 24.0  # default 1 day
