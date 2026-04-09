"""
Local web dashboard: wallet (L2 CLOB auth), 5-minute Up/Down markets, open orders.

Authentication follows https://docs.polymarket.com/api-reference/authentication
via py_clob_client (L1 EIP-712 derive → L2 HMAC headers).

Run: python main.py dashboard
"""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
import structlog
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from pathlib import Path

load_dotenv(Path(__file__).parent / ".env", encoding="utf-8-sig")

from config import TradingConfig

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import (
    AssetType,
    BalanceAllowanceParams,
    OpenOrderParams,
    OrderArgs,
    OrderType,
)
from py_clob_client.order_builder.constants import BUY, SELL

from calibration_store import (
    get_calibration_stats,
    init_calibration_db,
    record_signal_snapshot,
    try_resolve_pending_windows,
)
from signal_engine import annualized_vol_from_minute_closes, closes_from_bootstrap_points, compute_updown_signal

logger = structlog.get_logger()

FIVE_MIN = 300


def _dashboard_order_post_enabled() -> bool:
    """Allow placing/canceling orders from the dashboard UI (env toggle, safety)."""
    v = (os.getenv("DASHBOARD_ENABLE_ORDER_POST") or "").strip().lower()
    return v in ("true", "1", "yes", "on")


UPDOWN_ASSETS = ("btc", "eth", "sol", "xrp")

# Binance: historical 1m candles only (shape of the past window). Polymarket settles on Chainlink, not Binance.
BINANCE_SYMBOL: dict[str, str] = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "XRP": "XRPUSDT",
}
# Same assets on Polymarket RTDS — live line should match polymarket.com more closely than Binance trades.
CHAINLINK_SYMBOL: dict[str, str] = {
    "BTC": "btc/usd",
    "ETH": "eth/usd",
    "SOL": "sol/usd",
    "XRP": "xrp/usd",
}
# Polymarket Next.js loads `eventMetadata.priceToBeat` from this endpoint (same value as the website).
POLYMARKET_SERIES_SLUG: dict[str, str] = {
    "BTC": "btc-up-or-down-5m",
    "ETH": "eth-up-or-down-5m",
    "SOL": "sol-up-or-down-5m",
    "XRP": "xrp-up-or-down-5m",
}
POLYMARKET_USER_AGENT = "Mozilla/5.0 (compatible; polymarket-trader/1.0; +https://github.com/)"

app = Flask(__name__)
_cfg: Optional[TradingConfig] = None
_http: Optional[httpx.Client] = None
_clob: Optional[ClobClient] = None
_clob_lock = threading.Lock()
_clob_init_error: Optional[str] = None
_series_cache_lock = threading.Lock()
_series_events_cache: dict[str, tuple[float, list]] = {}


def _cfg_load() -> TradingConfig:
    global _cfg
    if _cfg is None:
        _cfg = TradingConfig()
    return _cfg


def _http_client() -> httpx.Client:
    global _http
    if _http is None:
        _http = httpx.Client(timeout=20.0)
    return _http


def _clob_client() -> Optional[ClobClient]:
    """Authenticated CLOB client (L1+L2); None if no private key or auth failed."""
    global _clob, _clob_init_error
    cfg = _cfg_load()
    if not cfg.poly_private_key or len(cfg.poly_private_key) < 10:
        return None
    with _clob_lock:
        if _clob is None:
            try:
                kwargs: dict[str, Any] = {
                    "host": cfg.poly_api_url,
                    "key": cfg.poly_private_key,
                    "chain_id": cfg.poly_chain_id,
                }
                if cfg.poly_signature_type is not None:
                    kwargs["signature_type"] = cfg.poly_signature_type
                if cfg.poly_funder_address:
                    kwargs["funder"] = cfg.poly_funder_address
                client = ClobClient(**kwargs)
                creds = client.create_or_derive_api_creds()
                client.set_api_creds(creds)
                _clob = client
                _clob_init_error = None
                logger.info("CLOB L2 authentication ready", address=client.get_address())
            except Exception as exc:
                _clob_init_error = str(exc)
                logger.warning("CLOB authentication failed", error=_clob_init_error)
                return None
        return _clob


def current_window_ts(t: Optional[int] = None) -> int:
    now = t if t is not None else int(time.time())
    return (now // FIVE_MIN) * FIVE_MIN


def _parse_json_field(val: Any) -> Any:
    if val is None:
        return None
    if isinstance(val, str):
        try:
            return json.loads(val)
        except json.JSONDecodeError:
            return val
    return val


def _parse_iso_to_ms(iso: Optional[str]) -> Optional[int]:
    if not iso:
        return None
    s = str(iso).replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _polymarket_series_events(series_slug: str) -> list:
    """Cached fetch of series events (large JSON); includes eventMetadata.priceToBeat when set."""
    now = time.time()
    with _series_cache_lock:
        ent = _series_events_cache.get(series_slug)
        if ent and now - ent[0] < 10.0:
            return ent[1]
    url = f"https://polymarket.com/api/series?slug={series_slug}&withMetadata=true"
    r = _http_client().get(url, headers={"User-Agent": POLYMARKET_USER_AGENT})
    r.raise_for_status()
    data = r.json()
    events = data.get("events") or []
    with _series_cache_lock:
        _series_events_cache[series_slug] = (now, events)
    return events


def polymarket_official_price_to_beat(series_slug: str, event_slug: str) -> Optional[float]:
    """Exact 'Price to beat' from polymarket.com when the API has populated eventMetadata."""
    try:
        for e in _polymarket_series_events(series_slug):
            if e.get("slug") != event_slug:
                continue
            em = e.get("eventMetadata") or {}
            raw = em.get("priceToBeat")
            if raw is None:
                return None
            return float(raw)
    except Exception as exc:
        logger.warning("polymarket_price_to_beat_failed", error=str(exc))
    return None


def _binance_klines_usdt(symbol: str, start_ms: int, end_ms: int) -> list[list[Any]]:
    r = _http_client().get(
        "https://api.binance.com/api/v3/klines",
        params={
            "symbol": symbol,
            "interval": "1m",
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": 30,
        },
    )
    r.raise_for_status()
    data = r.json()
    return data if isinstance(data, list) else []


def _resolution_close_binance(asset: str, window_ts: int) -> tuple[Optional[float], str]:
    """
    Binance 1m candle close with closeTime <= window end (USDT).
    Proxy for whether spot finished above strike — Polymarket may use Chainlink at a different timestamp.
    """
    sym = BINANCE_SYMBOL.get(asset.upper())
    if not sym:
        return None, "no_symbol"
    end_ms = (window_ts + FIVE_MIN) * 1000
    try:
        r = _http_client().get(
            "https://api.binance.com/api/v3/klines",
            params={
                "symbol": sym,
                "interval": "1m",
                "endTime": end_ms,
                "limit": 15,
            },
        )
        r.raise_for_status()
        kl = r.json()
    except Exception as exc:
        return None, str(exc)[:120]
    if not isinstance(kl, list) or not kl:
        return None, "empty_klines"
    best_ct = -1
    best_close: Optional[float] = None
    for row in kl:
        if not isinstance(row, (list, tuple)) or len(row) < 7:
            continue
        try:
            ct = int(row[6])
            cl = float(row[4])
        except (TypeError, ValueError, IndexError):
            continue
        if ct <= end_ms and ct > best_ct:
            best_ct = ct
            best_close = cl
    if best_close is not None:
        return best_close, "binance_1m_close"
    return None, "no_close_before_end"


def chart_bootstrap(asset: str, window_ts: int) -> dict[str, Any]:
    """
    Target line: Polymarket ``eventMetadata.priceToBeat`` when present (same as the website).
    For the *active* window that field is often null until later; the browser latches the first
    Chainlink RTDS tick at/after ``eventStartTime`` (see dashboard.html).

    Chart history remains Binance 1m closes (visual only).
    """
    asset_u = asset.upper()
    slug = f"{asset.lower()}-updown-5m-{window_ts}"
    sym = BINANCE_SYMBOL.get(asset_u)
    cl_sym = CHAINLINK_SYMBOL.get(asset_u)
    series_slug = POLYMARKET_SERIES_SLUG.get(asset_u)
    out: dict[str, Any] = {
        "ok": False,
        "asset": asset_u,
        "slug": slug,
        "binance_symbol": sym,
        "chainlink_symbol": cl_sym,
        "points": [],
        "strike": None,
        "strike_polymarket_official": None,
        "strike_binance_fallback": None,
        "strike_source": "none",
        "event_start_ms": None,
        "window_end_ms": None,
        "error": None,
    }
    if not sym:
        out["error"] = "unsupported asset"
        return out
    ev = fetch_gamma_event_by_slug(slug)
    if not ev:
        out["error"] = "gamma event not found"
        return out
    markets = ev.get("markets") or []
    if not markets:
        out["error"] = "no markets on event"
        return out
    m = markets[0]
    event_start_ms = _parse_iso_to_ms(m.get("eventStartTime"))
    window_end_ms = _parse_iso_to_ms(m.get("endDate"))
    if window_end_ms is None:
        window_end_ms = (window_ts + FIVE_MIN) * 1000
    if event_start_ms is None:
        event_start_ms = window_ts * 1000

    out["event_start_ms"] = event_start_ms
    out["window_end_ms"] = window_end_ms
    out["window_ts"] = window_ts
    out["title"] = ev.get("title") or m.get("question")

    now_ms = int(time.time() * 1000)
    end_ms = min(now_ms, window_end_ms)

    try:
        kl = _binance_klines_usdt(sym, event_start_ms, end_ms)
    except Exception as exc:
        out["error"] = str(exc)
        return out

    points: list[dict[str, Any]] = []
    strike: Optional[float] = None
    for row in kl:
        if not isinstance(row, (list, tuple)) or len(row) < 7:
            continue
        open_t = int(row[0])
        o = float(row[1])
        c = float(row[4])
        if strike is None and open_t <= event_start_ms < open_t + 60_000:
            strike = o
        points.append({"t": open_t + 60_000, "p": c})

    if strike is None and kl:
        try:
            strike = float(kl[0][1])
        except (TypeError, ValueError, IndexError):
            strike = None

    official: Optional[float] = None
    if series_slug:
        official = polymarket_official_price_to_beat(series_slug, slug)

    display_strike: Optional[float] = official if official is not None else strike
    if display_strike is not None:
        points.insert(0, {"t": event_start_ms, "p": display_strike})

    out["strike_polymarket_official"] = official
    out["strike_binance_fallback"] = strike
    out["strike"] = display_strike
    if official is not None:
        out["strike_source"] = "polymarket_api"
    elif strike is not None:
        out["strike_source"] = "binance_1m_open"
    else:
        out["strike_source"] = "none"
    out["points"] = points
    out["ok"] = True
    return out


def fetch_gamma_event_by_slug(slug: str) -> Optional[dict]:
    cfg = _cfg_load()
    url = f"{cfg.gamma_api_url}/events"
    r = _http_client().get(url, params={"slug": slug})
    if r.status_code != 200:
        return None
    data = r.json()
    if not data:
        return None
    return data[0] if isinstance(data, list) else data


def fetch_book_mid(token_id: str) -> dict[str, Any]:
    cfg = _cfg_load()
    url = f"{cfg.poly_api_url}/book"
    out: dict[str, Any] = {
        "best_bid": None,
        "best_ask": None,
        "mid": None,
        "last_trade_price": None,
    }
    try:
        r = _http_client().get(url, params={"token_id": token_id})
        r.raise_for_status()
        book = r.json()
        bids = book.get("bids") or []
        asks = book.get("asks") or []
        if bids:
            out["best_bid"] = max(float(b["price"]) for b in bids)
        if asks:
            out["best_ask"] = min(float(a["price"]) for a in asks)
        if out["best_bid"] is not None and out["best_ask"] is not None:
            out["mid"] = (out["best_bid"] + out["best_ask"]) / 2
        ltp = book.get("last_trade_price")
        if ltp is not None:
            out["last_trade_price"] = float(ltp)
    except Exception as exc:
        out["error"] = str(exc)
    return out


def build_updown_snapshot(window_ts: int) -> dict[str, Any]:
    """One 5-minute window across configured crypto assets."""
    rows = []
    for asset in UPDOWN_ASSETS:
        slug = f"{asset}-updown-5m-{window_ts}"
        ev = fetch_gamma_event_by_slug(slug)
        if not ev:
            rows.append(
                {
                    "asset": asset.upper(),
                    "slug": slug,
                    "ok": False,
                    "error": "event not found",
                }
            )
            continue
        markets = ev.get("markets") or []
        if not markets:
            rows.append({"asset": asset.upper(), "slug": slug, "ok": False, "error": "no markets"})
            continue
        m = markets[0]
        tokens_raw = _parse_json_field(m.get("clobTokenIds") or m.get("clob_token_ids"))
        outcomes_raw = _parse_json_field(m.get("outcomes"))
        if not tokens_raw or not isinstance(tokens_raw, list) or len(tokens_raw) < 2:
            rows.append({"asset": asset.upper(), "slug": slug, "ok": False, "error": "bad tokens"})
            continue
        outcomes = outcomes_raw if isinstance(outcomes_raw, list) else ["Up", "Down"]
        sides = []
        for i, label in enumerate(outcomes[:2]):
            tid = str(tokens_raw[i])
            book = fetch_book_mid(tid)
            sides.append(
                {
                    "outcome": label,
                    "token_id": tid,
                    "best_bid": book.get("best_bid"),
                    "best_ask": book.get("best_ask"),
                    "mid": book.get("mid"),
                    "last_trade_price": book.get("last_trade_price"),
                    "book_error": book.get("error"),
                }
            )
        order_min: Optional[float] = None
        oms = m.get("orderMinSize")
        if oms is not None:
            try:
                order_min = float(oms)
            except (TypeError, ValueError):
                order_min = None
        rows.append(
            {
                "asset": asset.upper(),
                "slug": slug,
                "ok": True,
                "title": ev.get("title") or m.get("question"),
                "question": m.get("question"),
                "event_url": f"https://polymarket.com/event/{slug}",
                "window_start": window_ts,
                "window_end": window_ts + FIVE_MIN,
                "event_start_time": m.get("eventStartTime"),
                "event_end_time": m.get("endDate"),
                "order_min_size": order_min,
                "sides": sides,
            }
        )
    return {
        "window_ts": window_ts,
        "window_end": window_ts + FIVE_MIN,
        "server_time": int(time.time()),
        "rows": rows,
    }


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/me")
def api_me():
    cfg = _cfg_load()
    body: dict[str, Any] = {
        "configured": cfg.is_configured,
        "needs_funder": cfg.needs_funder,
        "funder_set": bool(cfg.poly_funder_address),
        "signature_type": cfg.poly_signature_type,
        "dry_run_default": cfg.dry_run,
        "trading_enabled": _dashboard_order_post_enabled(),
    }
    client = _clob_client()
    if not client:
        body["authenticated"] = False
        if not cfg.is_configured:
            body["error"] = (
                "No private key in .env — set POLY_PRIVATE_KEY (or PRIVATE_KEY) in the "
                "project root .env and restart: python main.py dashboard"
            )
            if (os.getenv("PRIVATE_KEY") or "").strip() and not (
                os.getenv("POLY_PRIVATE_KEY") or ""
            ).strip():
                body["hint"] = (
                    "PRIVATE_KEY is set but POLY_PRIVATE_KEY is empty; the app now accepts "
                    "either. Restart the dashboard after saving .env."
                )
        else:
            body["error"] = (
                _clob_init_error
                or "CLOB client failed to initialize (check key, signature type, funder)."
            )
            if cfg.needs_funder and not cfg.poly_funder_address:
                body["hint"] = (
                    "POLY_SIGNATURE_TYPE is 1 or 2 — set POLY_FUNDER_ADDRESS to your Polymarket "
                    "proxy/deposit address (see README)."
                )
        return jsonify(body)
    body["authenticated"] = True
    body["signer_address"] = client.get_address()
    body["funder_address"] = cfg.poly_funder_address or None
    body["wallet_warnings"] = []
    if cfg.needs_funder and not (cfg.poly_funder_address or "").strip():
        body["wallet_warnings"].append(
            "POLY_SIGNATURE_TYPE is 1 or 2 (Magic / browser wallet). Set POLY_FUNDER_ADDRESS "
            "to your Polymarket deposit/proxy address (polymarket.com → settings) so CLOB balance "
            "matches the site. The header still shows your signer (exported key) address."
        )
    try:
        bal = client.get_balance_allowance(
            BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
        )
        body["collateral"] = bal
    except Exception as exc:
        body["balance_error"] = str(exc)
    if not body["trading_enabled"]:
        body["trading_gate_hint"] = (
            "Buy/Sell stays off until you set DASHBOARD_ENABLE_ORDER_POST=true (or 1/yes/on) in .env "
            "and restart: python main.py dashboard"
        )
    return jsonify(body)


@app.route("/api/updown")
def api_updown():
    w = request.args.get("window", type=int)
    if w is None:
        w = current_window_ts()
    snap = build_updown_snapshot(w)
    return jsonify({"current": snap})


@app.route("/api/signal")
def api_signal():
    """
    Decision-support signal: model fair prob vs CLOB mids (optional live spot/strike from UI).
    """
    init_calibration_db()
    try:
        n_res = try_resolve_pending_windows(
            now_ts=int(time.time()),
            window_length_sec=FIVE_MIN,
            fetch_close=_resolution_close_binance,
        )
        if n_res:
            logger.debug("calibration_resolved_windows", count=n_res)
    except Exception as exc:
        logger.warning("calibration_resolve_failed", error=str(exc))

    asset = (request.args.get("asset") or "BTC").upper()
    w = request.args.get("window", type=int)
    if w is None:
        w = current_window_ts()
    spot_q = request.args.get("spot", type=float)
    strike_q = request.args.get("strike", type=float)
    strike_quality = (request.args.get("strike_quality") or "unknown").strip().lower()
    holding_raw = (request.args.get("holding") or "").strip().lower()
    holding = holding_raw if holding_raw in ("up", "down") else None

    boot = chart_bootstrap(asset, w)
    if not boot.get("ok"):
        return jsonify({"ok": False, "error": boot.get("error")}), 400

    strike = strike_q if strike_q is not None and strike_q > 0 else boot.get("strike")
    if strike is None or float(strike) <= 0:
        return jsonify({"ok": False, "error": "strike_unavailable"}), 400

    now_ms = int(time.time() * 1000)
    window_end_ms = boot.get("window_end_ms") or (w + FIVE_MIN) * 1000
    tau_sec = max(0.0, (float(window_end_ms) - now_ms) / 1000.0)
    if tau_sec <= 1.0:
        return jsonify({"ok": False, "error": "window_closed"}), 400

    pts = boot.get("points") or []
    closes = closes_from_bootstrap_points(pts)
    sigma_ann = annualized_vol_from_minute_closes(closes)

    spot_src = "chainlink_ui"
    if spot_q is not None and spot_q > 0:
        spot = float(spot_q)
    elif closes:
        spot = float(closes[-1])
        spot_src = "binance_chart_close"
    else:
        return jsonify({"ok": False, "error": "spot_unavailable"}), 400

    snap = build_updown_snapshot(w)
    row = next((r for r in snap["rows"] if r.get("asset") == asset and r.get("ok")), None)
    if not row:
        return jsonify({"ok": False, "error": "market_unavailable"}), 400

    up = row["sides"][0]
    dn = row["sides"][1]
    sig = compute_updown_signal(
        spot=spot,
        spot_source=spot_src,
        strike=float(strike),
        strike_quality=strike_quality,
        tau_seconds=tau_sec,
        sigma_annual=sigma_ann,
        best_bid_up=up.get("best_bid"),
        best_ask_up=up.get("best_ask"),
        best_bid_down=dn.get("best_bid"),
        best_ask_down=dn.get("best_ask"),
        holding=holding,
    )
    try:
        record_signal_snapshot(
            asset=asset,
            window_ts=w,
            strike=float(strike),
            fair_p_up=sig.get("fair_p_up"),
            primary_signal=sig.get("primary"),
            confidence=sig.get("confidence"),
            spot=spot,
            mid_up=sig.get("mid_up"),
            mid_down=sig.get("mid_down"),
            edge_buy_up=sig.get("edge_buy_up"),
            edge_buy_down=sig.get("edge_buy_down"),
            sigma_annual=sigma_ann,
            tau_seconds=tau_sec,
        )
    except Exception as exc:
        logger.warning("calibration_snapshot_failed", error=str(exc))

    body = {
        "ok": True,
        "asset": asset,
        "window_ts": w,
        "tau_seconds": round(tau_sec, 2),
        "spot_used": spot,
        "spot_source": spot_src,
        "strike_used": float(strike),
        "strike_quality": strike_quality,
        "sigma_annual": round(sigma_ann, 4),
        "holding_used": holding,
        **sig,
    }
    return jsonify(body)


@app.route("/api/calibration")
def api_calibration():
    """Historical signal vs Binance-resolved outcomes (see calibration_store)."""
    init_calibration_db()
    try:
        try_resolve_pending_windows(
            now_ts=int(time.time()),
            window_length_sec=FIVE_MIN,
            fetch_close=_resolution_close_binance,
        )
    except Exception as exc:
        logger.warning("calibration_resolve_failed", error=str(exc))
    asset = (request.args.get("asset") or "").strip().upper() or None
    stats = get_calibration_stats(asset=asset, series_limit=100)
    return jsonify(stats)


@app.route("/api/chart-bootstrap")
def api_chart_bootstrap():
    """Minute-close series + strike for the live Up/Down chart (Binance; see disclaimer)."""
    asset = (request.args.get("asset") or "BTC").upper()
    w = request.args.get("window", type=int)
    if w is None:
        w = current_window_ts()
    return jsonify(chart_bootstrap(asset, w))


@app.route("/api/polymarket-strike")
def api_polymarket_strike():
    """Lightweight poll for official priceToBeat (cached series fetch on server)."""
    asset = (request.args.get("asset") or "BTC").upper()
    w = request.args.get("window", type=int)
    if w is None:
        w = current_window_ts()
    series = POLYMARKET_SERIES_SLUG.get(asset)
    if not series:
        return jsonify({"ok": False, "error": "unsupported asset"}), 400
    slug = f"{asset.lower()}-updown-5m-{w}"
    v = polymarket_official_price_to_beat(series, slug)
    return jsonify(
        {
            "ok": True,
            "slug": slug,
            "strike_polymarket_official": v,
        }
    )


@app.route("/api/orders")
def api_orders():
    client = _clob_client()
    if not client:
        return jsonify({"ok": False, "error": "not authenticated"}), 401
    try:
        orders = client.get_orders(OpenOrderParams())
        return jsonify({"ok": True, "orders": orders})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


def _conditional_balance_float(raw: Any) -> Optional[float]:
    """Best-effort parse of CLOB conditional balance (often micro-units)."""
    if not isinstance(raw, dict):
        return None
    b = raw.get("balance")
    if b is None:
        return None
    try:
        x = float(b)
    except (TypeError, ValueError):
        return None
    if x > 1_000_000:
        return x / 1e6
    return x


@app.route("/api/token-balance")
def api_token_balance():
    client = _clob_client()
    if not client:
        return jsonify({"ok": False, "error": "not authenticated"}), 401
    token_id = request.args.get("token_id")
    if not token_id:
        return jsonify({"ok": False, "error": "token_id required"}), 400
    try:
        raw = client.get_balance_allowance(
            BalanceAllowanceParams(asset_type=AssetType.CONDITIONAL, token_id=str(token_id))
        )
        bal = _conditional_balance_float(raw)
        return jsonify({"ok": True, "balance": bal, "raw": raw})
    except Exception as exc:
        logger.exception("token balance failed")
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/cancel-order", methods=["POST"])
def api_cancel_order():
    if not _dashboard_order_post_enabled():
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "Set DASHBOARD_ENABLE_ORDER_POST=true in .env to allow cancel from the dashboard.",
                }
            ),
            403,
        )
    client = _clob_client()
    if not client:
        return jsonify({"ok": False, "error": "not authenticated"}), 401
    data = request.get_json(force=True, silent=True) or {}
    order_id = data.get("order_id") or data.get("orderID")
    if not order_id:
        return jsonify({"ok": False, "error": "order_id required"}), 400
    try:
        resp = client.cancel(str(order_id))
        return jsonify({"ok": True, "response": resp})
    except Exception as exc:
        logger.exception("cancel failed")
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/order", methods=["POST"])
def api_order():
    if not _dashboard_order_post_enabled():
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "Set DASHBOARD_ENABLE_ORDER_POST=true in .env to allow placing orders from the dashboard.",
                }
            ),
            403,
        )
    client = _clob_client()
    if not client:
        return jsonify({"ok": False, "error": "not authenticated"}), 401
    data = request.get_json(force=True, silent=True) or {}
    token_id = data.get("token_id")
    side = (data.get("side") or "").upper()
    try:
        price = float(data["price"])
        size_shares = float(data["size_shares"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"ok": False, "error": "need price (float) and size_shares (float)"}), 400
    if not token_id or side not in ("BUY", "SELL"):
        return jsonify({"ok": False, "error": "token_id and side BUY|SELL required"}), 400
    if size_shares < 1:
        return jsonify({"ok": False, "error": "size_shares must be >= 1"}), 400
    oside = BUY if side == "BUY" else SELL
    try:
        args = OrderArgs(token_id=str(token_id), price=price, size=size_shares, side=oside)
        signed = client.create_order(args)
        resp = client.post_order(signed, OrderType.GTC)
        return jsonify({"ok": True, "response": resp})
    except Exception as exc:
        logger.exception("order failed")
        return jsonify({"ok": False, "error": str(exc)}), 500


def run_dashboard(host: str = "127.0.0.1", port: Optional[int] = None) -> None:
    if port is None:
        port = int(os.getenv("DASHBOARD_PORT", "8765"))
    init_calibration_db()
    print(f"\n  Dashboard: http://{host}:{port}")
    print(f"  Signal calibration DB: {Path(__file__).resolve().parent / 'data' / 'signal_calibration.sqlite'}")
    cfg = _cfg_load()
    if not cfg.is_configured:
        print("  Wallet: not configured — POLY_PRIVATE_KEY / PRIVATE_KEY missing in .env\n")
    else:
        cl = _clob_client()
        if cl:
            print(f"  Wallet (CLOB signer): {cl.get_address()}")
            if not _dashboard_order_post_enabled():
                print(
                    "  UI order buttons: OFF — set DASHBOARD_ENABLE_ORDER_POST=true (or 1) in .env to enable Buy/Sell."
                )
            if cfg.needs_funder and not (cfg.poly_funder_address or "").strip():
                print(
                    "  Note: set POLY_FUNDER_ADDRESS for signature types 1–2 (Magic / browser wallet)."
                )
        else:
            print(f"  Wallet: CLOB init failed — {_clob_init_error or 'unknown error'}")
    print("")
    app.run(host=host, port=port, debug=False, threaded=True)


if __name__ == "__main__":
    run_dashboard()
