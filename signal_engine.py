"""
Up/Down decision support: short-horizon digital prob + executable buy/sell edges.

Uses ask prices for buy edge and bid prices for exit edge when available.
Applies spot-vs-strike confluence so "Lean Up" aligns with structure unless mispricing is large.

Not investment advice; Polymarket settlement can differ from Binance proxies used in calibration.
"""
from __future__ import annotations

import math
from typing import Any, Literal, Optional

# Slightly stricter than v1 — fewer low-conviction leans.
EDGE_STRONG_BUY = 0.065
EDGE_MODEST_BUY = 0.038
EDGE_STRONG_SELL = 0.052
EDGE_MODEST_SELL = 0.03

# Structural lean only when book is thin: require clear moneyness.
STRUCT_UP_MIN_P = 0.57
STRUCT_DOWN_MAX_P = 0.43

# Allow tiny numerical slack vs strike (basis points).
STRIKE_SLACK_BP = 2.5


def norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def closes_from_bootstrap_points(points: list[dict[str, Any]], max_n: int = 24) -> list[float]:
    if not points:
        return []
    sorted_pts = sorted(points, key=lambda z: z.get("t", 0))
    out: list[float] = []
    for z in sorted_pts[-max_n:]:
        p = z.get("p")
        if p is not None and isinstance(p, (int, float)) and float(p) > 0:
            out.append(float(p))
    return out


def annualized_vol_from_minute_closes(closes: list[float]) -> float:
    """Log-return stdev per chart step (~1m), annualized (rough GBM vol)."""
    if len(closes) < 3:
        return 0.75
    lr: list[float] = []
    for i in range(1, len(closes)):
        a, b = closes[i - 1], closes[i]
        if a <= 0 or b <= 0:
            continue
        lr.append(math.log(b / a))
    if len(lr) < 2:
        return 0.75
    m = sum(lr) / len(lr)
    var = sum((x - m) ** 2 for x in lr) / (len(lr) - 1)
    sigma_1m = math.sqrt(max(var, 1e-18))
    return float(sigma_1m * math.sqrt(365.25 * 24 * 60))


def digital_prob_spot_above_strike(
    spot: float,
    strike: float,
    tau_seconds: float,
    sigma_annual: float,
) -> float:
    if strike <= 0 or spot <= 0 or tau_seconds <= 0:
        return 0.5
    tau_y = max(float(tau_seconds), 0.5) / (365.25 * 24 * 3600)
    sig = max(float(sigma_annual), 0.08)
    vol_sqrt_t = sig * math.sqrt(tau_y)
    if vol_sqrt_t < 1e-14:
        return 1.0 if spot > strike else 0.0
    drift_adj = -0.5 * sig * sig * tau_y
    d2 = (math.log(spot / strike) + drift_adj) / vol_sqrt_t
    p = norm_cdf(d2)
    return max(0.02, min(0.98, float(p)))


def mid_or_none(bid: Optional[float], ask: Optional[float]) -> Optional[float]:
    if bid is not None and ask is not None:
        return (float(bid) + float(ask)) / 2.0
    return None


def _bp_slack(strike: float) -> float:
    return float(strike) * STRIKE_SLACK_BP / 10_000.0


def _holding_norm(h: Optional[str]) -> Literal["up", "down", "none"]:
    if not h:
        return "none"
    x = str(h).strip().lower()
    if x in ("up", "u", "1"):
        return "up"
    if x in ("down", "d", "2"):
        return "down"
    return "none"


def compute_updown_signal(
    *,
    spot: float,
    spot_source: str,
    strike: float,
    strike_quality: str,
    tau_seconds: float,
    sigma_annual: float,
    best_bid_up: Optional[float],
    best_ask_up: Optional[float],
    best_bid_down: Optional[float],
    best_ask_down: Optional[float],
    holding: Optional[str] = None,
) -> dict[str, Any]:
    p_up = digital_prob_spot_above_strike(spot, strike, tau_seconds, sigma_annual)
    p_down = 1.0 - p_up

    mid_up = mid_or_none(best_bid_up, best_ask_up)
    mid_down = mid_or_none(best_bid_down, best_ask_down)

    # Executable buy edge: pay ask vs fair.
    edge_buy_up = (p_up - float(best_ask_up)) if best_ask_up is not None else None
    edge_buy_down = (p_down - float(best_ask_down)) if best_ask_down is not None else None
    # Fallback to mid if one-sided book
    if edge_buy_up is None and mid_up is not None:
        edge_buy_up = p_up - mid_up
    if edge_buy_down is None and mid_down is not None:
        edge_buy_down = p_down - mid_down

    # Sell / exit: bid vs fair (positive => bid rich vs model — better exit for long that token).
    edge_sell_up = (float(best_bid_up) - p_up) if best_bid_up is not None else None
    edge_sell_down = (float(best_bid_down) - p_down) if best_bid_down is not None else None
    if edge_sell_up is None and mid_up is not None:
        edge_sell_up = mid_up - p_up
    if edge_sell_down is None and mid_down is not None:
        edge_sell_down = mid_down - p_down

    slack = _bp_slack(strike)
    struct_above = spot >= strike - slack
    struct_below = spot <= strike + slack

    reasons: list[str] = []
    if strike > 0:
        spot_vs = ((spot / strike) - 1.0) * 100.0
        if spot > strike + slack:
            reasons.append(f"Spot {spot_vs:.3f}% above price to beat (structural tailwind for Up).")
        elif spot < strike - slack:
            reasons.append(f"Spot {abs(spot_vs):.3f}% below price to beat (structural tailwind for Down).")
        else:
            reasons.append("Spot ~on the strike — path + time decide; confluence filters tighten leans.")

    mins_left = tau_seconds / 60.0
    reasons.append(
        f"~{mins_left:.1f} min left · vol ~{sigma_annual * 100:.0f}% ann. from recent path (not a guarantee)."
    )

    if mid_up is not None and mid_down is not None:
        reasons.append(
            f"Book mids Up {mid_up * 100:.1f}¢ / Down {mid_down * 100:.1f}¢ · "
            f"model {p_up * 100:.1f}% / {p_down * 100:.1f}%."
        )
        if best_ask_up is not None and best_ask_down is not None:
            reasons.append(
                f"Executable buys: Up ask {float(best_ask_up) * 100:.1f}¢, Down ask {float(best_ask_down) * 100:.1f}¢."
            )

    # --- Buy primary (Lean Up / Down / Neutral) ---
    primary = "NEUTRAL"
    strength = "low"

    book_both = edge_buy_up is not None and edge_buy_down is not None
    eu = edge_buy_up if edge_buy_up is not None else -999.0
    ed = edge_buy_down if edge_buy_down is not None else -999.0

    up_strength = "low"
    down_strength = "low"
    up_from_book = False
    down_from_book = False

    if book_both:
        if eu >= EDGE_STRONG_BUY and eu >= ed + 0.008:
            up_from_book, up_strength = True, "high"
        elif eu >= EDGE_MODEST_BUY and eu > ed + 0.015:
            up_from_book, up_strength = True, "modest"
        if ed >= EDGE_STRONG_BUY and ed > eu + 0.008:
            down_from_book, down_strength = True, "high"
        elif ed >= EDGE_MODEST_BUY and ed > eu + 0.015:
            down_from_book, down_strength = True, "modest"

    up_override = edge_buy_up is not None and eu >= EDGE_STRONG_BUY + 0.02
    down_override = edge_buy_down is not None and ed >= EDGE_STRONG_BUY + 0.02

    up_ok = up_from_book and (struct_above or up_override)
    down_ok = down_from_book and (struct_below or down_override)

    if up_from_book and not struct_above and not up_override:
        reasons.append("Up ask looks cheap vs model, but spot is below strike — need large edge or wait.")
    if down_from_book and not struct_below and not down_override:
        reasons.append("Down ask looks cheap vs model, but spot is above strike — need large edge or wait.")

    up_struct = False
    down_struct = False
    if not up_from_book and not down_from_book:
        if struct_above and p_up >= STRUCT_UP_MIN_P:
            up_struct = True
            up_strength = "structure_only"
        elif struct_below and p_up <= STRUCT_DOWN_MAX_P:
            down_struct = True
            down_strength = "structure_only"
        elif p_up >= STRUCT_UP_MIN_P and not struct_above:
            reasons.append("Model tilts Up but spot is not above strike — no structural Lean Up.")
        elif p_up <= STRUCT_DOWN_MAX_P and not struct_below:
            reasons.append("Model tilts Down but spot is not below strike — no structural Lean Down.")

    pick_up = up_ok or up_struct
    pick_down = down_ok or down_struct
    if pick_up and pick_down:
        if eu >= ed:
            pick_down = False
        else:
            pick_up = False

    if pick_up:
        primary = "UP"
        strength = up_strength
        reasons.append("Lean Up: ask vs fair and/or spot-above-strike confluence for buying Up.")
    elif pick_down:
        primary = "DOWN"
        strength = down_strength
        reasons.append("Lean Down: ask vs fair and/or spot-below-strike confluence for buying Down.")
    else:
        reasons.append("No high-conviction buy lean — reduce size or wait for clearer executable edge.")

    # --- Sell / exit hints (bids vs fair) ---
    sell_up = "HOLD"
    sell_down = "HOLD"
    if edge_sell_up is not None:
        if edge_sell_up >= EDGE_STRONG_SELL:
            sell_up = "FAVORABLE_EXIT"
        elif edge_sell_up >= EDGE_MODEST_SELL:
            sell_up = "CONSIDER_EXIT"
        elif edge_sell_up <= -EDGE_STRONG_SELL:
            sell_up = "WEAK_BID"
    if edge_sell_down is not None:
        if edge_sell_down >= EDGE_STRONG_SELL:
            sell_down = "FAVORABLE_EXIT"
        elif edge_sell_down >= EDGE_MODEST_SELL:
            sell_down = "CONSIDER_EXIT"
        elif edge_sell_down <= -EDGE_STRONG_SELL:
            sell_down = "WEAK_BID"

    hv = _holding_norm(holding)
    sell_for_you = None
    sell_for_you_detail = None
    if hv == "up":
        if sell_up == "FAVORABLE_EXIT":
            sell_for_you = "SELL_UP_NOW"
            sell_for_you_detail = "Up bids look rich vs model — good zone to exit / take profit."
        elif sell_up == "CONSIDER_EXIT":
            sell_for_you = "CONSIDER_SELL_UP"
            sell_for_you_detail = "Mild exit edge on Up; partial trim or raise limit toward bid."
        elif sell_up == "WEAK_BID":
            sell_for_you = "HOLD_UP"
            sell_for_you_detail = "Up bids are cheap vs fair — avoid fire-selling; wait for better bid or window."
        else:
            sell_for_you = "HOLD_UP"
            sell_for_you_detail = "No strong exit signal on Up vs model."
    elif hv == "down":
        if sell_down == "FAVORABLE_EXIT":
            sell_for_you = "SELL_DOWN_NOW"
            sell_for_you_detail = "Down bids rich vs model — favorable exit zone."
        elif sell_down == "CONSIDER_EXIT":
            sell_for_you = "CONSIDER_SELL_DOWN"
            sell_for_you_detail = "Mild exit edge on Down."
        elif sell_down == "WEAK_BID":
            sell_for_you = "HOLD_DOWN"
            sell_for_you_detail = "Down bids weak vs fair — patience or higher limit."
        else:
            sell_for_you = "HOLD_DOWN"
            sell_for_you_detail = "No strong exit signal on Down vs model."

    if sell_for_you_detail:
        reasons.append(sell_for_you_detail)

    # Confidence
    confidence = 44
    if strength == "high":
        if primary == "UP" and edge_buy_up is not None:
            edge_mag = abs(edge_buy_up)
        elif primary == "DOWN" and edge_buy_down is not None:
            edge_mag = abs(edge_buy_down)
        else:
            edge_mag = max(abs(eu), abs(ed)) if book_both else 0.0
        confidence = min(88, 56 + int(edge_mag * 130))
    elif strength == "modest":
        confidence = 66
    elif strength == "structure_only":
        confidence = 52

    sq = (strike_quality or "").lower()
    if sq in ("binance_1m_open", "unknown", ""):
        confidence = max(22, confidence - 20)
        reasons.append("Strike uncertain — widen error bars; prefer book edge over pure structure.")
    elif sq == "latched":
        confidence = max(28, confidence - 5)

    if spot_source == "binance_chart_close":
        confidence = max(22, confidence - 16)
        reasons.append("Spot is stale chart proxy — connect RTDS for live Chainlink.")

    return {
        "fair_p_up": round(p_up, 4),
        "fair_p_down": round(p_down, 4),
        "mid_up": None if mid_up is None else round(mid_up, 4),
        "mid_down": None if mid_down is None else round(mid_down, 4),
        "edge_buy_up": None if edge_buy_up is None else round(edge_buy_up, 4),
        "edge_buy_down": None if edge_buy_down is None else round(edge_buy_down, 4),
        "edge_sell_up": None if edge_sell_up is None else round(edge_sell_up, 4),
        "edge_sell_down": None if edge_sell_down is None else round(edge_sell_down, 4),
        "primary": primary,
        "strength": strength,
        "confidence": max(0, min(100, int(confidence))),
        "reasons": reasons,
        "sell": {
            "up": sell_up,
            "down": sell_down,
            "for_holding": sell_for_you,
        },
    }
