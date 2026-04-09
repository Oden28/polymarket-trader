"""
SQLite log of signal snapshots vs post-hoc outcomes for dashboard calibration.

Outcomes use Binance 1m close vs logged strike (approximation — Polymarket may use Chainlink).
"""
from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

_DB_LOCK = threading.Lock()
_DB_PATH = Path(__file__).resolve().parent / "data" / "signal_calibration.sqlite"


def _connect() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH), timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn


def init_calibration_db() -> None:
    with _DB_LOCK:
        c = _connect()
        try:
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS window_log (
                    asset TEXT NOT NULL,
                    window_ts INTEGER NOT NULL,
                    strike REAL NOT NULL,
                    last_updated REAL NOT NULL,
                    fair_p_up REAL,
                    primary_signal TEXT,
                    confidence INTEGER,
                    spot REAL,
                    mid_up REAL,
                    mid_down REAL,
                    edge_buy_up REAL,
                    edge_buy_down REAL,
                    sigma_annual REAL,
                    tau_seconds REAL,
                    outcome_up_won INTEGER,
                    final_price REAL,
                    resolved_at REAL,
                    resolution_note TEXT,
                    PRIMARY KEY (asset, window_ts)
                )
                """
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_window_resolved ON window_log (outcome_up_won, window_ts)"
            )
            c.commit()
        finally:
            c.close()


def record_signal_snapshot(
    *,
    asset: str,
    window_ts: int,
    strike: float,
    fair_p_up: Optional[float],
    primary_signal: Optional[str],
    confidence: Optional[int],
    spot: Optional[float],
    mid_up: Optional[float],
    mid_down: Optional[float],
    edge_buy_up: Optional[float],
    edge_buy_down: Optional[float],
    sigma_annual: Optional[float],
    tau_seconds: Optional[float],
) -> None:
    """Latest /api/signal snapshot for this window (overwrites until resolved)."""
    asset_u = asset.upper()
    now = time.time()
    with _DB_LOCK:
        c = _connect()
        try:
            c.execute(
                """
                INSERT INTO window_log (
                    asset, window_ts, strike, last_updated,
                    fair_p_up, primary_signal, confidence, spot,
                    mid_up, mid_down, edge_buy_up, edge_buy_down,
                    sigma_annual, tau_seconds
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(asset, window_ts) DO UPDATE SET
                    last_updated = excluded.last_updated,
                    strike = CASE WHEN window_log.outcome_up_won IS NULL
                        THEN excluded.strike ELSE window_log.strike END,
                    fair_p_up = excluded.fair_p_up,
                    primary_signal = excluded.primary_signal,
                    confidence = excluded.confidence,
                    spot = excluded.spot,
                    mid_up = excluded.mid_up,
                    mid_down = excluded.mid_down,
                    edge_buy_up = excluded.edge_buy_up,
                    edge_buy_down = excluded.edge_buy_down,
                    sigma_annual = excluded.sigma_annual,
                    tau_seconds = excluded.tau_seconds
                WHERE window_log.outcome_up_won IS NULL
                """,
                (
                    asset_u,
                    window_ts,
                    float(strike),
                    now,
                    fair_p_up,
                    primary_signal,
                    confidence,
                    spot,
                    mid_up,
                    mid_down,
                    edge_buy_up,
                    edge_buy_down,
                    sigma_annual,
                    tau_seconds,
                ),
            )
            c.commit()
        finally:
            c.close()


def try_resolve_pending_windows(
    *,
    now_ts: int,
    window_length_sec: int,
    fetch_close: Callable[[str, int], tuple[Optional[float], str]],
) -> int:
    """
    Resolve rows whose window ended >= 45s ago and outcome is still null.
    Returns number of rows updated this pass.
    """
    deadline = now_ts - window_length_sec - 45
    updated = 0
    with _DB_LOCK:
        c = _connect()
        try:
            rows = c.execute(
                """
                SELECT asset, window_ts, strike FROM window_log
                WHERE outcome_up_won IS NULL AND window_ts <= ?
                ORDER BY window_ts ASC
                LIMIT 12
                """,
                (deadline,),
            ).fetchall()
            for row in rows:
                asset = row["asset"]
                wts = int(row["window_ts"])
                strike = float(row["strike"])
                close_p, note = fetch_close(asset, wts)
                if close_p is None or close_p <= 0:
                    continue
                up_won = 1 if close_p >= strike else 0
                cur = c.execute(
                    """
                    UPDATE window_log SET
                        outcome_up_won = ?,
                        final_price = ?,
                        resolved_at = ?,
                        resolution_note = ?
                    WHERE asset = ? AND window_ts = ? AND outcome_up_won IS NULL
                    """,
                    (up_won, close_p, time.time(), note, asset, wts),
                )
                if cur.rowcount:
                    updated += 1
            c.commit()
        finally:
            c.close()
    return updated


def get_calibration_stats(*, asset: Optional[str] = None, series_limit: int = 80) -> dict[str, Any]:
    init_calibration_db()
    asset_u = asset.upper() if asset else None
    where = "WHERE outcome_up_won IS NOT NULL"
    params: tuple[Any, ...] = ()
    if asset_u:
        where += " AND asset = ?"
        params = (asset_u,)

    with _DB_LOCK:
        c = _connect()
        try:
            total = c.execute(f"SELECT COUNT(*) AS n FROM window_log {where}", params).fetchone()["n"]
            if total == 0:
                return {
                    "ok": True,
                    "resolved_n": 0,
                    "series": [],
                    "buckets": [],
                    "lean_up": {},
                    "lean_down": {},
                    "brier_mean": None,
                }

            rows = c.execute(
                f"""
                SELECT asset, window_ts, fair_p_up, primary_signal, outcome_up_won, final_price, strike
                FROM window_log {where}
                ORDER BY window_ts ASC
                """,
                params,
            ).fetchall()

            preds = [float(r["fair_p_up"]) for r in rows if r["fair_p_up"] is not None]
            acts = [int(r["outcome_up_won"]) for r in rows if r["fair_p_up"] is not None]
            brier = None
            if preds and len(preds) == len(acts):
                brier = sum((acts[i] - preds[i]) ** 2 for i in range(len(preds))) / len(preds)

            def lean_stats(sig: str) -> dict[str, Any]:
                sub = [r for r in rows if (r["primary_signal"] or "") == sig]
                if not sub:
                    return {"n": 0, "hit_rate": None, "correct": 0}
                correct = sum(
                    1
                    for r in sub
                    if int(r["outcome_up_won"]) == (1 if sig == "UP" else 0)
                )
                return {"n": len(sub), "hit_rate": correct / len(sub), "correct": correct}

            # Reliability buckets on fair_p_up
            bucket_defs = [
                (0.0, 0.45, "0–45%"),
                (0.45, 0.50, "45–50%"),
                (0.50, 0.55, "50–55%"),
                (0.55, 0.60, "55–60%"),
                (0.60, 0.65, "60–65%"),
                (0.65, 0.70, "65–70%"),
                (0.70, 1.01, "70–100%"),
            ]
            buckets_out = []
            for lo, hi, label in bucket_defs:
                in_b = [
                    r
                    for r in rows
                    if r["fair_p_up"] is not None and lo <= float(r["fair_p_up"]) < hi
                ]
                if not in_b:
                    buckets_out.append({"label": label, "n": 0, "pred_mean": None, "actual_rate": None})
                    continue
                pred_mean = sum(float(r["fair_p_up"]) for r in in_b) / len(in_b)
                actual_rate = sum(int(r["outcome_up_won"]) for r in in_b) / len(in_b)
                buckets_out.append(
                    {
                        "label": label,
                        "n": len(in_b),
                        "pred_mean": round(pred_mean, 4),
                        "actual_rate": round(actual_rate, 4),
                    }
                )

            tail = rows[-series_limit:]
            series = [
                {
                    "window_ts": int(r["window_ts"]),
                    "asset": r["asset"],
                    "fair_p_up": float(r["fair_p_up"]) if r["fair_p_up"] is not None else None,
                    "up_won": int(r["outcome_up_won"]),
                    "primary": r["primary_signal"],
                }
                for r in tail
            ]

            return {
                "ok": True,
                "resolved_n": total,
                "brier_mean": None if brier is None else round(brier, 5),
                "lean_up": lean_stats("UP"),
                "lean_down": lean_stats("DOWN"),
                "neutral_n": len([r for r in rows if (r["primary_signal"] or "") == "NEUTRAL"]),
                "buckets": buckets_out,
                "series": series,
            }
        finally:
            c.close()
