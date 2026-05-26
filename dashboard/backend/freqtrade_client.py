"""Thin client for Freqtrade's REST API (read-only).

Reads credentials from the FastAPI backend's environment:
  FREQTRADE_API_URL       (default: http://127.0.0.1:8080)
  FREQTRADE_API_USER      (default: freqtrader)
  FREQTRADE_API_PASSWORD  (required for auth; if absent we return None)

Cached for 60 seconds so concurrent dashboard requests don't hammer Freqtrade.

All failures are swallowed → return None. The caller (performance route)
falls back to the latest DB performance_snapshots row.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional

import requests

log = logging.getLogger(__name__)

CACHE_TTL_SECONDS = 5.0   # tight enough for ~5s WS pushes; Freqtrade copes fine
DEFAULT_TIMEOUT = 3.0

_CACHE_LOCK = threading.Lock()
_CACHE: dict[str, tuple[float, Optional[dict]]] = {}


def _cfg() -> tuple[str, str, str]:
    url  = os.environ.get("FREQTRADE_API_URL", "http://127.0.0.1:8080").rstrip("/")
    user = os.environ.get("FREQTRADE_API_USER", "freqtrader")
    pw   = os.environ.get("FREQTRADE_API_PASSWORD", "")
    return url, user, pw


def _cached_get(path: str, timeout: float = DEFAULT_TIMEOUT) -> Optional[dict]:
    """GET a Freqtrade endpoint with basic auth, 60s in-memory cache."""
    url, user, pw = _cfg()
    if not pw:
        return None
    key = path
    now = time.time()
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if cached and (now - cached[0]) < CACHE_TTL_SECONDS:
            return cached[1]
    try:
        resp = requests.get(f"{url}{path}", auth=(user, pw), timeout=timeout)
        if resp.status_code != 200:
            log.debug("freqtrade %s -> %s", path, resp.status_code)
            with _CACHE_LOCK:
                _CACHE[key] = (now, None)
            return None
        data = resp.json()
    except Exception as exc:  # noqa: BLE001
        log.debug("freqtrade %s failed: %s", path, exc)
        with _CACHE_LOCK:
            _CACHE[key] = (now, None)
        return None
    with _CACHE_LOCK:
        _CACHE[key] = (now, data)
    return data


def fetch_balance() -> Optional[dict]:
    """Return the raw /api/v1/balance payload or None."""
    return _cached_get("/api/v1/balance")


def fetch_status() -> Optional[list]:
    """Return /api/v1/status — list of open trades; empty list if none."""
    data = _cached_get("/api/v1/status")
    return data if isinstance(data, list) else None


def fetch_profit() -> Optional[dict]:
    """Return /api/v1/profit — running PnL stats."""
    return _cached_get("/api/v1/profit")


def fetch_closed_trades(limit: int = 50) -> Optional[list]:
    """Return /api/v1/trades — list of closed trades, newest first.

    Freqtrade wraps the array in {"trades": [...], "trades_count": N, ...}.
    """
    limit = max(1, min(int(limit), 500))
    data = _cached_get(f"/api/v1/trades?limit={limit}")
    if not isinstance(data, dict):
        return None
    trades = data.get("trades")
    return trades if isinstance(trades, list) else []


# ---------------------------------------------------------------------------
# Mapping helpers: Freqtrade trade shape → dashboard shape.
# Dashboard format matches dashboard.backend.routes.positions._serialize()
# (the DB-backed Trade row shape) so the frontend doesn't need to change.
# ---------------------------------------------------------------------------
def _coin_from_pair(pair: str) -> str:
    """SOL/USDT:USDT  → 'SOL'."""
    if not pair:
        return ""
    return pair.split("/", 1)[0]


def _outcome_for(profit_abs: Optional[float], is_open: bool) -> str:
    if is_open:
        return "OPEN"
    if profit_abs is None:
        return "OPEN"
    return "WIN" if float(profit_abs) > 0 else "LOSS"


def _take_profit_price(open_rate: Optional[float], is_short: bool,
                       tp_pct: Optional[float] = None) -> Optional[float]:
    """Compute the take-profit price from entry + configured ROI.

    Freqtrade's /api/v1/status doesn't expose a take-profit price directly
    (it uses a time-based ROI table). We approximate from minimal_roi[0]
    which our config pins to TAKE_PROFIT_PCT (default 15%).
    """
    if open_rate is None:
        return None
    if tp_pct is None:
        from config.settings import TAKE_PROFIT_PCT
        tp_pct = float(TAKE_PROFIT_PCT)
    try:
        rate = float(open_rate)
    except (TypeError, ValueError):
        return None
    return rate * (1.0 - tp_pct) if is_short else rate * (1.0 + tp_pct)


def map_freqtrade_trade(t: dict) -> dict:
    """Convert one Freqtrade trade payload to the dashboard's trade dict.

    Handles both /status (open) and /trades (closed) shapes. Both formats use
    the same field names — the only differences are which fields have values.
    """
    profit_pct_pct = t.get("profit_pct")          # e.g. -0.03 means -0.03%
    profit_pct_frac = (float(profit_pct_pct) / 100.0) if profit_pct_pct is not None else None
    is_open = bool(t.get("is_open", True))
    is_short = bool(t.get("is_short", False))
    profit_abs = t.get("profit_abs")
    if profit_abs is None:
        profit_abs = t.get("close_profit_abs")

    open_rate = t.get("open_rate")
    current_rate = t.get("current_rate")
    amount = t.get("amount")
    # Position size at entry, in stake currency. Prefer Freqtrade's own
    # `stake_amount` (post-leverage account exposure) when present; fall back
    # to amount × open_rate.
    stake = t.get("stake_amount")
    if stake is None and amount is not None and open_rate is not None:
        try:
            stake = float(amount) * float(open_rate)
        except (TypeError, ValueError):
            stake = None

    # Current notional value: amount × current_rate.
    current_value = None
    if amount is not None and current_rate is not None:
        try:
            current_value = float(amount) * float(current_rate)
        except (TypeError, ValueError):
            current_value = None

    return {
        "id":               t.get("trade_id"),
        "coin":             _coin_from_pair(t.get("pair", "")),
        "side":             "SHORT" if is_short else "LONG",
        "entry_price":      open_rate,
        "current_price":    current_rate,
        "exit_price":       t.get("close_rate"),
        "quantity":         amount,
        "size_usdt":        stake,
        "current_value_usdt": current_value,
        "leverage":         int(t.get("leverage") or 1),
        "pnl_usd":          float(profit_abs) if profit_abs is not None else None,
        "pnl_pct":          profit_pct_frac,
        "stop_loss_price":  t.get("stop_loss_abs"),
        "take_profit_price": _take_profit_price(open_rate, is_short),
        "entry_ts":         t.get("open_date"),
        "exit_ts":          t.get("close_date"),
        "reason_in":        t.get("enter_tag") or "freqtrade",
        "reason_out":       t.get("exit_reason"),
        "outcome":          _outcome_for(profit_abs, is_open),
        "is_paper":         True,   # we run in dry_run; live mode flips this in the route
    }


def live_equity() -> Optional[float]:
    """Return the live equity, or None if Freqtrade unreachable.

    `value` is the fiat-converted total (when fiat_display_currency is set).
    When fiat conversion is disabled, Freqtrade reports value=0.0, so we
    fall back to `total` (the raw stake-currency sum).
    """
    bal = fetch_balance()
    if not bal:
        return None
    v = bal.get("value") or bal.get("total")
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def invalidate_cache() -> None:
    """Tests / manual refresh: drop all cached entries."""
    with _CACHE_LOCK:
        _CACHE.clear()
