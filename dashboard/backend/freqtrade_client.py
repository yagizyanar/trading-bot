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

CACHE_TTL_SECONDS = 60.0
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


def live_equity() -> Optional[float]:
    """Return the live equity in fiat (USD), or None if Freqtrade unreachable."""
    bal = fetch_balance()
    if not bal:
        return None
    # 'value' is the fiat-equivalent total across all currencies; fall back to 'total'.
    v = bal.get("value")
    if v is None:
        v = bal.get("total")
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def invalidate_cache() -> None:
    """Tests / manual refresh: drop all cached entries."""
    with _CACHE_LOCK:
        _CACHE.clear()
