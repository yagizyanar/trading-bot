"""SentiCrypt sentiment fetcher (senticrypt.com/api/v1/sentiment).

Free, no API key. Returns a daily aggregate sentiment score. The score range
varies by endpoint version; we normalise to [-1.0, +1.0].

Note: senticrypt's free tier serves BTC-wide market sentiment, not per-coin.
We use it as a *market-wide* sentiment baseline that gets blended into each
coin's unified score via the analyzer's weight scheme.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import requests

from config.settings import API_RETRY_ATTEMPTS, API_TIMEOUT_SECONDS, SENTICRYPT_URL

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SentiCryptReading:
    score: float          # normalised to [-1.0, +1.0]
    raw_score: float      # original API value (typically -1..1 already)
    ts: datetime
    sample_count: Optional[int]


def _normalise(raw: float) -> float:
    """Clip into [-1.0, +1.0] (SentiCrypt may occasionally return slight overshoots)."""
    if raw > 1.0:
        return 1.0
    if raw < -1.0:
        return -1.0
    return float(raw)


def fetch_senticrypt() -> Optional[SentiCryptReading]:
    """Fetch the most recent SentiCrypt sentiment record. None on failure."""
    for attempt in range(1, API_RETRY_ATTEMPTS + 1):
        try:
            resp = requests.get(SENTICRYPT_URL, timeout=API_TIMEOUT_SECONDS)
            resp.raise_for_status()
            payload = resp.json()
            if isinstance(payload, list):
                if not payload:
                    raise ValueError("empty payload")
                row = payload[-1]
            elif isinstance(payload, dict):
                row = payload
            else:
                raise TypeError(f"unexpected payload type: {type(payload).__name__}")

            raw = float(row.get("score") or row.get("mean") or 0.0)
            ts_raw = row.get("date") or row.get("timestamp")
            if isinstance(ts_raw, (int, float)):
                ts = datetime.fromtimestamp(float(ts_raw), tz=timezone.utc)
            elif isinstance(ts_raw, str):
                try:
                    ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
                except ValueError:
                    ts = datetime.now(timezone.utc)
            else:
                ts = datetime.now(timezone.utc)
            sample = row.get("count")
            sample = int(sample) if sample is not None else None
            return SentiCryptReading(
                score=_normalise(raw),
                raw_score=raw,
                ts=ts,
                sample_count=sample,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("senticrypt fetch attempt %s failed: %s", attempt, exc)
    return None
