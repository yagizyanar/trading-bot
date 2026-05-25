"""Fear & Greed Index fetcher (alternative.me).

Free, no API key. JSON shape:
  {"data": [{"value": "63", "value_classification": "Greed",
             "timestamp": "1706054400", ...}], "metadata": {...}}

Used as a *macro* multiplier on per-coin scores (sentiment-pipeline skill):
  0-25   Extreme Fear → ×0.5
  26-45  Fear         → ×0.75
  46-55  Neutral      → ×1.0
  56-75  Greed        → ×1.0
  76-100 Extreme Greed→ ×0.8 (contrarian caution)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import requests

from config.settings import API_RETRY_ATTEMPTS, API_TIMEOUT_SECONDS, FEAR_GREED_URL

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FearGreedReading:
    value: int                 # 0..100
    classification: str        # "Extreme Fear" .. "Extreme Greed"
    ts: datetime
    multiplier: float          # macro adjustment factor (skill table)


def _classify_to_multiplier(value: int) -> float:
    if value <= 25:
        return 0.5
    if value <= 45:
        return 0.75
    if value <= 55:
        return 1.0
    if value <= 75:
        return 1.0
    return 0.8


def fetch_fear_greed() -> Optional[FearGreedReading]:
    """Fetch the latest Fear & Greed reading. Returns None on failure."""
    for attempt in range(1, API_RETRY_ATTEMPTS + 1):
        try:
            resp = requests.get(
                FEAR_GREED_URL,
                params={"limit": 1, "format": "json"},
                timeout=API_TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
            payload = resp.json()
            row = payload["data"][0]
            value = int(row["value"])
            ts = datetime.fromtimestamp(int(row["timestamp"]), tz=timezone.utc)
            return FearGreedReading(
                value=value,
                classification=str(row.get("value_classification", "")),
                ts=ts,
                multiplier=_classify_to_multiplier(value),
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("fear_greed fetch attempt %s failed: %s", attempt, exc)
    return None
