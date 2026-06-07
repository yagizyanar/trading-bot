"""CoinGecko price-change source — the 7-day momentum signal.

Replaces yfinance (2026-06-03), which failed for 8/24 coins:
  - 1000-prefix Binance futures tickers (1000BONK/1000PEPE) have no Yahoo symbol
  - Yahoo disambiguates several symbols with CoinMarketCap-id suffixes that the
    naive `{COIN}-USD` mapping misses (APT/SUI/TAO/GMX)
  - newly rebranded / not-on-Yahoo (POL, S/Sonic)
CoinGecko resolves all 24 via stable ids and returns the 7-day % change in ONE
batched /coins/markets call (rate-limit friendly: 1 call per pipeline run).

Returned values are FRACTIONS (e.g. -0.11 for -11%) to match the old yfinance
convention consumed by analyzer._yfinance_to_signal.
"""
from __future__ import annotations

import logging
from typing import Iterable, Optional

import requests

from config.settings import (
    API_RETRY_ATTEMPTS,
    API_TIMEOUT_SECONDS,
    COINGECKO_MARKETS_URL,
)

log = logging.getLogger(__name__)

# Curated coin -> CoinGecko id. All 24 verified to resolve on /coins/markets
# (2026-06-03). 1000BONK/1000PEPE map to the underlying bonk/pepe (the "1000"
# is just Binance's futures price-denomination; % change is identical).
COINGECKO_IDS: dict[str, str] = {
    "SOL": "solana",
    "AVAX": "avalanche-2",
    "LINK": "chainlink",
    "DOT": "polkadot",
    "POL": "polygon-ecosystem-token",
    "INJ": "injective-protocol",
    "ARB": "arbitrum",
    "OP": "optimism",
    "APT": "aptos",
    "SUI": "sui",
    "NEAR": "near",
    "S": "sonic-3",
    "ATOM": "cosmos",
    "SAND": "the-sandbox",
    "MANA": "decentraland",
    "AXS": "axie-infinity",
    "DYDX": "dydx-chain",
    "GMX": "gmx",
    "WIF": "dogwifcoin",
    "1000BONK": "bonk",
    "1000PEPE": "pepe",
    "FET": "fetch-ai",
    "RENDER": "render-token",
    "TAO": "bittensor",
    "WLD": "worldcoin-wld",
    "UNI": "uniswap",
}


def fetch_price_changes_7d(coins: Iterable[str]) -> dict[str, float]:
    """Batch-fetch the 7-day price change (as a FRACTION) for `coins`.

    One CoinGecko /coins/markets call. Returns {coin: pct_fraction} for every
    coin that resolved and had a 7d value; coins missing an id or value are
    simply absent (the analyzer treats absence as a missing source and
    redistributes its weight). Returns {} on total request failure.
    """
    ids_map = {c: COINGECKO_IDS[c] for c in coins if c in COINGECKO_IDS}
    if not ids_map:
        return {}
    id_to_coin = {cid: c for c, cid in ids_map.items()}
    headers = {"User-Agent": "Mozilla/5.0 (compatible; trade-bot/1.0)"}
    for attempt in range(1, API_RETRY_ATTEMPTS + 1):
        try:
            resp = requests.get(
                COINGECKO_MARKETS_URL,
                params={
                    "vs_currency": "usd",
                    "ids": ",".join(sorted(set(ids_map.values()))),
                    "price_change_percentage": "7d",
                    "per_page": 250,
                },
                timeout=API_TIMEOUT_SECONDS,
                headers=headers,
            )
            resp.raise_for_status()
            rows = resp.json()
            out: dict[str, float] = {}
            for row in rows if isinstance(rows, list) else []:
                cid = row.get("id")
                pct = row.get("price_change_percentage_7d_in_currency")
                if cid in id_to_coin and pct is not None:
                    out[id_to_coin[cid]] = float(pct) / 100.0
            return out
        except Exception as exc:  # noqa: BLE001
            log.warning("coingecko price-change attempt %s failed: %s", attempt, exc)
    return {}


def fetch_price_change_7d(coin: str) -> Optional[float]:
    """Single-coin convenience wrapper (drop-in for the old yfinance fn)."""
    return fetch_price_changes_7d([coin]).get(coin)
