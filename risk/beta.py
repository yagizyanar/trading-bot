"""BTC-beta estimation for the aggregate net-beta cap (Item 6, 2026-06-03).

beta_to_btc = corr(coin, btc) * (vol_coin / vol_btc), computed on daily returns
over a trailing window. Used so the portfolio gate can limit the book's NET
directional exposure — preventing "10 correlated shorts" from becoming one
giant undiversified bet (the -44%-drawdown failure mode).

Neutral default of 1.0 on any data problem (better to assume full beta than to
under-count risk).
"""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

from sentiment.binance_data import fetch_binance_ohlcv

log = logging.getLogger(__name__)

BETA_WINDOW = 30   # trailing daily-return window


def compute_beta_to_btc(coin_close, btc_close, window: int = BETA_WINDOW) -> float:
    """Daily-return beta of a coin to BTC: corr * (coin_vol / btc_vol).

    Returns 1.0 (neutral) if data is insufficient or degenerate — under-counting
    beta would defeat the risk cap, so we default to "fully correlated".
    """
    try:
        cc = pd.Series(coin_close).astype(float).pct_change().dropna()
        bc = pd.Series(btc_close).astype(float).pct_change().dropna()
        n = min(len(cc), len(bc), window)
        if n < 10:
            return 1.0
        cc = cc.iloc[-n:].to_numpy()
        bc = bc.iloc[-n:].to_numpy()
        bvol = float(bc.std(ddof=1))
        cvol = float(cc.std(ddof=1))
        if bvol <= 0:
            return 1.0
        corr = float(np.corrcoef(cc, bc)[0, 1])
        if np.isnan(corr):
            return 1.0
        return float(corr * (cvol / bvol))
    except Exception as exc:  # noqa: BLE001
        log.warning("beta_to_btc compute failed: %s", exc)
        return 1.0


def compute_book_betas(coins: Iterable[str], days: int = 120) -> dict[str, float]:
    """Fetch daily candles and compute each coin's BTC-beta. {coin: beta}.

    Defaults every coin to 1.0 if BTC data is unavailable. ~1 fetch per coin +
    1 for BTC; runs hourly in market_evaluation (well within the cron window).
    """
    btc_df = fetch_binance_ohlcv("BTCUSDT", interval="1d", limit=days)
    if btc_df is None or btc_df.empty or "close" not in btc_df.columns:
        log.warning("BTC daily fetch failed — defaulting all betas to 1.0")
        return {c: 1.0 for c in coins}
    btc_close = btc_df["close"]
    out: dict[str, float] = {}
    for c in coins:
        df = fetch_binance_ohlcv(f"{c}USDT", interval="1d", limit=days)
        out[c] = (
            compute_beta_to_btc(df["close"], btc_close)
            if (df is not None and not df.empty and "close" in df.columns)
            else 1.0
        )
    return out
