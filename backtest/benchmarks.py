"""Benchmark strategies. The Markov+sentiment+technical strategy must beat
all three before going live (backtesting-protocol skill)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .metrics import PerformanceMetrics, compute_metrics


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    metrics: PerformanceMetrics
    daily_returns: pd.Series
    equity_curve: pd.Series


def _equity_from_returns(daily: pd.Series) -> pd.Series:
    return (1.0 + daily.fillna(0.0)).cumprod()


def buy_and_hold(close: pd.Series) -> BenchmarkResult:
    """100% long the whole period."""
    daily = close.pct_change().dropna()
    metrics = compute_metrics(trade_returns=[float(daily.sum())], daily_returns=daily.values.tolist())
    return BenchmarkResult(
        name="buy_and_hold",
        metrics=metrics,
        daily_returns=daily,
        equity_curve=_equity_from_returns(daily),
    )


def sma_200(close: pd.Series) -> BenchmarkResult:
    """Long when close > 200-day SMA, flat otherwise. Standard trend follower."""
    sma = close.rolling(200).mean()
    position = (close > sma).astype(float)
    daily_returns = close.pct_change().fillna(0.0) * position.shift(1).fillna(0.0)
    daily_returns = daily_returns.dropna()
    trade_returns = _segment_returns(position, close)
    metrics = compute_metrics(trade_returns=trade_returns, daily_returns=daily_returns.values.tolist())
    return BenchmarkResult(
        name="sma_200",
        metrics=metrics,
        daily_returns=daily_returns,
        equity_curve=_equity_from_returns(daily_returns),
    )


def random_entry(close: pd.Series, seed: int = 42, p: float = 0.5) -> BenchmarkResult:
    """Random long/flat with p probability of being long per day. Coin-flip baseline."""
    rng = np.random.default_rng(seed)
    position = pd.Series(rng.binomial(1, p, size=len(close)), index=close.index, dtype=float)
    daily_returns = close.pct_change().fillna(0.0) * position.shift(1).fillna(0.0)
    daily_returns = daily_returns.dropna()
    trade_returns = _segment_returns(position, close)
    metrics = compute_metrics(trade_returns=trade_returns, daily_returns=daily_returns.values.tolist())
    return BenchmarkResult(
        name="random_entry",
        metrics=metrics,
        daily_returns=daily_returns,
        equity_curve=_equity_from_returns(daily_returns),
    )


def _segment_returns(position: pd.Series, close: pd.Series) -> list[float]:
    """Convert a binary position series into per-trade percentage returns."""
    pos = position.fillna(0.0).astype(int).to_numpy()
    px = close.to_numpy()
    trade_returns: list[float] = []
    in_pos = False
    entry_px = 0.0
    for i in range(len(pos)):
        if pos[i] == 1 and not in_pos:
            entry_px = px[i]
            in_pos = True
        elif pos[i] == 0 and in_pos:
            trade_returns.append(float(px[i] / entry_px - 1.0))
            in_pos = False
    if in_pos and len(px) > 0:
        trade_returns.append(float(px[-1] / entry_px - 1.0))
    return trade_returns
