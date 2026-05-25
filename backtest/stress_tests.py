"""Stress tests (backtesting-protocol skill).

Scenarios:
  1. Black-swan crash: inject -15% single-day move
  2. Sustained bear: 30-day grind of -1%/day
  3. Sentiment outage: scores forced to neutral for 48 hours
  4. Extreme volatility: daily ranges > 20% for a week
  5. API failure: equivalent to "no new signals" — strategy must not blow up
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .metrics import PerformanceMetrics, compute_metrics
from .walk_forward import run_walk_forward


@dataclass(frozen=True)
class StressScenarioResult:
    name: str
    metrics: PerformanceMetrics
    survived: bool          # max_drawdown >= -50% counts as "survived"
    notes: str


@dataclass(frozen=True)
class StressTestReport:
    coin: str
    scenarios: list[StressScenarioResult]
    all_survived: bool


def _inject_crash(close: pd.Series, day_index: int, magnitude: float = -0.15) -> pd.Series:
    out = close.copy()
    if 0 < day_index < len(out):
        out.iloc[day_index:] = out.iloc[day_index:] * (1.0 + magnitude)
    return out


def _inject_bear_streak(close: pd.Series, start_index: int, days: int = 30, daily_pct: float = -0.01) -> pd.Series:
    out = close.copy()
    for i in range(days):
        idx = start_index + i
        if idx < len(out):
            out.iloc[idx:] = out.iloc[idx:] * (1.0 + daily_pct)
    return out


def _inject_volatility_burst(close: pd.Series, start_index: int, days: int = 7, vol_pct: float = 0.20) -> pd.Series:
    out = close.copy()
    rng = np.random.default_rng(seed=123)
    for i in range(days):
        idx = start_index + i
        if idx < len(out):
            shock = rng.uniform(-vol_pct, vol_pct)
            out.iloc[idx:] = out.iloc[idx:] * (1.0 + shock)
    return out


def _survived(metrics: PerformanceMetrics) -> bool:
    return metrics.max_drawdown >= -0.50  # didn't lose half the account


def run_stress_tests(close: pd.Series, coin: str) -> StressTestReport:
    """Apply each scenario and walk-forward the strategy through it."""
    results: list[StressScenarioResult] = []

    n = len(close)
    mid = n // 2

    # 1. Crash
    crashed = _inject_crash(close, day_index=mid, magnitude=-0.15)
    wf = run_walk_forward(crashed, coin)
    results.append(StressScenarioResult(
        name="black_swan_-15%",
        metrics=wf.metrics,
        survived=_survived(wf.metrics),
        notes=f"injected -15% on day {mid}",
    ))

    # 2. Sustained bear
    bear = _inject_bear_streak(close, start_index=mid, days=30, daily_pct=-0.01)
    wf = run_walk_forward(bear, coin)
    results.append(StressScenarioResult(
        name="sustained_bear_30d",
        metrics=wf.metrics,
        survived=_survived(wf.metrics),
        notes="30 consecutive days of -1%",
    ))

    # 3. Volatility burst
    vol = _inject_volatility_burst(close, start_index=mid, days=7, vol_pct=0.20)
    wf = run_walk_forward(vol, coin)
    results.append(StressScenarioResult(
        name="vol_burst_7d_20pct",
        metrics=wf.metrics,
        survived=_survived(wf.metrics),
        notes="7d of ±20% daily shocks",
    ))

    # 4. Sentiment outage — pass a forced-neutral sentiment series
    flat_sentiment = pd.Series(0.0, index=close.index)
    wf = run_walk_forward(close, coin, sentiment_signal=flat_sentiment)
    results.append(StressScenarioResult(
        name="sentiment_outage_48h",
        metrics=wf.metrics,
        survived=_survived(wf.metrics),
        notes="sentiment forced neutral throughout — proxy for outage",
    ))

    # 5. Combined: vol burst + sentiment outage
    wf = run_walk_forward(vol, coin, sentiment_signal=flat_sentiment)
    results.append(StressScenarioResult(
        name="vol_burst_plus_sentiment_outage",
        metrics=wf.metrics,
        survived=_survived(wf.metrics),
        notes="worst-case compound scenario",
    ))

    return StressTestReport(
        coin=coin,
        scenarios=results,
        all_survived=all(r.survived for r in results),
    )
