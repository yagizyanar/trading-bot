"""Performance metrics for backtesting.

Reports: Sharpe, Sortino, max drawdown, win rate, profit factor, avg win/loss.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from config.settings import (
    BACKTEST_MAX_DRAWDOWN,
    BACKTEST_MIN_PROFIT_FACTOR,
    BACKTEST_MIN_SHARPE,
    BACKTEST_MIN_WIN_RATE,
)


@dataclass(frozen=True)
class PerformanceMetrics:
    total_return: float
    annualised_return: float
    sharpe: float
    sortino: float
    max_drawdown: float          # negative number, e.g. -0.18 = -18%
    avg_drawdown: float
    win_rate: float
    loss_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    win_loss_ratio: float
    n_trades: int


def compute_metrics(
    trade_returns: Sequence[float],
    daily_returns: Optional[Sequence[float]] = None,
    periods_per_year: int = 252,
) -> PerformanceMetrics:
    """Compute the standard hedge-fund metric pack.

    `trade_returns` are per-trade P&L percentages (-1.0 .. +inf).
    `daily_returns` (optional) are equity-curve daily returns; if absent we
    derive them from trade_returns spread across the trades' duration.
    """
    if not trade_returns:
        return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    arr = np.array(trade_returns, dtype=float)
    n = len(arr)

    wins = arr[arr > 0]
    losses = arr[arr < 0]
    win_rate = float(len(wins) / n) if n else 0.0
    loss_rate = float(len(losses) / n) if n else 0.0
    gross_win = float(wins.sum())
    gross_loss = float(-losses.sum())
    profit_factor = float(gross_win / gross_loss) if gross_loss > 0 else math.inf
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    win_loss_ratio = float(abs(avg_win / avg_loss)) if avg_loss != 0 else math.inf

    equity = (1.0 + arr).cumprod()
    total_return = float(equity[-1] - 1.0)
    annualised_return = (
        float((1.0 + total_return) ** (periods_per_year / max(1, n)) - 1.0)
        if total_return > -1.0 else -1.0
    )

    if daily_returns is None:
        daily = arr
    else:
        daily = np.array(daily_returns, dtype=float)

    if len(daily) > 1 and daily.std(ddof=1) > 0:
        sharpe = float((daily.mean() / daily.std(ddof=1)) * math.sqrt(periods_per_year))
        downside = daily[daily < 0]
        downside_std = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
        sortino = (
            float((daily.mean() / downside_std) * math.sqrt(periods_per_year))
            if downside_std > 0 else math.inf
        )
    else:
        sharpe = 0.0
        sortino = 0.0

    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    max_dd = float(drawdown.min()) if len(drawdown) else 0.0
    avg_dd = float(drawdown[drawdown < 0].mean()) if (drawdown < 0).any() else 0.0

    return PerformanceMetrics(
        total_return=total_return,
        annualised_return=annualised_return,
        sharpe=sharpe,
        sortino=sortino if math.isfinite(sortino) else 0.0,
        max_drawdown=max_dd,
        avg_drawdown=avg_dd,
        win_rate=win_rate,
        loss_rate=loss_rate,
        profit_factor=profit_factor if math.isfinite(profit_factor) else 999.0,
        avg_win=avg_win,
        avg_loss=avg_loss,
        win_loss_ratio=win_loss_ratio if math.isfinite(win_loss_ratio) else 999.0,
        n_trades=n,
    )


def meets_minimum_requirements(metrics: PerformanceMetrics, min_trades: int = 100) -> tuple[bool, list[str]]:
    """Check the go-live gate from the backtesting-protocol skill.

    Returns (passed, failure_reasons).
    """
    failures: list[str] = []
    if metrics.win_rate < BACKTEST_MIN_WIN_RATE:
        failures.append(f"win_rate {metrics.win_rate:.2%} < {BACKTEST_MIN_WIN_RATE:.0%}")
    if metrics.profit_factor < BACKTEST_MIN_PROFIT_FACTOR:
        failures.append(f"profit_factor {metrics.profit_factor:.2f} < {BACKTEST_MIN_PROFIT_FACTOR}")
    if metrics.sharpe < BACKTEST_MIN_SHARPE:
        failures.append(f"sharpe {metrics.sharpe:.2f} < {BACKTEST_MIN_SHARPE}")
    if metrics.max_drawdown < -BACKTEST_MAX_DRAWDOWN:
        failures.append(f"max_drawdown {metrics.max_drawdown:.2%} below -{BACKTEST_MAX_DRAWDOWN:.0%}")
    if metrics.n_trades < min_trades:
        failures.append(f"n_trades {metrics.n_trades} < {min_trades}")
    if metrics.sharpe > 3.0:
        failures.append(f"sharpe {metrics.sharpe:.2f} > 3.0 — suspicious (overfit?)")
    return (len(failures) == 0, failures)
