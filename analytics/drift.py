"""Live-vs-backtest drift metrics (roadmap item 8).

Computes rolling performance metrics from Freqtrade's realized (closed) trades and
compares them to the backtest baselines, so we get an early warning when the live
edge decays / the strategy is overfit. Pure logic + a thin sqlite reader — the
routine layer handles persistence, the streak counter, and Telegram.

Metrics (default 30-day window):
  - rolling_sharpe        : annualized Sharpe of DAILY realized PnL (closed trades
                            bucketed by close-day; flat days count as 0).
  - win_rate              : fraction of closed trades with close_profit > 0.
  - avg_profit_per_trade  : mean close_profit (fraction) across the window.
  - actual_cost_bps       : realized round-trip exchange fees as bps of notional,
                            vs expected_cost_bps (the backtest cost assumption).

Backtest baselines: cross-regime Sharpe ~1.0; harness COST 0.0007/side (14 bps RT).
"""
from __future__ import annotations

import math
import sqlite3
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

# Backtest baselines / alert thresholds
BACKTEST_SHARPE_BASELINE = 1.0
EXPECTED_COST_PER_SIDE = 0.0007                       # fee+slippage assumed in OOS harness
EXPECTED_COST_BPS_RT = EXPECTED_COST_PER_SIDE * 2 * 1e4   # 14 bps round-trip

SHARPE_ALERT = 0.5
WINRATE_ALERT = 0.30
NEG_PROFIT_STREAK_ALERT = 7
MIN_TRADES_FOR_ALERT = 10        # don't alert on thin data (noisy metrics)
ANNUALIZATION = 365              # crypto trades every day


@dataclass
class DriftMetrics:
    window_days: int
    trades: int
    rolling_sharpe: float | None
    win_rate: float | None
    avg_profit_per_trade: float | None      # fraction (e.g. 0.012 = +1.2%/trade)
    actual_cost_bps: float | None           # realized round-trip fees, bps of notional
    expected_cost_bps: float
    avg_profit_negative: bool               # is avg_profit_per_trade < 0 this run


def _to_date(s):
    s = str(s).split(".")[0].replace("T", " ").replace("Z", "").strip()
    return datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S").date()


def load_closed_trades(db_path: str, since: datetime) -> list[dict]:
    """Closed trades with close_date >= since, from the Freqtrade sqlite."""
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            "SELECT close_date, close_profit, close_profit_abs, stake_amount, "
            "fee_open_cost, fee_close_cost, funding_fees "
            "FROM trades WHERE is_open=0 AND close_date IS NOT NULL AND close_date >= ? "
            "ORDER BY close_date",
            (since.strftime("%Y-%m-%d %H:%M:%S"),),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        con.close()


def _sharpe(daily_returns: list[float]) -> float | None:
    if len(daily_returns) < 2:
        return None
    sd = statistics.stdev(daily_returns)            # sample std (ddof=1), matches OOS harness
    if sd == 0:
        return None
    return statistics.fmean(daily_returns) / sd * math.sqrt(ANNUALIZATION)


def compute_metrics(trades: list[dict], window_days: int = 30, now: datetime | None = None) -> DriftMetrics:
    now = now or datetime.now(timezone.utc)
    n = len(trades)
    if n == 0:
        return DriftMetrics(window_days, 0, None, None, None, None, EXPECTED_COST_BPS_RT, False)

    wins = sum(1 for t in trades if (t.get("close_profit") or 0.0) > 0)
    win_rate = wins / n
    avg_profit = sum((t.get("close_profit") or 0.0) for t in trades) / n

    # Daily realized PnL bucketed by close-day, flat days = 0, over the full window.
    by_day: dict = {}
    for t in trades:
        d = _to_date(t["close_date"])
        by_day[d] = by_day.get(d, 0.0) + (t.get("close_profit_abs") or 0.0)
    series, cur, end = [], (now - timedelta(days=window_days)).date(), now.date()
    while cur <= end:
        series.append(by_day.get(cur, 0.0))
        cur += timedelta(days=1)
    sharpe = _sharpe(series)

    notional = sum((t.get("stake_amount") or 0.0) for t in trades)
    fees = sum((t.get("fee_open_cost") or 0.0) + (t.get("fee_close_cost") or 0.0) for t in trades)
    cost_bps = (fees / notional * 1e4) if notional > 0 else None

    return DriftMetrics(window_days, n, sharpe, win_rate, avg_profit, cost_bps,
                        EXPECTED_COST_BPS_RT, avg_profit < 0)


def evaluate_alerts(m: DriftMetrics, consecutive_negative_days: int) -> list[str]:
    """Which drift alerts fire. Sharpe/win-rate gated on >= MIN_TRADES_FOR_ALERT."""
    alerts: list[str] = []
    if m.trades >= MIN_TRADES_FOR_ALERT:
        if m.rolling_sharpe is not None and m.rolling_sharpe < SHARPE_ALERT:
            alerts.append(
                f"Rolling 30d Sharpe {m.rolling_sharpe:.2f} < {SHARPE_ALERT} "
                f"(backtest baseline ~{BACKTEST_SHARPE_BASELINE:.1f})")
        if m.win_rate is not None and m.win_rate < WINRATE_ALERT:
            alerts.append(f"30d win rate {m.win_rate:.0%} < {WINRATE_ALERT:.0%}")
    if consecutive_negative_days >= NEG_PROFIT_STREAK_ALERT:
        alerts.append(
            f"Avg profit/trade negative for {consecutive_negative_days} consecutive days")
    return alerts


def _f(x, nd=2):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "n/a"


def _esc(s: str) -> str:
    """Escape for Telegram HTML parse_mode (alerts are stored as plain text)."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def format_alert(m: DriftMetrics, streak: int, alerts: list[str]) -> str:
    bullets = "\n".join(f"• {_esc(a)}" for a in alerts)
    wr = f"{m.win_rate:.0%}" if m.win_rate is not None else "n/a"
    ap = f"{m.avg_profit_per_trade*100:.2f}%" if m.avg_profit_per_trade is not None else "n/a"
    return (
        "⚠️ <b>DRIFT ALERT</b> — live performance diverging from backtest\n"
        f"{bullets}\n"
        f"<i>30d window: {m.trades} trades · Sharpe {_f(m.rolling_sharpe)} · "
        f"win {wr} · avg {ap}/trade · fees {_f(m.actual_cost_bps,1)}bps "
        f"(exp {_f(m.expected_cost_bps,1)})</i>"
    )
