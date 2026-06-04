"""Daily live-vs-backtest drift monitor (roadmap item 8).

Reads Freqtrade's closed trades, computes rolling 30-day Sharpe / win-rate /
avg-profit-per-trade / fee-drag, persists a DriftSnapshot, and Telegram-alerts when
the live edge diverges from the backtest:
  - rolling 30d Sharpe < 0.5   (backtest baseline ~1.0)
  - 30d win rate < 30%
  - avg profit per trade negative for 7 consecutive daily runs

Run daily from cron. `--test-alert` sends a Telegram test message (verifies wiring).
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta, timezone

from config.settings import PROJECT_ROOT
from analytics.drift import (
    compute_metrics, evaluate_alerts, format_alert, load_closed_trades,
)
from notifications import send_telegram

log = logging.getLogger(__name__)

FREQTRADE_DB = str(PROJECT_ROOT / "user_data" / "tradesv3.sqlite")
WINDOW_DAYS = 30


def run_once(send_alerts: bool = True) -> dict:
    now = datetime.now(timezone.utc)
    trades = load_closed_trades(FREQTRADE_DB, now - timedelta(days=WINDOW_DAYS))
    m = compute_metrics(trades, WINDOW_DAYS, now)

    from database import DriftSnapshot, SessionLocal
    with SessionLocal() as session:
        prev = session.query(DriftSnapshot).order_by(DriftSnapshot.ts.desc()).first()
        prev_streak = int(prev.consecutive_negative_days) if prev else 0
        streak = (prev_streak + 1) if m.avg_profit_negative else 0
        alerts = evaluate_alerts(m, streak)
        session.add(DriftSnapshot(
            ts=now, window_days=m.window_days, trades_30d=m.trades,
            rolling_sharpe_30d=m.rolling_sharpe, win_rate_30d=m.win_rate,
            avg_profit_per_trade_30d=m.avg_profit_per_trade,
            actual_cost_bps=m.actual_cost_bps, expected_cost_bps=m.expected_cost_bps,
            consecutive_negative_days=streak,
            alerts="; ".join(alerts) if alerts else None,
        ))
        session.commit()

    log.info(
        "drift: trades=%d sharpe=%s winrate=%s avgprofit=%s fees_bps=%s neg_streak=%d alerts=%s",
        m.trades,
        f"{m.rolling_sharpe:.2f}" if m.rolling_sharpe is not None else "n/a",
        f"{m.win_rate:.2%}" if m.win_rate is not None else "n/a",
        f"{m.avg_profit_per_trade:.4f}" if m.avg_profit_per_trade is not None else "n/a",
        f"{m.actual_cost_bps:.1f}" if m.actual_cost_bps is not None else "n/a",
        streak, alerts or "none",
    )

    if send_alerts and alerts:
        ok = send_telegram(format_alert(m, streak, alerts))
        log.info("drift alert sent: %s (%d alert(s))", ok, len(alerts))

    return {"metrics": m, "streak": streak, "alerts": alerts}


def main() -> None:
    from routines.base import setup_routine_logging
    setup_routine_logging()
    if "--test-alert" in sys.argv:
        ok = send_telegram(
            "✅ <b>Drift monitor test</b> — Telegram alerting is wired correctly "
            "(roadmap item 8). This is a test, not a real drift event.")
        log.info("test-alert send_telegram returned: %s", ok)
        print(f"test alert sent: {ok}")
        return
    run_once()


if __name__ == "__main__":
    main()
