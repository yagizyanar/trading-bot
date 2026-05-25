"""Every-5-min position monitor.

Lighter than the hourly routines — doesn't read all memory files, doesn't run
the full circuit-breaker eval. Just iterates open trades and checks:

  - Stop loss hit  : pnl_pct <= -STOP_LOSS_PCT (default -5%)  → mark LOSS, close.
  - Take profit hit: pnl_pct >= +TAKE_PROFIT_PCT (default +15%) → mark WIN, close.
  - Abnormal move  : |last 5min price change| >= 3%           → alert to trade_log.

Lockfile is still respected. Errors are logged but don't crash the monitor.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from config.settings import STOP_LOSS_PCT, TAKE_PROFIT_PCT
from database import SessionLocal, Trade
from memory.memory_io import (
    append_lesson,
    append_trade,
    log_circuit_breaker,
)
from risk.lockfile import is_locked
from sentiment.binance_data import fetch_binance_ohlcv

log = logging.getLogger(__name__)

ANOMALOUS_MOVE_THRESHOLD = 0.03  # 3% in 5 minutes


@dataclass(frozen=True)
class MonitorResult:
    name: str
    started_at: datetime
    finished_at: datetime
    open_positions_checked: int
    stop_losses_triggered: int
    take_profits_triggered: int
    anomalies_logged: int
    skipped: bool
    skip_reason: Optional[str]
    error: Optional[str]


def _current_price(coin: str) -> Optional[float]:
    df = fetch_binance_ohlcv(f"{coin}USDT", interval="1m", limit=6)
    if df is None or df.empty:
        return None
    return float(df["close"].iloc[-1])


def _five_min_change(coin: str) -> Optional[float]:
    """5-minute return: close_now / close_5min_ago - 1."""
    df = fetch_binance_ohlcv(f"{coin}USDT", interval="1m", limit=6)
    if df is None or len(df) < 6:
        return None
    return float(df["close"].iloc[-1] / df["close"].iloc[-6] - 1.0)


def _close_position(session, trade: Trade, current_price: float, outcome: str, reason: str) -> None:
    """Mark a trade closed in-DB and append to trade_log.md."""
    now = datetime.now(timezone.utc)
    trade.exit_price = current_price
    trade.exit_ts = now
    if trade.side == "LONG":
        trade.pnl_usd = trade.quantity * (current_price - trade.entry_price)
        trade.pnl_pct = (current_price / trade.entry_price) - 1.0
    else:
        trade.pnl_usd = trade.quantity * (trade.entry_price - current_price)
        trade.pnl_pct = (trade.entry_price / current_price) - 1.0
    trade.outcome = outcome
    trade.reason_out = reason
    session.add(trade)
    append_trade(
        coin=trade.coin,
        direction=trade.side,
        entry=trade.entry_price,
        exit_price=current_price,
        quantity=trade.quantity,
        leverage=trade.leverage,
        pnl_usd=trade.pnl_usd,
        pnl_pct=trade.pnl_pct,
        reason_in=trade.reason_in or "n/a",
        reason_out=reason,
        outcome=outcome,
    )


def run() -> MonitorResult:
    started = datetime.now(timezone.utc)
    if is_locked():
        return MonitorResult(
            name="position_monitor",
            started_at=started, finished_at=datetime.now(timezone.utc),
            open_positions_checked=0, stop_losses_triggered=0,
            take_profits_triggered=0, anomalies_logged=0,
            skipped=True, skip_reason="lockfile present", error=None,
        )

    sl_count = 0
    tp_count = 0
    anom_count = 0
    checked = 0
    err: Optional[str] = None

    try:
        with SessionLocal() as session:
            open_trades = session.query(Trade).filter(Trade.outcome == "OPEN").all()
            checked = len(open_trades)

            for trade in open_trades:
                price = _current_price(trade.coin)
                if price is None:
                    continue

                if trade.side == "LONG":
                    pnl_pct = (price / trade.entry_price) - 1.0
                else:
                    pnl_pct = (trade.entry_price / price) - 1.0

                if pnl_pct <= -STOP_LOSS_PCT:
                    _close_position(session, trade, price, "LOSS",
                                    f"Stop-loss hit at {pnl_pct:.2%} (threshold -{STOP_LOSS_PCT:.0%})")
                    sl_count += 1
                    continue

                if pnl_pct >= TAKE_PROFIT_PCT:
                    _close_position(session, trade, price, "WIN",
                                    f"Take-profit hit at {pnl_pct:+.2%} (threshold +{TAKE_PROFIT_PCT:.0%})")
                    tp_count += 1
                    continue

                # Anomaly check — only logged, not actioned (manual judgment)
                move5 = _five_min_change(trade.coin)
                if move5 is not None and abs(move5) >= ANOMALOUS_MOVE_THRESHOLD:
                    anom_count += 1
                    log_circuit_breaker(
                        level="ANOMALY",
                        trigger=f"{trade.coin} moved {move5:+.2%} in 5min while position is OPEN (pnl_pct={pnl_pct:+.2%})",
                        equity=0.0,
                        extra=f"Side={trade.side} entry=${trade.entry_price:.4f} now=${price:.4f}",
                    )

            session.commit()

    except Exception as exc:  # noqa: BLE001
        err = str(exc)
        log.exception("position_monitor failed")
        try:
            append_lesson(
                observation=f"position_monitor crashed: {exc}",
                signal_involved="N/A",
                worked_or_failed="FAILED",
                action_next_time="Investigate Binance connectivity and DB",
            )
        except Exception:  # noqa: BLE001
            pass

    return MonitorResult(
        name="position_monitor",
        started_at=started,
        finished_at=datetime.now(timezone.utc),
        open_positions_checked=checked,
        stop_losses_triggered=sl_count,
        take_profits_triggered=tp_count,
        anomalies_logged=anom_count,
        skipped=False,
        skip_reason=None,
        error=err,
    )


def main() -> int:
    res = run()
    log.info("position_monitor: checked=%s sl=%s tp=%s anomalies=%s skipped=%s err=%s",
             res.open_positions_checked, res.stop_losses_triggered, res.take_profits_triggered,
             res.anomalies_logged, res.skipped, res.error)
    return 0 if res.error is None else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
