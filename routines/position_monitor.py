"""Every-1-min position monitor.

Reads OPEN trades from Freqtrade's REST API (the executor's ground truth)
and observes them. Falls back to our `trades` DB table if Freqtrade is
unreachable.

What we DO:
  - Detect 3%+ price moves in the last 5 minutes → append [ANOMALY] line
    to memory/trade_log.md.
  - Log [SL_NEAR] / [TP_NEAR] heads-up if PnL has crossed the stop-loss
    or take-profit threshold (Freqtrade will close the trade itself; we
    just observe).

What we do NOT do:
  - Actually close Freqtrade-managed trades. Freqtrade enforces stoploss
    (-5%) and minimal_roi (+15%) on its own internal loop (every 5s per
    `process_throttle_secs`), so duplicating closes here would race.

Lockfile is still respected. Errors are logged but don't crash the monitor.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from config.settings import STOP_LOSS_PCT, TAKE_PROFIT_PCT
from database import SessionLocal, Trade
from memory.memory_io import append_lesson, log_circuit_breaker
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
    stop_losses_triggered: int     # observed (Freqtrade enforces the close)
    take_profits_triggered: int    # observed (Freqtrade enforces the close)
    anomalies_logged: int
    source: str                    # "freqtrade" | "db" | "none"
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


def _check_one_freqtrade_trade(t: dict) -> tuple[int, int, int]:
    """Return (sl_observed, tp_observed, anomaly_logged) counts for one trade."""
    pair = t.get("pair", "") or ""
    coin = pair.split("/")[0]
    if not coin:
        return 0, 0, 0

    # Freqtrade reports profit_pct as a percentage number (e.g. -0.03 == -0.03%).
    raw_pct = t.get("profit_pct")
    pnl_pct = (float(raw_pct) / 100.0) if raw_pct is not None else 0.0
    is_short = bool(t.get("is_short", False))
    side = "SHORT" if is_short else "LONG"
    entry = t.get("open_rate")
    curr = t.get("current_rate")

    sl = tp = anom = 0

    if pnl_pct <= -STOP_LOSS_PCT:
        log_circuit_breaker(
            level="SL_NEAR",
            trigger=f"{coin} {side} at {pnl_pct:+.2%} ≤ -{STOP_LOSS_PCT:.0%} — Freqtrade will close",
            equity=0.0,
            extra=f"entry={entry} now={curr}",
        )
        sl = 1
    elif pnl_pct >= TAKE_PROFIT_PCT:
        log_circuit_breaker(
            level="TP_NEAR",
            trigger=f"{coin} {side} at {pnl_pct:+.2%} ≥ +{TAKE_PROFIT_PCT:.0%} — Freqtrade will close",
            equity=0.0,
            extra=f"entry={entry} now={curr}",
        )
        tp = 1

    # Independent anomaly check — useful even mid-trade, regardless of PnL
    move5 = _five_min_change(coin)
    if move5 is not None and abs(move5) >= ANOMALOUS_MOVE_THRESHOLD:
        log_circuit_breaker(
            level="ANOMALY",
            trigger=f"{coin} moved {move5:+.2%} in 5min while {side} position is OPEN (pnl_pct={pnl_pct:+.2%})",
            equity=0.0,
            extra=f"entry={entry} now={curr}",
        )
        anom = 1

    return sl, tp, anom


def _check_one_db_trade(session, trade: Trade) -> tuple[int, int, int]:
    """DB-backed fallback: same checks but using our trades-table row."""
    price = _current_price(trade.coin)
    if price is None:
        return 0, 0, 0

    if trade.side == "LONG":
        pnl_pct = (price / trade.entry_price) - 1.0
    else:
        pnl_pct = (trade.entry_price / price) - 1.0

    sl = tp = anom = 0

    if pnl_pct <= -STOP_LOSS_PCT:
        # DB-managed trade — we actually close it ourselves (Freqtrade isn't watching this row).
        from memory.memory_io import append_trade
        now = datetime.now(timezone.utc)
        trade.exit_price = price
        trade.exit_ts = now
        trade.pnl_pct = pnl_pct
        trade.pnl_usd = trade.quantity * (
            (price - trade.entry_price) if trade.side == "LONG" else (trade.entry_price - price)
        )
        trade.outcome = "LOSS"
        trade.reason_out = f"Stop-loss hit at {pnl_pct:.2%} (DB-managed trade)"
        session.add(trade)
        append_trade(
            coin=trade.coin, direction=trade.side, entry=trade.entry_price,
            exit_price=price, quantity=trade.quantity, leverage=trade.leverage,
            pnl_usd=trade.pnl_usd, pnl_pct=pnl_pct,
            reason_in=trade.reason_in or "n/a", reason_out=trade.reason_out,
            outcome="LOSS",
        )
        sl = 1
    elif pnl_pct >= TAKE_PROFIT_PCT:
        from memory.memory_io import append_trade
        now = datetime.now(timezone.utc)
        trade.exit_price = price
        trade.exit_ts = now
        trade.pnl_pct = pnl_pct
        trade.pnl_usd = trade.quantity * (
            (price - trade.entry_price) if trade.side == "LONG" else (trade.entry_price - price)
        )
        trade.outcome = "WIN"
        trade.reason_out = f"Take-profit hit at {pnl_pct:+.2%} (DB-managed trade)"
        session.add(trade)
        append_trade(
            coin=trade.coin, direction=trade.side, entry=trade.entry_price,
            exit_price=price, quantity=trade.quantity, leverage=trade.leverage,
            pnl_usd=trade.pnl_usd, pnl_pct=pnl_pct,
            reason_in=trade.reason_in or "n/a", reason_out=trade.reason_out,
            outcome="WIN",
        )
        tp = 1

    move5 = _five_min_change(trade.coin)
    if move5 is not None and abs(move5) >= ANOMALOUS_MOVE_THRESHOLD:
        log_circuit_breaker(
            level="ANOMALY",
            trigger=f"{trade.coin} moved {move5:+.2%} in 5min (pnl_pct={pnl_pct:+.2%})",
            equity=0.0,
            extra=f"Side={trade.side} entry=${trade.entry_price:.4f} now=${price:.4f}",
        )
        anom = 1

    return sl, tp, anom


def run() -> MonitorResult:
    started = datetime.now(timezone.utc)
    if is_locked():
        return MonitorResult(
            name="position_monitor",
            started_at=started, finished_at=datetime.now(timezone.utc),
            open_positions_checked=0, stop_losses_triggered=0,
            take_profits_triggered=0, anomalies_logged=0,
            source="none", skipped=True, skip_reason="lockfile present", error=None,
        )

    sl_count = tp_count = anom_count = checked = 0
    source = "none"
    err: Optional[str] = None

    # Prefer Freqtrade's view of open positions; fall back to our trades table
    # if Freqtrade is unreachable (then we may also need to actively close
    # SL/TP trades since nothing else is doing it).
    try:
        from dashboard.backend.freqtrade_client import fetch_status
        live = fetch_status()
    except Exception as exc:  # noqa: BLE001
        log.warning("freqtrade fetch_status failed: %s", exc)
        live = None

    try:
        if live is not None:
            source = "freqtrade"
            checked = len(live)
            for t in live:
                sl, tp, anom = _check_one_freqtrade_trade(t)
                sl_count += sl
                tp_count += tp
                anom_count += anom
        else:
            source = "db"
            with SessionLocal() as session:
                open_trades = session.query(Trade).filter(Trade.outcome == "OPEN").all()
                checked = len(open_trades)
                for tr in open_trades:
                    sl, tp, anom = _check_one_db_trade(session, tr)
                    sl_count += sl
                    tp_count += tp
                    anom_count += anom
                session.commit()
    except Exception as exc:  # noqa: BLE001
        err = str(exc)
        log.exception("position_monitor failed")
        try:
            append_lesson(
                observation=f"position_monitor crashed: {exc}",
                signal_involved="N/A",
                worked_or_failed="FAILED",
                action_next_time="Investigate Binance/Freqtrade/DB connectivity",
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
        source=source,
        skipped=False,
        skip_reason=None,
        error=err,
    )


def main() -> int:
    res = run()
    log.info(
        "position_monitor: source=%s checked=%s sl=%s tp=%s anomalies=%s skipped=%s err=%s",
        res.source, res.open_positions_checked, res.stop_losses_triggered,
        res.take_profits_triggered, res.anomalies_logged, res.skipped, res.error,
    )
    return 0 if res.error is None else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
