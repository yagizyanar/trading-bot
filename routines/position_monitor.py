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

from config.settings import MEMORY_DIR, STOP_LOSS_PCT, TAKE_PROFIT_PCT
from database import SessionLocal, Trade
from memory.memory_io import append_lesson, append_trade, log_circuit_breaker
from risk.lockfile import is_locked
from sentiment.binance_data import fetch_binance_ohlcv

log = logging.getLogger(__name__)

ANOMALOUS_MOVE_THRESHOLD = 0.03  # 3% in 5 minutes

# Watermark file storing the highest Freqtrade trade_id we've mirrored to
# trade_log.md. Prevents duplicate appends across restarts.
TRADE_LOG_WATERMARK = MEMORY_DIR / ".last_mirrored_trade_id"


def _read_watermark() -> int:
    if not TRADE_LOG_WATERMARK.exists():
        return 0
    try:
        return int(TRADE_LOG_WATERMARK.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        return 0


def _write_watermark(value: int) -> None:
    try:
        TRADE_LOG_WATERMARK.parent.mkdir(parents=True, exist_ok=True)
        TRADE_LOG_WATERMARK.write_text(str(value), encoding="utf-8")
    except OSError as exc:
        log.warning("watermark write failed: %s", exc)


def _format_freqtrade_ts(s: str | None) -> str:
    """Freqtrade '2026-05-26 14:32:28' or ISO → our '2026-05-26 14:32 UTC'."""
    if not s:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except (ValueError, TypeError):
        return s


def _mirror_closed_trades() -> int:
    """Append any Freqtrade-closed trades newer than the watermark to trade_log.md.

    Freqtrade owns trade state in its own SQLite; our memory/trade_log.md needs
    a copy so the bot's narrative log reflects reality. Idempotent via the
    `.last_mirrored_trade_id` watermark.

    Returns the count of new entries appended.
    """
    from dashboard.backend.freqtrade_client import fetch_closed_trades

    last_seen = _read_watermark()
    closed = fetch_closed_trades(limit=200)
    if not closed:
        return 0

    new_trades = [t for t in closed if int(t.get("trade_id") or 0) > last_seen]
    new_trades.sort(key=lambda t: int(t.get("trade_id") or 0))
    if not new_trades:
        return 0

    appended = 0
    max_id = last_seen
    for t in new_trades:
        pair = t.get("pair") or ""
        coin = pair.split("/", 1)[0] if pair else ""
        if not coin:
            continue
        try:
            close_profit_abs = float(t.get("close_profit_abs") or 0)
            close_profit_ratio = float(t.get("close_profit") or 0)
            append_trade(
                coin=coin,
                direction="SHORT" if t.get("is_short") else "LONG",
                entry=float(t.get("open_rate") or 0),
                exit_price=float(t.get("close_rate") or 0),
                quantity=float(t.get("amount") or 0),
                leverage=int(t.get("leverage") or 1),
                pnl_usd=close_profit_abs,
                pnl_pct=close_profit_ratio,
                reason_in=str(t.get("enter_tag") or "freqtrade"),
                reason_out=str(t.get("exit_reason") or "unknown"),
                outcome="WIN" if close_profit_abs > 0 else "LOSS",
                ts=_format_freqtrade_ts(t.get("close_date")),
            )
            appended += 1
            tid = int(t.get("trade_id") or 0)
            if tid > max_id:
                max_id = tid
        except Exception as exc:  # noqa: BLE001
            log.warning("mirror append failed for trade_id=%s pair=%s: %s",
                        t.get("trade_id"), pair, exc)

    if max_id > last_seen:
        _write_watermark(max_id)
    return appended


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

    # Mirror any newly-closed Freqtrade trades to memory/trade_log.md. Idempotent
    # — uses a trade_id watermark — so safe to run every minute.
    try:
        mirrored = _mirror_closed_trades()
        if mirrored:
            log.info("mirrored %s newly-closed trades to trade_log.md", mirrored)
    except Exception as exc:  # noqa: BLE001
        log.warning("mirror_closed_trades failed: %s", exc)

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
    from .base import setup_routine_logging
    setup_routine_logging()
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
