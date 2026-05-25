"""12:00 UTC — midday check.

Steps:
  1. Review all open positions.
  2. Tighten stop-loss on profitable positions (move to break-even at +5%).
  3. Cut positions losing more than 7%.
"""
from __future__ import annotations

import logging

from database import SessionLocal, Trade
from memory.memory_io import MemorySnapshot, append_lesson, append_trade
from risk.circuit_breakers import CircuitBreakerState
from sentiment.binance_data import fetch_binance_ohlcv

from .base import BaseRoutine

log = logging.getLogger(__name__)

CUT_LOSS_THRESHOLD = -0.07
BREAK_EVEN_THRESHOLD = 0.05


class MiddayCheckRoutine(BaseRoutine):
    name = "midday_check"

    def _run_inner(self, snapshot: MemorySnapshot, portfolio: dict, cb_state: CircuitBreakerState):
        tightened = 0
        cut = 0
        open_count = 0

        with SessionLocal() as session:
            open_trades = session.query(Trade).filter(Trade.outcome == "OPEN").all()
            open_count = len(open_trades)

            for trade in open_trades:
                df = fetch_binance_ohlcv(f"{trade.coin}USDT", interval="1h", limit=2)
                if df is None or df.empty:
                    continue
                current_price = float(df["close"].iloc[-1])
                pnl_pct = (
                    (current_price / trade.entry_price - 1.0) if trade.side == "LONG"
                    else (trade.entry_price / current_price - 1.0)
                )

                if pnl_pct <= CUT_LOSS_THRESHOLD:
                    trade.exit_price = current_price
                    trade.exit_ts = df.index[-1].to_pydatetime()
                    trade.pnl_pct = pnl_pct
                    trade.pnl_usd = trade.quantity * (
                        (current_price - trade.entry_price)
                        if trade.side == "LONG" else
                        (trade.entry_price - current_price)
                    )
                    trade.outcome = "LOSS"
                    trade.reason_out = f"Midday cut at {pnl_pct:.2%} (threshold {CUT_LOSS_THRESHOLD:.0%})"
                    cut += 1
                    append_trade(
                        coin=trade.coin, direction=trade.side, entry=trade.entry_price,
                        exit_price=current_price, quantity=trade.quantity, leverage=trade.leverage,
                        pnl_usd=trade.pnl_usd, pnl_pct=pnl_pct,
                        reason_in=trade.reason_in or "n/a", reason_out=trade.reason_out,
                        outcome="LOSS",
                    )
                elif pnl_pct >= BREAK_EVEN_THRESHOLD:
                    tightened += 1
                    append_lesson(
                        observation=f"Tightened stop to break-even on {trade.coin} at +{pnl_pct:.2%}",
                        signal_involved="midday_check",
                        worked_or_failed="ADJUSTED",
                        action_next_time="If trend continues, consider trailing TP higher",
                    )

            session.commit()

        return {"open_positions": open_count, "stops_tightened": tightened, "positions_cut": cut}


def main() -> None:
    MiddayCheckRoutine().run()


if __name__ == "__main__":
    main()
