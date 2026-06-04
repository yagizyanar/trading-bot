"""Item 3: empirical position-size distribution from the live signal_log, and the
implied USDT stake at a given account size. Usage: python -m backtest.live_size_probe [ACCOUNT] [DAYS]"""
from __future__ import annotations

import statistics as st
import sys
from datetime import datetime, timedelta, timezone

from database import SessionLocal, SignalLog

ACCOUNT = float(sys.argv[1]) if len(sys.argv) > 1 else 300.0
DAYS = int(sys.argv[2]) if len(sys.argv) > 2 else 7


def main():
    cut = datetime.now(timezone.utc) - timedelta(days=DAYS)
    s = SessionLocal()
    try:
        q = (s.query(SignalLog.position_size_pct)
             .filter(SignalLog.ts > cut, SignalLog.decision != "SKIP",
                     SignalLog.position_size_pct.isnot(None), SignalLog.position_size_pct > 0))
        pcts = sorted(float(r[0]) for r in q.all())
        print(f"Account={ACCOUNT:.0f} USDT; non-SKIP signals last {DAYS}d: {len(pcts)}")
        if pcts:
            print(f"  pct    min={pcts[0]:.4f}  med={st.median(pcts):.4f}  max={pcts[-1]:.4f}")
            print(f"  stake  min={pcts[0]*ACCOUNT:.2f}  med={st.median(pcts)*ACCOUNT:.2f}  max={pcts[-1]*ACCOUNT:.2f} USDT")
            for thr in (5.0, 10.0, 20.0):
                below = sum(1 for p in pcts if p * ACCOUNT < thr)
                print(f"  stakes < {thr:.0f} USDT: {below}/{len(pcts)} ({below/len(pcts):.0%})")
        rows = (s.query(SignalLog.coin, SignalLog.position_size_pct, SignalLog.decision, SignalLog.ts)
                .filter(SignalLog.ts > cut, SignalLog.decision != "SKIP")
                .order_by(SignalLog.ts.desc()).limit(16).all())
        print("  recent non-SKIP (coin, decision, pct, stake@acct):")
        for c, p, d, ts in rows:
            p = float(p or 0)
            print(f"    {c:10} {d:5} pct={p:.4f} stake={p*ACCOUNT:.2f}")
    finally:
        s.close()


if __name__ == "__main__":
    main()
