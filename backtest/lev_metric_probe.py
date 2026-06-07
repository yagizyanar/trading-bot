"""Which metric spreads into a real 1x/2x/3x leverage distribution?

Compares, across the live universe's latest signals: |markov| (current), |sentiment|,
the three-layer combined notional (= position_size_pct x leverage), and |funding rate|.
A good leverage differentiator has a HIGH coefficient of variation (spread), not a
cluster. Also shows the tier split the best candidate would produce.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from database import SessionLocal, SignalLog
from signals.three_layer import _leverage_from_signal
from backtest.oos_multiyear import _client

s = SessionLocal()
rows = s.query(SignalLog.coin, SignalLog.markov_signal, SignalLog.sentiment_score,
               SignalLog.position_size_pct, SignalLog.decision, SignalLog.ts) \
        .order_by(SignalLog.ts.desc()).limit(600).all()
s.close()
latest = {}
for c, m, se, p, d, ts in rows:
    if c not in latest:
        latest[c] = (float(m or 0), float(se or 0), float(p or 0), d)
fund = {t["symbol"]: float(t.get("lastFundingRate") or 0) for t in _client().futures_mark_price()}

rec = []
for c, (m, se, p, d) in latest.items():
    lev = _leverage_from_signal(m)
    rec.append(dict(coin=c, am=abs(m), asent=abs(se), sent=se,
                    notional=p * lev, afund=abs(fund.get(c + "USDT", 0))))


def report(name, key):
    a = np.array([r[key] for r in rec], float)
    mean = a.mean()
    cv = a.std() / mean if mean else 0.0
    rng = a.max() / a.min() if a.min() > 0 else float("inf")
    print(f"  {name:26} min={a.min():.4f}  med={np.median(a):.4f}  max={a.max():.4f}  CV={cv:.2f}  max/min={rng:>5.1f}x")


print(f"METRIC SPREAD across {len(rec)} coins (higher CV = better leverage differentiator):")
report("|markov| (current)", "am")
report("|sentiment|", "asent")
report("three-layer notional", "notional")
report("|funding rate|", "afund")

print("\nThree-layer notional -> tercile tiers (what the live book would actually get):")
vals = sorted(r["notional"] for r in rec)
q1, q2 = np.percentile(vals, [33.3, 66.7])
print(f"  proposed thresholds: 1x if <={q1:.4f}, 2x if <={q2:.4f}, else 3x")
counts = {1: 0, 2: 0, 3: 0}
for r in sorted(rec, key=lambda x: x["notional"]):
    t = 1 if r["notional"] <= q1 else (2 if r["notional"] <= q2 else 3)
    counts[t] += 1
    print(f"  {r['coin']:9} notional={r['notional']:.4f}  |mk|={r['am']:.2f}  sent={r['sent']:+.2f}  |fund|={r['afund']:.5f}  -> {t}x")
print(f"  tier counts: 1x={counts[1]} 2x={counts[2]} 3x={counts[3]}")
