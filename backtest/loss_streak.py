"""Forensic: the most-recent consecutive losing streak — clustering in time,
direction, leverage, realized loss, re-entry behaviour, and the BTC move that
triggered it. Read-only against tradesv3.sqlite + Binance klines.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sqlite3

import numpy as np
import pandas as pd
from binance.enums import HistoricalKlinesType

from backtest.oos_multiyear import _client

con = sqlite3.connect("user_data/tradesv3.sqlite")
df = pd.read_sql("select pair,is_short,leverage,open_rate,close_rate,close_profit,close_profit_abs,"
                 "open_date,close_date,exit_reason,max_rate,min_rate from trades where is_open=0",
                 con, parse_dates=["open_date", "close_date"])
con.close()
df = df.sort_values("close_date").reset_index(drop=True)
df["coin"] = df["pair"].str.split("/").str[0]
df["dir"] = np.where(df["is_short"] == 1, "SHORT", "LONG")
df["loss"] = df["close_profit"] < 0
df["px_move"] = (df["close_rate"] / df["open_rate"] - 1) * 100   # SIGNED BY PRICE

streaks, cur = [], []
for i in range(len(df)):
    if df.loc[i, "loss"]:
        cur.append(i)
    else:
        if cur:
            streaks.append(cur); cur = []
if cur:
    streaks.append(cur)
s = df.loc[max(streaks, key=len)].copy() if streaks else df.iloc[0:0]

print("=" * 96)
print(f"LONGEST CONSECUTIVE LOSING STREAK: {len(s)} trades   (total closed trades in book: {len(df)})")
print("=" * 96)
span = (s["close_date"].max() - s["close_date"].min()).total_seconds() / 60
opspan = (s["open_date"].max() - s["open_date"].min()).total_seconds() / 60
print(f"CLOSE window:  {s['close_date'].min():%Y-%m-%d %H:%M:%S} -> {s['close_date'].max():%H:%M:%S} UTC  (span {span:.0f} min)")
print(f"OPEN window:   {s['open_date'].min():%H:%M:%S} -> {s['open_date'].max():%H:%M:%S} UTC  (built over {opspan:.0f} min)")
print(f"directions:    {dict(s['dir'].value_counts())}")
print(f"exit reasons:  {dict(s['exit_reason'].value_counts())}")
print(f"leverage:      {dict(s['leverage'].value_counts())}")
print(f">>> REALIZED LOSS: ${s['close_profit_abs'].sum():+.2f}   "
      f"(avg ${s['close_profit_abs'].mean():+.3f}/trade, avg ret-on-margin {s['close_profit'].mean()*100:+.1f}%)")
# re-entry: trades opened AFTER the first stop in the window = chasing the reversal
first_close = s["close_date"].min()
reentries = (s["open_date"] > first_close).sum()
print(f">>> RE-ENTRIES into the reversal (opened after the first stop): {reentries} of {len(s)}")

print("\n  close_time   coin       dir   lev   price_move   pnl$     opened")
print("  " + "-" * 74)
for _, r in s.iterrows():
    print(f"  {r['close_date']:%H:%M:%S}   {r['coin']:9} {r['dir']:5} {r['leverage']:.0f}x   {r['px_move']:+6.2f}%   "
          f"${r['close_profit_abs']:+.3f}   {r['open_date']:%m-%d %H:%M}")

# --- BTC move that triggered it ---
d0 = (s["close_date"].min() - pd.Timedelta(hours=3))
d1 = (s["close_date"].max() + pd.Timedelta(minutes=30))
try:
    kl = _client().get_historical_klines("BTCUSDT", "5m", d0.strftime("%Y-%m-%d %H:%M:%S"),
                                         d1.strftime("%Y-%m-%d %H:%M:%S"), klines_type=HistoricalKlinesType.FUTURES)
    b = pd.DataFrame(kl, columns=["t", "o", "h", "l", "c", "v", "ct", "qv", "n", "tb", "tq", "ig"])
    b["dt"] = pd.to_datetime(b["t"], unit="ms", utc=True)
    for col in ("o", "h", "l", "c"):
        b[col] = b[col].astype(float)
    low_row = b.loc[b["l"].idxmin()]
    high_row = b.loc[b["h"].idxmax()]
    print("\n--- BTC around the streak (the reversal the shorts faced) ---")
    print(f"  BTC LOW ${b['l'].min():,.0f} @ {low_row['dt']:%m-%d %H:%M}  ->  HIGH ${b['h'].max():,.0f} @ {high_row['dt']:%m-%d %H:%M}")
    print(f"  grind-up low->high: +{(b['h'].max() / b['l'].min() - 1) * 100:.2f}%  (shorts stop out as price RISES)")
except Exception as e:
    print("BTC fetch failed:", e)
print("=" * 96)
