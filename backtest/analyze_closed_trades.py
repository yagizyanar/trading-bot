"""Closed-trade P&L attribution from Freqtrade's tradesv3.sqlite (dry-run book).

LONG vs SHORT win-rate / P&L, by-coin breakdown, exit-reason mix, and a
signal-vs-timing-vs-market decomposition:
  - price_move  = close/open-1 SIGNED BY PRICE (was the market up/down during the hold?)
  - raw_move    = price_move adjusted for direction (did the position's bet pay off? unleveraged)
  - MFE / MAE   = max favorable / adverse excursion via max_rate/min_rate (entry vs exit timing)
close_profit is leverage-amplified (ratio on margin); raw_move and $abs are the clean measures.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
from binance.enums import HistoricalKlinesType
from backtest.oos_multiyear import _client

DB = "user_data/tradesv3.sqlite"
pd.set_option("display.width", 200)


def load():
    con = sqlite3.connect(DB)
    df = pd.read_sql("select * from trades where is_open=0", con, parse_dates=["open_date", "close_date"])
    con.close()
    df["coin"] = df["pair"].str.split("/").str[0]
    df["dir"] = np.where(df["is_short"] == 1, "SHORT", "LONG")
    df["dur_h"] = (df["close_date"] - df["open_date"]).dt.total_seconds() / 3600.0
    sgn = np.where(df["is_short"] == 1, -1.0, 1.0)
    df["price_move"] = df["close_rate"] / df["open_rate"] - 1.0          # signed by price (market direction)
    df["raw_move"] = df["price_move"] * sgn                              # direction-adjusted (bet payoff, unleveraged)
    long = df["is_short"] == 0
    df["mfe"] = np.where(long, df["max_rate"] / df["open_rate"] - 1, df["open_rate"] / df["min_rate"] - 1)
    df["mae"] = np.where(long, df["min_rate"] / df["open_rate"] - 1, df["open_rate"] / df["max_rate"] - 1)
    df["win"] = df["close_profit"] > 0
    return df


def _btc_window(d0, d1):
    try:
        kl = _client().get_historical_klines("BTCUSDT", "1d", d0, d1, klines_type=HistoricalKlinesType.FUTURES)
        if kl:
            return float(kl[-1][4]) / float(kl[0][1]) - 1.0
    except Exception:
        pass
    return float("nan")


def run():
    df = load()
    d0, d1 = df["open_date"].min(), df["close_date"].max()
    print("=" * 100)
    print(f"CLOSED-TRADE ANALYSIS — {len(df)} trades, {d0:%Y-%m-%d} to {d1:%Y-%m-%d}  (~{(d1-d0).days} days)")
    print("=" * 100)
    print(f"OVERALL: win={df['win'].mean():.0%}  net=${df['close_profit_abs'].sum():+,.0f}  "
          f"funding=${df['funding_fees'].fillna(0).sum():+,.1f}  avgLev={df['leverage'].mean():.2f}  "
          f"BTC over window={_btc_window(d0.strftime('%Y-%m-%d'), (d1+pd.Timedelta(days=1)).strftime('%Y-%m-%d')):+.1%}")

    print("\nLONG vs SHORT  (raw_move = direction-adjusted price move, UNLEVERAGED; close% is lev-amplified):")
    print(f"  {'dir':<6}{'n':>5}{'win%':>7}{'rawMove%':>10}{'close%(lev)':>12}{'avg$':>9}{'total$':>10}"
          f"{'dur_h':>7}{'lev':>6}{'MFE%':>8}{'MAE%':>8}{'pxMove%':>9}")
    for d in ("LONG", "SHORT"):
        g = df[df["dir"] == d]
        print(f"  {d:<6}{len(g):>5}{g['win'].mean()*100:>6.0f}%{g['raw_move'].mean()*100:>9.2f}%"
              f"{g['close_profit'].mean()*100:>11.1f}%{g['close_profit_abs'].mean():>9.2f}{g['close_profit_abs'].sum():>10.1f}"
              f"{g['dur_h'].mean():>7.1f}{g['leverage'].mean():>6.2f}{g['mfe'].mean()*100:>7.1f}%{g['mae'].mean()*100:>7.1f}%"
              f"{g['price_move'].mean()*100:>8.2f}%")

    print("\nBY COIN x DIRECTION  (sorted by total$ asc — biggest losers first):")
    print(f"  {'coin':<9}{'dir':<6}{'n':>4}{'win%':>7}{'rawMove%':>10}{'pxMove%':>9}{'total$':>10}")
    by = df.groupby(["coin", "dir"]).agg(n=("id", "size"), win=("win", "mean"), raw=("raw_move", "mean"),
                                         px=("price_move", "mean"), tot=("close_profit_abs", "sum")).reset_index()
    for _, r in by.sort_values("tot").iterrows():
        print(f"  {r['coin']:<9}{r['dir']:<6}{int(r['n']):>4}{r['win']*100:>6.0f}%{r['raw']*100:>9.2f}%"
              f"{r['px']*100:>8.2f}%{r['tot']:>10.1f}")

    print("\nEXIT REASON x DIRECTION  (n / win% / avg$):")
    er = df.groupby(["exit_reason", "dir"]).agg(n=("id", "size"), win=("win", "mean"),
                                                avg=("close_profit_abs", "mean")).reset_index()
    print(f"  {'exit_reason':<24}{'LONG (n/win%/avg$)':>26}{'SHORT (n/win%/avg$)':>26}")
    for reason in sorted(df["exit_reason"].dropna().unique()):
        cell = {}
        for d in ("LONG", "SHORT"):
            row = er[(er["exit_reason"] == reason) & (er["dir"] == d)]
            cell[d] = f"{int(row['n'].iloc[0])}/{row['win'].iloc[0]*100:.0f}%/{row['avg'].iloc[0]:+.1f}" if len(row) else "-"
        print(f"  {reason:<24}{cell['LONG']:>26}{cell['SHORT']:>26}")

    print("\nTIMING via MFE/MAE — did losers go favorable first (exit problem) or immediately adverse (entry/market)?")
    for d in ("LONG", "SHORT"):
        los = df[(df["dir"] == d) & (~df["win"])]
        won = df[(df["dir"] == d) & (df["win"])]
        print(f"  {d}: losers n={len(los)} avgMFE={los['mfe'].mean()*100:+.1f}% avgMAE={los['mae'].mean()*100:+.1f}% "
              f"dur={los['dur_h'].mean():.1f}h  |  winners n={len(won)} avgMFE={won['mfe'].mean()*100:+.1f}% dur={won['dur_h'].mean():.1f}h")
    print("=" * 100)


if __name__ == "__main__":
    run()
