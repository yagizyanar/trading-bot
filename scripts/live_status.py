"""Live system status (read-only): real account equity, open positions with
unrealized P&L, closed-trade summary since go-live, and concentration.

Run: .venv/bin/python scripts/live_status.py
"""
from __future__ import annotations

import os
import sqlite3
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backtest.oos_multiyear import _client

START_EQUITY = 366.64   # wallet at go-live 2026-06-06 ~12:40 UTC
DB = "user_data/tradesv3.sqlite"


def run():
    cli = _client()
    a = cli.futures_account()
    wallet = float(a["totalWalletBalance"]); avail = float(a["availableBalance"])
    upnl = float(a["totalUnrealizedProfit"]); equity = wallet + upnl
    print("=" * 80)
    print("LIVE SYSTEM STATUS")
    print("=" * 80)
    chg = equity - START_EQUITY
    print(f"ACCOUNT  equity=${equity:.2f}  wallet=${wallet:.2f}  unrealized=${upnl:+.2f}  free=${avail:.2f}")
    print(f"  since go-live (${START_EQUITY:.2f}):  {chg:+.2f} USDT  ({chg / START_EQUITY * 100:+.2f}%)")

    mark_map = {t["symbol"]: float(t["markPrice"]) for t in cli.futures_mark_price()}
    pos = [p for p in a["positions"] if float(p["positionAmt"]) != 0]
    print(f"\nOPEN POSITIONS ({len(pos)})  [sorted worst->best uPnL]")
    print(f"  {'coin':9}{'side':6}{'lev':4}{'entry':>11}{'mark':>11}{'move%':>8}{'uPnL$':>9}{'notional$':>10}")
    tot_notional = 0.0
    for p in sorted(pos, key=lambda x: float(x["unrealizedProfit"])):
        sym = p["symbol"].replace("USDT", ""); amt = float(p["positionAmt"])
        entry = float(p["entryPrice"]); u = float(p["unrealizedProfit"])
        mark = mark_map.get(p["symbol"], entry); lev = p["leverage"]
        side = "SHORT" if amt < 0 else "LONG"
        notional = abs(amt) * mark; tot_notional += notional
        move = ((entry - mark) / entry if amt < 0 else (mark - entry) / entry) * 100
        print(f"  {sym:9}{side:6}{lev:>3}x{entry:>11.4g}{mark:>11.4g}{move:>+7.2f}%{u:>+9.2f}{notional:>10.1f}")
    longs = sum(1 for p in pos if float(p["positionAmt"]) > 0)
    print(f"  concentration: {len(pos) - longs} SHORT / {longs} LONG | total notional ${tot_notional:.0f} "
          f"({tot_notional / equity * 100:.0f}% of equity) | margin used ${wallet - avail:.2f}")

    con = sqlite3.connect(DB)
    rows = con.execute("select pair,is_short,close_profit_abs,exit_reason,close_date,open_date "
                       "from trades where is_open=0 order by close_date").fetchall()
    con.close()
    print(f"\nCLOSED TRADES SINCE GO-LIVE ({len(rows)})")
    if rows:
        wins = sum(1 for r in rows if r[2] > 0); realized = sum(r[2] for r in rows)
        print(f"  win rate {wins}/{len(rows)} ({wins / len(rows) * 100:.0f}%)   realized P&L ${realized:+.2f}")
        for side, isshort in (("LONG", 0), ("SHORT", 1)):
            sr = [r for r in rows if r[1] == isshort]
            if sr:
                w = sum(1 for r in sr if r[2] > 0)
                print(f"    {side:5}: {len(sr):>2} trades, {w}/{len(sr)} win, ${sum(r[2] for r in sr):+.2f}")
        print("  exit mix:", dict(Counter(r[3] for r in rows).most_common()))
        best = max(rows, key=lambda r: r[2]); worst = min(rows, key=lambda r: r[2])
        print(f"  best  {best[0].split('/')[0]:9} ${best[2]:+.2f} ({best[3]})")
        print(f"  worst {worst[0].split('/')[0]:9} ${worst[2]:+.2f} ({worst[3]})")
    else:
        print("  (none yet)")
    print("=" * 80)


if __name__ == "__main__":
    run()
