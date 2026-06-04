"""Diagnostic: how the LIVE exit stack (2% trail / +15% ROI / -5% stop on 15m bars)
actually behaves, + whether config-secrets.json overrides roi/trailing/whitelist."""
import json
import os
import sqlite3

for path in ("config-secrets.json",):
    if os.path.exists(path):
        d = json.load(open(path))
        present = [k for k in ("minimal_roi", "trailing_stop", "trailing_stop_positive",
                               "trailing_stop_positive_offset", "stoploss", "order_types", "timeframe")
                   if k in d]
        print("config-secrets overrides present:", present)
        for k in present:
            if k != "order_types":
                print("   ", k, "=", d[k])
        ex = d.get("exchange", {})
        wl = ex.get("pair_whitelist")
        print("   secrets whitelist:", f"len={len(wl)} LINK={any('LINK' in p for p in wl)}" if wl else "none")
    else:
        print("no config-secrets.json in", os.getcwd())

db = "user_data/tradesv3.sqlite"
if os.path.exists(db):
    con = sqlite3.connect(db)
    cur = con.cursor()
    print("\nEXIT REASONS (reason, n, avg%, min%, max%):")
    for r in cur.execute(
        "SELECT exit_reason, count(*), round(avg(close_profit)*100,2), "
        "round(min(close_profit)*100,2), round(max(close_profit)*100,2) "
        "FROM trades WHERE is_open=0 GROUP BY exit_reason ORDER BY count(*) DESC"
    ).fetchall():
        print("  ", r)
    n, avg, wins = cur.execute(
        "SELECT count(*), round(avg(close_profit)*100,2), "
        "round(100.0*sum(CASE WHEN close_profit>0 THEN 1 ELSE 0 END)/count(*),1) "
        "FROM trades WHERE is_open=0"
    ).fetchone()
    print(f"ALL closed: n={n} avg={avg}% winrate={wins}%")
    print("profit buckets:")
    for lo, hi, lbl in [(-100, -3, "<-3%"), (-3, 0, "-3..0%"), (0, 3, "0..3%"),
                        (3, 8, "3..8%"), (8, 100, ">8%")]:
        c = cur.execute("SELECT count(*) FROM trades WHERE is_open=0 AND close_profit>=? AND close_profit<?",
                        (lo / 100.0, hi / 100.0)).fetchone()[0]
        print(f"   {lbl:>8}: {c}")
    con.close()
else:
    print("no", db)
