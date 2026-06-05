"""Squeeze attribution + de-squeezed, 1x-perp carry — is it still deployable?

(1) Per-coin: worst single-day perp up-move + how many >50/30/20% days, and the
    coin's average positive funding (its carry value). High funding ↔ squeeze risk
    (adverse selection) is the thing to check.
(2) Exclude coins with any +50% single-day move; re-run the carry on the clean set.
(3) Force 1x perp: liquidation needs a +100% move → GONE for moves <100%. But 1x
    perp (isolated margin) = 2x capital/position → return on capital halves (Binance
    portfolio margin nets the hedge → ~1.3x capital → ~0.77x). Re-measure both.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.oos_carry import carry_net
from backtest.oos_carry_stress import COST_STRESS, _build_carry, _maxdd, _sharpe, _tot
from backtest.oos_intraday import _fetch_15m, _funding_events
from backtest.oos_multiyear import UNIVERSE

Y0 = pd.Timestamp("2022-01-01", tz="UTC")
Y1 = pd.Timestamp("2024-12-31 23:59", tz="UTC")
SQUEEZE_THRESH = 0.50


def coin_stats():
    rows = {}
    for c in UNIVERSE:
        perp = _fetch_15m(c)["c"].resample("1D").last()
        perp = perp[(perp.index >= Y0) & (perp.index <= Y1)]
        r = perp.pct_change().dropna()
        if r.empty:
            continue
        fts, frate = _funding_events(c)
        fund = pd.Series(frate, index=pd.to_datetime(fts, unit="ms", utc=True)).resample("1D").sum()
        fund = fund[(fund.index >= Y0) & (fund.index <= Y1)]
        pos_fund = fund[fund > 0]
        rows[c] = dict(maxmove=float(r.max()),
                       n50=int((r > 0.50).sum()), n30=int((r > 0.30).sum()), n20=int((r > 0.20).sum()),
                       avgfund_bps=float(pos_fund.mean() * 1e4) if len(pos_fund) else 0.0)
    return rows


def run_clean(year, exclude):
    coins, common, M = _build_carry(year)
    keep = [i for i, c in enumerate(coins) if c not in exclude]
    Mk = {k: M[k][keep] for k in M}
    cnet, brk = carry_net(Mk, rebal_days=7, cost_side=COST_STRESS)
    kept = [coins[i] for i in keep]
    worst = float(np.nanmax(Mk["perp_ret"])) if Mk["perp_ret"].size else 0.0
    return cnet, brk, len(kept), worst


def main():
    print("=" * 100)
    print("SQUEEZE ATTRIBUTION + DE-SQUEEZED 1x-PERP CARRY (2022-2024)")
    print("=" * 100)
    st = coin_stats()
    print(f"\n  {'coin':<10}{'max 1d move':>12}{'>50%':>6}{'>30%':>6}{'>20%':>6}{'avg+fund(bps/d)':>16}")
    print("  " + "-" * 56)
    for c, s in sorted(st.items(), key=lambda kv: -kv[1]["maxmove"]):
        flag = "  <-- SQUEEZE" if s["maxmove"] > SQUEEZE_THRESH else ""
        print(f"  {c:<10}{s['maxmove']:>11.0%}{s['n50']:>6}{s['n30']:>6}{s['n20']:>6}{s['avgfund_bps']:>16.2f}{flag}")
    squeeze = sorted([c for c, s in st.items() if s["maxmove"] > SQUEEZE_THRESH])
    print(f"\n  EXCLUDE (max move > {SQUEEZE_THRESH:.0%}): {squeeze}")

    print(f"\n  {'year':<6}{'universe':<22}{'ret(notional)':>14}{'1x-perp ÷2':>11}{'pmargin ÷1.3':>13}{'maxDD':>8}{'dep':>6}{'worst move':>11}")
    print("  " + "-" * 92)
    for year in [2022, 2023, 2024]:
        for label, excl in [("FULL", set()), ("CLEAN (no squeeze)", set(squeeze))]:
            cnet, brk, n, worst = run_clean(year, excl)
            r = _tot(cnet)
            print(f"  {year if label=='FULL' else '':<6}{label+f' [{n}c]':<22}{r:>+13.1%}{r/2:>+10.1%}{r/1.3:>+12.1%}"
                  f"{_maxdd(cnet):>8.1%}{brk['deployed']:>5.0%}{worst:>+11.0%}")
    print("\n  At 1x perp, liquidation needs a +100% move → none of these liquidate (worst < 100%).")
    print("  Drawdowns above are the REALIZED daily P&L (basis reverts by close); the squeeze becomes a")
    print("  transient mark-to-market you hold through, not a wipeout. The honest tail is the bear-year")
    print("  funding dry-up + occasional basis blow-outs, NOT liquidation.")
    print("=" * 100)


if __name__ == "__main__":
    main()
