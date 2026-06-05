"""15m intraday: trailing 4%/+5% (NEW) vs 8%/+10% (OLD) — apples-to-apples.

Isolates the trailing change on the DEPLOYED book (15m + 1d cooldown + dyn-lev
1/2/3x, items 5-7 OFF / FLIP=0, equal-weight clean set). Shares the expensive
signal+data computation across both configs, runs 2022/23/24, and reports
return · Sharpe · maxDD · exit mix · fee drag · trades · avg duration.

Sanity: the OLD (8%/+10%) plain-cooldown book should reproduce the previously
recorded -11.1% / +30.5% / -43.9%. The dyn-lev book equals the plain book unless
a bar gaps past the margin (liq>0) — so liq=0 confirms the leverage no-op.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

import backtest.oos_intraday as H
from backtest.oos_intraday import (
    UNIVERSE, YEARS, COOLDOWN_BARS,
    _daily_targets, _fetch_15m, _funding_events, _bars_arrays,
    replay_intraday, _portfolio, _sharpe, _maxdd, _exitmix,
)

CONFIGS = [("4%/+5% NEW", 0.04, 0.05), ("8%/+10% OLD", 0.08, 0.10)]
REGIME = {2022: "BEAR", 2023: "RECOVERY", 2024: "BULL"}


def _metrics(port, trades, fee, reasons):
    durs = [t["dur"] for t in trades] or [0]
    return dict(
        ret=float(np.prod(1 + port.to_numpy()) - 1),
        sharpe=_sharpe(port), dd=_maxdd(port), n=len(trades),
        dur=float(np.mean(durs)) / 24.0, fee=float(np.mean(fee)) * 100.0,
        mix=_exitmix(reasons), liq=reasons.get("liquidation", 0),
    )


def _row(label, m):
    print(f"  {label:<14}{m['ret']:>+9.1%}{m['sharpe']:>8.2f}{m['dd']:>8.1%}"
          f"{m['n']:>7}tr{m['dur']:>7.1f}d{m['fee']:>6.1f}%   [{m['mix']}]")


def run():
    print("=" * 104)
    print("15m INTRADAY — TRAILING 4%/+5% (NEW) vs 8%/+10% (OLD)")
    print("items 5-7 OFF (FLIP=0) · stop -5% · 1d cooldown · DYN LEV 1/2/3x (notional held constant) · equal-weight")
    print("headline rows = DEPLOYED book (15m + cooldown + dyn-lev); equal-weight, NOT the sized book")
    print("=" * 104)
    grand = {}
    for year in YEARS:
        coindata = []
        for coin in UNIVERSE:
            targets, strength = _daily_targets(coin, year, return_strength=True)
            if not targets:
                continue
            bars = _fetch_15m(coin)
            bars = bars[(bars.index >= pd.Timestamp(f"{year}-01-01", tz="UTC")) &
                        (bars.index <= pd.Timestamp(f"{year}-12-31 23:59", tz="UTC"))]
            if len(bars) < 20000:
                continue
            fts, frate = _funding_events(coin)
            coindata.append((coin, targets, strength, _bars_arrays(bars), fts, frate))

        print(f"\n{year} [{REGIME[year]}]  ({len(coindata)} coins)")
        print(f"  {'config':<14}{'return':>9}{'Sharpe':>8}{'maxDD':>8}{'trades':>8}{'avgdur':>8}{'fee':>7}   exit mix")
        print("  " + "-" * 96)
        grand[year] = {}
        for name, tp, ta in CONFIGS:
            H.TRAIL_PCT, H.TRAIL_ACTIVATE = tp, ta
            cd, cdtr, cdfe, cdrs = [], [], [], Counter()    # plain cooldown (no dyn-lev) — baseline/sanity
            dl, dltr, dlfe, dlrs = [], [], [], Counter()    # cooldown + dyn-lev — DEPLOYED
            for coin, targets, strength, arr, fts, frate in coindata:
                r, t, f, _, rs = replay_intraday(arr, targets, fts, frate, cooldown_bars=COOLDOWN_BARS)
                cd.append(r.rename(coin)); cdtr += t; cdfe.append(f); cdrs += rs
                r, t, f, _, rs = replay_intraday(arr, targets, fts, frate, COOLDOWN_BARS,
                                                 strength=strength, dynamic_lev=True)
                dl.append(r.rename(coin)); dltr += t; dlfe.append(f); dlrs += rs
            m_dl = _metrics(_portfolio(dl), dltr, dlfe, dlrs)
            m_cd = _metrics(_portfolio(cd), cdtr, cdfe, cdrs)
            m_dl["_noop"] = (abs(m_dl["ret"] - m_cd["ret"]) < 1e-9 and m_dl["liq"] == 0)
            m_dl["_cd_ret"] = m_cd["ret"]
            _row(name, m_dl)
            grand[year][name] = m_dl

        n, o = grand[year]["4%/+5% NEW"], grand[year]["8%/+10% OLD"]
        print(f"  {'Δ new-old':<14}{n['ret']-o['ret']:>+9.1%}{n['sharpe']-o['sharpe']:>+8.2f}"
              f"{n['dd']-o['dd']:>+8.1%}{n['n']-o['n']:>+7}tr")
        print(f"  no-op check: dyn-lev==1x? NEW={'yes' if n['_noop'] else 'NO'} (liq {n['liq']}) "
              f"OLD={'yes' if o['_noop'] else 'NO'} (liq {o['liq']})  |  "
              f"OLD plain-cooldown baseline (vs memory -11.1/+30.5/-43.9): {o['_cd_ret']:+.1%}")

    print("\n" + "=" * 104)
    print("SUMMARY — DEPLOYED (dyn-lev) book, NEW 4%/+5% vs OLD 8%/+10%")
    print(f"  {'year':<10}{'NEW ret':>10}{'OLD ret':>10}{'Δret':>9}{'NEW Sh':>8}{'OLD Sh':>8}{'NEW DD':>9}{'OLD DD':>9}")
    for year in YEARS:
        n, o = grand[year]["4%/+5% NEW"], grand[year]["8%/+10% OLD"]
        print(f"  {year:<10}{n['ret']:>+10.1%}{o['ret']:>+10.1%}{n['ret']-o['ret']:>+9.1%}"
              f"{n['sharpe']:>8.2f}{o['sharpe']:>8.2f}{n['dd']:>+9.1%}{o['dd']:>+9.1%}")
    print("=" * 104)


if __name__ == "__main__":
    run()
