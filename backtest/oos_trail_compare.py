"""15m intraday: trailing-stop bake-off — 4%/+5% vs 6%/+8% vs 8%/+10%.

Isolates the trailing change on the DEPLOYED book (15m + 1d cooldown + dyn-lev
1/2/3x, items 5-7 OFF / FLIP=0, equal-weight clean set). Shares the expensive
signal+data computation across all configs, runs 2022/23/24, reports
return · Sharpe · maxDD · exit mix · fee drag · trades · avg duration, then ranks
by 3-year compounded return and names the winner.

Sanity: the 8%/+10% book should reproduce -11.1 / +30.5 / -43.9. Dyn-lev == plain
book unless a bar gaps past margin (liq>0), so liq=0 = leverage no-op.
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

# (label, TRAIL_PCT, TRAIL_ACTIVATE)  — Freqtrade requires activation > trail
CONFIGS = [("4%/+5%", 0.04, 0.05), ("6%/+8%", 0.06, 0.08), ("8%/+10%", 0.08, 0.10)]
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
    print(f"  {label:<12}{m['ret']:>+9.1%}{m['sharpe']:>8.2f}{m['dd']:>8.1%}"
          f"{m['n']:>7}tr{m['dur']:>7.1f}d{m['fee']:>6.1f}%   [{m['mix']}]")


def run():
    print("=" * 104)
    print("15m INTRADAY — TRAILING BAKE-OFF: 4%/+5% vs 6%/+8% vs 8%/+10%")
    print("items 5-7 OFF (FLIP=0) · stop -5% · 1d cooldown · DYN LEV 1/2/3x (notional constant) · equal-weight")
    print("rows = DEPLOYED book (15m + cooldown + dyn-lev); equal-weight, NOT the sized book")
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
        print(f"  {'config':<12}{'return':>9}{'Sharpe':>8}{'maxDD':>8}{'trades':>8}{'avgdur':>8}{'fee':>7}   exit mix")
        print("  " + "-" * 96)
        grand[year] = {}
        for name, tp, ta in CONFIGS:
            H.TRAIL_PCT, H.TRAIL_ACTIVATE = tp, ta
            cd, cdtr, cdfe, cdrs = [], [], [], Counter()
            dl, dltr, dlfe, dlrs = [], [], [], Counter()
            for coin, targets, strength, arr, fts, frate in coindata:
                r, t, f, _, rs = replay_intraday(arr, targets, fts, frate, cooldown_bars=COOLDOWN_BARS)
                cd.append(r.rename(coin)); cdtr += t; cdfe.append(f); cdrs += rs
                r, t, f, _, rs = replay_intraday(arr, targets, fts, frate, COOLDOWN_BARS,
                                                 strength=strength, dynamic_lev=True)
                dl.append(r.rename(coin)); dltr += t; dlfe.append(f); dlrs += rs
            m_dl = _metrics(_portfolio(dl), dltr, dlfe, dlrs)
            m_cd = _metrics(_portfolio(cd), cdtr, cdfe, cdrs)
            m_dl["_noop"] = (abs(m_dl["ret"] - m_cd["ret"]) < 1e-9 and m_dl["liq"] == 0)
            _row(name, m_dl)
            grand[year][name] = m_dl
        liqs = " ".join(f"{n}={grand[year][n]['liq']}" for n, _, _ in CONFIGS)
        print(f"  no-op check (dyn-lev liquidations, want all 0): {liqs}")

    # ---- 3-year roll-up + winner ----
    print("\n" + "=" * 104)
    print("3-YEAR ROLL-UP — compounded return · avg Sharpe · avg fee drag · total trades · per-year return")
    print(f"  {'config':<12}{'comp.ret':>10}{'avgSh':>8}{'avgFee':>9}{'trades':>9}    {'2022':>8}{'2023':>8}{'2024':>8}")
    print("  " + "-" * 88)
    roll = {}
    for name, _, _ in CONFIGS:
        rets = [grand[y][name]["ret"] for y in YEARS]
        roll[name] = dict(
            comp=float(np.prod([1 + r for r in rets]) - 1),
            sh=float(np.mean([grand[y][name]["sharpe"] for y in YEARS])),
            fee=float(np.mean([grand[y][name]["fee"] for y in YEARS])),
            tr=sum(grand[y][name]["n"] for y in YEARS),
        )
        print(f"  {name:<12}{roll[name]['comp']:>+10.1%}{roll[name]['sh']:>8.2f}{roll[name]['fee']:>8.1f}%"
              f"{roll[name]['tr']:>9}    " + "".join(f"{r:>+8.1%}" for r in rets))
    winner = max(CONFIGS, key=lambda c: roll[c[0]]["comp"])[0]
    best_sh = max(CONFIGS, key=lambda c: roll[c[0]]["sh"])[0]
    print("  " + "-" * 88)
    print(f"  WINNER (3yr compounded return): {winner}  "
          f"[comp {roll[winner]['comp']:+.1%} · Sharpe {roll[winner]['sh']:.2f} · fee {roll[winner]['fee']:.1f}% · {roll[winner]['tr']}tr]")
    print(f"  best avg Sharpe: {best_sh} ({roll[best_sh]['sh']:.2f})"
          + ("  — agrees with winner" if best_sh == winner else "  — DISAGREES with return winner, judgment call"))
    print("=" * 104)


if __name__ == "__main__":
    run()
