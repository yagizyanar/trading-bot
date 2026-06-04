"""Item 1 (MEASURE → deploy winner): take-profit redesign on 2022-2024 OOS.

Exit regimes, deployed sizing otherwise (vol 0.05, net-beta 3.0, items 5/6, hyst),
all with the flat -5% hard stop (proven best vs ATR in the prior run):
  S1  TP +15% fixed            (CURRENT deployed)
  S2  no TP                    (let winners run to signal/stop)
  T3  trailing 3% from peak
  T5  trailing 5% from peak
  T8  trailing 8% from peak

Trailing TP: once a trade is up >= X, exit when profit gives back X from its
high-water mark (Freqtrade trailing_stop_positive=X, offset=X). TP/stop are
cumulative-from-entry, capped at the exit level (resting-order model; optimistic
on gaps — the -5% stop + costs partially offset). Reports portfolio return /
Sharpe / maxDD AND trade-level win-rate / profit-factor / count (per the
backtesting-protocol). Pick the regime-ROBUST winner, not the single-year max.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.oos_2024_items import (
    ENTRY_GATE, FLIP, _atr_pct_series, _raw_signals, _rolling_beta, run_portfolio,
)
from backtest.oos_multiyear import UNIVERSE, _spot

TARGET = 0.05
BUDGET = 3.0
VOL_MIN, VOL_MAX = 0.25, 2.0


def _volm(atr):
    safe = np.where(atr > 0, atr, 1.0)
    return np.clip(np.where(atr > 0, TARGET / safe, 1.0), VOL_MIN, VOL_MAX)


def _derive_path_exits(raw, rets, atr_pct, in_sample, *, stop_param, tp_pct=None, trail=None):
    """Positions/gross/trades with flat stop + (fixed or trailing) take-profit.

    Returns (positions, gross, trades) where trades is the list of realized
    cumulative per-trade returns (unit ±1, pre-sizing) used for win-rate/PF.
    """
    n = len(raw)
    positions = np.zeros(n)
    gross = np.full(n, np.nan)
    trades: list[float] = []
    pos = 0.0           # position held INTO day t (±1)
    h = 0.0             # cumulative directional return since entry
    peak = 0.0          # high-water mark of h since entry
    stop_level = 0.0
    for t in range(in_sample, n):
        positions[t] = pos
        if pos != 0.0:
            new_h = h + rets[t] * pos
            if new_h <= -stop_level:                                   # hard stop
                gross[t] = -stop_level - h
                trades.append(-stop_level)
                pos = 0.0; h = 0.0; peak = 0.0
                continue
            new_peak = max(peak, new_h)
            if trail is not None and new_peak >= trail and (new_peak - new_h) >= trail:  # trailing TP
                gross[t] = (new_peak - trail) - h
                trades.append(new_peak - trail)
                pos = 0.0; h = 0.0; peak = 0.0
                continue
            if tp_pct is not None and new_h >= tp_pct:                 # fixed TP
                gross[t] = tp_pct - h
                trades.append(tp_pct)
                pos = 0.0; h = 0.0; peak = 0.0
                continue
            gross[t] = rets[t] * pos
            h = new_h; peak = new_peak
        else:
            gross[t] = 0.0
        # signal decision (deployed hysteresis)
        s = raw[t]
        desired = 1.0 if s > ENTRY_GATE else (-1.0 if s < -ENTRY_GATE else 0.0)
        if pos == 0.0:
            target = desired
        elif desired != 0.0 and desired != pos:
            target = desired if abs(s) >= FLIP else pos
        else:
            target = pos
        if target != pos:
            if pos != 0.0:                  # signal-driven close of the open trade
                trades.append(h)
            pos = target
            h = 0.0; peak = 0.0
            if pos != 0.0:
                stop_level = stop_param
    return positions, gross, trades


def _per_coin(year, lookback=365):
    es = pd.Timestamp(f"{year}-01-01", tz="UTC")
    ee = pd.Timestamp(f"{year}-12-31", tz="UTC")
    ss = es - pd.Timedelta(days=lookback + 5)
    btc = _spot("BTCUSDT")["close"]
    per = {}
    for c in UNIVERSE:
        df = _spot(f"{c}USDT")
        if df is None:
            continue
        sl = df[(df.index >= ss) & (df.index <= ee)]
        if int((sl.index < es).sum()) < lookback or \
           int(((sl.index >= es) & (sl.index <= ee)).sum()) < 350:
            continue
        close = sl["close"]
        per[c] = dict(
            idx=close.index,
            raw=_raw_signals(close, lookback),
            rets=close.pct_change().fillna(0.0).to_numpy(),
            atr=_atr_pct_series(sl).to_numpy(),
            beta=_rolling_beta(close, btc).to_numpy(),
            mom=close.pct_change(20).abs().fillna(0.0).to_numpy(),
            lookback=lookback,
        )
    return per, es, ee


def _eval_common(per, coins, es, ee):
    common = None
    for c in coins:
        idx = per[c]["idx"]
        evi = idx[(idx >= es) & (idx <= ee)]
        common = evi if common is None else common.intersection(evi)
    return common.sort_values()


def _trade_stats(trades):
    if not trades:
        return 0, 0.0, 0.0
    t = np.array(trades)
    wins = t[t > 0]; losses = t[t < 0]
    win_rate = len(wins) / len(t)
    gl = -losses.sum()
    pf = (wins.sum() / gl) if gl > 0 else float("inf")
    return len(t), win_rate, pf


def run_scenario(per, coins, common, *, stop_param, tp_pct=None, trail=None):
    cols = {"pos": [], "gross": [], "volm": [], "beta": [], "mom": []}
    all_trades: list[float] = []
    for c in coins:
        d = per[c]
        pos, gross, trades = _derive_path_exits(
            d["raw"], d["rets"], d["atr"], d["lookback"],
            stop_param=stop_param, tp_pct=tp_pct, trail=trail)
        all_trades.extend(trades)
        frame = pd.DataFrame({
            "pos": pos, "gross": np.nan_to_num(gross), "volm": _volm(d["atr"]),
            "beta": d["beta"], "mom": d["mom"],
        }, index=d["idx"]).reindex(common)
        for k in cols:
            cols[k].append(frame[k].fillna(0.0).to_numpy())
    M = {k: np.vstack(v) for k, v in cols.items()}
    res = run_portfolio(coins, common, M["pos"], M["gross"], M["volm"], M["beta"], M["mom"],
                        item5=True, item6=True, budget=BUDGET)
    res["n_trades"], res["win_rate"], res["pf"] = _trade_stats(all_trades)
    return res


SCENARIOS = [
    ("S1 TP+15 fixed (CURRENT)", dict(stop_param=0.05, tp_pct=0.15)),
    ("S2 no TP", dict(stop_param=0.05)),
    ("T3 trail 3%", dict(stop_param=0.05, trail=0.03)),
    ("T5 trail 5%", dict(stop_param=0.05, trail=0.05)),
    ("T8 trail 8%", dict(stop_param=0.05, trail=0.08)),
]


def main():
    print("=" * 100)
    print("ITEM 1 — TRAILING-TP REDESIGN (2022-2024 OOS, deployed sizing, flat -5% stop) — MEASURE")
    print("=" * 100)
    for year in [2022, 2023, 2024]:
        per, es, ee = _per_coin(year)
        coins = list(per.keys())
        common = _eval_common(per, coins, es, ee)
        regime = {2022: "BEAR", 2023: "RECOVERY", 2024: "BULL"}[year]
        print(f"\n{year} [{regime}]  ({len(coins)} coins, {len(common)} days)")
        print(f"  {'scenario':<26}{'return':>10}{'Sharpe':>8}{'maxDD':>8}{'trades':>8}{'win%':>7}{'PF':>7}")
        print("  " + "-" * 74)
        for label, kw in SCENARIOS:
            r = run_scenario(per, coins, common, **kw)
            print(f"  {label:<26}{r['total_return']:>+9.1%}{r['sharpe']:>8.2f}{r['max_dd']:>8.1%}"
                  f"{r['n_trades']:>8}{r['win_rate']:>6.0%}{r['pf']:>7.2f}")
    print("=" * 100)


if __name__ == "__main__":
    main()
