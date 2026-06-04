"""Item 1 (MEASURE ONLY): take-profit + stop redesign on the 2022-2024 OOS harness.

Four exit regimes, deployed sizing otherwise (vol 0.05, net-beta 3.0, items 5/6 on):
  S1  TP +15% , flat -5% stop            (CURRENT deployed behaviour)
  S2  no TP   , flat -5% stop            (let winners run)
  S3  TP +15% , 2x-ATR stop              (wider, vol-scaled stop)
  S4  no TP   , 2x-ATR stop              (let winners run + wider stop)

TP/stop are tracked as CUMULATIVE return since entry (resting-order model: the
exit day's return is capped exactly at the +15% / -stop level), unlike the old
single-day stop. Signal entry/flip uses the deployed hysteresis logic; after a
TP/stop exit the signal may re-enter next day (matches Freqtrade minimal_roi +
re-entry). Reports return / Sharpe / maxDD per scenario per year. Ship nothing.
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


def _derive_path_exits(raw, rets, atr_pct, in_sample, *, stop_kind, stop_param, tp_pct):
    """Positions/gross with cumulative-from-entry TP and stop.

    stop_kind="flat" -> stop_level = stop_param (e.g. 0.05)
    stop_kind="atr"  -> stop_level = stop_param * ATR%-at-entry (e.g. 2.0 x ATR)
    tp_pct=None -> no take-profit (trailing/signal exit only)
    """
    n = len(raw)
    positions = np.zeros(n)
    gross = np.full(n, np.nan)
    pos = 0.0            # position held coming INTO day t (±1)
    h = 0.0             # cumulative directional return since entry
    stop_level = 0.0
    for t in range(in_sample, n):
        positions[t] = pos
        if pos != 0.0:
            day = rets[t] * pos
            new_h = h + day
            if new_h <= -stop_level:                       # stop hit → cap at level
                gross[t] = -stop_level - h
                pos = 0.0
                h = 0.0
                continue
            if tp_pct is not None and new_h >= tp_pct:     # take-profit hit → cap at level
                gross[t] = tp_pct - h
                pos = 0.0
                h = 0.0
                continue
            gross[t] = day
            h = new_h
        else:
            gross[t] = 0.0
        # signal decision for next day (deployed hysteresis)
        s = raw[t]
        desired = 1.0 if s > ENTRY_GATE else (-1.0 if s < -ENTRY_GATE else 0.0)
        if pos == 0.0:
            target = desired
        elif desired != 0.0 and desired != pos:
            target = desired if abs(s) >= FLIP else pos
        else:
            target = pos
        if target != pos:
            pos = target
            if pos != 0.0:                                 # fresh entry / flip → reset trade
                h = 0.0
                stop_level = stop_param if stop_kind == "flat" else stop_param * max(atr_pct[t], 1e-4)
    return positions, gross


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
        mask = (idx >= es) & (idx <= ee)
        evi = idx[mask]
        common = evi if common is None else common.intersection(evi)
    return common.sort_values()


def run_scenario(per, coins, common, es, ee, *, stop_kind, stop_param, tp_pct):
    cols = {"pos": [], "gross": [], "volm": [], "beta": [], "mom": []}
    for c in coins:
        d = per[c]
        pos, gross = _derive_path_exits(d["raw"], d["rets"], d["atr"], d["lookback"],
                                        stop_kind=stop_kind, stop_param=stop_param, tp_pct=tp_pct)
        frame = pd.DataFrame({
            "pos": pos, "gross": np.nan_to_num(gross), "volm": _volm(d["atr"]),
            "beta": d["beta"], "mom": d["mom"],
        }, index=d["idx"]).reindex(common)
        for k in cols:
            cols[k].append(frame[k].fillna(0.0).to_numpy())
    M = {k: np.vstack(v) for k, v in cols.items()}
    return run_portfolio(coins, common, M["pos"], M["gross"], M["volm"], M["beta"], M["mom"],
                         item5=True, item6=True, budget=BUDGET)


SCENARIOS = [
    ("S1 TP+15 / flat-5  (CURRENT)", dict(stop_kind="flat", stop_param=0.05, tp_pct=0.15)),
    ("S2 noTP   / flat-5", dict(stop_kind="flat", stop_param=0.05, tp_pct=None)),
    ("S3 TP+15  / 2xATR", dict(stop_kind="atr", stop_param=2.0, tp_pct=0.15)),
    ("S4 noTP   / 2xATR", dict(stop_kind="atr", stop_param=2.0, tp_pct=None)),
]


def main():
    print("=" * 92)
    print("ITEM 1 — TP + STOP REDESIGN (2022-2024 OOS, deployed sizing) — MEASURE ONLY")
    print("=" * 92)
    for year in [2022, 2023, 2024]:
        per, es, ee = _per_coin(year)
        coins = list(per.keys())
        common = _eval_common(per, coins, es, ee)
        print(f"\n{year}  ({len(coins)} coins, {len(common)} days)")
        print(f"  {'scenario':<30}{'return':>12}{'Sharpe':>9}{'maxDD':>9}")
        print("  " + "-" * 58)
        for label, kw in SCENARIOS:
            r = run_scenario(per, coins, common, es, ee, **kw)
            print(f"  {label:<30}{r['total_return']:>+11.1%}{r['sharpe']:>9.2f}{r['max_dd']:>9.1%}")
    print("=" * 92)


if __name__ == "__main__":
    main()
