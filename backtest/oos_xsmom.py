"""Cross-sectional momentum vs the current time-series (Markov) momentum, 2022-2024.
MEASURE ONLY.

Cross-sectional sleeve: each day rank coins by trailing 20-day return, long the top
quartile / short the bottom quartile, dollar-neutral, equal-weight within each leg,
rebalanced daily (causal: rank on the 20d return through t-1, realize day-t return).
Daily cost on turnover. No per-name stop (pure factor).

Time-series sleeve = the deployed directional book (run_portfolio: Markov ±1,
hysteresis, vol 0.05, net-beta 3.0, items 5/6) — its daily net return stream.

Reports per year: return / Sharpe / maxDD for each sleeve, corr(XS, TS), and the
50/50 combined book. Universe = the coins with clean 2022-24 spot history (NOT all
24 — newer listings lack the lookback; same constraint as every multi-year run).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from backtest.oos_2024_items import (
    COST, STOP, _atr_pct_series, _derive_path, _raw_signals, _rolling_beta, run_portfolio,
)
from backtest.oos_multiyear import UNIVERSE, _spot

TARGET = 0.05
BUDGET = 3.0
VOL_MIN, VOL_MAX = 0.25, 2.0


def _volm(atr):
    safe = np.where(atr > 0, atr, 1.0)
    return np.clip(np.where(atr > 0, TARGET / safe, 1.0), VOL_MIN, VOL_MAX)


def _sharpe(x):
    s = np.asarray(x, dtype=float)
    s = s[np.isfinite(s)]
    sd = s.std(ddof=1) if len(s) > 1 else 0.0
    return float(s.mean() / sd * math.sqrt(365)) if sd > 0 else 0.0


def _maxdd(net):
    eq = np.cumprod(1.0 + np.nan_to_num(net))
    rm = np.maximum.accumulate(eq)
    return float(((eq - rm) / rm).min())


def _ret(net):
    return float(np.prod(1.0 + np.nan_to_num(net)) - 1.0)


def build_year(year, lookback=365):
    es = pd.Timestamp(f"{year}-01-01", tz="UTC")
    ee = pd.Timestamp(f"{year}-12-31", tz="UTC")
    ss = es - pd.Timedelta(days=lookback + 25)
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
        rets = close.pct_change().fillna(0.0).to_numpy()
        raw = _raw_signals(close, lookback)
        hp, hg = _derive_path(raw, rets, lookback, STOP, "hysteresis")
        d = pd.DataFrame({
            "hp": hp, "hg": np.nan_to_num(hg),
            "atr": _atr_pct_series(sl).to_numpy(),
            "beta": _rolling_beta(close, btc).to_numpy(),
            "mom": close.pct_change(20).abs().fillna(0.0).to_numpy(),
            "mom20": close.pct_change(20).to_numpy(),     # signed 20d return (causal); kept NaN-aware
            "ret": rets,
        }, index=close.index)
        per[c] = d[(d.index >= es) & (d.index <= ee)]
    coins = list(per.keys())
    common = None
    for c in coins:
        common = per[c].index if common is None else common.intersection(per[c].index)
    common = common.sort_values()

    def mat(col, fill):
        out = np.vstack([per[c].reindex(common)[col].to_numpy() for c in coins])
        return np.nan_to_num(out) if fill else out

    M = {k: mat(k, True) for k in ("hp", "hg", "atr", "beta", "mom", "ret")}
    M["mom20"] = mat("mom20", False)          # keep NaN so the ranker can skip undefined
    return coins, common, M


def xsmom_net(M, k, cost=COST):
    """Dollar-neutral cross-sectional momentum: long top-k / short bottom-k by 20d return."""
    RET, SIG = M["ret"], M["mom20"]
    nc, nd = RET.shape
    net = np.zeros(nd)
    prev = np.zeros(nc)
    for d in range(nd):
        if d < 1:
            continue
        s = SIG[:, d - 1]                      # rank on 20d return through t-1 (causal)
        valid = np.where(np.isfinite(s))[0]
        if len(valid) < 2 * k:
            continue
        order = valid[np.argsort(s[valid])]    # ascending: losers first, winners last
        shorts, longs = order[:k], order[-k:]
        w = np.zeros(nc)
        for i in longs:
            w[i] = 0.5 / k
        for i in shorts:
            w[i] = -0.5 / k
        net[d] = float(np.sum(w * RET[:, d])) - cost * float(np.abs(w - prev).sum())
        prev = w
    return net


def main():
    print("=" * 92)
    print("CROSS-SECTIONAL vs TIME-SERIES MOMENTUM (2022-2024) — MEASURE ONLY")
    print("=" * 92)
    for year in [2022, 2023, 2024]:
        coins, common, M = build_year(year)
        n = len(coins)
        k = max(3, round(n / 4))               # ~top/bottom quartile, >=3 names/leg
        regime = {2022: "BEAR", 2023: "RECOVERY", 2024: "BULL"}[year]
        ts = run_portfolio(coins, common, M["hp"], M["hg"], _volm(M["atr"]), M["beta"], M["mom"],
                           item5=True, item6=True, budget=BUDGET)
        ts_net = np.nan_to_num(ts["net"])
        xs_net = xsmom_net(M, k)
        corr = float(np.corrcoef(ts_net, xs_net)[0, 1])
        comb = 0.5 * ts_net + 0.5 * xs_net

        print(f"\n{year} [{regime}]  ({n} coins, k={k}/leg, {len(common)} days)")
        print(f"  {'sleeve':<26}{'return':>11}{'Sharpe':>9}{'maxDD':>9}")
        print("  " + "-" * 55)
        print(f"  {'TIME-SERIES (current)':<26}{_ret(ts_net):>+10.1%}{_sharpe(ts_net):>9.2f}{_maxdd(ts_net):>9.1%}")
        print(f"  {'CROSS-SECTIONAL':<26}{_ret(xs_net):>+10.1%}{_sharpe(xs_net):>9.2f}{_maxdd(xs_net):>9.1%}")
        print(f"  {'corr(XS, TS)':<26}{corr:>+10.2f}")
        print(f"  {'COMBINED 50/50':<26}{_ret(comb):>+10.1%}{_sharpe(comb):>9.2f}{_maxdd(comb):>9.1%}")
    print("=" * 92)


if __name__ == "__main__":
    main()
