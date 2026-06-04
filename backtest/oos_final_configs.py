"""Final config bake-off: 3 configs × 3 regimes, drawdown lock modeled.

  1 Deployed   : items 5-7 ON, TARGET_DAILY_VOL=0.035, NET_BETA_BUDGET=3.0, CB -10%
  2 Aggressive : items 5-7 OFF (baseline), CB -30%
  3 Middle     : items 5-7 ON, TARGET_DAILY_VOL=0.05, NET_BETA_BUDGET=3.0, CB -30%

Realized return = return after the drawdown lock halts trading. Sharpe is over the
actively-traded window. Reuses the oos_2024_items engine + oos_multiyear spot data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.oos_2024_items import (
    STOP, _atr_pct_series, _derive_path, _raw_signals, _rolling_beta, run_portfolio,
)
from backtest.oos_multiyear import UNIVERSE, _spot

VOL_MIN, VOL_MAX = 0.25, 2.0


def build_full(year: int, lookback: int = 365):
    eval_start = pd.Timestamp(f"{year}-01-01", tz="UTC")
    eval_end = pd.Timestamp(f"{year}-12-31", tz="UTC")
    slice_start = eval_start - pd.Timedelta(days=lookback + 5)
    btc = _spot("BTCUSDT")["close"]
    per: dict[str, pd.DataFrame] = {}
    for c in UNIVERSE:
        df = _spot(f"{c}USDT")
        if df is None:
            continue
        sl = df[(df.index >= slice_start) & (df.index <= eval_end)]
        if int((sl.index < eval_start).sum()) < lookback or \
           int(((sl.index >= eval_start) & (sl.index <= eval_end)).sum()) < 350:
            continue
        close = sl["close"]
        rets = close.pct_change().fillna(0.0).to_numpy()
        raw = _raw_signals(close, lookback)
        bp, bg = _derive_path(raw, rets, lookback, STOP, "baseline")
        hp, hg = _derive_path(raw, rets, lookback, STOP, "hysteresis")
        d = pd.DataFrame({
            "bp": bp, "bg": np.nan_to_num(bg), "hp": hp, "hg": np.nan_to_num(hg),
            "atr": _atr_pct_series(sl).to_numpy(),
            "beta": _rolling_beta(close, btc).to_numpy(),
            "mom": close.pct_change(20).abs().fillna(0.0).to_numpy(),
        }, index=close.index)
        per[c] = d[(d.index >= eval_start) & (d.index <= eval_end)]
    coins = list(per.keys())
    common = None
    for c in coins:
        common = per[c].index if common is None else common.intersection(per[c].index)
    common = common.sort_values()

    def mat(col):
        return np.vstack([per[c].reindex(common)[col].fillna(0.0).to_numpy() for c in coins])

    return coins, common, {k: mat(k) for k in ("bp", "bg", "hp", "hg", "atr", "beta", "mom")}


def _volm(ATR, target):
    safe = np.where(ATR > 0, ATR, 1.0)
    return np.clip(np.where(ATR > 0, target / safe, 1.0), VOL_MIN, VOL_MAX)


CONFIGS = [
    # label, path, item5, item6, target_vol, budget, dd_lock
    ("1 Deployed  (ON, 3.5%, bud3, CB-10%)", "hyst", True,  True,  0.035, 3.0, 0.10),
    ("2 Aggressive(OFF baseline,    CB-30%)", "base", False, False, None,  3.0, 0.30),
    ("3 Middle    (ON, 5.0%, bud3, CB-30%)", "hyst", True,  True,  0.050, 3.0, 0.30),
    ("4 Aggr+items(ON, 5.0%, bud8, CB-30%)", "hyst", True,  True,  0.050, 8.0, 0.30),
]


def main() -> None:
    print("=" * 98)
    print("FINAL CONFIG BAKE-OFF  |  realized return after drawdown lock  |  Binance spot")
    print("=" * 98)
    print(f"  {'year':<6}{'config':<40}{'realized ret':>13}{'Sharpe':>8}{'maxDD':>8}{'CB trigger':>21}")
    print("-" * 98)
    for year, regime in [(2022, "bear"), (2023, "recovery"), (2024, "bull")]:
        coins, common, M = build_full(year)
        ONES = np.ones_like(M["atr"])
        print(f"  {year} {regime}  ({len(coins)} coins)")
        for label, path, i5, i6, tgt, bud, dl in CONFIGS:
            POS = M["hp"] if path == "hyst" else M["bp"]
            GROSS = M["hg"] if path == "hyst" else M["bg"]
            VM = _volm(M["atr"], tgt) if (i5 and tgt) else ONES
            r = run_portfolio(coins, common, POS, GROSS, VM, M["beta"], M["mom"],
                              item5=i5, item6=i6, budget=bud, dd_lock=dl)
            cb = (f"{common[r['locked_day']].strftime('%b %d')} @peak+{r['peak_return']:.0%}"
                  if r["locked"] else "no trigger")
            print(f"  {'':<6}{label:<40}{r['total_return']:>+12.1%}{r['sharpe_active']:>8.2f}"
                  f"{r['max_dd']:>8.1%}{cb:>21}")
        print("-" * 98)
    print("=" * 98)


if __name__ == "__main__":
    main()
