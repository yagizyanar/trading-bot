"""Circuit-breaker (drawdown-lock) impact on the BASELINE — realized return.

The other backtests ignore the live drawdown lock, which HALTS trading the first
time equity is down `dd_lock` from its peak (writes TRADING_LOCKED.txt; manual
restart). This measures what you'd ACTUALLY realize on the baseline (items off)
under three lock settings, across bear/recovery/bull:
    -10% (current DRAWDOWN_LOCK_PCT),  -30% (relaxed),  none.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.oos_2024_items import (
    STOP, _atr_pct_series, _derive_path, _raw_signals, _rolling_beta, run_portfolio,
)
from backtest.oos_multiyear import UNIVERSE, _spot


def build_baseline(year: int, lookback: int = 365):
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
        d = pd.DataFrame({
            "bp": bp, "bg": np.nan_to_num(bg),
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

    return coins, common, mat("bp"), mat("bg"), mat("beta"), mat("mom")


def main() -> None:
    print("=" * 92)
    print("CIRCUIT-BREAKER (drawdown-lock) IMPACT ON BASELINE — realized return after lock")
    print("=" * 92)
    settings = [("CB -10% (current)", 0.10), ("CB -30% (relaxed)", 0.30), ("No CB", None)]
    print(f"  {'year/regime':<18}{'CB setting':<20}{'realized ret':>13}{'realized DD':>13}{'locked?':>26}")
    print("-" * 92)
    for year, regime in [(2022, "bear"), (2023, "recovery"), (2024, "bull")]:
        coins, common, POS, GROSS, BETA, MOM = build_baseline(year)
        ONES = np.ones_like(POS, dtype=float)
        nd = len(common)
        for label, dl in settings:
            r = run_portfolio(coins, common, POS, GROSS, ONES, BETA, MOM,
                              item5=False, item6=False, budget=3.0, dd_lock=dl)
            if r["locked"]:
                day = r["locked_day"]
                when = common[day].strftime("%b %d")
                info = f"locked {when} (day {day}/{nd}, peak +{r['peak_return']:.0%})"
            else:
                info = "never locked (full year)"
            tag = f"{year} {regime}" if label == settings[0][0] else ""
            print(f"  {tag:<18}{label:<20}{r['total_return']:>+12.1%}{r['max_dd']:>12.1%}  {info:<24}")
        print("-" * 92)
    print("=" * 92)
    print("Reminder: this is the BASELINE (items 5-7 OFF). The deployed config holds DD to")
    print("~5-13%, so a -10% lock interacts very differently with it.")


if __name__ == "__main__":
    main()
