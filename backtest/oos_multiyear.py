"""Multi-year OOS across regimes (2021-2024), items OFF vs deployed (2026-06-03).

Reuses the engine from oos_2024_items.py but fetches long Binance SPOT history
(futures klines only reach ~2022-04) and auto-selects each year's valid universe
(coins with `lookback` days of pre-year history + a near-full eval year).

DATA REALITY (spot listing dates): the 13-coin majors set is a 2024 survivor
universe; several tokens didn't exist earlier.
  2024: 13 coins (clean, 365d lookback)
  2023: 11 coins (no OP/APT — listed 2022; <365d pre-2023)
  2022: 10 coins (no OP/APT/DYDX)
  2021: DEGRADED — majors mostly listed mid-2020, so only ~6 coins have even
        ~130d of pre-2021 history. Run with a SHORT 130d lookback; treat as
        weak/indicative only (thin Markov matrix, tiny universe, staggered).

Scenarios per year:
  Baseline  = items off (equal 1-slot, gate-0.2 path, no beta cap).
  Deployed  = items 5+6+7, TARGET_DAILY_VOL=0.035, NET_BETA_BUDGET=3.0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import BINANCE_API_KEY, BINANCE_SECRET_KEY
from backtest.oos_2024_items import (
    STOP, _atr_pct_series, _derive_path, _raw_signals, _rolling_beta, run_portfolio,
)

UNIVERSE = ["SOL", "AVAX", "LINK", "DOT", "INJ", "OP", "APT",
            "NEAR", "ATOM", "SAND", "MANA", "AXS", "DYDX"]
TARGET = 0.035
BUDGET = 3.0
VOL_MIN, VOL_MAX = 0.25, 2.0

_CLI = None
_CACHE: dict[str, pd.DataFrame] = {}


def _client():
    global _CLI
    if _CLI is None:
        from binance.client import Client
        _CLI = Client(BINANCE_API_KEY, BINANCE_SECRET_KEY)
    return _CLI


def _spot(sym: str):
    if sym in _CACHE:
        return _CACHE[sym]
    kl = _client().get_historical_klines(sym, "1d", "2020-01-01", "2025-01-01")
    if not kl:
        _CACHE[sym] = None
        return None
    df = pd.DataFrame(kl, columns=["t", "open", "high", "low", "close", "volume",
                                   "ct", "qv", "n", "tb", "tq", "ig"])
    df["date"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    _CACHE[sym] = df.set_index("date")[["open", "high", "low", "close", "volume"]]
    return _CACHE[sym]


def build_year(year: int, lookback: int):
    eval_start = pd.Timestamp(f"{year}-01-01", tz="UTC")
    eval_end = pd.Timestamp(f"{year}-12-31", tz="UTC")
    slice_start = eval_start - pd.Timedelta(days=lookback + 5)
    btc_close = _spot("BTCUSDT")["close"]

    per: dict[str, pd.DataFrame] = {}
    for c in UNIVERSE:
        df = _spot(f"{c}USDT")
        if df is None:
            continue
        sl = df[(df.index >= slice_start) & (df.index <= eval_end)]
        pre = int((sl.index < eval_start).sum())
        yr = int(((sl.index >= eval_start) & (sl.index <= eval_end)).sum())
        if pre < lookback or yr < 350:        # insufficient lookback or year coverage
            continue
        close = sl["close"]
        rets = close.pct_change().fillna(0.0).to_numpy()
        raw = _raw_signals(close, lookback)
        bp, bg = _derive_path(raw, rets, lookback, STOP, "baseline")
        hp, hg = _derive_path(raw, rets, lookback, STOP, "hysteresis")
        d = pd.DataFrame({
            "bp": bp, "bg": np.nan_to_num(bg), "hp": hp, "hg": np.nan_to_num(hg),
            "atr": _atr_pct_series(sl).to_numpy(),
            "beta": _rolling_beta(close, btc_close).to_numpy(),
            "mom": close.pct_change(20).abs().fillna(0.0).to_numpy(),
        }, index=close.index)
        per[c] = d[(d.index >= eval_start) & (d.index <= eval_end)]

    coins = list(per.keys())
    if len(coins) < 2:
        return coins, None, None, 0
    common = None
    for c in coins:
        common = per[c].index if common is None else common.intersection(per[c].index)
    common = common.sort_values()

    def mat(col):
        return np.vstack([per[c].reindex(common)[col].fillna(0.0).to_numpy() for c in coins])

    BP, BG, HP, HG = mat("bp"), mat("bg"), mat("hp"), mat("hg")
    ATR, BETA, MOM = mat("atr"), mat("beta"), mat("mom")
    ONES = np.ones_like(ATR)
    safe = np.where(ATR > 0, ATR, 1.0)
    VOLM = np.clip(np.where(ATR > 0, TARGET / safe, 1.0), VOL_MIN, VOL_MAX)

    base = run_portfolio(coins, common, BP, BG, ONES, BETA, MOM, item5=False, item6=False, budget=BUDGET)
    dep = run_portfolio(coins, common, HP, HG, VOLM, BETA, MOM, item5=True, item6=True, budget=BUDGET)
    return coins, base, dep, len(common)


def main() -> None:
    print("=" * 90)
    print("MULTI-YEAR OOS  |  Binance spot  |  items OFF vs deployed (tgt 3.5%, budget 3.0)")
    print("=" * 90)
    plan = [(2021, 130, "bull (DEGRADED — short lookback, tiny universe)"),
            (2022, 365, "bear"),
            (2023, 365, "recovery"),
            (2024, 365, "bull")]
    print(f"  {'year':<6}{'regime / universe':<52}{'total ret':>10}{'Sharpe':>8}{'maxDD':>8}{'fee$':>7}")
    print("-" * 90)
    for year, lb, regime in plan:
        coins, base, dep, nd = build_year(year, lb)
        tag = f"{regime}  [{len(coins)} coins, {lb}d lookback, {nd}d]"
        if base is None:
            print(f"  {year:<6}{tag:<52}{'— not enough data —':>33}")
            continue
        print(f"  {year:<6}{('  '+regime):<52}")
        print(f"  {'':<6}{('    baseline (items off)  ['+str(len(coins))+' coins, '+str(lb)+'d]'):<52}"
              f"{base['total_return']:>+9.1%}{base['sharpe']:>8.2f}{base['max_dd']:>8.1%}{base['fee']:>7,.0f}")
        print(f"  {'':<6}{'    deployed (5+6+7, 3.5%/3.0)':<52}"
              f"{dep['total_return']:>+9.1%}{dep['sharpe']:>8.2f}{dep['max_dd']:>8.1%}{dep['fee']:>7,.0f}")
        print(f"  {'':<6}{('    coins: '+','.join(coins)):<52}")
        print("-" * 90)
    print("=" * 90)


if __name__ == "__main__":
    main()
