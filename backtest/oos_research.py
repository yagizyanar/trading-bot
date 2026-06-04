"""Research (MEASURE ONLY, ship nothing): funding-carry sleeve + regime-confidence
gate, 2022-2024.

TASK 1 — funding carry sleeve: dollar-neutral, short the highest-funding coins /
long the lowest-funding (signal = -funding). Report standalone Sharpe, correlation
to the momentum sleeve, and the combined 50/50 portfolio Sharpe.

TASK 2 — confidence gate: skip the signal when regime confidence
(max next-state prob) < 0.7. Compare momentum sleeve with vs without the gate.

Reuses the oos_2024_items engine + oos_multiyear spot data. Funding history comes
from Binance fapi (full history, unlike OI/L-S which are 30d-capped).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from config.settings import MARKOV_THRESHOLD, MARKOV_WINDOW
from markov.regime_detector import build_transition_matrix, label_regimes, signal_from_matrix
from backtest.oos_2024_items import (
    COST, STOP, _atr_pct_series, _derive_path, _rolling_beta, run_portfolio,
)
from backtest.oos_multiyear import UNIVERSE, _client, _spot

VOL_MIN, VOL_MAX = 0.25, 2.0
TARGET = 0.05
BUDGET = 3.0
CONF_THRESH = 0.7
K_CARRY = 3

_FUND: dict[str, pd.Series] = {}


def funding_daily(coin: str):
    """Full Binance funding history → daily (sum of the 3 8h payments). Cached."""
    sym = f"{coin}USDT"
    if sym in _FUND:
        return _FUND[sym]
    cli = _client()
    rows = []
    start = int(pd.Timestamp("2020-01-01", tz="UTC").timestamp() * 1000)
    end = int(pd.Timestamp("2025-01-01", tz="UTC").timestamp() * 1000)
    while start < end:
        try:
            batch = cli.futures_funding_rate(symbol=sym, startTime=start, endTime=end, limit=1000)
        except Exception:
            batch = []
        if not batch:
            break
        rows += batch
        if len(batch) < 1000:
            break
        start = int(batch[-1]["fundingTime"]) + 1
    if not rows:
        _FUND[sym] = None
        return None
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["fr"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    _FUND[sym] = df.set_index("date")["fr"].resample("1D").sum()
    return _FUND[sym]


def raw_signals_conf(close: pd.Series, in_sample: int):
    """Raw Markov signal AND confidence (max next-state prob) per day."""
    close = close.reset_index(drop=True)
    n = len(close)
    raw = np.full(n, np.nan)
    conf = np.full(n, np.nan)
    for t in range(in_sample, n):
        train = close.iloc[t - in_sample:t]
        labels = label_regimes(train, window=MARKOV_WINDOW, threshold=MARKOV_THRESHOLD)
        if labels.empty:
            raw[t], conf[t] = 0.0, 0.0
        else:
            P = build_transition_matrix(labels)
            st = int(labels.iloc[-1])
            raw[t] = float(signal_from_matrix(P, st))
            conf[t] = float(np.max(P[st]))
    return raw, conf


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


def build_year(year: int, lookback: int = 365):
    es = pd.Timestamp(f"{year}-01-01", tz="UTC")
    ee = pd.Timestamp(f"{year}-12-31", tz="UTC")
    ss = es - pd.Timedelta(days=lookback + 5)
    btc = _spot("BTCUSDT")["close"]
    per: dict[str, pd.DataFrame] = {}
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
        raw, conf = raw_signals_conf(close, lookback)
        hp, hg = _derive_path(raw, rets, lookback, STOP, "hysteresis")
        hpg, hgg = _derive_path(raw, rets, lookback, STOP, "hysteresis",
                                conf=conf, conf_thresh=CONF_THRESH)
        fund = funding_daily(c)
        fr = fund.reindex(close.index).fillna(0.0).to_numpy() if fund is not None else np.zeros(len(close))
        d = pd.DataFrame({
            "hp": hp, "hg": np.nan_to_num(hg), "hpg": hpg, "hgg": np.nan_to_num(hgg),
            "conf": np.nan_to_num(conf, nan=1.0),
            "atr": _atr_pct_series(sl).to_numpy(),
            "beta": _rolling_beta(close, btc).to_numpy(),
            "mom": close.pct_change(20).abs().fillna(0.0).to_numpy(),
            "ret": rets, "fund": fr,
        }, index=close.index)
        per[c] = d[(d.index >= es) & (d.index <= ee)]
    coins = list(per.keys())
    common = None
    for c in coins:
        common = per[c].index if common is None else common.intersection(per[c].index)
    common = common.sort_values()

    def mat(col):
        return np.vstack([per[c].reindex(common)[col].fillna(0.0).to_numpy() for c in coins])

    M = {k: mat(k) for k in ("hp", "hg", "hpg", "hgg", "conf", "atr", "beta", "mom", "ret", "fund")}
    return coins, common, M


def _volm(ATR, t):
    safe = np.where(ATR > 0, ATR, 1.0)
    return np.clip(np.where(ATR > 0, t / safe, 1.0), VOL_MIN, VOL_MAX)


def carry_net(M, k=K_CARRY, cost=COST):
    """Dollar-neutral funding carry: short top-k funding, long bottom-k (signal = -funding)."""
    RET, FUND = M["ret"], M["fund"]
    nc, nd = RET.shape
    net = np.zeros(nd)
    prev = np.zeros(nc)
    fund_only = np.zeros(nd)
    for d in range(nd):
        if d < 1:
            continue
        fp = FUND[:, d - 1]                    # rank by prior-day funding (causal)
        valid = np.where(np.isfinite(fp))[0]
        if len(valid) < 2 * k:
            continue
        order = valid[np.argsort(fp[valid])]   # ascending funding
        longs, shorts = order[:k], order[-k:]
        w = np.zeros(nc)
        for i in longs:
            w[i] = 0.5 / k
        for i in shorts:
            w[i] = -0.5 / k
        g = float(np.sum(w * (RET[:, d] - FUND[:, d])))   # price P&L + funding collected
        fund_only[d] = float(np.sum(-w * FUND[:, d]))
        net[d] = g - cost * float(np.abs(w - prev).sum())
        prev = w
    return net, fund_only


def main() -> None:
    print("=" * 96)
    print("RESEARCH: funding carry sleeve + regime-confidence gate (2022-2024) — MEASURE ONLY")
    print("=" * 96)
    for year in [2022, 2023, 2024]:
        coins, common, M = build_year(year)
        nd = len(common)
        mom = run_portfolio(coins, common, M["hp"], M["hg"], _volm(M["atr"], TARGET),
                            M["beta"], M["mom"], item5=True, item6=True, budget=BUDGET)
        momg = run_portfolio(coins, common, M["hpg"], M["hgg"], _volm(M["atr"], TARGET),
                             M["beta"], M["mom"], item5=True, item6=True, budget=BUDGET)
        cnet, fund_only = carry_net(M)
        m_net = np.nan_to_num(mom["net"])
        corr = float(np.corrcoef(m_net, cnet)[0, 1])
        comb = 0.5 * m_net + 0.5 * cnet
        gate_rate = float((M["conf"] < CONF_THRESH).mean())

        print(f"\n{year}  ({len(coins)} coins, {nd} days)")
        print("-" * 96)
        print("  TASK 2 — confidence gate (<0.7):")
        print(f"    momentum (deployed)    ret {mom['total_return']:>+8.1%}  Sharpe {_sharpe(m_net):>5.2f}  DD {_maxdd(m_net):>7.1%}")
        print(f"    + confidence gate      ret {momg['total_return']:>+8.1%}  Sharpe {_sharpe(np.nan_to_num(momg['net'])):>5.2f}  DD {_maxdd(np.nan_to_num(momg['net'])):>7.1%}   (gate fires {gate_rate:.0%} of coin-days)")
        print("  TASK 1 — funding carry sleeve (dollar-neutral, k=3):")
        print(f"    carry standalone       ret {_ret(cnet):>+8.1%}  Sharpe {_sharpe(cnet):>5.2f}  DD {_maxdd(cnet):>7.1%}   (funding-only Sharpe {_sharpe(fund_only):.2f})")
        print(f"    corr(carry, momentum)  {corr:>+.2f}")
        print(f"    COMBINED 50/50         ret {_ret(comb):>+8.1%}  Sharpe {_sharpe(comb):>5.2f}  DD {_maxdd(comb):>7.1%}")
    print("=" * 96)


if __name__ == "__main__":
    main()
