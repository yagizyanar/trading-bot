"""Auto-research loop — grid-search RSI/ADX/funding filters on the 20d-momentum signal.

Methodology (leakage-proof):
  - Base signal = sign(20-day return)  (the Markov proxy; "no filters" = baseline).
  - Filters GATE the base signal (a trade only survives if it passes all filters):
      ADX  : skip if ADX < adx_min          (choppy market — momentum unreliable)
      RSI  : skip LONG if RSI >= overbought; skip SHORT if RSI <= oversold (don't chase extremes)
      Fund : skip LONG if funding in TOP x% of trailing-90d; skip SHORT if BOTTOM x% (avoid crowded side)
  - All features are ROLLING (computed from past data only); positions are shifted +1 day
    before applying returns → no lookahead.
  - OPTIMIZE on 2022-2023, then evaluate the chosen combo on HELD-OUT 2024.
  - Overfit guard: any combo with in-sample (2022-23) Sharpe > 3.0 is REJECTED as implausible.
  - Report: best combo, its 2024 OOS return/Sharpe/maxDD, vs the no-filter baseline.
"""
import itertools
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

from signals.technical import _rsi
from backtest.oos_multiyear import _spot, UNIVERSE
from backtest.oos_intraday import _funding_events

MOM = 20
RSI_OB = [60, 65, 70, 75, 80]
RSI_OS = [20, 25, 30, 35, 40]
ADX_MIN = [15, 20, 25, 30]
FUND_PCT = [0.10, 0.15, 0.20, 0.25, 0.30]
TRAIN = ("2022-01-01", "2023-12-31")
OOS = ("2024-01-01", "2024-12-31")


def _adx(high, low, close, period=14):
    up, down = high.diff(), -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)
    tr = pd.concat([(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    pdi = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    mdi = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def build_features(coin):
    df = _spot(coin + "USDT")
    if df is None:
        return None
    df = df[(df.index >= "2021-08-01") & (df.index <= "2024-12-31")]
    if len(df) < 400 or df.index.min() > pd.Timestamp("2021-09-01", tz="UTC"):
        return None  # require full 2022-24 history for a clean train/OOS universe
    close, high, low = df["close"], df["high"], df["low"]
    mom = close / close.shift(MOM) - 1
    feat = pd.DataFrame({
        "signal": np.sign(mom),
        "ret": close.pct_change(),
        "rsi": _rsi(close, 14),
        "adx": _adx(high, low, close, 14),
    })
    fts, frate = _funding_events(coin)
    if len(fts):
        f = pd.Series(frate, index=pd.to_datetime(fts, unit="ms", utc=True)).resample("1D").mean()
        f = f.reindex(close.index).ffill(limit=2)
        feat["fund_rank"] = f.rolling(90, min_periods=30).apply(lambda w: (w <= w[-1]).mean(), raw=True)
    else:
        feat["fund_rank"] = np.nan
    return feat


def _metrics(port, lo, hi):
    r = port[(port.index >= lo) & (port.index <= hi)].dropna().to_numpy()
    if len(r) < 30:
        return None
    sd = r.std(ddof=1)
    eq = np.cumprod(1 + r)
    dd = float(((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)).min())
    return dict(sharpe=float(r.mean() / sd * math.sqrt(365)) if sd > 0 else 0.0,
                ret=float(np.prod(1 + r) - 1), dd=dd, n=len(r))


print("building rolling features (no lookahead)...")
feats = {c: f for c in UNIVERSE if (f := build_features(c)) is not None}
coins = list(feats)
idx = sorted(set().union(*[set(f.index) for f in feats.values()]))
idx = pd.DatetimeIndex(idx)


def _M(col):
    return np.column_stack([feats[c][col].reindex(idx).to_numpy() for c in coins])


SIG, RET, RSI, ADX, FR = _M("signal"), _M("ret"), _M("rsi"), _M("adx"), _M("fund_rank")
print(f"  {len(coins)} coins, {len(idx)} days ({idx.min():%Y-%m} .. {idx.max():%Y-%m})")


def port_series(rsi_ob, rsi_os, adx_min, fund_pct, filters=True):
    POS = SIG.copy()
    if filters:
        POS[(ADX < adx_min)] = 0.0
        POS[(SIG > 0) & (RSI >= rsi_ob)] = 0.0
        POS[(SIG < 0) & (RSI <= rsi_os)] = 0.0
        POS[(SIG > 0) & (FR >= 1 - fund_pct)] = 0.0
        POS[(SIG < 0) & (FR <= fund_pct)] = 0.0
    POS = np.where(np.isnan(SIG), np.nan, POS)
    shifted = np.vstack([np.full((1, POS.shape[1]), np.nan), POS[:-1]])  # position from prior day
    strat = shifted * RET
    return pd.Series(np.nanmean(strat, axis=1), index=idx)


# baseline (no filters)
bp = port_series(0, 0, 0, 0, filters=False)
base_tr, base_oos = _metrics(bp, *TRAIN), _metrics(bp, *OOS)

# grid
rows = []
for ob, os_, ax, fp in itertools.product(RSI_OB, RSI_OS, ADX_MIN, FUND_PCT):
    p = port_series(ob, os_, ax, fp)
    tr, oo = _metrics(p, *TRAIN), _metrics(p, *OOS)
    if tr and oo:
        rows.append(dict(ob=ob, os=os_, adx=ax, fund=fp, tr=tr, oo=oo))

overfit = [r for r in rows if r["tr"]["sharpe"] > 3.0]
clean = [r for r in rows if r["tr"]["sharpe"] <= 3.0]
clean.sort(key=lambda r: -r["tr"]["sharpe"])
best = clean[0]

print("\n" + "=" * 96)
print(f"AUTO-RESEARCH: {len(rows)} combos | optimize 2022-23, evaluate held-out 2024 | overfit-reject Sharpe>3.0")
print("=" * 96)
print(f"BASELINE (no filters, raw 20d-momentum):")
print(f"  train 2022-23:  Sharpe {base_tr['sharpe']:+.2f}  ret {base_tr['ret']:+.1%}  maxDD {base_tr['dd']:.1%}")
print(f"  OOS   2024   :  Sharpe {base_oos['sharpe']:+.2f}  ret {base_oos['ret']:+.1%}  maxDD {base_oos['dd']:.1%}")
print(f"\nOverfit-rejected (train Sharpe > 3.0): {len(overfit)} / {len(rows)} combos")

print(f"\nBEST combo by TRAIN Sharpe (non-overfit):  RSI_OB={best['ob']} RSI_OS={best['os']} ADX_min={best['adx']} FUND_pct={best['fund']:.0%}")
print(f"  train 2022-23:  Sharpe {best['tr']['sharpe']:+.2f}  ret {best['tr']['ret']:+.1%}  maxDD {best['tr']['dd']:.1%}")
print(f"  >>> OOS 2024 :  Sharpe {best['oo']['sharpe']:+.2f}  ret {best['oo']['ret']:+.1%}  maxDD {best['oo']['dd']:.1%}  <<< HELD-OUT")
oos_flag = " ⚠ >3.0 — leakage suspect!" if best["oo"]["sharpe"] > 3.0 else ""
print(f"  OOS leakage sanity: best OOS Sharpe {best['oo']['sharpe']:+.2f}{oos_flag}")

print(f"\nbest OOS vs baseline OOS:  Sharpe {best['oo']['sharpe']:+.2f} vs {base_oos['sharpe']:+.2f}  |  "
      f"ret {best['oo']['ret']:+.1%} vs {base_oos['ret']:+.1%}  |  maxDD {best['oo']['dd']:.1%} vs {base_oos['dd']:.1%}")

print("\nTop-8 by train Sharpe — does train-selection survive OOS? (the honest overfitting tell):")
print(f"  {'RSI_OB/OS':<11}{'ADX':>5}{'FUND':>6}{'trainSh':>9}{'OOSsh':>8}{'OOSret':>9}")
for r in clean[:8]:
    print(f"  {str(r['ob'])+'/'+str(r['os']):<11}{r['adx']:>5}{r['fund']*100:>5.0f}%{r['tr']['sharpe']:>9.2f}"
          f"{r['oo']['sharpe']:>8.2f}{r['oo']['ret']:>+9.1%}")
# correlation between train and OOS Sharpe across all combos = generalization signal
trs = np.array([r["tr"]["sharpe"] for r in clean])
oos = np.array([r["oo"]["sharpe"] for r in clean])
corr = float(np.corrcoef(trs, oos)[0, 1]) if len(clean) > 2 else float("nan")
print(f"\ncorr(train Sharpe, OOS Sharpe) across {len(clean)} combos = {corr:+.2f}  "
      f"(near 0 / negative ⇒ optimization does NOT generalize = noise-fitting)")
print("=" * 96)
