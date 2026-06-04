"""Why does momentum work in 2023 but fail 2022/2024? Beta-vs-alpha + regime-gate test."""
import pandas as pd, numpy as np, math
from backtest.oos_2024_items import _raw_signals, _derive_path
from backtest.oos_multiyear import _spot, UNIVERSE

def sh(r):
    s = np.asarray(r, float); s = s[np.isfinite(s)]; sd = s.std(ddof=1)
    return float(s.mean()/sd*math.sqrt(365)) if sd > 0 else 0.0
def tot(r): return float(np.prod(1+np.nan_to_num(np.asarray(r, float)))-1)

for year in [2022, 2023, 2024]:
    es = pd.Timestamp(f"{year}-01-01", tz="UTC"); ee = pd.Timestamp(f"{year}-12-31", tz="UTC")
    ss = es - pd.Timedelta(days=380)
    data = {}
    for c in UNIVERSE:
        df = _spot(f"{c}USDT")
        if df is None: continue
        sl = df[(df.index >= ss) & (df.index <= ee)]
        if int((sl.index < es).sum()) < 365 or int(((sl.index >= es) & (sl.index <= ee)).sum()) < 350: continue
        close = sl["close"]; rets = close.pct_change().fillna(0).to_numpy()
        pos, gross = _derive_path(_raw_signals(close, 365), rets, 365, 99.0, "hysteresis")
        data[c] = pd.DataFrame({"ret": rets, "pos": pos, "gross": np.nan_to_num(gross)}, index=sl.index)
    coins = list(data.keys()); common = None
    for c in coins: common = data[c].index if common is None else common.intersection(data[c].index)
    common = common.sort_values()
    RET = pd.concat([data[c].reindex(common)["ret"] for c in coins], axis=1).fillna(0)
    POS = pd.concat([data[c].reindex(common)["pos"] for c in coins], axis=1).fillna(0)
    GR = pd.concat([data[c].reindex(common)["gross"] for c in coins], axis=1).fillna(0)
    basket = (1 + RET.mean(axis=1)).cumprod()
    gate = (basket.pct_change(20).shift(1) > 0).astype(float)            # causal: basket 20d uptrend
    em = (common >= es) & (common <= ee)
    bh, sig = RET.mean(axis=1)[em], GR.mean(axis=1)[em]
    longleg = GR.where(POS > 0, 0.0).mean(axis=1)[em]
    shortleg = GR.where(POS < 0, 0.0).mean(axis=1)[em]
    gated = (GR.mean(axis=1) * gate)[em]
    print(f"{year}: B&H {tot(bh):+5.0%}(Sh{sh(bh):+.2f}) | signal {tot(sig):+5.0%}(Sh{sh(sig):+.2f}) | "
          f"LONG-leg {tot(longleg):+5.0%}  SHORT-leg {tot(shortleg):+5.0%} | avg-net {POS.mean(axis=1)[em].mean():+.2f} | "
          f"corr(sig,mkt) {float(sig.corr(bh)):+.2f} | basket-GATED {tot(gated):+5.0%}(Sh{sh(gated):+.2f}, traded {gate[em].mean():.0%} of days)")
