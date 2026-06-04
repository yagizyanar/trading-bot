"""Dissect run_portfolio: isolate the look-ahead in the CAP=10 MOM[d] selection."""
import numpy as np, pandas as pd
import backtest.oos_2024_items as O

btc = O._fetch("BTCUSDT")["close"].astype(float)
per = {}
for c in O.UNIVERSE:
    df = O._fetch(f"{c}USDT")
    if df is None: continue
    df = df[(df.index >= O.LOOKBACK_START) & (df.index <= O.EVAL_END)]
    close = df["close"].astype(float); rets = close.pct_change().fillna(0).to_numpy()
    raw = O._raw_signals(close, O.IN_SAMPLE)
    hp, hg = O._derive_path(raw, rets, O.IN_SAMPLE, O.STOP, "hysteresis")
    per[c] = pd.DataFrame({"hp": hp, "hg": np.nan_to_num(hg), "atr": O._atr_pct_series(df).to_numpy(),
        "beta": O._rolling_beta(close, btc).to_numpy(),
        "mom": close.pct_change(20).abs().fillna(0).to_numpy()}, index=close.index)
    per[c] = per[c][(per[c].index >= O.EVAL_START) & (per[c].index <= O.EVAL_END)]
coins = list(per.keys()); common = None
for c in coins: common = per[c].index if common is None else common.intersection(per[c].index)
common = common.sort_values()
def mat(col): return np.vstack([per[c].reindex(common)[col].fillna(0).to_numpy() for c in coins])
HP, HG, ATR, BETA, MOM = mat("hp"), mat("hg"), mat("atr"), mat("beta"), mat("mom")
ONES = np.ones_like(ATR)
MOMc = np.concatenate([MOM[:, :1], MOM[:, :-1]], axis=1)   # causal: MOM[d-1]
def volm(tv):
    safe = np.where(ATR > 0, ATR, 1.0); return np.clip(np.where(ATR > 0, tv/safe, 1.0), O.VOL_NORM_MIN, O.VOL_NORM_MAX)
def show(lbl, **kw):
    r = O.run_portfolio(coins, common, HP, HG, kw.pop("V"), BETA, kw.pop("M"), **kw)
    print(f"  {lbl:<44} ret {r['total_return']:+8.1%}  Sharpe {r['sharpe']:6.2f}  maxDD {r['max_dd']:7.1%}  avg_active {r['avg_active']:.1f}")

print("== DEPLOYED vs CAUSAL (only the selection criterion's causality differs) ==")
show("E deployed: CAP10 MOM[d], item5+6",  V=volm(.05), M=MOM,  item5=True,  item6=True,  budget=3.0)
show("F causal:   CAP10 MOM[d-1], item5+6", V=volm(.05), M=MOMc, item5=True,  item6=True,  budget=3.0)
print("== isolate the selection alone (items OFF) ==")
O.CAP = 99; O.PER_SLOT = 1.0/13
show("A no-selection CAP=99, items OFF",    V=ONES,      M=MOM,  item5=False, item6=False, budget=3.0)
O.CAP = 10; O.PER_SLOT = 0.1
show("B CAP10 MOM[d] (lookahead), items OFF",  V=ONES,   M=MOM,  item5=False, item6=False, budget=3.0)
show("C CAP10 MOM[d-1] (causal), items OFF",   V=ONES,   M=MOMc, item5=False, item6=False, budget=3.0)
print("== vol-norm / beta-cap contributions (MOM[d]) ==")
show("D CAP10 MOM[d] + vol-norm only",      V=volm(.05), M=MOM,  item5=True,  item6=False, budget=3.0)
