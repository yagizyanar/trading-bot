import pandas as pd, numpy as np, math
from backtest.oos_2024_items import _raw_signals, _derive_path
from backtest.oos_multiyear import _spot, UNIVERSE
from backtest.oos_intraday import _daily_targets, signal_only_returns

def sh(r):
    r=np.asarray(r,float); sd=r.std(ddof=1)
    return float(r.mean()/sd*math.sqrt(365)) if sd>0 else 0.0

es=pd.Timestamp("2024-01-01",tz="UTC"); ee=pd.Timestamp("2024-12-31",tz="UTC"); ss=es-pd.Timedelta(days=375)
oos,mine,coins=[],[],[]
for c in UNIVERSE:
    df=_spot(f"{c}USDT")
    if df is None: continue
    sl=df[(df.index>=ss)&(df.index<=ee)]
    if int((sl.index<es).sum())<365 or int(((sl.index>=es)&(sl.index<=ee)).sum())<350: continue
    close=sl["close"]; rets=close.pct_change().fillna(0).to_numpy()
    raw=_raw_signals(close,365)
    pos,gross=_derive_path(raw,rets,365,99.0,"hysteresis")  # no-stop pure signal
    sr=pd.Series(np.nan_to_num(gross),index=sl.index); sr=sr[(sr.index>=es)&(sr.index<=ee)]
    oos.append(sr.rename(c))
    dc=close[(close.index>=es)&(close.index<=ee)]
    mine.append(signal_only_returns(dc,_daily_targets(c,2024),cost=0.0).rename(c))
    coins.append(c)
po=pd.concat(oos,axis=1).fillna(0).mean(axis=1); pm=pd.concat(mine,axis=1).fillna(0).mean(axis=1)
print("coins",len(coins))
print("OOS_2024_items signal-only EW: ret",round(float((1+po).prod()-1),3),"sharpe",round(sh(po),2))
print("oos_intraday    signal-only EW: ret",round(float((1+pm).prod()-1),3),"sharpe",round(sh(pm),2))
print("corr(daily streams)",round(float(po.corr(pm)),3))
# also: simple long-only EW buy&hold of the 13 alts in 2024 for context
bh=[]
for c in coins:
    df=_spot(f"{c}USDT"); sl=df[(df.index>=es)&(df.index<=ee)]
    bh.append(sl["close"].pct_change().fillna(0).rename(c))
pbh=pd.concat(bh,axis=1).fillna(0).mean(axis=1)
print("LONG-ONLY buy&hold EW 13 alts 2024: ret",round(float((1+pbh).prod()-1),3))
