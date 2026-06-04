"""PHASE 1 GATE — delta-neutral funding-carry backtest + combined carry+momentum.

The directional carry failed because it took the legs' PRICE risk. The right vehicle
is per-coin cash-and-carry: SHORT the perp (collects funding) + LONG the spot (hedges
price) → delta-neutral, you harvest the funding. Per unit notional, daily P&L =
    funding[d]  +  (spot_ret[d] - perp_ret[d])   - costs
    └ collected ┘   └ basis residual (≈0, the hedge) ┘

Causal: rank coins by TRAILING funding through d-1, hold the top-K with trailing
funding > FUND_MIN (long-spot/short-perp; sit flat when nothing qualifies), realize
day-d funding + basis. Realistic 2-leg costs on turnover (spot 0.1% + perp 0.05% per
side). Daily granularity (carry is a multi-day hold — no intraday stops needed).
Conservative: capital = spot notional, no leverage. Reports the funding/basis/cost
breakdown (is the return real carry or basis noise?), then combines with the causal
momentum sleeve. Universe = 13-coin set; 2022-2024. ALL DATA CACHED.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from backtest.oos_2024_items import (
    STOP, _atr_pct_series, _derive_path, _raw_signals, _rolling_beta, run_portfolio,
)
from backtest.oos_multiyear import UNIVERSE, _spot
from backtest.oos_intraday import _fetch_15m, _funding_events
from signals.three_layer import VOL_NORM_MAX, VOL_NORM_MIN

SPOT_FEE = 0.0010          # 0.10%/side spot taker (conservative)
PERP_FEE = 0.0005          # 0.05%/side perp taker
COST_SIDE = SPOT_FEE + PERP_FEE     # 0.15% per side (both legs of the pair)
K = 5                      # max simultaneous carry positions
FUND_WINDOW = 3            # trailing-funding lookback (days), causal
FUND_MIN = 0.0             # only hold coins whose trailing daily funding > this
TARGET, BUDGET = 0.05, 3.0


def _build_year(year, lookback=365):
    es = pd.Timestamp(f"{year}-01-01", tz="UTC"); ee = pd.Timestamp(f"{year}-12-31", tz="UTC")
    ss = es - pd.Timedelta(days=lookback + 15)
    btc = _spot("BTCUSDT")["close"]
    per = {}
    for c in UNIVERSE:
        sp = _spot(f"{c}USDT")
        if sp is None:
            continue
        sl = sp[(sp.index >= ss) & (sp.index <= ee)]
        if int((sl.index < es).sum()) < lookback or int(((sl.index >= es) & (sl.index <= ee)).sum()) < 300:
            continue
        close = sl["close"]; rets = close.pct_change().fillna(0.0).to_numpy()
        raw = _raw_signals(close, lookback)
        hp, hg = _derive_path(raw, rets, lookback, STOP, "hysteresis")
        perp = _fetch_15m(c)["c"].resample("1D").last()
        fts, frate = _funding_events(c)
        fund = pd.Series(frate, index=pd.to_datetime(fts, unit="ms", utc=True)).resample("1D").sum()
        d = pd.DataFrame({
            "spot_ret": rets, "hp": hp, "hg": np.nan_to_num(hg),
            "atr": _atr_pct_series(sl).to_numpy(), "beta": _rolling_beta(close, btc).to_numpy(),
            "mom": close.pct_change(20).abs().fillna(0.0).to_numpy(),
        }, index=close.index)
        d["perp"] = perp.reindex(close.index)
        d["perp_ret"] = d["perp"].pct_change()
        d["fund"] = fund.reindex(close.index).fillna(0.0)
        per[c] = d[(d.index >= es) & (d.index <= ee)]
    coins = list(per.keys()); common = None
    for c in coins:
        common = per[c].index if common is None else common.intersection(per[c].index)
    common = common.sort_values()

    def mat(col, fill0):
        out = np.vstack([per[c].reindex(common)[col].to_numpy() for c in coins])
        return np.nan_to_num(out) if fill0 else out

    M = {k: mat(k, True) for k in ("hp", "hg", "atr", "beta", "mom", "spot_ret")}
    M["fund"] = mat("fund", False)
    M["perp_ret"] = mat("perp_ret", False)
    return coins, common, M


FUND_ENTER = 0.0001        # enter a coin when trailing daily funding > 0.01%/day (~3.7%/yr)
FUND_EXIT = 0.0            # hold until trailing funding turns non-positive (hysteresis)


def carry_net(M, k=K, rebal_days=7, fund_enter=FUND_ENTER, fund_exit=FUND_EXIT, cost_side=COST_SIDE):
    """Weekly-rebalanced delta-neutral carry with funding hysteresis (hold, don't churn)."""
    FUND, SPOT_RET, PERP_RET = M["fund"], M["spot_ret"], M["perp_ret"]
    basis = SPOT_RET - PERP_RET
    nc, nd = FUND.shape
    tf = np.full((nc, nd), np.nan)              # causal trailing funding (through d-1)
    for d in range(1, nd):
        seg = FUND[:, max(0, d - FUND_WINDOW):d]
        if seg.shape[1] > 0:
            tf[:, d] = np.nanmean(seg, axis=1)
    net = np.zeros(nd); w = np.zeros(nc); prev = np.zeros(nc); held = []
    fund_tot = basis_tot = cost_tot = 0.0; deployed = np.zeros(nd); rebals = 0
    for d in range(nd):
        if d % rebal_days == 0:
            f = tf[:, d]
            ok = lambda i: np.isfinite(f[i]) and np.isfinite(basis[i, d])
            keep = [i for i in held if ok(i) and f[i] > fund_exit]                 # hysteresis: hold
            cand = sorted([i for i in range(nc) if ok(i) and f[i] > fund_enter and i not in keep],
                          key=lambda i: -f[i])
            held = keep + cand[:max(0, k - len(keep))]
            w = np.zeros(nc)
            for i in held:
                w[i] = 1.0 / len(held)
            rebals += 1
        wd = w.copy()
        bad = ~(np.isfinite(basis[:, d]) & np.isfinite(FUND[:, d]))
        wd[bad] = 0.0
        fc = float(np.sum(wd * np.nan_to_num(FUND[:, d])))
        bp = float(np.sum(wd * np.nan_to_num(basis[:, d])))
        cst = cost_side * float(np.abs(w - prev).sum())
        net[d] = fc + bp - cst
        fund_tot += fc; basis_tot += bp; cost_tot += cst; deployed[d] = wd.sum()
        prev = w
    return net, dict(fund=fund_tot, basis=basis_tot, cost=cost_tot, deployed=float(deployed.mean()), rebals=rebals)


def _volm(atr):
    safe = np.where(atr > 0, atr, 1.0)
    return np.clip(np.where(atr > 0, TARGET / safe, 1.0), VOL_NORM_MIN, VOL_NORM_MAX)


def _sharpe(r):
    r = np.asarray(r, float); s = r[np.isfinite(r)]; sd = s.std(ddof=1)
    return float(s.mean() / sd * math.sqrt(365)) if sd > 0 else 0.0


def _tot(r): return float(np.prod(1 + np.nan_to_num(np.asarray(r, float))) - 1)


def _maxdd(r):
    eq = np.cumprod(1 + np.nan_to_num(np.asarray(r, float))); rm = np.maximum.accumulate(eq)
    return float(((eq - rm) / rm).min())


def main():
    print("=" * 104)
    print(f"PHASE 1 — DELTA-NEUTRAL FUNDING CARRY (short-perp+long-spot) + COMBINED w/ MOMENTUM, 2022-2024")
    print(f"  K={K} positions · trailing funding {FUND_WINDOW}d · cost {COST_SIDE:.2%}/side (spot {SPOT_FEE:.1%}+perp {PERP_FEE:.2%}) · no leverage")
    print("=" * 104)
    for year in [2022, 2023, 2024]:
        coins, common, M = _build_year(year)
        print(f"\n{year}  ({len(coins)} coins, {len(common)} days)")
        print("  CARRY rebalance-frequency sweep (cost is the whole story):")
        cnet7 = None
        for rd, tag in [(1, "daily  "), (7, "WEEKLY "), (30, "monthly")]:
            cnet, brk = carry_net(M, rebal_days=rd)
            if rd == 7:
                cnet7 = cnet
            print(f"    {tag} rebal  ret {_tot(cnet):>+7.1%}  Sh {_sharpe(cnet):>5.2f}  DD {_maxdd(cnet):>6.1%}  dep {brk['deployed']:>4.0%}"
                  f"  [fund {brk['fund']:>+6.1%} / basis {brk['basis']:>+6.1%} / cost {-brk['cost']:>6.1%}]  ({brk['rebals']} rebals)")
        mom = run_portfolio(coins, common, M["hp"], M["hg"], _volm(M["atr"]), M["beta"], M["mom"],
                            item5=True, item6=True, budget=BUDGET)
        mnet = np.nan_to_num(mom["net"])
        corr = float(np.corrcoef(cnet7, mnet)[0, 1])
        vc, vm = np.std(cnet7, ddof=1), np.std(mnet, ddof=1)
        wc = (1 / vc) / ((1 / vc) + (1 / vm)) if vc > 0 and vm > 0 else 0.5
        rp = wc * cnet7 + (1 - wc) * mnet
        print(f"  MOMENTUM (run_portfolio; OPTIMISTIC-daily — realistic 15m is NEG 2/3yr):"
              f"  ret {_tot(mnet):>+7.1%}  Sh {_sharpe(mnet):>5.2f}  DD {_maxdd(mnet):>6.1%}")
        print(f"  corr(weekly-carry, momentum) {corr:>+.2f}  |  COMBINED risk-parity (carry wt {wc:.0%}):"
              f"  ret {_tot(rp):>+7.1%}  Sh {_sharpe(rp):>5.2f}  DD {_maxdd(rp):>6.1%}")
    print("=" * 104)


if __name__ == "__main__":
    main()
