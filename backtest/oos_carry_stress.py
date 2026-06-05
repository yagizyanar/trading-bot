"""PHASE 1.5 — tail-stress the delta-neutral carry; honest combined with REALISTIC 15m momentum.

The calm-period Sharpe (6-11) hides that carry is short-volatility. Here we add the
frictions and tail events the daily backtest omits, and pair the carry with the
REALISTIC 15m momentum (not the optimistic run_portfolio), to get the
drawdown-aware edge.

  - slippage: +0.10%/side on top of fees → 0.25%/side total
  - funding-flip scenario: a 7-day window at -0.4%/day funding while deployed
  - squeeze scenario: a +12% intraday perp-vs-spot basis spike while deployed
  - liquidation: worst 1-day adverse perp move vs the threshold at leverage L
  - combined: weekly-carry(stressed) + realistic 15m momentum (equal-weight, cooldown)
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from backtest.oos_carry import carry_net, COST_SIDE, FUND_WINDOW
from backtest.oos_intraday import _bars_arrays, _daily_targets, _fetch_15m, _funding_events, replay_intraday
from backtest.oos_multiyear import UNIVERSE, _spot

SLIP = 0.0010                       # 0.10%/side slippage per leg-pair
COST_STRESS = COST_SIDE + SLIP      # 0.25%/side (fees 0.15 + slippage 0.10)
FLIP_RATE, FLIP_DAYS = -0.004, 7    # funding-flip scenario: -0.4%/day for 7 days
SQUEEZE = 0.12                      # +12% intraday basis spike scenario


def _sharpe(r):
    r = np.asarray(r, float); s = r[np.isfinite(r)]; sd = s.std(ddof=1)
    return float(s.mean() / sd * math.sqrt(365)) if sd > 0 else 0.0
def _tot(r): return float(np.prod(1 + np.nan_to_num(np.asarray(r, float))) - 1)
def _maxdd(r):
    eq = np.cumprod(1 + np.nan_to_num(np.asarray(r, float))); rm = np.maximum.accumulate(eq)
    return float(((eq - rm) / rm).min())


def _build_carry(year):
    es = pd.Timestamp(f"{year}-01-01", tz="UTC"); ee = pd.Timestamp(f"{year}-12-31", tz="UTC")
    ss = es - pd.Timedelta(days=FUND_WINDOW + 8)
    per = {}
    for c in UNIVERSE:
        sp = _spot(f"{c}USDT")
        if sp is None:
            continue
        sl = sp[(sp.index >= ss) & (sp.index <= ee)]
        if int(((sl.index >= es) & (sl.index <= ee)).sum()) < 300:
            continue
        close = sl["close"]
        perp = _fetch_15m(c)["c"].resample("1D").last()
        fts, frate = _funding_events(c)
        fund = pd.Series(frate, index=pd.to_datetime(fts, unit="ms", utc=True)).resample("1D").sum()
        d = pd.DataFrame({"spot_ret": close.pct_change().fillna(0.0)}, index=close.index)
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
    M = {"spot_ret": mat("spot_ret", True), "fund": mat("fund", False), "perp_ret": mat("perp_ret", False)}
    return coins, common, M


def _mom15(year, common):
    """Realistic 15m momentum, equal-weight, cooldown — aligned to carry's `common` dates."""
    cd = [d.date() for d in common]
    series = []
    for c in UNIVERSE:
        tg = _daily_targets(c, year)
        if not tg:
            continue
        bars = _fetch_15m(c)
        bars = bars[(bars.index >= pd.Timestamp(f"{year}-01-01", tz="UTC")) &
                    (bars.index <= pd.Timestamp(f"{year}-12-31 23:59", tz="UTC"))]
        if len(bars) < 20000:
            continue
        fts, frate = _funding_events(c)
        r, *_ = replay_intraday(_bars_arrays(bars), tg, fts, frate, cooldown_bars=96)
        series.append(r.rename(c))
    ew = pd.concat(series, axis=1).fillna(0.0).mean(axis=1)
    return np.array([float(ew.get(d, 0.0)) for d in cd])


def main():
    print("=" * 104)
    print("PHASE 1.5 — TAIL-STRESS THE CARRY + REALISTIC-MOMENTUM COMBINED, 2022-2024")
    print(f"  slippage +{SLIP:.2%}/side (total {COST_STRESS:.2%}) · funding-flip {FLIP_RATE:.1%}/d x{FLIP_DAYS}d · squeeze +{SQUEEZE:.0%}")
    print("=" * 104)
    worst_up_all = 0.0
    for year in [2022, 2023, 2024]:
        coins, common, M = _build_carry(year)
        cnet, brk = carry_net(M, rebal_days=7, cost_side=COST_STRESS)
        mnet = _mom15(year, common)
        dep = brk["deployed"]
        # tail scenarios (what a bad event costs at this deployment)
        flip = FLIP_RATE * FLIP_DAYS * dep
        sqz = -SQUEEZE * dep
        stressed_dd = _maxdd(cnet) + sqz             # historical DD + a squeeze hitting the trough
        worst_up = float(np.nanmax(M["perp_ret"]))   # worst 1-day perp up-move (squeeze proxy)
        worst_up_all = max(worst_up_all, worst_up)
        # combined: carry + realistic momentum
        corr = float(np.corrcoef(cnet, mnet)[0, 1])
        c80 = 0.8 * cnet + 0.2 * mnet
        vc, vm = np.std(cnet, ddof=1), np.std(mnet, ddof=1)
        wc = (1 / vc) / ((1 / vc) + (1 / vm)) if vc > 0 and vm > 0 else 0.5
        rp = wc * cnet + (1 - wc) * mnet
        print(f"\n{year}  ({len(coins)} coins, {len(common)} days, carry deployed {dep:.0%})")
        print(f"  CARRY (stressed: fees+slippage)   ret {_tot(cnet):>+7.1%}  Sharpe {_sharpe(cnet):>5.2f}  maxDD {_maxdd(cnet):>6.1%}")
        print(f"    tail what-ifs:  funding-flip {flip:>+5.1%}   squeeze {sqz:>+6.1%}   → stressed maxDD {stressed_dd:>6.1%}   (worst 1d perp move +{worst_up:.0%})")
        print(f"  MOMENTUM (realistic 15m+cooldown) ret {_tot(mnet):>+7.1%}  Sharpe {_sharpe(mnet):>5.2f}  maxDD {_maxdd(mnet):>6.1%}")
        print(f"  corr(carry,mom) {corr:>+.2f}  |  COMBINED 80/20  ret {_tot(c80):>+7.1%} Sh {_sharpe(c80):>5.2f} DD {_maxdd(c80):>6.1%}"
              f"  |  risk-parity(carry {wc:.0%})  ret {_tot(rp):>+7.1%} Sh {_sharpe(rp):>5.2f} DD {_maxdd(rp):>6.1%}")
    print("\n  LIQUIDATION (perp short, worst 1-day up-move +{:.0%} across all years):".format(worst_up_all))
    for L in [2, 3, 5]:
        thr = 1.0 / L
        print(f"    perp leverage {L}x → liquidation at ~+{thr:.0%} adverse;  worst move "
              f"{'WOULD LIQUIDATE' if worst_up_all > thr else 'survives'}")
    print("=" * 104)


if __name__ == "__main__":
    main()
