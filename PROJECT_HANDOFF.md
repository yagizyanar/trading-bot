# trade-sentiment-markov — Project Handoff

**Last updated:** 2026-06-03
**Status:** Paper-trading (dry_run=true) on Binance Futures via Freqtrade. Live, profitable on a small sample. **Planned: switch to REAL money ~June 5 — see §0 prerequisites first.**
**Equity:** ~$10,566 (balance.total) / +$582 all-in PnL · 187 closed trades · 77% win rate · 15 open positions
**Project root:** `C:\Users\yagiz\trade`
**VPS:** `178.105.141.217` (Hetzner, Ubuntu 24.04, Nuremberg)
**Dashboard:** http://178.105.141.217/ (HTTP basic auth — creds in `C:\Users\yagiz\.trade_dashboard_creds.txt`)
**Tests:** 148 passing locally / **127 on the VPS** (the 21-test gap is `tests/test_backtest.py` — backtest harness is research-only, not deployed; both suites 0 failures as of 2026-06-03).

---

## 0. ⚠️ GOING LIVE (~June 5) — DO THESE FIRST

**✅ Done 2026-06-03 (readiness pass):** forced **1× leverage** (item 6 — `MAX_LEVERAGE=1`, now actually enforced at the strategy `leverage()` callback; it was previously a dead constant); **SSH hardened to key-only + fail2ban** (item 3 — password SSH disabled, see §7 for the new key auth); committed the deployed VPS state (git HEAD `e5c4dcf`); deleted a stale 372 MB log. Comprehensive health check = GREEN (services, 127 tests, stops firing, equity reconciles).
**⛔ Still required before real money:** rotate Binance keys (item 1), flip `dry_run=false` (item 4). HTTPS + password rotation (items 2–3) recommended but lower priority now that SSH is key-only.

The bot is still `dry_run=true`. Before any real money:

**Security (non-negotiable — credentials from the build transcript are compromised):**
1. **Rotate Binance API keys** → regenerate with **Futures-trade only, WITHDRAWALS DISABLED, IP-restricted to 178.105.141.217**. This means a leaked key can trade but **cannot drain funds**. Update `/opt/trading-bot/.env` + `config/config-secrets.json`, restart freqtrade.
2. Rotate VPS root password, Postgres `trader` pw, dashboard basic-auth, Freqtrade API pw.
3. (Recommended) HTTPS via certbot + ~~SSH key auth + disable password login~~ **(✅ done 2026-06-03 — key-only, fail2ban active; HTTPS still TODO)**.

**Flip to live:**
4. `config/config.json` and `config/config-secrets.json`: `"dry_run": false`. Restart freqtrade.

**Risk discipline (honest — see §3 for why):**
5. **Start TINY** — money you'd be 100% fine losing entirely ($100–300), not savings. This is *tuition* to validate live execution, not an investment.
6. ~~**Keep leverage at 1×** initially~~ **(✅ done 2026-06-03 — `MAX_LEVERAGE=1`, enforced at the strategy `leverage()` callback; `_choose_leverage` still logs 2× intent but execution is clamped. Raise back to 2 to restore.)** The edge is ~Sharpe 1; amplifying it just amplifies variance.
7. **The strategy is net-short beta (−0.45 to −0.85).** It made its paper gains because crypto fell. **It will lose on a sustained rally.** Going live = consciously betting crypto keeps trending down/choppy.
8. Watch the first days closely. Kill-switch: `mv TRADING_LOCKED.txt.disabled TRADING_LOCKED.txt` or `systemctl stop trading-bot-freqtrade`.
9. Compare live fills vs paper for weeks — slippage/funding/partial-fills are exactly what paper can't show and what a Sharpe-1 edge has little margin to absorb.

---

## 1. What this is

24/7 crypto-futures paper-trading bot: Markov regime detection (observable, 20-day rolling-return labels) on daily candles drives **direction + base size**; sentiment scales size; technical contradiction shaves size; executed via Freqtrade (custom `FreqAISentimentStrategy` reading a Postgres `signal_log` table); risk via circuit breakers + sector correlation caps + stops; surfaced in a React dashboard with 5s WebSocket updates.

**Honest characterization (validated this session): the edge is ~Sharpe 1 trend/beta-timing, NOT stock-picking skill.** See §4.

---

## 2. SESSION CHANGELOG (2026-05-26 → 06-03)

### Go-live readiness pass (2026-06-03, committed on VPS as `e5c4dcf`)
- **Forced 1× leverage** — `config/settings.py::MAX_LEVERAGE` 2→1; strategy `leverage()` now `min(lev, max_leverage, MAX_LEVERAGE)`. Fixes a latent bug: `MAX_LEVERAGE` was only read by the dead `risk.position_manager.decide_leverage`, never by the live path (`_choose_leverage`). `config.json::leverage` 2→1 too. Test `test_decide_leverage_requires_both_conditions` made cap-aware (asserts `MAX_LEVERAGE`, not literal 2).
- **SSH hardened** — installed ed25519 key, disabled password auth (`/etc/ssh/sshd_config.d/10-hardening.conf`, sorts before `50-cloud-init.conf`), `PermitRootLogin prohibit-password`, installed+enabled **fail2ban** (sshd jail; immediately banned a live brute-forcer). `.ssh_helper.py` now supports `VPS_KEYFILE`.
- **Deploy hygiene** — committed the dirty VPS working tree (40 files; base64 deploys had left git HEAD stale at `2bb46cd`) so a stray `git reset/checkout` can't revert it; deleted stale 372 MB `freqtrade.log-2026-05-27`.
- Verified VPS running code == local git (md5-identical, 6 critical files); full health check GREEN.

### Three fixes (2026-06-03, committed on VPS as `e7a1932`; 128 tests pass)
- **Volume-anomaly bug FIXED** — `sentiment/binance_data.py::volume_anomaly` used the still-forming current candle (`iloc[-1]`) as `recent`. At the top of the hour (when sentiment_update/market_evaluation run) that bar is near-empty → `spike≈0` → score pinned to **−1.0 for every coin**, killing all per-coin differentiation. Now uses the last *closed* bar (`iloc[-2]`). Verified live: 12 coins → 12 distinct values, range −0.72…+1.0, none at floor. +regression test `test_volume_anomaly_ignores_in_progress_bar`.
- **+15% take-profit RE-ENABLED** — `minimal_roi {"0":100}`→`{"0":0.15}` (reverses `c63ac5f`). Caveat: `TAKE_PROFIT_PCT` (settings.py) still 1.0 so dashboard/alerts don't reflect it (display-only — see §3).
- **Capacity realigned to documented risk rules** — `max_open_trades` 15→10, `tradable_balance_ratio` 0.75→0.50, **and** `MAX_CAPITAL_DEPLOYED_PCT` 0.75→0.50 (the gate must equal tradable_balance_ratio or the 0.14%-stake clamp returns — §5). Side effect: 15 positions were open under the old cap; Freqtrade won't force-close, so the book drifts down to 10 / 50% as positions close.

### Memory/logging fixes
- `7a6b6fc` Routines read live Freqtrade state; **mirror closed trades into `memory/trade_log.md`** (was never populated); added `setup_routine_logging()` so routine `log.info` reaches `/var/log` (logs were 0 bytes); `day_close` queries Freqtrade API not the empty legacy `trades` table; new `append_daily_summary()`.
- `c6b810a` **Fix mirror watermark bug** — single-int high-water-mark skipped out-of-order closes (a low-id trade closing after a high-id one). Now a **set** of mirrored trade_ids (`memory/.mirrored_trade_ids`).

### Signal / strategy
- `7dbfcef` **Hyperliquid broke** (CDN leaderboard addresses no longer map to clearinghouseState — all top traders return 0 positions). Replaced the "hyperliquid" sentiment slot with **Binance Futures `topLongShortPositionRatio`** (per-coin, all 24 coins, follow-smart-money). DB column/slot name kept for stability.
- `2ded7a6` **Split L1** into `L1-MAJOR` (SOL,AVAX,NEAR,DOT) and `L1-ALT` (APT,SUI,ATOM,S) — 8 L1 coins competing for 2 slots was the top bottleneck (counterfactual: ~$100 of 15h alpha lost). Effective L1 ceiling 2→4.
- `e9e819f` Lower TP 15%→10%; lower **2× sentiment threshold 0.3→0.2** (old 0.3 was nearly unreachable; 2×-eligible coins went 1→9).
- `8944563` **market_evaluation gate now processes candidates in signal-conviction order** (highest position_size_pct first) so the strongest setup wins a contested sector slot, not whoever's first in the coin list.
- `c63ac5f` **Disabled fixed ROI exit** (`minimal_roi`→`{"0":100}`); trailing-stop is now the only profit-taking mechanism (+ hard −5% stop + signal flip). `a76f6b2` dashboard renders TP as "—" when disabled.
- `76b7a25` Regime window **730→365 days** (sweep showed window immaterial; 365 nominal-best — marginal/reversible).

### Capacity / sizing
- `73b8dc6` **Universe 18→24 coins** (+MEME: WIF,1000BONK,1000PEPE; +AI: FET,RENDER,TAO). `max_open_trades` 10→15.
- `13ec830` `MAX_CAPITAL_DEPLOYED_PCT` 0.50→0.75. `f2e20ea` `tradable_balance_ratio` 0.50→0.75 (the two were mismatched — Freqtrade was silently clamping new entries; one trade opened at 0.14% instead of 5%). `f92f6df` `MAX_OPEN_POSITIONS` 10→15 (internal gate was capping at 10 while Freqtrade allowed 15). `52db230` both now **derived from config.json** (single source of truth, no drift).

### Dashboard / ops
- `8df3ce8` "Next routine" widget shows all routines firing at the same minute.
- `a6a30e3` **Portfolio chart persists across refresh** — new `snapshot_writer` cron (every 5 min) writes `performance_snapshots`; chart pre-loads 30d history.
- `a644518` / `9bcdfcc` Closed table: + Size(USDT), + Opened/Closed timestamps (UTC).
- `ce8e5ac` **System Alerts panel** — 15 health checks (trading/sentiment/data/system), green/yellow/red, 5-min cache, in WS payload. `dacd67d` tuned binance_link to retry-once + degraded TTL after a false-positive flap.
- `f432352` Health follow-ups: logrotate for freqtrade.log (was 389 MB), real deployed% in market_context.md, pruned stale FTM/MATIC rows.
- `0e3af14` **Fix closed-positions table hiding newest trades** — Freqtrade `/api/v1/trades?limit=N` returns OLDEST-first, so the newest fell off as history grew. Now fetch full history, sort newest-first.
- `f65dcaf` Closed table: **pagination ("Load more", 50/page, all 187 reachable), leverage column, scrollable**.

### Backtest harness (new — `backtest/daily_walk_forward.py`)
- `44c129f` Faithful **daily walk-forward** with transaction costs (the old `walk_forward.py` held one position 180d, no costs — kept for persistence research). No-look-ahead asserted by test.
- `bc8a63a` continuous vol-normalized momentum signal + pluggable `signal_fn`.
- `5dd682f` `market_neutral_portfolio` + `market_beta`.
- `93c565b` `efficiency_ratio` + `regime_scaled_portfolio`.
- `d88f5b9` ATR volatility-adaptive stops.

---

## 3. CURRENT SYSTEM STATE (exact parameters)

**Universe:** 24 coins / 8 sectors (cap 2 per sector via `CORRELATED_SECTOR_LIMIT=3`):
| Sector | Coins |
|---|---|
| L1-MAJOR | SOL, AVAX, NEAR, DOT |
| L1-ALT | APT, SUI, ATOM, S |
| L2 | POL, ARB, OP |
| ORACLE | LINK |
| DEFI | INJ, DYDX, GMX |
| GAMING | SAND, MANA, AXS |
| MEME | WIF, 1000BONK, 1000PEPE |
| AI | FET, RENDER, TAO |

**Risk / sizing (`config/settings.py` + `config/config.json`):**
- `max_open_trades` = **10**, `MAX_OPEN_POSITIONS` = **10** (both from config.json::max_open_trades; realigned from 15 to the documented "max 10 positions" rule 2026-06-03)
- `MAX_CAPITAL_DEPLOYED_PCT` = **0.50**, `tradable_balance_ratio` = **0.50** (realigned from 0.75 to the documented "max 50% capital" rule; the two MUST stay equal — see §5)
- Base sizes: 5% / 3% / 1% (Markov full/medium/small) × sentiment mult (0.25–1.0) × tech (0.75 if contradiction) × regime (0.5 sideways)
- `MAX_LEVERAGE` = **1** (was 2; clamped at the strategy `leverage()` callback for go-live 2026-06-03), `SENTIMENT_2X_THRESHOLD` = 0.2 (the 2× rule still computes — LONG in Bull/Euphoria w/ sent>+0.2, SHORT in Bear w/ sent<−0.2 — but execution is capped at 1× while `MAX_LEVERAGE=1`)
- `stoploss` = −0.05 (hard), `minimal_roi` = `{"0": 0.15}` (**+15% take-profit RE-ENABLED 2026-06-03** — matches the strategy's own `minimal_roi` attr; coexists with trailing stop + −5% stop + signal flip, whichever fires first), trailing 2% engaging at +3% profit
- `TAKE_PROFIT_PCT` = 1.0 (settings.py — **still display-only & stale**: drives the dashboard TP column + position_monitor TP_NEAR alerts, NOT the exit. With ROI now at 0.15 the dashboard shows TP as "—" and no TP_NEAR alerts fire. Set to 0.15 to make the UI/alerts reflect the live +15% exit), timeframe = 15m, regime window = 365 days

**Sentiment sources (`sentiment/analyzer.py` weights):** news 0.30 (**fixed 2026-06-03** — now free no-key RSS feeds via CoinTelegraph/CoinDesk/Decrypt + FinBERT; functional but per-coin coverage is sparse ~2–4/24, missing coins redistribute), volume 0.20, long_short 0.20, funding 0.15, "yfinance" 0.10 (**fixed 2026-06-03** — slot now = CoinGecko 7d % change, batched, **all 24 coins covered**; field name kept for DB stability), "hyperliquid" 0.05 (= Binance top-trader ratio).

**Infra:** 5 services active (freqtrade, backend, nginx, postgres, cron). Cron: pre_market, sentiment_update (2h), market_evaluation (1h), midday_check, day_close, weekly_review, position_monitor (1min), **snapshot_writer (5min, new)**. `dry_run = true`.

---

## 4. BACKTEST RESULTS — 6 experiments, ALL negative

Harness: `backtest/daily_walk_forward.py` (daily walk-forward, rolling re-estimate, 0.15%/side costs, −5% stop, cap-15; majors-10 = survivorship-controlled universe; same eval-days for fair comparison). Look-ahead verified absent.

| # | Experiment | Result | Verdict |
|---|---|---|---|
| 1 | Regime window 252/365/540/730 | Sharpe ~2.3 regardless (same eval-days) | Immaterial |
| 2 | Continuous vol-normalized signal | best tie 2.30 vs 2.32; 0/10 coins improved | No improvement |
| 3 | Market-neutral (long top-k / short bottom-k) | Sharpe 1.8 best, beta→0 | Lower — revealed edge=beta |
| 4 | Regime-risk exposure scaling (Efficiency Ratio) | all 12 configs lower Sharpe; cuts DD −26.5%→~−20% | Drawdown knob, not Sharpe |
| 5 | ATR volatility-adaptive stops | worse than fixed −5% on Sharpe, win rate, avg loss | Fixed −5% is better |
| 6 | Tighter stops −2%…−5% | monotonic to boundary, Sharpe 5.4 (impossible) | **Artifact** — daily bars can't test stops |

**Edge conclusion:** Real but modest — **~Sharpe 1.0–1.2 after survivorship + realistic costs + position cap** (the naive backtest shows Sharpe ~3, which is inflated). The "Markov signal" is effectively **20-day momentum** (the transition matrix is diagonal-dominant → signal collapses to sign-of-trailing-return; magnitude carries no per-coin info — hence every coin reading ≈ −0.87 in a downtrend). Directional Sharpe is **largely short-beta** (−0.45 to −0.85): a liability when crypto rallies. Headline numbers flattered by a down-only sample + survivor coins (WIF/BONK/PEPE).

**Memory:** full detail in `~/.claude/.../memory/project_edge_backtest.md`.

---

## 5. WHAT WAS LEARNED

1. **The edge is trend/beta-timing, ~Sharpe 1, not skill.** Six signal/risk experiments produced zero improvements — **signal engineering on daily OHLCV is exhausted.** Stop tuning parameters expecting Sharpe gains.
2. **The backtest can't validly test stops** (intraday mechanism on daily bars → produces artifacts). Stop questions need 15-min data or live.
3. **Validate-before-ship works.** Two self-corrections were caught by measurement, not belief: (a) a "window is too slow" claim that was a confounded-period artifact; (b) a tempting tighter-stop "win" that was a Sharpe-5.4 daily-bar artifact. Neither shipped.
4. **Don't trust Freqtrade result ordering.** Bit twice — out-of-order trade closes (watermark bug) and oldest-first `/api/v1/trades` (closed-table bug). Fetch enough to cover everything and sort yourself.
5. **Keep capacity knobs consistent.** The position-sizing bug (0.14% stake) and the 10-vs-15 cap mismatch both came from two settings that must agree; now derived from one source.

---

## 6. NEXT STEPS

**Immediate (this week — going live):** §0 checklist. Rotate keys (no-withdrawal!), flip dry_run, start tiny + 1×, monitor.

**To actually grow the edge (post-live, in priority order — NOT more signal tweaks):**
1. **Live validation with small capital** — the only source of new information now (slippage, funding, fills, the unvalidated sentiment layer). This is the 3/10→4/10 step.
2. **15-minute intraday backtest** — the only valid way to test stop tightness (the daily harness can't). A real build.
3. **External-data alpha** (on-chain flows / paid feeds) — the only genuinely-new edge source. Months + cost.
4. **Market-neutral as a separate sleeve** (not a replacement) — deliberate beta reduction for regime-robustness; accepts lower Sharpe.
5. `regime_scaled_portfolio` is available as a **drawdown knob** if a smoother curve ever matters more than Sharpe.

---

## 7. OPERATIONAL REFERENCE

**SSH:** `178.105.141.217` root (password — rotate per §0). Paramiko helper: `C:\Users\yagiz\trade\.ssh_helper.py` (reads `VPS_HOST`/`VPS_USER`/`VPS_PASSWORD` env). On VPS the Freqtrade API pw is in `/opt/trading-bot/config/config-secrets.json`.

**Restart:** `systemctl restart trading-bot-freqtrade trading-bot-backend`
**Watch:** `journalctl -u trading-bot-freqtrade -f` · logs in `/var/log/trading-bot/` (now logrotated)
**Kill trading:** `mv /opt/trading-bot/TRADING_LOCKED.txt.disabled /opt/trading-bot/TRADING_LOCKED.txt` (unlock: `rm` it)
**Tests:** `cd /opt/trading-bot && .venv/bin/python -m pytest tests/ -q` (148 passing)
**Deploy:** VPS has no git auth — changes pushed this session via base64-over-SSH (`echo '<b64>' | base64 -d > file`), then `systemctl restart` / `npm run build` in `dashboard/frontend`. GitHub remote `github.com/yagizyanar/trading-bot` (push from Windows).

**Key files:** signals/three_layer.py (`_choose_leverage`, `SENTIMENT_2X_THRESHOLD`) · config/settings.py (universe, sectors, risk constants) · config/config.json (Freqtrade: max_open_trades, stoploss, minimal_roi, trailing, tradable_balance_ratio) · risk/position_manager.py (`CORRELATED_SECTOR_LIMIT`, `can_open_position`) · routines/market_evaluation.py (the hourly heart) · backtest/daily_walk_forward.py (edge harness).

**Known non-blocking issues:** news per-coin coverage is sparse (~2–4/24 coins per run — general crypto RSS rarely names niche alts; missing coins redistribute the 0.30 weight; per-coin tag feeds would improve it); FreqAI present but never validated AND not in the decision path (no trained model; predictions never read by entry/exit — treat as off); closed-table pagination uses numeric offset (rare skip if a trade closes mid-paging — cosmetic). (Resolved 2026-06-03: cryptocurrency.cv 402 → RSS; yfinance 8/24 failures → CoinGecko; volume_anomaly −1.0 saturation.)

---
End of handoff. Bot alive at http://178.105.141.217/. **Before real money: §0. The edge is real but thin (~Sharpe 1, beta-dependent) — size accordingly.**
