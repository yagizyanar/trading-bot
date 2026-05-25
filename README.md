# trade-sentiment-markov

A crypto futures trading bot for Binance, combining:

- **Observable Markov regime detection** (Bull / Bear / Sideways / Crash / Euphoria)
- **Multi-source sentiment** (Fear & Greed Index, SentiCrypt, cryptocurrency.cv news + FinBERT, yfinance, Binance volume anomaly)
- **Technical indicators** (RSI, MACD, Bollinger Bands, EMA, volume spike)
- **Three-layer signal confirmation** + position sizing
- **Granular circuit breakers** + hard drawdown lockfile
- **Memory architecture** (4 markdown files read/written by every routine)
- **Walk-forward backtesting** with benchmarks and stress tests
- **FastAPI + React dashboard**

**Default mode is paper trading (`dry_run: true`).** Switching to live trading is a single line change in [config/config.json](config/config.json).

---

## Table of contents

1. [Installation](#installation)
2. [Binance API keys](#binance-api-keys)
3. [Run paper trading](#run-paper-trading)
4. [Switch to live trading](#switch-to-live-trading)
5. [Run the dashboard](#run-the-dashboard)
6. [Markov regime output](#markov-regime-output)
7. [Circuit breakers](#circuit-breakers)
8. [Scheduled routines](#scheduled-routines)
9. [Backtesting](#backtesting)
10. [Testing](#testing)
11. [Project layout](#project-layout)

---

## Installation

### 1. Python environment

Requires Python 3.10+. From the project root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> **Windows note (TA-Lib):** This project intentionally avoids `TA-Lib` (it requires a C compiler on Windows). All indicators are implemented in pure NumPy/Pandas inside [signals/technical.py](signals/technical.py).

### 2. Freqtrade (Phase 2)

Freqtrade is the execution engine. We deliberately do **not** auto-install it — clone alongside this project and install in the same venv:

```powershell
cd ..
git clone https://github.com/freqtrade/freqtrade.git
cd freqtrade
pip install -e ".[all]"
cd ..\trade
```

Verify FreqAI is enabled:

```powershell
freqtrade --version
freqtrade list-strategies --strategy-path strategies
```

### 3. PostgreSQL

Install PostgreSQL 14+ locally (or via Docker), create a database and user matching your `.env`:

```sql
CREATE USER trade WITH PASSWORD 'change_me';
CREATE DATABASE trade OWNER trade;
GRANT ALL PRIVILEGES ON DATABASE trade TO trade;
```

Then initialise the schema:

```powershell
python -c "from database.migrations import init_db; init_db()"
```

### 4. Dashboard frontend

```powershell
cd dashboard\frontend
npm install
cd ..\..
```

---

## Binance API keys

1. Visit https://www.binance.com/en/my/settings/api-management
2. Create an API key. Restrict it to **Futures Read** + **Futures Trade**. Disable withdrawals.
3. Copy `BINANCE_API_KEY` and `BINANCE_SECRET_KEY` into a `.env` file at the project root:

```
BINANCE_API_KEY=your_key_here
BINANCE_SECRET_KEY=your_secret_here

POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=trade
POSTGRES_USER=trade
POSTGRES_PASSWORD=change_me

DRY_RUN=true
TIMEZONE=UTC
LOG_LEVEL=INFO
```

The `.env` is git-ignored. A template lives in [`.env.example`](.env.example).

---

## Run paper trading

In separate terminals (all with the venv activated):

**Terminal 1 — Freqtrade**

```powershell
freqtrade trade --config config\config.json --strategy FreqAISentimentStrategy --strategy-path strategies
```

The strategy reads our routine decisions from PostgreSQL and gates entries/exits accordingly.

**Terminal 2 — routines scheduler**

```powershell
python -m routines.scheduler
```

This registers the 6 cron jobs:

| Time (UTC)    | Routine            | Purpose                                         |
| ------------- | ------------------ | ----------------------------------------------- |
| 00:00 daily   | pre_market         | News scan + Fear & Greed pull                   |
| 04:00 daily   | sentiment_update   | Refresh all 5 sources, persist unified scores   |
| 08:00 daily   | market_evaluation  | Markov regime + 3-layer gate + signal_log write |
| 12:00 daily   | midday_check       | Tighten stops, cut losers > 7%                  |
| 16:00 daily   | day_close          | Daily PnL snapshot, memory update               |
| 20:00 Sunday  | weekly_review      | Sharpe / win-rate / FreqAI retrain sentinel     |

**Terminal 3 — dashboard backend**

```powershell
uvicorn dashboard.backend.main:app --host 127.0.0.1 --port 8000 --reload
```

**Terminal 4 — dashboard frontend**

```powershell
cd dashboard\frontend
npm run dev
```

Open <http://localhost:5173>.

---

## Switch to live trading

1. **Verify your paper-trading runs have been profitable for at least 2 weeks.**
2. **Verify backtests pass minimum requirements:** `python -m backtest.walk_forward` (see [Backtesting](#backtesting) below).
3. Edit [config/config.json](config/config.json) — change a single line:
   ```json
   "dry_run": false
   ```
4. Restart Freqtrade. That's it.

The Python codebase reads `DRY_RUN` from `.env` separately for its own paper/live awareness — set it consistently:
```
DRY_RUN=false
```

---

## Run the dashboard

The dashboard is read-only — it polls the FastAPI backend and the database. It does not execute trades.

It shows:

- Bot status (RUNNING / LOCKED / dry-run flag)
- Fear & Greed widget
- Next routine countdown
- Drawdown indicator (visual bar; locks at 10%)
- Portfolio value chart (30-day history with peak line)
- Markov regime per coin (color-coded)
- Sentiment grid (per-coin unified score)
- Open and closed positions tables
- Memory files viewer (live read of the 4 markdown files)

WebSocket at `/ws` pushes a light status snapshot every 5 seconds.

---

## Markov regime output

The bot uses the [`markov-hedge-fund-method`](C:\Users\yagiz\.claude\skills\markov-hedge-fund-method\SKILL.md) skill as its base. For every coin and every routine cycle:

1. Pull daily candles from Binance Futures.
2. Label each day Bull / Sideways / Bear based on the 20-day rolling return.
3. Estimate a 3×3 transition matrix by MLE counting.
4. Read the *current* state's row: probabilities of moving to Bear / Sideways / Bull next.
5. Compute the **Markov signal** = `P(next=Bull | current) − P(next=Bear | current)`.
6. Layer Crash / Euphoria on top:
   - `Crash` if 20-day return < −6% AND realised vol > 75th percentile
   - `Euphoria` if 20-day return > +6% AND realised vol > 75th percentile

Example dashboard read:

```
SOL    Bull   72%    signal +0.41
LINK   Sideways 51%  signal +0.05
NEAR   Crash  84%    signal -0.62
```

The signal feeds the position-sizing tier (see [signals/position_sizing.py](signals/position_sizing.py)):

```
|signal| > 0.5  → 5% of capital
|signal| > 0.3  → 3% of capital
|signal| > 0.2  → 1% of capital
|signal| ≤ 0.2  → no trade
```

---

## Circuit breakers

All thresholds live in [config/settings.py](config/settings.py) and are evaluated by [risk/circuit_breakers.py](risk/circuit_breakers.py) at the start of every routine:

| Trigger                        | Action                                             |
| ------------------------------ | -------------------------------------------------- |
| Daily loss > 2%                | Cut new position sizes in half                     |
| Daily loss > 3%                | Close all positions, block new entries today       |
| Daily loss > 5%                | Bot pauses for the day                             |
| Weekly loss > 5%               | Reduce sizes 50% for the rest of the week          |
| Weekly loss > 8%               | Close all, no new trades this week                 |
| **10% drawdown from peak**     | **Write `TRADING_LOCKED.txt` — manual restart required** |

The lockfile contains the trigger reason, peak equity, current equity, and drawdown. The bot will refuse to trade until you investigate and delete it.

There is a pre-disarmed placeholder at `TRADING_LOCKED.txt.disabled` — rename it to `TRADING_LOCKED.txt` to manually halt the bot.

---

## Scheduled routines

Every routine inherits [`BaseRoutine`](routines/base.py), which enforces:

1. **Lockfile check.** If `TRADING_LOCKED.txt` exists → abort.
2. **Memory read.** Pull all 4 memory files into a `MemorySnapshot`.
3. **Circuit-breaker evaluation** against the latest equity snapshot.
4. **Routine logic** (specific to each routine).
5. **Memory write.** Overwrite `market_context.md`, append to others.
6. **Completion log.**

Errors inside `_run_inner` are caught, logged to `lessons_learned.md`, and the scheduler keeps running.

---

## Backtesting

```powershell
python -c "
from sentiment.binance_data import fetch_binance_ohlcv
from backtest.walk_forward import run_walk_forward
from backtest.benchmarks import buy_and_hold, sma_200, random_entry
from backtest.stress_tests import run_stress_tests
from backtest.metrics import meets_minimum_requirements

df = fetch_binance_ohlcv('SOLUSDT', interval='1d', limit=1500)
close = df['close'].dropna()

wf  = run_walk_forward(close, 'SOL')
bh  = buy_and_hold(close)
sma = sma_200(close)
rnd = random_entry(close)

print(f'Walk-forward Sharpe: {wf.metrics.sharpe:.2f}, max DD: {wf.metrics.max_drawdown:.2%}')
print(f'Buy & hold  Sharpe: {bh.metrics.sharpe:.2f}')
print(f'SMA-200     Sharpe: {sma.metrics.sharpe:.2f}')
print(f'Random      Sharpe: {rnd.metrics.sharpe:.2f}')

passed, fails = meets_minimum_requirements(wf.metrics)
print('PASSED' if passed else f'FAILED: {fails}')

print(run_stress_tests(close, 'SOL').all_survived)
"
```

**Minimum requirements to go live** (per [`backtesting-protocol`](C:\Users\yagiz\AppData\Roaming\Claude\local-agent-mode-sessions\skills-plugin\...) skill):

- Win rate > 50%
- Profit factor > 1.5
- Sharpe > 1.0
- Max drawdown < 20%
- Total trades > 100
- Beats all three benchmarks (buy & hold, SMA-200, random)
- Sharpe > 3.0 → flagged as suspicious (likely overfit)

---

## Testing

```powershell
pytest tests/ -v
```

Tests in `tests/` cover:

- Memory I/O contract (append-only, overwrite)
- Circuit breaker tiers (all 6 severity levels)
- Position sizing, leverage selection, correlation gate
- Technical indicators + 3-layer signal gate
- Sentiment helpers (Fear & Greed multiplier, FinBERT aggregation, volume anomaly)
- Markov regime detector against synthetic bull/bear series
- Backtest metrics, benchmarks, walk-forward, stress tests
- Routine base contract (lockfile guard, error→lesson logging)
- Config sanity (18 coins, ordering of thresholds)

External-dependent tests (Binance / Yahoo / web APIs) are *not* exercised by default — they would require network and credentials.

---

## Project layout

```
trade/
├── config/                      .env loader, all constants, Freqtrade config.json
├── memory/                      4 markdown files + memory_io.py
├── database/                    SQLAlchemy models, connection, migrations
├── sentiment/                   5 fetchers + analyzer
├── markov/                      Wrapper around the markov-hedge-fund-method skill
├── signals/                     Technical indicators + 3-layer gate + sizing
├── risk/                        Circuit breakers, lockfile, position rules
├── strategies/                  Freqtrade strategy (FreqAISentimentStrategy)
├── routines/                    BaseRoutine + 6 routines + APScheduler entry
├── backtest/                    walk_forward, benchmarks, stress_tests, metrics
├── dashboard/
│   ├── backend/                 FastAPI app + WebSocket
│   └── frontend/                React + Vite + Recharts + Tailwind (via CDN)
├── tests/                       pytest suite
├── .env                         (you create this; git-ignored)
├── .env.example                 template
├── requirements.txt
└── TRADING_LOCKED.txt.disabled  placeholder; rename to activate kill switch
```

---

## Safety notes

- The `.env` file is in `.gitignore` — never commit it.
- The Binance API key should be **trade-only**, **no withdrawals**, IP-restricted if possible.
- Always run paper trading first.
- The 10% drawdown lockfile is a hard floor — when it fires, **stop and review** before resuming.
- Any sentiment source can fail without halting the bot; the unified score redistributes weights across whichever sources are available.
