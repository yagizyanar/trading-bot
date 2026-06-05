"""Central settings: .env loader, constants, target coin list, thresholds.

All other modules import constants from here so config changes are one-place.
Never hardcode API keys anywhere else.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
ENV_PATH: Final[Path] = PROJECT_ROOT / ".env"
load_dotenv(ENV_PATH)


def _require(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Required env var {name} missing from {ENV_PATH}")
    return val


BINANCE_API_KEY: Final[str] = _require("BINANCE_API_KEY")
BINANCE_SECRET_KEY: Final[str] = _require("BINANCE_SECRET_KEY")

POSTGRES_HOST: Final[str] = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT: Final[int] = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB: Final[str] = os.getenv("POSTGRES_DB", "trade")
POSTGRES_USER: Final[str] = os.getenv("POSTGRES_USER", "trade")
POSTGRES_PASSWORD: Final[str] = os.getenv("POSTGRES_PASSWORD", "trade")

DATABASE_URL: Final[str] = (
    f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

DRY_RUN: Final[bool] = os.getenv("DRY_RUN", "true").lower() == "true"
TIMEZONE: Final[str] = os.getenv("TIMEZONE", "UTC")
LOG_LEVEL: Final[str] = os.getenv("LOG_LEVEL", "INFO").upper()

# ---------------------------------------------------------------------------
# Single source of truth: load Freqtrade's config.json once and derive the
# Python-side constants from it. Avoids drift between config.json (authoritative
# for Freqtrade itself) and settings.py (used by our routines + risk gates).
# Falls back to sensible defaults if the file is missing / unreadable so tests
# and fresh deploys don't blow up before config.json exists.
# ---------------------------------------------------------------------------
def _load_freqtrade_config() -> dict:
    import json
    cfg_path = PROJECT_ROOT / "config" / "config.json"
    try:
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

_FREQTRADE_CONFIG: Final[dict] = _load_freqtrade_config()

# Paper wallet — must match Freqtrade's dry_run_wallet so equity reconciles
# with the trade-by-trade positions view (see PROJECT_HANDOFF.md §5 commit
# 2bb46cd: equity = DRY_RUN_WALLET + closed_pnl + sum(open_profit_abs)).
DRY_RUN_WALLET: Final[float] = float(_FREQTRADE_CONFIG.get("dry_run_wallet", 10000.0))

LOCKFILE_PATH: Final[Path] = PROJECT_ROOT / "TRADING_LOCKED.txt"
MEMORY_DIR: Final[Path] = PROJECT_ROOT / "memory"
LOGS_DIR: Final[Path] = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

TARGET_COINS: Final[tuple[str, ...]] = (
    # MATIC → POL (Polygon rebrand, 2024) and FTM → S (Sonic rebrand, 2024/25).
    # Binance Futures discontinued the old tickers; the new symbols below are
    # the active perpetual pairs.
    # LINK dropped 2026-06-04: its $20 Binance min-notional vs a $15 max stake
    # at $300 forced every LINK trade to ~7% (2x intended) or rejection. Re-add
    # when the account is >= ~$700 (see project_tier1_findings_2026-06-04).
    "SOL", "AVAX", "DOT", "POL", "INJ", "ARB", "OP", "APT", "SUI",
    "NEAR", "S", "ATOM", "SAND", "MANA", "AXS", "DYDX", "GMX",
    # MEME sector (added 2026-05-28). BONK and PEPE perpetuals are
    # 1000x-scaled on Binance — the contract symbols are 1000BONKUSDT and
    # 1000PEPEUSDT. We use those names as the TARGET_COINS keys directly so
    # the pair-construction expressions `f"{coin}USDT"` produce the correct
    # Binance symbol without special-casing.
    "WIF", "1000BONK", "1000PEPE",
    # AI sector (added 2026-05-28).
    "FET", "RENDER", "TAO",
)
QUOTE: Final[str] = "USDT"
PAIRS: Final[tuple[str, ...]] = tuple(f"{c}/{QUOTE}" for c in TARGET_COINS)

SECTOR_MAP: Final[dict[str, str]] = {
    "POL": "L2", "ARB": "L2", "OP": "L2",
    # L1 split (2026-05-27): 8 L1 coins competing for 2 slots was the bot's
    # most-binding constraint — 121/256 cap-blocks in a 15h window, ~$100
    # of missed counterfactual PnL. Split by market-cap tier so the
    # correlation-cap principle (≤2 active per cluster) still applies but
    # the effective L1 ceiling doubles to 4. Majors move together with
    # BTC dominance + ETH beta; alts trade more on idiosyncratic narratives.
    "SOL": "L1-MAJOR", "AVAX": "L1-MAJOR", "NEAR": "L1-MAJOR", "DOT": "L1-MAJOR",
    "APT": "L1-ALT",   "SUI": "L1-ALT",   "ATOM": "L1-ALT",   "S": "L1-ALT",
    "INJ": "DEFI", "DYDX": "DEFI", "GMX": "DEFI",
    "SAND": "GAMING", "MANA": "GAMING", "AXS": "GAMING",
    # MEME and AI sectors added 2026-05-28. Cap stays at 2 per sector via
    # the global CORRELATED_SECTOR_LIMIT.
    "WIF": "MEME", "1000BONK": "MEME", "1000PEPE": "MEME",
    "FET": "AI",   "RENDER": "AI",      "TAO": "AI",
}

# Authoritative leverage ceiling — capped at the strategy's leverage() callback
# (strategies/freqai_sentiment_strategy.py). Set to 1 on 2026-06-03 to force 1x
# for the real-money go-live (handoff §0.6): the edge is ~Sharpe 1, amplifying it
# just amplifies variance. Raise back to 2 once live execution is validated.
MAX_LEVERAGE: Final[int] = 3   # dynamic leverage 1/2/3x by |signal| (2026-06-05); was 1
# Read directly from config.json::max_open_trades so the internal gate
# (risk/position_manager.py::can_open_position) and Freqtrade itself can
# never drift apart. Fallback 10 only kicks in if config.json is missing.
MAX_OPEN_POSITIONS: Final[int] = int(_FREQTRADE_CONFIG.get("max_open_trades", 10))
# Documented risk rule: "max 50% of total capital in positions at any time."
# MUST equal config.json::tradable_balance_ratio — the internal gate
# (can_open_position) and Freqtrade's allocation are measured against the same
# denominator (total equity), so a higher gate than tradable_balance_ratio lets
# the bot keep opening past Freqtrade's fundable balance, which clamps new
# entries to tiny sizes (the historical 0.14%-stake bug). Realigned 0.75→0.50
# on 2026-06-03 to match the documented rule + the lowered tradable_balance_ratio.
MAX_CAPITAL_DEPLOYED_PCT: Final[float] = 0.50
# Item 6 (2026-06-03): aggregate net-BTC-beta cap. The book's net directional
# exposure, in full-position-equivalents (position_size / SIGNAL_FULL_PCT)
# weighted by each coin's beta-to-BTC, may not exceed ±this. 3.0 ≈ "max 3 full
# positions of net one-way beta" — stops 10 correlated same-direction trades
# from becoming one giant undiversified bet (the -44% drawdown failure mode).
# Item 6 net-beta cap DISABLED 2026-06-05 (user request, baseline config): 1e9 =
# effectively unlimited one-way exposure. Set back to 3.0 to re-enable the cap.
NET_BETA_BUDGET: Final[float] = 1e9
STOP_LOSS_PCT: Final[float] = 0.05
# +15% take-profit. MUST match config/config.json::minimal_roi[0] (= 0.15):
# Freqtrade enforces the actual ROI exit, while this constant only drives the
# dashboard TP-price column and position_monitor TP_NEAR alerts. Keeping them
# aligned makes the UI/alerts reflect the real exit. Re-enabled 1.0→0.15 on
# 2026-06-03 alongside minimal_roi.
TAKE_PROFIT_PCT: Final[float] = 0.15

DAILY_LOSS_HALVE_PCT: Final[float] = 0.02
DAILY_LOSS_CLOSE_PCT: Final[float] = 0.03
DAILY_LOSS_PAUSE_PCT: Final[float] = 0.05
WEEKLY_LOSS_REDUCE_PCT: Final[float] = 0.05
WEEKLY_LOSS_STOP_PCT: Final[float] = 0.08
# Drawdown-from-peak lock: writes TRADING_LOCKED.txt (halts, manual restart).
# Raised 0.10→0.20 on 2026-06-03: with items 5-7 holding the book's natural
# drawdown to ~13% (Config 3), a 10% lock tripped during NORMAL operation in
# volatile years and froze realized return well below potential (2024: +39% vs
# +90%). 20% sits above the strategy's normal DD so the lock is a true
# catastrophe backstop, not an everyday tripwire. (Multi-year OOS analysis.)
DRAWDOWN_LOCK_PCT: Final[float] = 0.30   # loosened 0.20→0.30 2026-06-05 (user request)

SIGNAL_FULL_SIZE: Final[float] = 0.50
SIGNAL_MEDIUM_SIZE: Final[float] = 0.30
SIGNAL_SMALL_SIZE: Final[float] = 0.20
SIGNAL_FULL_PCT: Final[float] = 0.05
SIGNAL_MEDIUM_PCT: Final[float] = 0.03
SIGNAL_SMALL_PCT: Final[float] = 0.01

SENTIMENT_BULL_THRESHOLD: Final[float] = 0.2
SENTIMENT_BEAR_THRESHOLD: Final[float] = -0.2
LEVERAGE_SENTIMENT_THRESHOLD: Final[float] = 0.3

MARKOV_WINDOW: Final[int] = 20
MARKOV_THRESHOLD: Final[float] = 0.02
MARKOV_MIN_TRAIN: Final[int] = 252

FEAR_GREED_URL: Final[str] = "https://api.alternative.me/fng/"
# News: cryptocurrency.cv went paywalled (402) 2026-06. Replaced with free,
# no-key RSS feeds (2026-06-03). CoinTelegraph is the verified-reliable backbone;
# the others are best-effort and skipped gracefully if a feed is down/changes.
NEWS_RSS_FEEDS: Final[tuple[str, ...]] = (
    "https://cointelegraph.com/rss",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://decrypt.co/feed",
)
# CoinGecko replaces flaky yfinance for the 7-day price-change signal (2026-06-03):
# yfinance failed for 8/24 coins (Yahoo ticker disambiguation, 1000-prefix futures
# tickers, not-listed). CoinGecko resolves all 24 via stable ids in one batch call.
COINGECKO_MARKETS_URL: Final[str] = "https://api.coingecko.com/api/v3/coins/markets"
HYPERLIQUID_URL: Final[str] = "https://api.hyperliquid.xyz/info"

API_TIMEOUT_SECONDS: Final[float] = 15.0
API_RETRY_ATTEMPTS: Final[int] = 2

VOLUME_SPIKE_MULTIPLIER: Final[float] = 2.0
TECHNICAL_RSI_OVERSOLD: Final[float] = 35.0
TECHNICAL_RSI_OVERBOUGHT: Final[float] = 65.0

BACKTEST_IN_SAMPLE_DAYS: Final[int] = 252
BACKTEST_OUT_SAMPLE_DAYS: Final[int] = 180
BACKTEST_MIN_WIN_RATE: Final[float] = 0.50
BACKTEST_MIN_PROFIT_FACTOR: Final[float] = 1.5
BACKTEST_MIN_SHARPE: Final[float] = 1.0
BACKTEST_MAX_DRAWDOWN: Final[float] = 0.20
