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
    "SOL", "AVAX", "LINK", "DOT", "POL", "INJ", "ARB", "OP", "APT", "SUI",
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
    "LINK": "ORACLE",
    "INJ": "DEFI", "DYDX": "DEFI", "GMX": "DEFI",
    "SAND": "GAMING", "MANA": "GAMING", "AXS": "GAMING",
    # MEME and AI sectors added 2026-05-28. Cap stays at 2 per sector via
    # the global CORRELATED_SECTOR_LIMIT.
    "WIF": "MEME", "1000BONK": "MEME", "1000PEPE": "MEME",
    "FET": "AI",   "RENDER": "AI",      "TAO": "AI",
}

MAX_LEVERAGE: Final[int] = 2
# Read directly from config.json::max_open_trades so the internal gate
# (risk/position_manager.py::can_open_position) and Freqtrade itself can
# never drift apart. Fallback 10 only kicks in if config.json is missing.
MAX_OPEN_POSITIONS: Final[int] = int(_FREQTRADE_CONFIG.get("max_open_trades", 10))
MAX_CAPITAL_DEPLOYED_PCT: Final[float] = 0.75
STOP_LOSS_PCT: Final[float] = 0.05
# Effectively disabled — exit logic is trailing stop + hard stoploss + signal
# flip only. 1.0 means "TP at 100% profit", which is never reached. Matches
# config/config.json::minimal_roi[0] = 100 (Freqtrade interprets as 10000%).
# Dashboard TP-price display and position_monitor TP_NEAR alerts read this
# constant — keeping them aligned avoids spurious "TP at $X" rendering.
TAKE_PROFIT_PCT: Final[float] = 1.0

DAILY_LOSS_HALVE_PCT: Final[float] = 0.02
DAILY_LOSS_CLOSE_PCT: Final[float] = 0.03
DAILY_LOSS_PAUSE_PCT: Final[float] = 0.05
WEEKLY_LOSS_REDUCE_PCT: Final[float] = 0.05
WEEKLY_LOSS_STOP_PCT: Final[float] = 0.08
DRAWDOWN_LOCK_PCT: Final[float] = 0.10

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
CRYPTO_NEWS_URL: Final[str] = "https://cryptocurrency.cv/api/news"
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
