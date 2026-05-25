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

LOCKFILE_PATH: Final[Path] = PROJECT_ROOT / "TRADING_LOCKED.txt"
MEMORY_DIR: Final[Path] = PROJECT_ROOT / "memory"
LOGS_DIR: Final[Path] = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

TARGET_COINS: Final[tuple[str, ...]] = (
    "SOL", "AVAX", "LINK", "DOT", "MATIC", "INJ", "ARB", "OP", "APT", "SUI",
    "NEAR", "FTM", "ATOM", "SAND", "MANA", "AXS", "DYDX", "GMX",
)
QUOTE: Final[str] = "USDT"
PAIRS: Final[tuple[str, ...]] = tuple(f"{c}/{QUOTE}" for c in TARGET_COINS)

SECTOR_MAP: Final[dict[str, str]] = {
    "MATIC": "L2", "ARB": "L2", "OP": "L2",
    "SOL": "L1", "AVAX": "L1", "DOT": "L1", "APT": "L1", "SUI": "L1",
    "NEAR": "L1", "FTM": "L1", "ATOM": "L1",
    "LINK": "ORACLE",
    "INJ": "DEFI", "DYDX": "DEFI", "GMX": "DEFI",
    "SAND": "GAMING", "MANA": "GAMING", "AXS": "GAMING",
}

MAX_LEVERAGE: Final[int] = 2
MAX_OPEN_POSITIONS: Final[int] = 10
MAX_CAPITAL_DEPLOYED_PCT: Final[float] = 0.50
STOP_LOSS_PCT: Final[float] = 0.05
TAKE_PROFIT_PCT: Final[float] = 0.15

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
