"""System-health alerts for the dashboard.

Computes a list of structured alert dicts covering:
- Trading health (positions opened, sizes, leverage, win rate, drawdown)
- Sentiment pipeline health (per-source coverage, stuck values)
- Data freshness (memory files, signal_log, sentiment_scores cadence)
- System health (services, Freqtrade-Binance link, cron gap detection)

Each alert has severity green/yellow/red. Cached 5 minutes server-side —
all dashboard reads (REST + WebSocket) share the same cache so the UI is
consistent regardless of access path.

`started_at` is tracked per alert id so the UI can show how long an
issue has been live within a backend session. Resets when an alert
clears (transitions to green) or when the backend restarts.
"""
from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from config.settings import MEMORY_DIR, TARGET_COINS
from database import (
    SessionLocal, SentimentScore, RegimeState, SignalLog, PerformanceSnapshot,
)
from signals.three_layer import _choose_leverage

from .freqtrade_client import (
    fetch_balance, fetch_closed_trades, fetch_status,
    invalidate_cache as _invalidate_freqtrade_cache,
)

log = logging.getLogger(__name__)

# 5 minutes when everything is green (low overhead), 60 seconds when any
# alert is non-green (so a transient that recovers within ~1 minute clears
# from the dashboard quickly instead of sticking around for 5 min).
_CACHE_TTL_GREEN_SECONDS = 300.0
_CACHE_TTL_DEGRADED_SECONDS = 60.0
_CACHE_LOCK = threading.Lock()
_CACHE: dict = {"ts": 0.0, "alerts": []}

_FIRST_SEEN_LOCK = threading.Lock()
_FIRST_SEEN: dict[str, str] = {}

SEVERITY_ORDER = {"red": 3, "yellow": 2, "green": 1}


@dataclass
class Alert:
    id: str
    severity: str            # "red" | "yellow" | "green"
    category: str            # "trading" | "sentiment" | "data" | "system"
    title: str
    detail: str = ""
    suggested_action: str = ""
    started_at: Optional[str] = None


def _stamp_first_seen(alert: Alert) -> Alert:
    """Set started_at to when this alert id was first observed non-green."""
    if alert.severity == "green":
        with _FIRST_SEEN_LOCK:
            _FIRST_SEEN.pop(alert.id, None)
        alert.started_at = None
        return alert
    now_iso = datetime.now(timezone.utc).isoformat()
    with _FIRST_SEEN_LOCK:
        if alert.id not in _FIRST_SEEN:
            _FIRST_SEEN[alert.id] = now_iso
        alert.started_at = _FIRST_SEEN[alert.id]
    return alert


def _parse_freqtrade_ts(s: str) -> Optional[datetime]:
    if not s:
        return None
    try:
        iso = str(s).replace(" ", "T")
        if not iso.endswith("Z") and "+" not in iso[-6:]:
            iso = iso + "+00:00"
        return datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Trading-health checks
# ---------------------------------------------------------------------------
def _check_positions_opened_24h() -> Alert:
    now = datetime.now(timezone.utc)
    cutoff_24h = now - timedelta(hours=24)
    cutoff_12h = now - timedelta(hours=12)
    opens_24h = 0
    opens_12h = 0
    try:
        for t in (fetch_status() or []) + (fetch_closed_trades(limit=200) or []):
            dt = _parse_freqtrade_ts(t.get("open_date", ""))
            if dt:
                if dt >= cutoff_24h:
                    opens_24h += 1
                if dt >= cutoff_12h:
                    opens_12h += 1
    except Exception as exc:
        return Alert("positions_opened_24h", "yellow", "trading",
                     "Could not check entry cadence",
                     f"Freqtrade API error: {exc}",
                     "Confirm Freqtrade is reachable")
    if opens_12h == 0:
        return Alert("positions_opened_24h", "red", "trading",
                     "No new positions in 12+ hours",
                     f"opens_24h={opens_24h}, opens_12h={opens_12h}",
                     "Check signal_log for SKIP reasons + sector cap saturation")
    return Alert("positions_opened_24h", "green", "trading",
                 "Position cadence normal",
                 f"{opens_24h} opens in 24h, {opens_12h} in 12h")


def _check_position_sizes() -> Alert:
    try:
        status = fetch_status() or []
        bal = fetch_balance() or {}
        equity = float(bal.get("total") or bal.get("value") or 0)
    except Exception:
        status, equity = [], 0.0
    if equity <= 0:
        # Equity unavailable means Freqtrade balance is unreachable — that's
        # already reported by binance_link. Return green here to avoid
        # double-reporting the same underlying transient.
        return Alert("position_sizes", "green", "trading",
                     "Position size check deferred (equity unavailable)",
                     "binance_link will report the underlying issue")
    outliers = []
    for t in status:
        try:
            stake = float(t.get("stake_amount") or 0)
            pair = (t.get("pair") or "?").split("/")[0]
            pct = stake / equity
            if pct > 0.06:
                outliers.append(f"{pair} {pct*100:.2f}% (>6%)")
            elif pct < 0.005 and stake > 0:
                outliers.append(f"{pair} {pct*100:.2f}% (<0.5%)")
        except (TypeError, ValueError, ZeroDivisionError):
            continue
    if outliers:
        return Alert("position_sizes", "red", "trading",
                     "Position sizes out of [0.5%, 6%] range",
                     "; ".join(outliers),
                     "Inspect signal_log.position_size_pct + custom_stake_amount")
    return Alert("position_sizes", "green", "trading",
                 f"All {len(status)} position sizes within [0.5%, 6%] of equity")


def _check_leverage_24h(session: Session) -> Alert:
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=24)
    lev2_in_24h = 0
    try:
        for t in (fetch_status() or []) + (fetch_closed_trades(limit=200) or []):
            dt = _parse_freqtrade_ts(t.get("open_date", ""))
            if dt and dt >= cutoff and int(t.get("leverage") or 1) == 2:
                lev2_in_24h += 1
    except Exception:
        pass

    qualifying_coins = 0
    try:
        sl_sub = session.query(SignalLog.coin, func.max(SignalLog.ts).label("m")) \
            .group_by(SignalLog.coin).subquery()
        sl_rows = session.query(SignalLog) \
            .join(sl_sub, (SignalLog.coin == sl_sub.c.coin) & (SignalLog.ts == sl_sub.c.m)).all()
        rg_sub = session.query(RegimeState.coin, func.max(RegimeState.ts).label("m")) \
            .group_by(RegimeState.coin).subquery()
        rg_rows = session.query(RegimeState) \
            .join(rg_sub, (RegimeState.coin == rg_sub.c.coin) & (RegimeState.ts == rg_sub.c.m)).all()
        rg_by = {r.coin: r for r in rg_rows}
        for sl in sl_rows:
            if sl.coin not in TARGET_COINS:
                continue
            rg = rg_by.get(sl.coin)
            if not rg:
                continue
            direction = sl.decision if sl.decision != "SKIP" else (
                "SHORT" if sl.markov_signal < 0 else "LONG"
            )
            if _choose_leverage(direction, float(sl.sentiment_score), rg.regime) == 2:
                qualifying_coins += 1
    except Exception as exc:
        log.debug("leverage_24h: qualifying-coins query failed: %s", exc)

    if lev2_in_24h == 0 and qualifying_coins > 0:
        return Alert("leverage_24h", "yellow", "trading",
                     "No 2x trades opened in 24h despite qualifying coins",
                     f"{qualifying_coins} coins currently qualify for 2x but none opened",
                     "Check sector caps / max_open_trades — may be blocking 2x-eligible entries")
    return Alert("leverage_24h", "green", "trading",
                 "Leverage activity normal",
                 f"{lev2_in_24h} 2x trades in 24h, {qualifying_coins} coins qualifying now")


def _check_win_rate() -> Alert:
    try:
        closed = fetch_closed_trades(limit=10) or []
    except Exception:
        return Alert("win_rate", "yellow", "trading",
                     "Could not compute win rate", "Freqtrade API error",
                     "Check Freqtrade health")
    if len(closed) < 5:
        return Alert("win_rate", "green", "trading",
                     "Insufficient sample for win rate",
                     f"{len(closed)} closed trades (need >=5)")
    wins = sum(1 for t in closed if float(t.get("close_profit_abs") or 0) > 0)
    pct = wins / len(closed)
    if pct < 0.40:
        return Alert("win_rate", "red", "trading",
                     f"Win rate last {len(closed)} trades: {pct*100:.0f}%",
                     f"{wins}/{len(closed)} wins (< 40% threshold)",
                     "Review strategy + signal quality + check if regime shifted")
    return Alert("win_rate", "green", "trading",
                 f"Win rate {pct*100:.0f}%",
                 f"{wins}/{len(closed)} wins (last 10 trades)")


def _check_drawdown(session: Session) -> Alert:
    try:
        latest = session.query(PerformanceSnapshot) \
            .order_by(PerformanceSnapshot.ts.desc()).first()
    except Exception:
        return Alert("drawdown", "yellow", "trading",
                     "Could not read drawdown", "DB error", "Check Postgres")
    if not latest:
        return Alert("drawdown", "green", "trading", "No drawdown data yet")
    dd = float(latest.drawdown_pct or 0)
    if dd >= 0.08:
        return Alert("drawdown", "red", "trading",
                     f"Drawdown {dd*100:.2f}% (>=8%)",
                     f"Peak ${latest.peak_equity:.2f}, current ${latest.total_equity:.2f}",
                     "Lockfile activates at 10%; consider manual halt if conditions worsen")
    if dd >= 0.05:
        return Alert("drawdown", "yellow", "trading",
                     f"Drawdown {dd*100:.2f}% (>=5%)",
                     f"Peak ${latest.peak_equity:.2f}, current ${latest.total_equity:.2f}",
                     "Monitor closely; circuit breakers trigger at 8% / 10%")
    return Alert("drawdown", "green", "trading",
                 f"Drawdown {dd*100:.2f}% (within bounds)")


# ---------------------------------------------------------------------------
# Sentiment-health checks
# ---------------------------------------------------------------------------
# news_score deliberately excluded — cryptocurrency.cv free tier broken since
# 2026-05-25 and analyzer redistributes its weight automatically.
SENTIMENT_COLUMNS = ["fear_greed", "volume_anomaly", "yfinance_change",
                     "long_short_ratio", "funding_rate", "hyperliquid_score"]


def _check_sentiment_sources(session: Session) -> Alert:
    try:
        latest_ts = session.query(func.max(SentimentScore.ts)).scalar()
        if latest_ts is None:
            return Alert("sentiment_sources", "red", "sentiment",
                         "No sentiment data at all", "sentiment_scores empty",
                         "Check sentiment_update routine")
        rows = session.query(SentimentScore).filter(SentimentScore.ts == latest_ts).all()
    except Exception:
        return Alert("sentiment_sources", "yellow", "sentiment",
                     "Sentiment DB error", "", "Check Postgres connection")
    dead = []
    partial = []
    for col in SENTIMENT_COLUMNS:
        pop = sum(1 for r in rows if getattr(r, col) is not None)
        if pop == 0:
            dead.append(col)
        elif pop < len(rows) * 0.6:  # below 60% coverage = partial outage
            partial.append(f"{col}({pop}/{len(rows)})")
    if dead:
        return Alert("sentiment_sources", "red", "sentiment",
                     f"Source(s) returning nothing for all 18 coins: {', '.join(dead)}",
                     f"Cycle {latest_ts.isoformat()}",
                     "Inspect sentiment/<source>.py + upstream API status")
    if partial:
        return Alert("sentiment_sources", "yellow", "sentiment",
                     f"Source(s) partial coverage: {', '.join(partial)}",
                     "Analyzer will redistribute weights",
                     "Check upstream API for affected source")
    return Alert("sentiment_sources", "green", "sentiment",
                 f"All {len(SENTIMENT_COLUMNS)} tracked sources alive",
                 "news_score excluded (known broken — cryptocurrency.cv)")


def _check_sentiment_stuck(session: Session) -> Alert:
    try:
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=2)
        latest_ts = session.query(func.max(SentimentScore.ts)).scalar()
        if not latest_ts:
            return Alert("sentiment_stuck", "green", "sentiment", "No sentiment data")
        latest_rows = session.query(SentimentScore).filter(SentimentScore.ts == latest_ts).all()
        old_ts_row = session.query(SentimentScore.ts) \
            .filter(SentimentScore.ts <= cutoff) \
            .order_by(SentimentScore.ts.desc()).first()
        if not old_ts_row:
            return Alert("sentiment_stuck", "green", "sentiment",
                         "Not enough history to detect stuck values",
                         "Bot may be too new (< 2h of data)")
        old_ts = old_ts_row[0]
        old_rows = session.query(SentimentScore).filter(SentimentScore.ts == old_ts).all()
        old_by = {r.coin: r for r in old_rows}
        unchanged = 0
        compared = 0
        for r in latest_rows:
            old = old_by.get(r.coin)
            if old:
                compared += 1
                try:
                    if abs(float(r.unified) - float(old.unified)) < 1e-9:
                        unchanged += 1
                except (TypeError, ValueError):
                    continue
        if compared > 0 and unchanged >= compared * 0.9:
            return Alert("sentiment_stuck", "red", "sentiment",
                         f"Unified sentiment stuck — {unchanged}/{compared} coins identical to 2h ago",
                         f"Latest {latest_ts.isoformat()} vs reference {old_ts.isoformat()}",
                         "Check sentiment_update routine + upstream APIs for caching")
        return Alert("sentiment_stuck", "green", "sentiment",
                     "Sentiment moving",
                     f"{compared - unchanged}/{compared} coins changed since 2h ago")
    except Exception:
        return Alert("sentiment_stuck", "yellow", "sentiment",
                     "Stuck-check errored", "", "Check backend logs")


# ---------------------------------------------------------------------------
# Data-freshness checks
# ---------------------------------------------------------------------------
def _check_trade_log_fresh() -> Alert:
    path = MEMORY_DIR / "trade_log.md"
    if not path.exists():
        return Alert("trade_log_fresh", "red", "data",
                     "trade_log.md missing", "", "Restore from git or create from template")
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    now = datetime.now(timezone.utc)
    age = now - mtime
    cutoff = now - timedelta(hours=24)
    closed_in_24h = 0
    try:
        for t in fetch_closed_trades(limit=200) or []:
            cd = _parse_freqtrade_ts(t.get("close_date", ""))
            if cd and cd >= cutoff:
                closed_in_24h += 1
    except Exception:
        pass
    if age > timedelta(hours=24) and closed_in_24h > 0:
        return Alert("trade_log_fresh", "red", "data",
                     "trade_log.md stale despite closed trades",
                     f"Last write {age.total_seconds()/3600:.1f}h ago; {closed_in_24h} trades closed in 24h",
                     "Check position_monitor._mirror_closed_trades")
    return Alert("trade_log_fresh", "green", "data",
                 "trade_log.md fresh",
                 f"Last write {age.total_seconds()/3600:.1f}h ago")


def _check_market_context_fresh() -> Alert:
    path = MEMORY_DIR / "market_context.md"
    if not path.exists():
        return Alert("market_context_fresh", "red", "data",
                     "market_context.md missing", "", "Run market_evaluation manually")
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    age_min = (datetime.now(timezone.utc) - mtime).total_seconds() / 60
    if age_min > 120:
        return Alert("market_context_fresh", "yellow", "data",
                     f"market_context.md stale ({age_min:.0f}m old)",
                     "Refreshes hourly via market_evaluation routine",
                     "journalctl -u cron --since '2h ago' | grep market_evaluation")
    return Alert("market_context_fresh", "green", "data",
                 f"market_context.md fresh ({age_min:.0f}m old)")


def _check_signal_log_coverage(session: Session) -> Alert:
    try:
        latest_ts = session.query(func.max(SignalLog.ts)).scalar()
        if not latest_ts:
            return Alert("signal_log_coverage", "red", "data",
                         "signal_log empty", "", "Check market_evaluation routine")
        n = session.query(func.count(SignalLog.id)) \
            .filter(SignalLog.ts == latest_ts) \
            .filter(SignalLog.coin.in_(TARGET_COINS)).scalar() or 0
    except Exception:
        return Alert("signal_log_coverage", "yellow", "data",
                     "signal_log query failed", "", "Check Postgres")
    target = len(TARGET_COINS)
    if n < target:
        return Alert("signal_log_coverage", "red", "data",
                     f"signal_log latest cycle missing coins: {n}/{target}",
                     f"Latest cycle {latest_ts.isoformat()}",
                     "Check sentiment + regime data per missing coin")
    return Alert("signal_log_coverage", "green", "data",
                 f"signal_log full coverage ({n}/{target})")


def _check_sentiment_coverage(session: Session) -> Alert:
    try:
        latest_ts = session.query(func.max(SentimentScore.ts)).scalar()
        if not latest_ts:
            return Alert("sentiment_coverage", "yellow", "data",
                         "sentiment_scores empty")
        coins = {r[0] for r in session.query(SentimentScore.coin)
                 .filter(SentimentScore.ts == latest_ts).distinct().all()}
    except Exception:
        return Alert("sentiment_coverage", "yellow", "data",
                     "sentiment coverage check errored")
    missing = set(TARGET_COINS) - coins
    if missing:
        return Alert("sentiment_coverage", "yellow", "data",
                     f"{len(missing)} coin(s) missing from latest sentiment cycle",
                     f"missing: {sorted(missing)}",
                     "Check sentiment_update per-coin failures")
    return Alert("sentiment_coverage", "green", "data",
                 f"All {len(TARGET_COINS)} coins in latest sentiment cycle")


# ---------------------------------------------------------------------------
# System-health checks
# ---------------------------------------------------------------------------
_SERVICES = [
    "trading-bot-freqtrade.service",
    "trading-bot-backend.service",
    "nginx",
    "postgresql",
    "cron",
]


def _check_services() -> Alert:
    inactive = []
    for svc in _SERVICES:
        try:
            r = subprocess.run(
                ["systemctl", "is-active", svc],
                capture_output=True, text=True, timeout=3,
            )
            if r.stdout.strip() != "active":
                inactive.append(svc)
        except FileNotFoundError:
            return Alert("services", "yellow", "system",
                         "systemctl not available",
                         "Cannot verify service state from this process",
                         "Manual: systemctl status trading-bot-*")
        except subprocess.TimeoutExpired:
            return Alert("services", "yellow", "system",
                         f"systemctl timed out checking {svc}",
                         "", "Investigate system load")
        except Exception as exc:
            return Alert("services", "yellow", "system",
                         "Cannot query systemctl",
                         str(exc), "Verify systemctl is callable")
    if inactive:
        return Alert("services", "red", "system",
                     f"Service(s) NOT active: {', '.join(inactive)}",
                     "",
                     f"systemctl restart {inactive[0]}")
    return Alert("services", "green", "system",
                 f"All {len(_SERVICES)} services active")


def _check_binance_connection() -> Alert:
    """Retry-once design to avoid false positives from transient blips.

    The freqtrade_client caches None for 5 seconds on a single failed call,
    so a brief blip would otherwise propagate to RED for one full alerts
    cycle. Approach: try once; if it returns None, invalidate the
    freqtrade_client cache and retry after 500ms. Only RED if BOTH attempts
    fail. If the second attempt succeeds, mark green with a note (so the UI
    shows it as healthy but the operator knows there was a glitch).
    """
    import time as _time
    try:
        bal = fetch_balance()
    except Exception as exc:
        return Alert("binance_link", "red", "system",
                     "Freqtrade API unreachable",
                     str(exc), "systemctl status trading-bot-freqtrade")
    if bal is not None:
        return Alert("binance_link", "green", "system",
                     "Freqtrade <-> Binance link healthy",
                     f"total={bal.get('total','?')}")
    # First call returned None — could be a real outage or a transient blip
    # whose None is sitting in the 5-second freqtrade_client cache.
    try:
        _invalidate_freqtrade_cache()
    except Exception:
        pass
    _time.sleep(0.5)
    try:
        bal_retry = fetch_balance()
    except Exception as exc:
        return Alert("binance_link", "red", "system",
                     "Freqtrade API unreachable on retry",
                     str(exc), "systemctl status trading-bot-freqtrade")
    if bal_retry is not None:
        return Alert("binance_link", "green", "system",
                     "Freqtrade <-> Binance link recovered after transient",
                     f"first call returned None, retry succeeded; total={bal_retry.get('total','?')}")
    return Alert("binance_link", "red", "system",
                 "Freqtrade not returning balance data",
                 "fetch_balance() returned None on first call AND retry",
                 "Check Freqtrade-Binance connectivity in freqtrade.log; restart trading-bot-freqtrade if persistent")


def _check_market_evaluation_gap(session: Session) -> Alert:
    try:
        latest = session.query(func.max(SignalLog.ts)).scalar()
    except Exception:
        return Alert("market_evaluation_gap", "yellow", "system",
                     "DB error checking market_evaluation cadence")
    if not latest:
        return Alert("market_evaluation_gap", "red", "system",
                     "market_evaluation has never run",
                     "signal_log empty", "Check cron")
    age_min = (datetime.now(timezone.utc) - latest).total_seconds() / 60
    if age_min > 70:
        return Alert("market_evaluation_gap", "red", "system",
                     f"market_evaluation last ran {age_min:.0f}m ago (>70m gap)",
                     f"Latest signal_log {latest.isoformat()}",
                     "journalctl -u cron --since '2h ago' | grep market_evaluation")
    return Alert("market_evaluation_gap", "green", "system",
                 f"market_evaluation last ran {age_min:.0f}m ago")


def _check_sentiment_update_gap(session: Session) -> Alert:
    try:
        latest = session.query(func.max(SentimentScore.ts)).scalar()
    except Exception:
        return Alert("sentiment_update_gap", "yellow", "system",
                     "DB error checking sentiment_update cadence")
    if not latest:
        return Alert("sentiment_update_gap", "red", "system",
                     "sentiment_update has never run", "", "Check cron")
    age_min = (datetime.now(timezone.utc) - latest).total_seconds() / 60
    if age_min > 130:
        return Alert("sentiment_update_gap", "red", "system",
                     f"sentiment_update last ran {age_min:.0f}m ago (>130m gap)",
                     f"Latest sentiment_scores {latest.isoformat()}",
                     "journalctl -u cron --since '3h ago' | grep sentiment_update")
    return Alert("sentiment_update_gap", "green", "system",
                 f"sentiment_update last ran {age_min:.0f}m ago")


# ---------------------------------------------------------------------------
# Orchestration + caching
# ---------------------------------------------------------------------------
def _compute_alerts_fresh() -> list[dict]:
    """Run every check. Returns a list including the green ones so the UI
    can show 'X of Y systems normal' summaries."""
    alerts: list[Alert] = []
    try:
        with SessionLocal() as session:
            checks = [
                lambda: _check_positions_opened_24h(),
                lambda: _check_position_sizes(),
                lambda: _check_leverage_24h(session),
                lambda: _check_win_rate(),
                lambda: _check_drawdown(session),
                lambda: _check_sentiment_sources(session),
                lambda: _check_sentiment_stuck(session),
                lambda: _check_trade_log_fresh(),
                lambda: _check_market_context_fresh(),
                lambda: _check_signal_log_coverage(session),
                lambda: _check_sentiment_coverage(session),
                lambda: _check_services(),
                lambda: _check_binance_connection(),
                lambda: _check_market_evaluation_gap(session),
                lambda: _check_sentiment_update_gap(session),
            ]
            for check in checks:
                try:
                    a = check()
                    alerts.append(_stamp_first_seen(a))
                except Exception as exc:
                    log.exception("alerts: check raised: %s", exc)
    except Exception as exc:
        log.exception("alerts: outer compute failed: %s", exc)
    alerts.sort(key=lambda a: (-SEVERITY_ORDER.get(a.severity, 0), a.category, a.id))
    return [asdict(a) for a in alerts]


def get_alerts() -> list[dict]:
    """Cached read. TTL is 5 minutes when all green, 60 seconds when any
    alert is non-green — so transient red/yellow flakes clear faster.
    """
    now = time.time()
    with _CACHE_LOCK:
        cached_alerts = _CACHE["alerts"]
        cached_ts = _CACHE["ts"]
    if cached_alerts:
        any_non_green = any(a["severity"] != "green" for a in cached_alerts)
        ttl = _CACHE_TTL_DEGRADED_SECONDS if any_non_green else _CACHE_TTL_GREEN_SECONDS
        if (now - cached_ts) < ttl:
            return list(cached_alerts)
    fresh = _compute_alerts_fresh()
    with _CACHE_LOCK:
        _CACHE["ts"] = now
        _CACHE["alerts"] = fresh
    return list(fresh)


def invalidate_cache() -> None:
    """For tests / manual refresh."""
    with _CACHE_LOCK:
        _CACHE["ts"] = 0.0
        _CACHE["alerts"] = []
