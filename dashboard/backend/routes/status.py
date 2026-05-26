"""Bot status + next-routine countdown.

The "next routine" times are computed by reading the actual cron file at
/etc/cron.d/trading-bot, so the dashboard always matches what cron will fire.
The read is cached for 30s to avoid disk I/O on every request.

Falls back to a hardcoded default schedule when the cron file isn't readable
(dev machines, pytest, fresh deploys before cron is provisioned).
"""
from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter

from config.settings import DRY_RUN
from risk.lockfile import is_locked, lockfile_reason

log = logging.getLogger(__name__)
router = APIRouter()

CRON_FILE = Path("/etc/cron.d/trading-bot")
CRON_CACHE_TTL = 30.0
_CRON_CACHE: dict = {"ts": 0.0, "specs": None}

# Default fallback used when CRON_FILE is missing / unreadable.
# Keep in rough sync with what's actually installed on the VPS.
_FALLBACK_SPECS: list[tuple[str, str]] = [
    ("pre_market",        "0 0 * * 1-5"),
    ("sentiment_update",  "0 */2 * * *"),
    ("market_evaluation", "0 * * * *"),
    ("midday_check",      "0 12 * * 1-5"),
    ("day_close",         "0 16 * * 1-5"),
    ("weekly_review",     "0 20 * * 0"),
]

# `position_monitor` fires every minute — it would always dominate the
# "next routine" widget and isn't user-meaningful. Hide it from the picker
# but include other intra-day routines.
HIDDEN_FROM_NEXT_WIDGET = {"position_monitor"}


# ---------------------------------------------------------------------------
# Cron parsing
# ---------------------------------------------------------------------------
def _field_match(spec: str, val: int) -> bool:
    """Does this single cron field match `val`? Supports `*`, `N`, `A-B`, `*/N`, `A,B,C`."""
    if spec == "*":
        return True
    if spec.startswith("*/"):
        try:
            step = int(spec[2:])
            return step > 0 and val % step == 0
        except ValueError:
            return False
    for part in spec.split(","):
        if "-" in part:
            try:
                a, b = part.split("-", 1)
                if int(a) <= val <= int(b):
                    return True
            except ValueError:
                continue
        else:
            try:
                if int(part) == val:
                    return True
            except ValueError:
                continue
    return False


def _matches(dt: datetime, m: str, h: str, dom: str, mon: str, dow: str) -> bool:
    """5-field cron match against a datetime. Cron dow convention: 0=Sun..6=Sat."""
    py_dow = (dt.weekday() + 1) % 7  # Python: Mon=0..Sun=6  →  cron: Sun=0..Sat=6
    return (
        _field_match(m, dt.minute)
        and _field_match(h, dt.hour)
        and _field_match(dom, dt.day)
        and _field_match(mon, dt.month)
        and _field_match(dow, py_dow)
    )


def _next_fire(cron_spec: str, after: datetime,
               max_search_minutes: int = 8 * 24 * 60) -> Optional[datetime]:
    """Smallest datetime > `after` matching the 5-field cron expression.

    Returns None if no match within `max_search_minutes` minutes (default 8 days).
    """
    parts = cron_spec.split()
    if len(parts) < 5:
        return None
    m, h, dom, mon, dow = parts[:5]
    t = after.replace(second=0, microsecond=0) + timedelta(minutes=1)
    for _ in range(max_search_minutes):
        if _matches(t, m, h, dom, mon, dow):
            return t
        t += timedelta(minutes=1)
    return None


# ---------------------------------------------------------------------------
# Cron file IO
# ---------------------------------------------------------------------------
_ROUTINE_RE = re.compile(r"routines\.(\w+)")
_ENV_LINE_RE = re.compile(r"^[A-Z_]+\s*=")


def _read_cron_specs() -> list[tuple[str, str]]:
    """Return list of (routine_name, cron_spec) from CRON_FILE. Fallback on failure."""
    if not CRON_FILE.exists():
        return list(_FALLBACK_SPECS)

    try:
        text = CRON_FILE.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        log.warning("could not read %s: %s", CRON_FILE, exc)
        return list(_FALLBACK_SPECS)

    specs: list[tuple[str, str]] = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or _ENV_LINE_RE.match(s):
            continue
        parts = s.split(None, 6)
        if len(parts) < 7:
            continue
        cron_spec = " ".join(parts[:5])
        cmd = parts[6]
        m = _ROUTINE_RE.search(cmd)
        if not m:
            continue
        specs.append((m.group(1), cron_spec))

    return specs or list(_FALLBACK_SPECS)


def _cached_specs() -> list[tuple[str, str]]:
    now = time.time()
    if (now - _CRON_CACHE["ts"]) < CRON_CACHE_TTL and _CRON_CACHE["specs"] is not None:
        return _CRON_CACHE["specs"]
    specs = _read_cron_specs()
    _CRON_CACHE["ts"] = now
    _CRON_CACHE["specs"] = specs
    return specs


def _invalidate_cache() -> None:
    """For tests / manual refresh."""
    _CRON_CACHE["ts"] = 0.0
    _CRON_CACHE["specs"] = None


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------
@router.get("/")
def status() -> dict:
    locked = is_locked()
    locked_reason_text = lockfile_reason() if locked else None
    now = datetime.now(timezone.utc)

    specs = _cached_specs()
    upcoming: list[dict] = []
    for name, cron_spec in specs:
        if name in HIDDEN_FROM_NEXT_WIDGET:
            continue
        nxt = _next_fire(cron_spec, now)
        if nxt is not None:
            upcoming.append({
                "name": name,
                "cron": cron_spec,
                "next_run": nxt.isoformat(),
            })
    upcoming.sort(key=lambda x: x["next_run"])
    next_one = upcoming[0] if upcoming else {"name": "unknown", "cron": None, "next_run": None}

    state = "LOCKED" if locked else "RUNNING"
    return {
        "state": state,
        "dry_run": DRY_RUN,
        "locked": locked,
        "locked_reason": locked_reason_text,
        "now": now.isoformat(),
        "next_routine": next_one,
        "upcoming": upcoming,
    }
