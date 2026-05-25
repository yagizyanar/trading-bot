"""Bot status + next-routine countdown."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter

from config.settings import DRY_RUN
from risk.lockfile import is_locked, lockfile_reason

router = APIRouter()


# Cron schedule (UTC) for the 6 routines
_SCHEDULE = [
    ("pre_market", 0, 0),
    ("sentiment_update", 4, 0),
    ("market_evaluation", 8, 0),
    ("midday_check", 12, 0),
    ("day_close", 16, 0),
    # weekly_review is Sunday 20:00 — handled below
]


def _next_run(now: datetime, hour: int, minute: int) -> datetime:
    candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate <= now:
        candidate += timedelta(days=1)
    return candidate


def _next_weekly(now: datetime) -> datetime:
    candidate = now.replace(hour=20, minute=0, second=0, microsecond=0)
    days_until_sunday = (6 - now.weekday()) % 7  # Monday=0..Sunday=6
    candidate += timedelta(days=days_until_sunday)
    if candidate <= now:
        candidate += timedelta(days=7)
    return candidate


@router.get("/")
def status() -> dict:
    locked = is_locked()
    locked_reason = lockfile_reason() if locked else None
    now = datetime.now(timezone.utc)

    upcoming = [
        {"name": name, "next_run": _next_run(now, h, m).isoformat()}
        for name, h, m in _SCHEDULE
    ]
    upcoming.append({"name": "weekly_review", "next_run": _next_weekly(now).isoformat()})
    upcoming.sort(key=lambda x: x["next_run"])
    next_one = upcoming[0]

    state = "LOCKED" if locked else "RUNNING"

    return {
        "state": state,
        "dry_run": DRY_RUN,
        "locked": locked,
        "locked_reason": locked_reason,
        "now": now.isoformat(),
        "next_routine": next_one,
        "upcoming": upcoming,
    }
