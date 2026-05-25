"""APScheduler entry point — registers all 6 routines on their cron slots.

Run with:
    python -m routines.scheduler

Use Ctrl+C to stop. The scheduler runs in foreground so you can see logs.
"""
from __future__ import annotations

import logging
import signal
import sys
from typing import Callable

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from pytz import utc

from config.settings import LOG_LEVEL
from .day_close import DayCloseRoutine
from .market_evaluation import MarketEvaluationRoutine
from .midday_check import MiddayCheckRoutine
from .pre_market import PreMarketRoutine
from .sentiment_update import SentimentUpdateRoutine
from .weekly_review import WeeklyReviewRoutine

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)
log = logging.getLogger("scheduler")


def _wrap(routine_cls) -> Callable[[], None]:
    def _job():
        try:
            res = routine_cls().run()
            log.info("Routine %s finished: success=%s extra=%s", res.name, res.success, res.extra)
        except Exception:  # noqa: BLE001
            log.exception("Routine %s crashed at scheduler level", routine_cls.__name__)
    return _job


def build_scheduler() -> BlockingScheduler:
    scheduler = BlockingScheduler(timezone=utc)
    scheduler.add_job(_wrap(PreMarketRoutine),         CronTrigger(hour=0,  minute=0, timezone=utc), id="pre_market")
    scheduler.add_job(_wrap(SentimentUpdateRoutine),   CronTrigger(hour=4,  minute=0, timezone=utc), id="sentiment_update")
    scheduler.add_job(_wrap(MarketEvaluationRoutine),  CronTrigger(hour=8,  minute=0, timezone=utc), id="market_evaluation")
    scheduler.add_job(_wrap(MiddayCheckRoutine),       CronTrigger(hour=12, minute=0, timezone=utc), id="midday_check")
    scheduler.add_job(_wrap(DayCloseRoutine),          CronTrigger(hour=16, minute=0, timezone=utc), id="day_close")
    scheduler.add_job(_wrap(WeeklyReviewRoutine),      CronTrigger(day_of_week="sun", hour=20, minute=0, timezone=utc), id="weekly_review")
    return scheduler


def main() -> int:
    scheduler = build_scheduler()
    log.info("Scheduler starting with %d jobs", len(scheduler.get_jobs()))
    for job in scheduler.get_jobs():
        log.info("  - %s next run: %s", job.id, job.next_run_time)

    def _stop(_signum, _frame):
        log.info("Stopping scheduler...")
        scheduler.shutdown(wait=False)
        sys.exit(0)

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    scheduler.start()
    return 0


if __name__ == "__main__":
    sys.exit(main())
