"""Best-effort Telegram alerts for ops / risk events.

Separate from Freqtrade's built-in trade telegram (entry/exit/stop): this is
for things Freqtrade doesn't know about — circuit-breaker fires, backup
failures, watchdog events. Reads TELEGRAM_TOKEN + TELEGRAM_CHAT_ID from the
environment (.env on the VPS). NEVER raises — alerting must not break trading.
"""
from __future__ import annotations

import logging
import os

import requests

log = logging.getLogger(__name__)


def send_telegram(text: str) -> bool:
    """Send a Telegram message. Returns True on success, False if unconfigured
    or on any error (logged, never raised)."""
    token = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        log.debug("Telegram not configured (TELEGRAM_TOKEN/CHAT_ID unset); skipping alert")
        return False
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=10,
        )
        resp.raise_for_status()
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("Telegram alert failed: %s", exc)
        return False
