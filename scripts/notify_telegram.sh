#!/usr/bin/env bash
# Best-effort Telegram alert for cron/ops scripts (backups, watchdog).
# Reads TELEGRAM_TOKEN + TELEGRAM_CHAT_ID from /opt/trading-bot/.env.
# Usage: notify_telegram.sh "message text"
# Never fails hard — alerting must not break the caller.
set -u
MSG="${1:-(no message)}"
ENV_FILE="${TRADE_ENV_FILE:-/opt/trading-bot/.env}"
if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
fi
if [ -z "${TELEGRAM_TOKEN:-}" ] || [ -z "${TELEGRAM_CHAT_ID:-}" ]; then
  echo "notify_telegram: TELEGRAM_TOKEN/CHAT_ID unset; skipping"
  exit 0
fi
if curl -s -m 10 -o /dev/null -X POST \
     "https://api.telegram.org/bot${TELEGRAM_TOKEN}/sendMessage" \
     --data-urlencode "chat_id=${TELEGRAM_CHAT_ID}" \
     --data-urlencode "text=${MSG}" \
     -d "parse_mode=HTML" -d "disable_web_page_preview=true"; then
  echo "notify_telegram: sent"
else
  echo "notify_telegram: send failed"
fi
exit 0
