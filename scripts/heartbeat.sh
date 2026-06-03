#!/usr/bin/env bash
# External watchdog (replaces systemd WatchdogSec — freqtrade 2026.4 does NOT
# emit sd_notify, so Type=notify/WatchdogSec would crash-loop it). Runs every
# minute from cron. Strictly stronger than the in-process watchdog: it also
# catches whole-VPS death / network partitions (via healthchecks.io absence).
#
#   - probes the Freqtrade REST API (/api/v1/ping)
#   - healthy  → ping HEALTHCHECK_URL (success), reset the failure counter
#   - down     → ping HEALTHCHECK_URL/fail; after 2 consecutive failures
#                (~2 min) restart freqtrade + Telegram alert (hang recovery)
#
# Env (from /opt/trading-bot/.env): HEALTHCHECK_URL (optional — healthchecks.io
#   ping URL), TELEGRAM_TOKEN/CHAT_ID (optional — alerts). Degrades gracefully
#   if unset: auto-restart-on-hang still works without them.
set -uo pipefail

ROOT=/opt/trading-bot
LOG=/var/log/trading-bot/heartbeat.log
FAILFILE="$ROOT/.heartbeat_fails"
NOTIFY="$ROOT/scripts/notify_telegram.sh"
API="http://127.0.0.1:8080/api/v1/ping"
RESTART_AFTER=2

if [ -f "$ROOT/.env" ]; then
  set -a; # shellcheck disable=SC1091
  . "$ROOT/.env"; set +a
fi

if curl -fsS -m 8 "$API" >/dev/null 2>&1; then
  echo 0 >"$FAILFILE" 2>/dev/null
  [ -n "${HEALTHCHECK_URL:-}" ] && curl -fsS -m 8 "$HEALTHCHECK_URL" >/dev/null 2>&1
  exit 0
fi

# --- unhealthy ---
N=$(cat "$FAILFILE" 2>/dev/null || echo 0)
N=$((N + 1))
echo "$N" >"$FAILFILE" 2>/dev/null
echo "$(date -u +%FT%TZ) heartbeat: freqtrade API DOWN (consecutive=$N)" >>"$LOG"
[ -n "${HEALTHCHECK_URL:-}" ] && curl -fsS -m 8 "$HEALTHCHECK_URL/fail" >/dev/null 2>&1

if [ "$N" -ge "$RESTART_AFTER" ]; then
  echo "$(date -u +%FT%TZ) heartbeat: restarting freqtrade after $N consecutive failures" >>"$LOG"
  [ -x "$NOTIFY" ] && bash "$NOTIFY" "🔴 Watchdog: freqtrade API down ${N}× on $(hostname) — restarting" >/dev/null 2>&1
  systemctl restart trading-bot-freqtrade
  echo 0 >"$FAILFILE" 2>/dev/null
fi
exit 0
