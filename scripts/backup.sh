#!/usr/bin/env bash
# Encrypted off-box backup of the trading bot's databases.
#   1. pg_dump of the trading_bot Postgres DB
#   2. copy of Freqtrade's tradesv3.sqlite
#   3. tar + gpg symmetric (AES256) encrypt with GPG_BACKUP_PASSPHRASE
#   4. upload to Backblaze B2 (env creds)
#   5. local retention (keep last 8 ~= 2 days at 6h cadence)
#   6. Telegram alert on FAILURE
#
# Env (from /opt/trading-bot/.env): GPG_BACKUP_PASSPHRASE (required to run),
#   B2_KEY_ID, B2_APP_KEY, B2_BUCKET (required to upload),
#   TELEGRAM_TOKEN, TELEGRAM_CHAT_ID (optional, for failure alerts).
# Until GPG_BACKUP_PASSPHRASE is set, the script no-ops cleanly (exit 0).
set -uo pipefail

ROOT=/opt/trading-bot
LOG=/var/log/trading-bot/backup.log
BACKUP_DIR="$ROOT/backups"
B2_BIN="$ROOT/.b2venv/bin/b2"
NOTIFY="$ROOT/scripts/notify_telegram.sh"
SQLITE_SRC="$ROOT/user_data/tradesv3.sqlite"
KEEP=8

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { echo "$(ts) backup: $*" >>"$LOG"; }
fail() {
  log "FAIL: $*"
  [ -x "$NOTIFY" ] && bash "$NOTIFY" "🔴 Backup FAILED on $(hostname): $*" >/dev/null 2>&1
  exit 1
}

mkdir -p "$BACKUP_DIR"

# Load env (B2 / GPG / Telegram creds)
if [ -f "$ROOT/.env" ]; then
  set -a; # shellcheck disable=SC1091
  . "$ROOT/.env"; set +a
fi

if [ -z "${GPG_BACKUP_PASSPHRASE:-}" ]; then
  log "SKIP: GPG_BACKUP_PASSPHRASE not set — backups not configured yet (no-op)"
  exit 0
fi

STAMP=$(date -u +%Y%m%d-%H%M%S)
WORK="$BACKUP_DIR/.work-$STAMP"
mkdir -p "$WORK"
trap 'rm -rf "$WORK"' EXIT

# 1. Postgres dump (as the postgres superuser — no password needed)
if ! sudo -u postgres pg_dump trading_bot >"$WORK/trading_bot.sql" 2>>"$LOG"; then
  fail "pg_dump trading_bot"
fi

# 2. SQLite snapshot (consistent .backup if sqlite3 present, else cp)
if [ -f "$SQLITE_SRC" ]; then
  if command -v sqlite3 >/dev/null 2>&1; then
    sqlite3 "$SQLITE_SRC" ".backup '$WORK/tradesv3.sqlite'" 2>>"$LOG" || cp "$SQLITE_SRC" "$WORK/tradesv3.sqlite"
  else
    cp "$SQLITE_SRC" "$WORK/tradesv3.sqlite"
  fi
else
  log "WARN: $SQLITE_SRC not found (skipping sqlite)"
fi

# 3. tar + gpg symmetric encrypt
ARCHIVE="$BACKUP_DIR/trade-backup-$STAMP.tar.gz"
ENC="$ARCHIVE.gpg"
tar -czf "$ARCHIVE" -C "$WORK" . 2>>"$LOG" || fail "tar"
if ! gpg --batch --yes --passphrase "$GPG_BACKUP_PASSPHRASE" \
        --symmetric --cipher-algo AES256 -o "$ENC" "$ARCHIVE" 2>>"$LOG"; then
  rm -f "$ARCHIVE"; fail "gpg encrypt"
fi
rm -f "$ARCHIVE"  # never keep plaintext archive
SIZE=$(du -h "$ENC" | cut -f1)

# 4. Upload to Backblaze B2 (if configured)
if [ -n "${B2_KEY_ID:-}" ] && [ -n "${B2_APP_KEY:-}" ] && [ -n "${B2_BUCKET:-}" ]; then
  export B2_APPLICATION_KEY_ID="$B2_KEY_ID"
  export B2_APPLICATION_KEY="$B2_APP_KEY"
  if "$B2_BIN" file upload "$B2_BUCKET" "$ENC" "trade-backups/$(basename "$ENC")" >>"$LOG" 2>&1; then
    log "OK: uploaded $(basename "$ENC") ($SIZE) to B2 bucket $B2_BUCKET"
  else
    fail "b2 upload"
  fi
else
  log "SKIP upload: B2 creds not set — local encrypted backup kept ($ENC, $SIZE)"
fi

# 5. Local retention
# shellcheck disable=SC2012
ls -1t "$BACKUP_DIR"/trade-backup-*.tar.gz.gpg 2>/dev/null | tail -n +$((KEEP + 1)) | xargs -r rm -f
log "OK: backup complete ($(basename "$ENC"), $SIZE)"
