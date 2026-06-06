#!/usr/bin/env bash
# GO-LIVE — archive the dry-run book, then start Freqtrade FRESH on REAL money.
#
#   Run ONLY when you mean it:
#       bash scripts/go_live.sh I-UNDERSTAND-REAL-MONEY
#
# Two fail-safe guards run BEFORE anything is touched (no mutation until both pass):
#   1. the exact confirm token must be $1
#   2. the USDⓈ-M futures wallet must hold > 0 USDT (refuses on an empty account)
#
# Then, in order:
#   stop bot -> final encrypted archive (sqlite + pg) to backups/ + B2
#            -> move tradesv3.sqlite aside (Freqtrade recreates an empty book)
#            -> set config.json dry_run=false  -> start  -> verify
#
# Prepared 2026-06-06. NOT wired to any cron/timer — manual, deliberate run only.
# After it succeeds, commit config.json (VPS + local + GitHub) for traceability.
set -uo pipefail
ROOT=/opt/trading-bot
SVC=trading-bot-freqtrade.service
STAMP=$(date -u +%Y-%m-%d)
cd "$ROOT"

# --- Guard 1: explicit confirmation token ---
if [ "${1:-}" != "I-UNDERSTAND-REAL-MONEY" ]; then
  echo "REFUSED: this goes LIVE with real money."
  echo "  Re-run:  bash scripts/go_live.sh I-UNDERSTAND-REAL-MONEY"
  exit 2
fi

# --- Guard 2: real funds present (fail-safe: any error => 0 => refuse) ---
BAL=$(.venv/bin/python -c "from backtest.oos_multiyear import _client; print(float(_client().futures_account()['availableBalance']))" 2>/dev/null || echo 0)
echo "USDⓈ-M futures availableBalance = ${BAL} USDT"
if ! awk "BEGIN{exit !(${BAL}>0)}"; then
  echo "REFUSED: futures wallet is empty (\$${BAL}). Transfer USDT to USDⓈ-M Futures first."
  echo "  No changes were made. Bot still running dry-run."
  exit 3
fi

echo ">>> Both guards passed — going live in 5s. Ctrl-C to abort."
sleep 5

# --- 1. stop the bot ---
echo ">>> stopping $SVC"
systemctl stop "$SVC"; sleep 3

# --- 2. final encrypted archive of the dry-run book (sqlite + pg) -> backups/ + B2 ---
echo ">>> final archive of dry-run book"
set -a; . ./.env; set +a
WORK="backups/.golive-$STAMP"; mkdir -p "$WORK"
[ -f user_data/tradesv3.sqlite ] && sqlite3 user_data/tradesv3.sqlite ".backup '$WORK/tradesv3.dryrun-final-$STAMP.sqlite'"
sudo -u postgres pg_dump trading_bot > "$WORK/trading_bot-final-$STAMP.sql" 2>/dev/null || echo "WARN: pg_dump skipped"
tar -czf "backups/dryrun-final-$STAMP.tar.gz" -C "$WORK" .
if [ -n "${GPG_BACKUP_PASSPHRASE:-}" ]; then
  gpg --batch --yes --passphrase "$GPG_BACKUP_PASSPHRASE" --symmetric --cipher-algo AES256 \
      -o "backups/dryrun-final-$STAMP.tar.gz.gpg" "backups/dryrun-final-$STAMP.tar.gz" && rm -f "backups/dryrun-final-$STAMP.tar.gz"
  if [ -n "${B2_KEY_ID:-}" ] && [ -n "${B2_APP_KEY:-}" ] && [ -n "${B2_BUCKET:-}" ]; then
    export B2_APPLICATION_KEY_ID="$B2_KEY_ID" B2_APPLICATION_KEY="$B2_APP_KEY"
    .b2venv/bin/b2 file upload "$B2_BUCKET" "backups/dryrun-final-$STAMP.tar.gz.gpg" "dryrun-archive/dryrun-final-$STAMP.tar.gz.gpg" >/dev/null 2>&1 \
      && echo ">>> final archive uploaded to B2" || echo "WARN: B2 upload failed (local copy kept)"
  fi
else
  echo "WARN: no GPG passphrase — final archive left UNencrypted as backups/dryrun-final-$STAMP.tar.gz"
fi
rm -rf "$WORK"

# --- 3. clean book: move the dry-run trades DB aside (Freqtrade recreates an empty one) ---
echo ">>> starting fresh book (moving dry-run sqlite aside)"
[ -f user_data/tradesv3.sqlite ] && mv user_data/tradesv3.sqlite "user_data/tradesv3.predeploy-$STAMP.sqlite"

# --- 4. flip dry_run -> false (string replace preserves formatting; asserts current = true) ---
.venv/bin/python -c "p='config/config.json'; s=open(p).read(); assert '\"dry_run\": true' in s, 'dry_run is already not true — aborting flip'; open(p,'w').write(s.replace('\"dry_run\": true', '\"dry_run\": false', 1)); print('>>> config.json dry_run -> false')"
# keep the bot's internal flag (config.settings.DRY_RUN, read from .env) in sync,
# else the dashboard/routines stay stuck on 'paper'/dry_run_wallet (2026-06-06 fix).
if grep -q "^DRY_RUN=" .env 2>/dev/null; then sed -i "s/^DRY_RUN=.*/DRY_RUN=false/" .env; else printf "\nDRY_RUN=false\n" >> .env; fi
echo ">>> .env DRY_RUN -> false (dashboard/routines mode flag)"

# --- 5. start + verify ---
echo ">>> starting $SVC (LIVE)"
systemctl start "$SVC"; sleep 20
echo "service:        $(systemctl is-active "$SVC")"
echo "config dry_run: $(.venv/bin/python -c "import json;print(json.load(open('config/config.json'))['dry_run'])")"
echo "--- startup log (errors / mode) ---"
journalctl -u "$SVC" --since "30 seconds ago" --no-pager 2>/dev/null | grep -iE "dry.?run|live|trading mode|error|exception|balance" | grep -vi verbosity | tail -12
echo ">>> LIVE. Watch the first entries closely; remember leverage can be up to 3x and the net-beta cap is OFF on this config."
