#!/usr/bin/env bash
set -euo pipefail

# Grandfather-father-son PostgreSQL/TimescaleDB backups for EnsembleLegends.
# Retention policy:
#   - daily:   keep the last 7 days
#   - weekly:  keep the last 8 weeks (covers every week of the current month)
#   - monthly: keep the last 24 months

CONTAINER_NAME="${CONTAINER_NAME:-ensemblelegends-timescaledb}"
POSTGRES_USER="${POSTGRES_USER:-betting}"
POSTGRES_DB="${POSTGRES_DB:-betting}"
BACKUP_ROOT="${BACKUP_ROOT:-/data/inzynierka/db_backups}"
DAILY_RETENTION_DAYS="${DAILY_RETENTION_DAYS:-7}"
WEEKLY_RETENTION_DAYS="${WEEKLY_RETENTION_DAYS:-56}"
MONTHLY_RETENTION_DAYS="${MONTHLY_RETENTION_DAYS:-730}"

DATE_UTC="$(date -u +%F)"
STAMP_UTC="$(date -u +%Y%m%dT%H%M%SZ)"
ISO_WEEK="$(date -u +%G-W%V)"
MONTH="$(date -u +%Y-%m)"
DAY_OF_WEEK="$(date -u +%u)"  # 1=Monday
DAY_OF_MONTH="$(date -u +%d)"

DAILY_DIR="$BACKUP_ROOT/daily"
WEEKLY_DIR="$BACKUP_ROOT/weekly"
MONTHLY_DIR="$BACKUP_ROOT/monthly"
TMP_DIR="$BACKUP_ROOT/.tmp"
LOG_FILE="$BACKUP_ROOT/backup.log"
LOCK_FILE="$BACKUP_ROOT/.backup.lock"

mkdir -p "$DAILY_DIR" "$WEEKLY_DIR" "$MONTHLY_DIR" "$TMP_DIR"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG_FILE"
}

cleanup_old_backups() {
  find "$DAILY_DIR" -type f \( -name '*.dump' -o -name '*.sha256' \) -mtime +"$DAILY_RETENTION_DAYS" -delete
  find "$WEEKLY_DIR" -type f \( -name '*.dump' -o -name '*.sha256' \) -mtime +"$WEEKLY_RETENTION_DAYS" -delete
  find "$MONTHLY_DIR" -type f \( -name '*.dump' -o -name '*.sha256' \) -mtime +"$MONTHLY_RETENTION_DAYS" -delete
}

(
  flock -n 9 || { log "Another backup run is already active; exiting."; exit 0; }

  if ! docker inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
    log "ERROR: container $CONTAINER_NAME does not exist."
    exit 1
  fi

  if ! docker exec "$CONTAINER_NAME" pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB" >/dev/null 2>&1; then
    log "ERROR: database $POSTGRES_DB in $CONTAINER_NAME is not ready."
    exit 1
  fi

  filename="${POSTGRES_DB}_${STAMP_UTC}.dump"
  container_tmp="/tmp/$filename"
  tmp_file="$TMP_DIR/$filename"
  daily_file="$DAILY_DIR/$filename"

  log "Starting backup: $filename"
  docker exec "$CONTAINER_NAME" pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc -Z 6 -f "$container_tmp"
  docker cp "$CONTAINER_NAME:$container_tmp" "$tmp_file"
  docker exec "$CONTAINER_NAME" rm -f "$container_tmp" >/dev/null

  mv "$tmp_file" "$daily_file"
  sha256sum "$daily_file" > "$daily_file.sha256"
  log "Created daily backup: $daily_file ($(du -h "$daily_file" | awk '{print $1}'))"

  # Weekly snapshot every Monday. If this is the first run in a new week but not
  # Monday, create a weekly snapshot anyway so the current week is protected.
  if [[ "$DAY_OF_WEEK" == "1" ]] || ! compgen -G "$WEEKLY_DIR/${POSTGRES_DB}_${ISO_WEEK}_*.dump" >/dev/null; then
    weekly_file="$WEEKLY_DIR/${POSTGRES_DB}_${ISO_WEEK}_${STAMP_UTC}.dump"
    ln "$daily_file" "$weekly_file" 2>/dev/null || cp -p "$daily_file" "$weekly_file"
    sha256sum "$weekly_file" > "$weekly_file.sha256"
    log "Created weekly backup: $weekly_file"
  fi

  # Monthly snapshot on the first day of the month. Also create one if this
  # month has no snapshot yet (useful when enabling backups mid-month).
  if [[ "$DAY_OF_MONTH" == "01" ]] || ! compgen -G "$MONTHLY_DIR/${POSTGRES_DB}_${MONTH}_*.dump" >/dev/null; then
    monthly_file="$MONTHLY_DIR/${POSTGRES_DB}_${MONTH}_${STAMP_UTC}.dump"
    ln "$daily_file" "$monthly_file" 2>/dev/null || cp -p "$daily_file" "$monthly_file"
    sha256sum "$monthly_file" > "$monthly_file.sha256"
    log "Created monthly backup: $monthly_file"
  fi

  cleanup_old_backups
  log "Backup finished. Retention: daily=${DAILY_RETENTION_DAYS}d weekly=${WEEKLY_RETENTION_DAYS}d monthly=${MONTHLY_RETENTION_DAYS}d."
) 9>"$LOCK_FILE"
