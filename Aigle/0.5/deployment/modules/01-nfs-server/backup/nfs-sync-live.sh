#!/bin/bash
# Live (zero-downtime) half of module 01's two-phase backup, VIE01-369
# Phase 1. See docs/backup-restore-v0.5.md's "01" section for the design
# rationale: this host is plain ext4 (no LVM/ZFS/Btrfs snapshot
# primitive), so a persistent, incrementally-updated rsync mirror is what
# makes nfs-freeze-archive.sh's actual frozen window cheap -- its cost
# tracks how much changed since this script's last run, not total
# dataset size.
#
# Intended to run on a schedule (e.g. hourly cron) and touches nothing
# under NFS_BASE_PATH -- only reads from it. Safe to run concurrently with
# anything except nfs-freeze-archive.sh itself: if a freeze is in
# progress, this run is skipped (not an error, exit 0) rather than
# waiting -- the next scheduled run catches up.
#
# Usage: nfs-sync-live.sh [--env-file <path>]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE_DIR="$SCRIPT_DIR/.."
# shellcheck source=./nfs-backup-common.sh
source "$SCRIPT_DIR/nfs-backup-common.sh"

ENV_FILE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --env-file) ENV_FILE="$2"; shift ;;
    esac
    shift
done

load_env "$MODULE_DIR" "$ENV_FILE"
init_backup_config

LOCK_WAIT_SECONDS="${NFS_SYNC_LOCK_WAIT_SECONDS:-10}"
exec {LOCK_FD}>"$LOCK_FILE"
if ! flock -w "$LOCK_WAIT_SECONDS" "$LOCK_FD"; then
    log "freeze-archive is running, skipping this live sync -- next scheduled run will catch up"
    exit 0
fi

log "live sync: $NFS_BASE_PATH -> $MIRROR_PATH (paths: $INCLUDE_PATHS)"
run_mirror_pass
log "live sync done: $RSYNC_FILES_TRANSFERRED file(s) transferred, $RSYNC_BYTES_TRANSFERRED byte(s), $RSYNC_FILES_DELETED deleted"
