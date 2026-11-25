#!/bin/bash
# Frozen half of module 01's two-phase backup, VIE01-369 Phase 1. The only
# step of the two that actually stops writes. See docs/backup-restore-v0.5.md's
# "01" section for the full design rationale (why not a single tar, why
# not an offen sidecar, why quiescence checks instead of a hard freeze-only
# design).
#
# Deliberately does NOT touch any container's lifecycle (no docker
# stop/start) -- that was this design's first version, dropped because a
# container that doesn't come back up turns a backup into an outage, with
# no good recovery. Instead, this calls module 13's own
# /admin/maintenance/enable|disable (X-Internal-Token gated) to make write
# routes answer 503 while reads keep working -- covers 04's and 26's
# synchronous HTTP write paths (both confirmed to go through the gateway),
# which is everything actually in scope after the exclusions above. Does
# NOT cover SeaweedFS's own background vacuum/balance/erasure-coding
# worker, which writes independently of any gateway request -- open item,
# see docs.
#
# Order of operations:
#   1. Take the nfs-backup lock (holds it through step 6 -- the next
#      scheduled nfs-sync-live.sh run simply waits/skips, see v1 design
#      note in docs).
#   2. Quiescence gate: if the mirror is meaningfully behind (by last
#      live-sync's own reported stats, optionally Kafka consumer lag),
#      do another live pass and recheck, capped at a few retries --
#      shortens the real write-frozen window below.
#   3. Enable write-freeze at the gateway (disabled on EXIT no matter how
#      this script ends, via trap -- and the gateway's own side sets a
#      TTL regardless, so a crashed caller can't leave it frozen forever).
#   4. One more small rsync pass to catch the now-tiny remaining delta.
#   5. Disable write-freeze immediately -- steps 6-7 run with writes
#      already unblocked.
#   6. tar + age-encrypt the now-static mirror.
#   7. Prune generations older than NFS_BACKUP_RETENTION_DAYS.
#
# Usage: nfs-freeze-archive.sh [--env-file <path>] [--dest <dir>] [--force]
#   --force  skip the quiescence gate's retry loop (go straight to
#            freezing writes after one check, with a warning if not
#            quiescent) -- for manual/on-demand runs, not the cron path
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE_DIR="$SCRIPT_DIR/.."
# shellcheck source=./nfs-backup-common.sh
source "$SCRIPT_DIR/nfs-backup-common.sh"

ENV_FILE=""
DEST=""
FORCE=false
while [ $# -gt 0 ]; do
    case "$1" in
        --env-file) ENV_FILE="$2"; shift ;;
        --dest)     DEST="$2"; shift ;;
        --force)    FORCE=true ;;
    esac
    shift
done

load_env "$MODULE_DIR" "$ENV_FILE"
init_backup_config

RETENTION_DAYS="${NFS_BACKUP_RETENTION_DAYS:-30}"
DEST="${DEST:-${BACKUP_LOCAL_ARCHIVE_PATH:-$MODULE_DIR/backup-archive-default}}"
mkdir -p "$DEST"
PREFIX="01-nfs-"

LOCK_WAIT_SECONDS="${NFS_BACKUP_LOCK_TIMEOUT_SECONDS:-300}"
QUIESCENCE_MAX_BYTES="${NFS_BACKUP_QUIESCENCE_MAX_BYTES:-104857600}"       # 100MiB
QUIESCENCE_MAX_FILES="${NFS_BACKUP_QUIESCENCE_MAX_FILES:-200}"
QUIESCENCE_MAX_RETRIES="${NFS_BACKUP_QUIESCENCE_MAX_RETRIES:-3}"
GATEWAY_URL="${NFS_BACKUP_GATEWAY_URL:-}"
GATEWAY_INTERNAL_TOKEN="${NFS_BACKUP_GATEWAY_INTERNAL_TOKEN:-}"
GATEWAY_FREEZE_TTL_SECONDS="${NFS_BACKUP_GATEWAY_FREEZE_TTL_SECONDS:-600}"
GATEWAY_TIMEOUT_SECONDS="${NFS_BACKUP_GATEWAY_TIMEOUT_SECONDS:-10}"

# --------------- 1. Lock, held through the archive step (step 6) ---------------
exec {LOCK_FD}>"$LOCK_FILE"
log "waiting for nfs-backup lock (up to ${LOCK_WAIT_SECONDS}s)..."
if ! flock -w "$LOCK_WAIT_SECONDS" "$LOCK_FD"; then
    log "ERROR: could not acquire lock -- another freeze-archive run in progress?"
    exit 1
fi
log "lock acquired"

# --------------- 2. Quiescence gate ---------------
# Kafka lag and SeaweedFS garbage-ratio checks are best-effort secondary
# signals, not hard requirements -- skipped silently if unconfigured or
# unreachable. Primary signal is the live-sync's own last-reported
# transfer size, which always exists once nfs-sync-live.sh has run once.
_kafka_lag_nonzero() {
    [ -n "${NFS_BACKUP_PROMETHEUS_URL:-}" ] || return 1   # not configured: treat as "can't tell", not blocking
    [ -n "${NFS_BACKUP_KAFKA_CONSUMER_GROUPS:-}" ] || return 1
    local group total=0 lag
    for group in $NFS_BACKUP_KAFKA_CONSUMER_GROUPS; do
        lag="$(curl -fsS --max-time 5 \
            "${NFS_BACKUP_PROMETHEUS_URL}/api/v1/query" \
            --data-urlencode "query=sum(kafka_consumergroup_lag{consumergroup=\"$group\"})" \
            2>/dev/null | grep -oP '"value":\[[0-9.]+,"\K[0-9.]+' || true)"
        total=$((total + ${lag%.*}))
    done 2>/dev/null || true
    [ "$total" -gt 0 ]
}

_seaweedfs_garbage_high() {
    [ -n "${NFS_BACKUP_SEAWEEDFS_STATUS_URL:-}" ] || return 1
    # Informational only (see docs: balance/EC have no confirmed disable
    # switch yet) -- never blocks, just logged.
    local ratio
    ratio="$(curl -fsS --max-time 5 "$NFS_BACKUP_SEAWEEDFS_STATUS_URL" 2>/dev/null | \
        grep -oP '"DeletedByteCount":\K[0-9]+' || true)"
    if [ -n "$ratio" ]; then
        log "  (informational) SeaweedFS DeletedByteCount=$ratio -- see docs, not a blocking check"
    fi
    return 1
}

quiescence_check_once() {
    local reason=""
    if [ -f "$STATS_FILE" ]; then
        # shellcheck disable=SC1090
        source "$STATS_FILE"
        if [ "${BYTES_TRANSFERRED:-0}" -gt "$QUIESCENCE_MAX_BYTES" ]; then
            reason="last sync transferred ${BYTES_TRANSFERRED} bytes (> ${QUIESCENCE_MAX_BYTES})"
        elif [ "${FILES_TRANSFERRED:-0}" -gt "$QUIESCENCE_MAX_FILES" ]; then
            reason="last sync transferred ${FILES_TRANSFERRED} files (> ${QUIESCENCE_MAX_FILES})"
        fi
    else
        reason="no prior live-sync stats found -- run nfs-sync-live.sh at least once before freezing"
    fi
    if [ -z "$reason" ] && _kafka_lag_nonzero; then
        reason="Kafka consumer lag for $NFS_BACKUP_KAFKA_CONSUMER_GROUPS is not zero"
    fi
    _seaweedfs_garbage_high || true
    if [ -n "$reason" ]; then
        log "  not quiescent yet: $reason"
        return 1
    fi
    return 0
}

if $FORCE; then
    quiescence_check_once || log "  --force: proceeding anyway"
else
    retries="$QUIESCENCE_MAX_RETRIES"
    until quiescence_check_once; do
        if [ "$retries" -le 0 ]; then
            log "  giving up on quiescence after $QUIESCENCE_MAX_RETRIES retries -- proceeding anyway"
            break
        fi
        log "  doing one more live pass before re-checking ($retries retries left)..."
        run_mirror_pass
        retries=$((retries - 1))
    done
fi

# --------------- 3/5. Enable/disable write-freeze at the gateway ---------------
FROZEN=false
UNFROZEN=false

disable_write_freeze() {
    $FROZEN || return 0
    $UNFROZEN && return 0
    UNFROZEN=true
    log "disabling write-freeze at the gateway"
    curl -fsS --max-time "$GATEWAY_TIMEOUT_SECONDS" -X POST \
        -H "X-Internal-Token: $GATEWAY_INTERNAL_TOKEN" \
        "$GATEWAY_URL/admin/maintenance/disable" >/dev/null \
        || log "WARNING: disabling write-freeze failed -- it carries its own ${GATEWAY_FREEZE_TTL_SECONDS}s TTL server-side and will lift on its own, but check the gateway manually"
}
trap disable_write_freeze EXIT

if [ -n "$GATEWAY_URL" ]; then
    log "enabling write-freeze at the gateway (ttl=${GATEWAY_FREEZE_TTL_SECONDS}s)"
    curl -fsS --max-time "$GATEWAY_TIMEOUT_SECONDS" -X POST \
        -H "X-Internal-Token: $GATEWAY_INTERNAL_TOKEN" \
        "$GATEWAY_URL/admin/maintenance/enable?ttl_seconds=$GATEWAY_FREEZE_TTL_SECONDS" >/dev/null
    FROZEN=true
else
    log "NFS_BACKUP_GATEWAY_URL is unset -- not freezing writes; archive may be taken from a live tree (confirm this is intentional)"
fi

# --------------- 4. Final catch-up pass ---------------
log "final catch-up pass..."
run_mirror_pass
log "catch-up done: $RSYNC_FILES_TRANSFERRED file(s), $RSYNC_BYTES_TRANSFERRED byte(s), $RSYNC_FILES_DELETED deleted"

# --------------- 5. Unfreeze now, before the slow archive step ---------------
disable_write_freeze

# --------------- 6. tar + age-encrypt the now-static mirror ---------------
TIMESTAMP="$(date -u +%Y-%m-%dT%H-%M-%S)"
TAR_PATH="$DEST/${PREFIX}${TIMESTAMP}.tar.gz"
log "archiving $MIRROR_PATH -> $TAR_PATH"
tar -czf "$TAR_PATH" -C "$MIRROR_PATH" --exclude=".nfs-backup-state" $INCLUDE_PATHS

if [ -n "${AGE_PUBLIC_KEYS:-}" ]; then
    log "encrypting with age..."
    age -r "$AGE_PUBLIC_KEYS" -o "$TAR_PATH.age" "$TAR_PATH"
    rm -f "$TAR_PATH"
    TAR_PATH="$TAR_PATH.age"
fi
log "wrote $TAR_PATH"

# --------------- 7. Retention ---------------
# BACKUP_PRUNING_PREFIX convention, same as every offen sidecar -- scoped
# to this module's own prefix so it never touches another module's
# backups sitting in the same destination.
find "$DEST" -maxdepth 1 -name "${PREFIX}*.tar.gz*" -mtime "+${RETENTION_DAYS}" -print -delete
