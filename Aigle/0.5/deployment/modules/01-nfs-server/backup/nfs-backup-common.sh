# shellcheck shell=bash
# Shared env-loading, locking, and rsync-pass helpers for
# nfs-sync-live.sh / nfs-freeze-archive.sh (VIE01-369 Phase 1, module 01).
# Sourced, not executed directly.

# --------------- Safely load a .env file (no `source`/`eval` on its content) ---------------
# Same parser as init-nfs.sh -- .env values are data, not shell code, and
# ${VAR:-default} is a real pattern root .env uses (e.g.
# GATEWAY_KEYCLOAK_PUBLIC_URL), not just one variable name.
_expand_vars() {
    local rest="$1" out="" pre inner var default
    while [[ "$rest" == *'${'*'}'* ]]; do
        pre="${rest%%\$\{*}"
        rest="${rest#*\$\{}"
        inner="${rest%%\}*}"
        rest="${rest#*\}}"
        if [[ "$inner" == *:-* ]]; then
            var="${inner%%:-*}"
            default="${inner#*:-}"
        else
            var="$inner"
            default=""
        fi
        out+="$pre${!var:-$default}"
    done
    out+="$rest"
    printf '%s' "$out"
}

safe_source_env() {
    local file="$1" line key value
    while IFS= read -r line || [ -n "$line" ]; do
        line="${line%$'\r'}"
        [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
        [[ "$line" =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
        key="${BASH_REMATCH[1]}"
        value="${BASH_REMATCH[2]}"
        if [[ "$value" =~ ^\"(.*)\"[[:space:]]*$ ]]; then
            value="${BASH_REMATCH[1]}"
        elif [[ "$value" =~ ^\'(.*)\'[[:space:]]*$ ]]; then
            value="${BASH_REMATCH[1]}"
        else
            value="${value%%[[:space:]]#*}"
            value="${value%"${value##*[![:space:]]}"}"
        fi
        export "$key=$(_expand_vars "$value")"
    done < "$file"
}

load_env() {
    # Same precedence as build.py's _env_file_args(): exactly one file
    # wins, never merged.
    local module_dir="$1" env_file="$2"
    if [ -n "$env_file" ] && [ -f "$env_file" ]; then
        safe_source_env "$env_file"
    elif [ -f "$module_dir/../.env" ]; then
        safe_source_env "$module_dir/../.env"
    elif [ -f "$module_dir/.env" ]; then
        safe_source_env "$module_dir/.env"
    fi
}

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

# --------------- Config shared by both scripts, with defaults ---------------
init_backup_config() {
    : "${NFS_BASE_PATH:?NFS_BASE_PATH must be set (in .env)}"
    : "${NFS_BACKUP_MIRROR_PATH:?NFS_BACKUP_MIRROR_PATH must be set (in .env) -- ideally on separate physical storage from NFS_BASE_PATH, see docs/backup-restore-v0.5.md}"

    # Only seaweedfs/* (04's blobs) and memvid/storage (26's data) --
    # aiml/{tmp,data} and media/processing are re-buildable scratch/cache,
    # deliberately out of scope (see docs/backup-restore-v0.5.md's "01"
    # section).
    INCLUDE_PATHS="${NFS_BACKUP_INCLUDE_PATHS:-seaweedfs memvid/storage}"

    MIRROR_PATH="$NFS_BACKUP_MIRROR_PATH"
    mkdir -p "$MIRROR_PATH"
    STATE_DIR="${NFS_BACKUP_STATE_DIR:-$MIRROR_PATH/.nfs-backup-state}"
    mkdir -p "$STATE_DIR"
    # shellcheck disable=SC2034  # used by nfs-sync-live.sh/nfs-freeze-archive.sh, which source this file
    LOCK_FILE="${NFS_BACKUP_LOCK_FILE:-$STATE_DIR/nfs-backup.lock}"
    STATS_FILE="$STATE_DIR/last-sync-stats"
}

# --------------- One rsync pass over every include path ---------------
# Mirrors $NFS_BASE_PATH/<p> -> $MIRROR_PATH/<p> for each p in
# $INCLUDE_PATHS, aggregates rsync's own --stats output (no separate
# dry-run needed), and writes it to $STATS_FILE for the quiescence gate
# to read. Sets RSYNC_FILES_TRANSFERRED / RSYNC_BYTES_TRANSFERRED /
# RSYNC_FILES_DELETED as a side effect.
run_mirror_pass() {
    RSYNC_FILES_TRANSFERRED=0
    RSYNC_BYTES_TRANSFERRED=0
    RSYNC_FILES_DELETED=0

    local p src dst stats_out n rc
    local -a extra_args
    for p in $INCLUDE_PATHS; do
        src="$NFS_BASE_PATH/$p/"
        dst="$MIRROR_PATH/$p/"
        mkdir -p "$dst"
        if [ ! -d "$NFS_BASE_PATH/$p" ]; then
            log "  (skip) $NFS_BASE_PATH/$p does not exist yet"
            continue
        fi
        extra_args=()
        if [ "$p" = "seaweedfs" ]; then
            # seaweedfs/backup/ is module 04's own existing filer.backup
            # mirror (docker-compose.yml's seaweedfs-backup service) --
            # a secondary, derived copy of data this script already backs
            # up directly, so it needs no backup of its own. It also holds
            # SeaweedFS-internal zero-byte placeholder files with mode
            # 0000 -- confirmed live (`ls -la` on the real files). When
            # NFS_BASE_PATH is itself network storage (not local disk),
            # reading those fails with EACCES even for root: the remote
            # NFS server maps this host's "root" to an unprivileged user
            # for permission checks, so mode 0000 denies everyone.
            # Exclude it rather than chase that.
            extra_args=(--exclude=/backup)
        fi
        # --no-owner --no-group: confirmed live against a real remote NFS
        # export (BACKUP_LOCAL_ARCHIVE_PATH pointed off-host, exactly the
        # deployment this is designed for) -- plain -a's chown/chgrp of
        # the destination to match the source's owner/group is rejected
        # by that export ("Operation not permitted"), which silently
        # dropped every single file's transfer (rsync exit 23, "some
        # files/attrs were not transferred") without this flag. Owner/
        # group fidelity through backup/restore was never load-bearing
        # here anyway -- init-nfs.sh already chmod 777's this whole tree.
        set +e
        stats_out="$(rsync -a --no-owner --no-group --delete --stats "${extra_args[@]}" "$src" "$dst")"
        rc=$?
        set -e
        if [ "$rc" -ne 0 ]; then
            log "  ERROR: rsync exited $rc syncing $src -> $dst"
            printf '%s\n' "$stats_out" >&2
            return "$rc"
        fi

        n="$(printf '%s\n' "$stats_out" | grep -oP 'Number of (regular )?files transferred: \K[\d,]+' | tr -d ',')"
        RSYNC_FILES_TRANSFERRED=$((RSYNC_FILES_TRANSFERRED + ${n:-0}))
        n="$(printf '%s\n' "$stats_out" | grep -oP 'Number of deleted files: \K[\d,]+' | tr -d ',')"
        RSYNC_FILES_DELETED=$((RSYNC_FILES_DELETED + ${n:-0}))
        n="$(printf '%s\n' "$stats_out" | grep -oP 'Total transferred file size: \K[\d,]+' | tr -d ',')"
        RSYNC_BYTES_TRANSFERRED=$((RSYNC_BYTES_TRANSFERRED + ${n:-0}))
    done

    printf 'TIMESTAMP=%s\nFILES_TRANSFERRED=%d\nBYTES_TRANSFERRED=%d\nFILES_DELETED=%d\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$RSYNC_FILES_TRANSFERRED" "$RSYNC_BYTES_TRANSFERRED" "$RSYNC_FILES_DELETED" \
        > "$STATS_FILE"
}
