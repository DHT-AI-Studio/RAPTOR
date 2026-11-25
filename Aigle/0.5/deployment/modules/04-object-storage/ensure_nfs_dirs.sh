#!/bin/bash
# Automatically check/create NFS subdirectories

set -euo pipefail

# --------------- Safely load a .env file (no `source`/`eval` on its content) ---------------
# .env values are data, not shell code — an unfilled placeholder like
# NFS_SERVER=<your_nfs_server_ip> is fine for docker compose's own parser,
# but `source`ing it as bash treats < and > as redirection operators and
# crashes with "syntax error near unexpected token `newline'". Any value
# could similarly contain `;`, `|`, backticks, or $(...) — sourcing a
# config file as executable code is a real (if low-risk here) injection
# surface too. Parse KEY=VALUE by hand instead; only ${VAR} references are
# expanded (against already-loaded vars), nothing else is interpreted.
_expand_vars() {
    local rest="$1" out="" pre var
    while [[ "$rest" == *'${'*'}'* ]]; do
        pre="${rest%%\$\{*}"
        rest="${rest#*\$\{}"
        var="${rest%%\}*}"
        rest="${rest#*\}}"
        out+="$pre${!var-}"
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

# --------------- Load environment variables ---------------
# Same precedence as build.py's _env_file_args(): exactly one file wins,
# never merged. A module-local .env is only a fallback for standalone
# use (no root .env at all) — if root .env exists, it alone applies, so
# a stale/unfilled module-local .env can't silently override a value
# root already got right.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/../.env" ]; then
    safe_source_env "$SCRIPT_DIR/../.env"
elif [ -f "$SCRIPT_DIR/.env" ]; then
    safe_source_env "$SCRIPT_DIR/.env"
fi

NFS_SERVER="${NFS_SERVER:?NFS_SERVER not set}"
NFS_EXPORT="${NFS_EXPORT:?NFS_EXPORT not set}"
TMP_MNT="${TMP_MNT:?TMP_MNT not set}"
BASE_DIR="${BASE_DIR:?BASE_DIR not set}"
SUB_DIRS="${SUB_DIRS:?SUB_DIRS not set}"

echo "NFS Server: $NFS_SERVER"
echo "NFS Export: $NFS_EXPORT"
echo "Temporary Mount Point: $TMP_MNT"
echo "Base Directory: $BASE_DIR"
echo "Sub Directories: $SUB_DIRS"

# Create temporary mount point
sudo mkdir -p "$TMP_MNT"

# Auto-unmount trap
trap 'echo "Unmounting $TMP_MNT..."; sudo umount -f "$TMP_MNT" || true' EXIT

echo "Mounting $NFS_SERVER:$NFS_EXPORT to $TMP_MNT..."
sudo mount -t nfs -o nfsvers=4 "$NFS_SERVER:$NFS_EXPORT" "$TMP_MNT"

# Ensure base_dir is clean
if [ -d "$TMP_MNT/$BASE_DIR" ]; then
    echo "Removing existing base directory: $BASE_DIR"
    sudo rm -rf "$TMP_MNT/$BASE_DIR"
fi

# Create base directory
echo "Creating base directory: $BASE_DIR"
sudo mkdir -p "$TMP_MNT/$BASE_DIR"

# Create subdirs
for sub in $SUB_DIRS; do
    TARGET="$TMP_MNT/$BASE_DIR/$sub"
    if [ ! -d "$TARGET" ]; then
        echo "Creating: $BASE_DIR/$sub"
        sudo mkdir -p "$TARGET"
        sudo chmod 777 "$TARGET"
    else
        echo "Already exists: $BASE_DIR/$sub"
    fi
done

echo "All done. NFS directories ready under $NFS_EXPORT/$BASE_DIR"

