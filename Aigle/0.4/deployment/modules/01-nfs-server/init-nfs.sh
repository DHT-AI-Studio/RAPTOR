#!/bin/bash
# Initialize NFS storage directories
# Creates the directory structure needed for SeaweedFS and other services
# Usage: init-nfs.sh [--clean]
#   --clean  Remove and recreate all directories (used by build.py --delete)

set -euo pipefail

CLEAN=false
ENV_FILE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --clean)    CLEAN=true ;;
        --env-file) ENV_FILE="$2"; shift ;;
    esac
    shift
done

# --------------- Get .env file ---------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOTENV_DIR="$SCRIPT_DIR"

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
if [ -n "$ENV_FILE" ] && [ -f "$ENV_FILE" ]; then
    safe_source_env "$ENV_FILE"
elif [ -f "$DOTENV_DIR/../.env" ]; then
    safe_source_env "$DOTENV_DIR/../.env"
elif [ -f "$DOTENV_DIR/.env" ]; then
    safe_source_env "$DOTENV_DIR/.env"
fi


STORAGE_DIR="${NFS_BASE_PATH}"

# Default values
BASE_DIR="${BASE_DIR:-seaweedfs}"
SUB_DIRS="${SUB_DIRS:-admin s3 backup filer vol1 vol2 vol3 vol4 master1 master2 master3}"

echo "=========================================="
echo "NFS Storage Initialization Script"
echo "=========================================="
echo "Storage Directory: $STORAGE_DIR"
echo "Base Directory: $BASE_DIR"
echo "Sub Directories: $SUB_DIRS"
echo ""

# Create storage directory if it doesn't exist
if [ ! -d "$STORAGE_DIR" ]; then
    echo "Creating storage directory: $STORAGE_DIR"
    mkdir -p "$STORAGE_DIR"
fi

# ---------------- 多 module 支援 ----------------
INDEX=1
while true; do
    # 動態取得 BASE_DIR 與 SUB_DIRS
    BASE_VAR="BASE_DIR"
    SUB_VAR="SUB_DIRS"

    if [ $INDEX -gt 1 ]; then
        BASE_VAR="BASE_DIR_$INDEX"
        SUB_VAR="SUB_DIRS_$INDEX"
    fi

    BASE="${!BASE_VAR:-}"   # 取得變數內容
    SUBS="${!SUB_VAR:-}"

    # 沒設定 base 就結束
    if [ -z "$BASE" ]; then
        break
    fi

    BASE_PATH="$STORAGE_DIR/$BASE"
    echo ""
    echo "Processing base directory: $BASE"

    # 如果 base 目錄存在：--clean 時刪除，否則跳過
    if [ -d "$BASE_PATH" ]; then
        if [ "$CLEAN" = true ]; then
            echo "Removing: $BASE_PATH"
            rm -rf "$BASE_PATH"
        else
            echo "Already exists, skipping: $BASE_PATH"
        fi
    fi

    mkdir -p "$BASE_PATH"

    # 建立子目錄
    for sub in $SUBS; do
        TARGET="$BASE_PATH/$sub"
        if [ ! -d "$TARGET" ]; then
            echo "  Creating: $BASE/$sub"
            mkdir -p "$TARGET"
            chmod 777 "$TARGET"
        else
            echo "  Already exists: $BASE/$sub"
        fi
    done

    INDEX=$((INDEX + 1))
done

# Set permissions
echo ""
echo "Setting permissions..."
chmod -R 777 "$STORAGE_DIR"

echo ""
echo "=========================================="
echo "NFS Storage initialization complete!"
echo "=========================================="
echo "Storage location: $STORAGE_DIR"
echo ""
echo "Next steps:"
echo "1. Start the NFS server: docker compose up -d"
echo "2. Verify NFS export: showmount -e localhost"
echo "3. Confirm NFS_SERVER=<this host ip> in dependent modules' .env"
echo ""


ENV_path="$SCRIPT_DIR"
# 如果檔案最後一行不是空行，就補一個換行符
[ -n "$(tail -c1 "$ENV_path/.env" 2>/dev/null)" ] && echo "" >> "$ENV_path/.env"
sed -i '/^NFS_BASE_PATH=/d' "$ENV_path/.env"
echo "NFS_BASE_PATH=$STORAGE_DIR" >> "$ENV_path/.env"