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

# --------------- Load environment variables ---------------
# 1. Explicit --env-file (passed by build.py, same as docker compose)
if [ -n "$ENV_FILE" ] && [ -f "$ENV_FILE" ]; then
    set -a; source "$ENV_FILE"; set +a
else
    # Fallback: global env (deployment/modules/.env)
    if [ -f "$DOTENV_DIR/../.env" ]; then
        set -a; source "$DOTENV_DIR/../.env"; set +a
    fi
fi
# 2. Module-local .env always overrides last (machine-specific values)
if [ -f "$DOTENV_DIR/.env" ]; then
    set -a; source "$DOTENV_DIR/.env"; set +a
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