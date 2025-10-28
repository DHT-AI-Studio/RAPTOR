#!/bin/bash
# Automatically check/create NFS subdirectories

set -euo pipefail

# Load environment variables
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
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

