# Module 01: NFS Server

NFS v4 server providing persistent storage for SeaweedFS volumes.

## Services

| Container | Port | Purpose |
|-----------|------|---------|
| `raptor-nfs-server` | 2049 | NFS v4 server |

> **Note:** Ports 111 (RPC portmapper), 32765, 32767 are commented out in docker-compose.yml.
> Port 111 conflicts with the host's rpcbind service — do not expose it.

## First-Time Setup

Run once before the first deployment to create NFS storage directories:

```bash
sudo ./init-nfs.sh
```

This script reads `.env` and creates the directory structure under `NFS_ROOT_PATH`, then writes `NFS_BASE_PATH` back to `.env`.

Required `.env` variables:

```env
NFS_ROOT_PATH=/opt/home/dht/storage
BASE_DIR=seaweedfs
SUB_DIRS="admin s3 backup filer vol1 vol2 vol3 vol4 master1 master2 master3"
BASE_DIR_2=aiml
SUB_DIRS_2="tmp data"
```

## Deploy

```bash
# Via build.py (recommended)
python build.py -m 01

# Or directly
docker compose up -d
```

## Verify

```bash
# Check container status
docker compose ps

# Check NFS exports
showmount -e localhost

# Check mounted directories inside container
docker exec raptor-nfs-server ls /nfs-share
```

## Configuration

- **Host storage path**: `NFS_BASE_PATH` (set by `init-nfs.sh`)
- **Container mount**: `/nfs-share`
- **Network**: `raptor` (external)

## Notes

- Must be deployed before module 04 (SeaweedFS uses NFS-backed volumes)
- `init-nfs.sh` is interactive — it will prompt before removing existing directories
