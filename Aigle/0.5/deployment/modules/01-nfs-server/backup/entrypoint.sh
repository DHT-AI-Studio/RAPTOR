#!/bin/sh
# Writes busybox crond's crontab and runs it in the foreground as PID 1.
#
# Plain cron (busybox's included) starts each job with almost none of the
# daemon's own environment -- a classic gotcha, confirmed live on this
# image. The fix: crond's own crontab file supports plain NAME=value lines
# before the schedule lines, which it re-exports into every job it runs
# (not shell-sourced, just its own simple parser) -- so re-declare
# everything the two scripts might read there instead of relying on
# inheritance.
#
# Job output goes to /proc/1/fd/1 and /proc/1/fd/2 -- this process *is*
# PID 1 (crond -f), so that's this container's own stdout/stderr, which
# `docker logs` already shows.
set -eu

CRONTAB=/etc/crontabs/root
: > "$CRONTAB"

env | grep -E '^(NFS_|BACKUP_|AGE_PUBLIC_KEYS=|TZ=)' >> "$CRONTAB" || true

echo "${NFS_SYNC_CRON:-0 * * * *} /scripts/nfs-sync-live.sh >>/proc/1/fd/1 2>>/proc/1/fd/2" >> "$CRONTAB"
echo "${NFS_BACKUP_CRON:-0 3 * * *} /scripts/nfs-freeze-archive.sh >>/proc/1/fd/1 2>>/proc/1/fd/2" >> "$CRONTAB"

echo "nfs-backup sidecar starting, crontab:"
cat "$CRONTAB"

exec crond -f -d 8
