# VIE01-369: sidecar for nfs-sync-live.sh + nfs-freeze-archive.sh.
# No application code lives in this image -- all three scripts (this one's
# own sibling files) are bind-mounted in at runtime (see docker-compose.yml's
# nfs-backup service), not COPYed at build time, so changing them never
# needs an image rebuild.
FROM alpine:3.20

# busybox's own crond (already in this base image) is the scheduler --
# age/rsync/curl/bash/flock(util-linux)/tar are the only real dependencies
# the two scripts need beyond coreutils.
RUN apk add --no-cache bash rsync age curl tzdata util-linux coreutils grep

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
