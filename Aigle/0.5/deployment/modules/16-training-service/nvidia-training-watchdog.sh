#!/bin/sh
# nvidia-training-watchdog.sh — GPU watchdog for the training service.
#
# Runs as a lightweight *sidecar container* (see docker-compose.yml service
# `training-watchdog`), NOT host cron — so it starts/stops with the module via
# `docker compose up/down` and needs no host sudo or /etc/cron.d entry.
#
# Why: Module 16's healthcheck marks the container "unhealthy" when
#   `torch.cuda.is_available()` turns false — i.e. CUDA/NVML goes stale after
#   long uptime or the GPU drops off. This loop detects that and recovers by
#   restarting the container, which re-initializes the CUDA context.
#
# Config (env):
#   WATCH_CONTAINER   container to watch/restart   (default: raptor-training-service)
#   CHECK_INTERVAL    seconds between checks        (default: 30)
#   RECOVER_COOLDOWN  seconds to wait after a restart before checking again
#                     (default: 90 — must exceed the container's health start_period)
set -eu

WATCH_CONTAINER="${WATCH_CONTAINER:-raptor-training-service}"
CHECK_INTERVAL="${CHECK_INTERVAL:-30}"
RECOVER_COOLDOWN="${RECOVER_COOLDOWN:-90}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] watchdog: $*"; }

log "started — watching '$WATCH_CONTAINER' every ${CHECK_INTERVAL}s"

while true; do
    # `.State.Health.Status` is empty/absent if the container has no healthcheck
    # or is gone; guard so the loop never dies.
    status="$(docker inspect "$WATCH_CONTAINER" --format '{{.State.Health.Status}}' 2>/dev/null || echo missing)"

    if [ "$status" = "unhealthy" ]; then
        log "'$WATCH_CONTAINER' is UNHEALTHY — recovering (restart + NVML refresh)"
        # Best-effort persistence-mode refresh; may be a no-op from inside a
        # container, so never fail the loop on it.
        docker exec "$WATCH_CONTAINER" nvidia-smi -pm 1 >/dev/null 2>&1 || true
        if docker restart "$WATCH_CONTAINER" >/dev/null 2>&1; then
            log "restarted '$WATCH_CONTAINER'; cooling down ${RECOVER_COOLDOWN}s"
        else
            log "restart of '$WATCH_CONTAINER' FAILED"
        fi
        sleep "$RECOVER_COOLDOWN"
    fi

    sleep "$CHECK_INTERVAL"
done
