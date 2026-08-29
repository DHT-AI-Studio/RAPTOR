#!/bin/sh
# nvidia-gpu-watchdog.sh — shared GPU watchdog for co-located GPU worker containers.
#
# Runs as a lightweight *sidecar container* (see docker-compose.yml service
# `gpu-watchdog`), watching every container listed in WATCH_CONTAINERS. Each
# watched container must expose a Docker healthcheck that fails when
# CUDA/NVML goes stale (see the `torch.cuda.is_available()` healthchecks on
# the GPU services in modules 09/10/11/12).
#
# One watchdog per GPU host, not per module — modules 09/10/11/12 are
# normally deployed together on the same GPU host (see BUILD.md), so a single
# sidecar can recover all of them via the local Docker socket. Pattern ported
# from module 16's training-watchdog sidecar.
#
# Config (env):
#   WATCH_CONTAINERS  comma or space separated container names to watch
#   CHECK_INTERVAL    seconds between sweeps            (default: 30)
#   RECOVER_COOLDOWN  seconds to wait after any restart  (default: 90 — must
#                     exceed the watched containers' health start_period)
set -eu

CHECK_INTERVAL="${CHECK_INTERVAL:-30}"
RECOVER_COOLDOWN="${RECOVER_COOLDOWN:-90}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] gpu-watchdog: $*"; }

# Accept either comma- or space-separated container lists.
containers="$(echo "${WATCH_CONTAINERS:?WATCH_CONTAINERS must be set}" | tr ',' ' ')"

log "started — watching [$containers] every ${CHECK_INTERVAL}s"

while true; do
    restarted_any=0
    for c in $containers; do
        # `.State.Health.Status` is empty/absent if the container has no
        # healthcheck or is gone; guard so the loop never dies.
        status="$(docker inspect "$c" --format '{{.State.Health.Status}}' 2>/dev/null || echo missing)"

        if [ "$status" = "unhealthy" ]; then
            log "'$c' is UNHEALTHY — recovering (restart + NVML refresh)"
            # Best-effort persistence-mode refresh; may be a no-op from
            # inside a container, so never fail the sweep on it.
            docker exec "$c" nvidia-smi -pm 1 >/dev/null 2>&1 || true
            if docker restart "$c" >/dev/null 2>&1; then
                log "restarted '$c'"
                restarted_any=1
            else
                log "restart of '$c' FAILED"
            fi
        fi
    done

    if [ "$restarted_any" = "1" ]; then
        log "cooling down ${RECOVER_COOLDOWN}s"
        sleep "$RECOVER_COOLDOWN"
    fi
    sleep "$CHECK_INTERVAL"
done
