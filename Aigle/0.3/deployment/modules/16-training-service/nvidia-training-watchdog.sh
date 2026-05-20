#!/bin/bash
# nvidia-training-watchdog.sh
# Called by /etc/cron.d/nvidia-training-watchdog every 5 minutes.
# Restarts aigle-training-service if its Docker healthcheck reports "unhealthy".

LOG=/var/log/nvidia-training-watchdog.log
CONTAINER=aigle-training-service

STATUS=$(docker inspect "$CONTAINER" --format='{{.State.Health.Status}}' 2>/dev/null)

if [ "$STATUS" = "unhealthy" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Container is unhealthy — refreshing NVML and restarting..." >> "$LOG"
    nvidia-smi -pm 1 >> "$LOG" 2>&1
    docker restart "$CONTAINER" >> "$LOG" 2>&1
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Restart complete." >> "$LOG"
fi
