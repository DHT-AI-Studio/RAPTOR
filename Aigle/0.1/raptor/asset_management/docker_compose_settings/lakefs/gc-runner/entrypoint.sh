#!/bin/bash
set -e

: "${GC_CRON_SCHEDULE:=0 3 * * *}"

LOG_FILE="/var/log/gc.log"
GC_SCRIPT="/usr/local/bin/run-gc.sh"
CRON_FILE="/etc/cron.d/gc-cron"
ENV_FILE="/tmp/gc-env.sh"

touch "$LOG_FILE"
chmod 0644 "$LOG_FILE"

echo "[GC] Setting cron schedule: $GC_CRON_SCHEDULE" | tee -a "$LOG_FILE"

if [[ ! -f "$GC_SCRIPT" ]]; then
  echo "[ERROR] $GC_SCRIPT does not exist" | tee -a "$LOG_FILE"
  exit 1
fi
if [[ ! -x "$GC_SCRIPT" ]]; then
  echo "[ERROR] $GC_SCRIPT is not executable" | tee -a "$LOG_FILE"
  exit 1
fi

echo '#!/bin/bash' > "$ENV_FILE"
printenv | grep -E 'LAKEFS|S3|AWS|TZ' | sed 's/^/export /' >> "$ENV_FILE"
echo 'export JAVA_HOME=/opt/java/openjdk' >> "$ENV_FILE"
chmod 0644 "$ENV_FILE"

cat > "$CRON_FILE" <<EOF
SHELL=/bin/bash
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/spark/bin

${GC_CRON_SCHEDULE} root . $ENV_FILE && $GC_SCRIPT >> $LOG_FILE 2>&1
EOF

chmod 0644 "$CRON_FILE"

echo "[GC] Starting cron daemon..." | tee -a "$LOG_FILE"
exec cron -f
