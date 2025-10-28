#!/bin/bash
set -e

# --- Truncate log if too big, keep last N lines ---
LOG_FILE="/var/log/gc.log"
MAX_LINES=10000

if [ -f "$LOG_FILE" ]; then
    LINE_COUNT=$(wc -l < "$LOG_FILE")
    if [ "$LINE_COUNT" -gt "$MAX_LINES" ]; then
        echo "[GC] Log file exceeded $MAX_LINES lines, truncating..." | tee "$LOG_FILE.tmp"
        tail -n $MAX_LINES "$LOG_FILE" >> "$LOG_FILE.tmp"
        mv "$LOG_FILE.tmp" "$LOG_FILE"
    fi
fi

: "${LAKEFS_REPOSITORY:?Environment variable LAKEFS_REPOSITORY is required}"
: "${LAKEFS_ENDPOINT:?Environment variable LAKEFS_ENDPOINT is required}"
: "${LAKEFS_ACCESS_KEY:?Environment variable LAKEFS_ACCESS_KEY is required}"
: "${LAKEFS_SECRET_KEY:?Environment variable LAKEFS_SECRET_KEY is required}"
: "${S3_ENDPOINT:?Environment variable S3_ENDPOINT is required}"
: "${AWS_ACCESS_KEY:?Environment variable AWS_ACCESS_KEY is required}"
: "${AWS_SECRET_KEY:?Environment variable AWS_SECRET_KEY is required}"

echo "[GC] Starting lakeFS garbage collection for repository ${LAKEFS_REPOSITORY}..."
echo "[GC] s3 endpoint: ${S3_ENDPOINT}"

SPARK_SUBMIT="/opt/spark/bin/spark-submit"

if [[ ! -x "$SPARK_SUBMIT" ]]; then
  echo "[ERROR] spark-submit not found or not executable: $SPARK_SUBMIT" >&2
  exit 1
fi

"$SPARK_SUBMIT" \
    --conf "spark.hadoop.lakefs.api.url=${LAKEFS_ENDPOINT}/api/v1" \
    --conf "spark.hadoop.lakefs.api.access_key=${LAKEFS_ACCESS_KEY}" \
    --conf "spark.hadoop.lakefs.api.secret_key=${LAKEFS_SECRET_KEY}" \
    --conf "spark.hadoop.fs.s3a.endpoint=${S3_ENDPOINT}" \
    --conf "spark.hadoop.fs.s3a.access.key=${AWS_ACCESS_KEY}" \
    --conf "spark.hadoop.fs.s3a.secret.key=${AWS_SECRET_KEY}" \
    --conf "spark.hadoop.fs.s3a.connection.ssl.enabled=false" \
    --conf "spark.hadoop.fs.s3a.path.style.access=true" \
    --conf "spark.hadoop.com.amazonaws.services.s3a.enableV4=true" \
    --packages org.apache.hadoop:hadoop-aws:3.3.4 \
    --class io.treeverse.gc.GarbageCollection \
    http://treeverse-clients-us-east.s3-website-us-east-1.amazonaws.com/lakefs-spark-client/0.15.0/lakefs-spark-client-assembly-0.15.0.jar \
    "${LAKEFS_REPOSITORY}"

echo "[GC] Finished lakeFS garbage collection."