#!/usr/bin/env bash
#
# Module 14 alert smoke test.
#
# Proves end-to-end that GPUMemoryPressure and KafkaConsumerLag actually reach an
# oncall receiver:
#
#   synthetic metrics -> Prometheus (real alert_rules.yml)
#                     -> Alertmanager (real alertmanager.yml)
#                     -> webhook sink
#
# Both rules keep their shipped `for:` durations (5m and 10m), so a full pass takes
# roughly 12 minutes. Nothing is stubbed except the webhook destination.
#
# Usage:  sudo bash deployment/modules/14-monitoring/test/alert_smoke_test.sh
# Output: test/alert_smoke_test.log  (also printed to the terminal)

set -uo pipefail

cd "$(dirname "$(readlink -f "$0")")" || exit 2

COMPOSE_FILE="docker-compose.test.yml"
PROJECT="raptor-alerttest"
LOG="alert_smoke_test.log"
SINK_OUT="out/received.jsonl"

PROM="http://localhost:19090"
ALERTMANAGER="http://localhost:19093"
ALERTS=("GPUMemoryPressure" "KafkaConsumerLag")

PENDING_TIMEOUT=120   # rule loaded and matching the fixture series
FIRING_TIMEOUT=900    # KafkaConsumerLag has for: 10m
SINK_TIMEOUT=240      # plus Alertmanager group_wait: 30s

: >"$LOG"
exec > >(tee -a "$LOG") 2>&1

say() { echo "[$(date '+%H:%M:%S')] $*"; }
fail() { echo; echo "RESULT: FAIL — $*"; dump_diag; teardown; exit 1; }

dump_diag() {
  echo
  echo "───── diagnostics ─────"
  for svc in prometheus alertmanager fixture; do
    echo "--- $svc (last 40 lines) ---"
    docker compose -p "$PROJECT" -f "$COMPOSE_FILE" logs --tail=40 "$svc" 2>&1
  done
  echo "--- $SINK_OUT ---"
  [ -f "$SINK_OUT" ] && cat "$SINK_OUT" || echo "(sink received nothing)"
  echo "───────────────────────"
}

# Scoped cleanup. Only ever touches the raptor-alerttest project — never a blanket
# `docker system prune`, which would take the production Raptor stack with it.
teardown() {
  say "Tearing down test stack..."
  docker compose -p "$PROJECT" -f "$COMPOSE_FILE" down -v --remove-orphans >/dev/null 2>&1

  # Belt and braces: anything left behind matching the test's own names.
  local leftovers
  leftovers="$(docker ps -aq --filter "name=raptor-alerttest-" 2>/dev/null)"
  if [ -n "$leftovers" ]; then
    say "Removing leftover test containers..."
    docker rm -f $leftovers >/dev/null 2>&1
  fi
  docker network rm "${PROJECT}_default" >/dev/null 2>&1
  docker volume ls -q --filter "name=^${PROJECT}_" 2>/dev/null | while read -r v; do
    [ -n "$v" ] && docker volume rm "$v" >/dev/null 2>&1
  done

  # Verify nothing of ours survived, and prove production was untouched.
  local remaining
  remaining="$(docker ps -aq --filter "name=raptor-alerttest-" 2>/dev/null | wc -l)"
  if [ "$remaining" -eq 0 ]; then
    say "Cleanup OK — no raptor-alerttest containers, networks or volumes remain."
  else
    say "WARNING — $remaining raptor-alerttest container(s) still present:"
    docker ps -a --filter "name=raptor-alerttest-" --format '  {{.Names}}\t{{.Status}}'
  fi
  say "Production containers still running: $(docker ps -q 2>/dev/null | wc -l)"
}

# state_of ALERTNAME -> inactive | pending | firing | absent | unavailable
state_of() {
  curl -sf --max-time 5 "$PROM/api/v1/rules" 2>/dev/null | python3 -c '
import json, sys
name = sys.argv[1]
try:
    data = json.load(sys.stdin)
except Exception:
    print("unavailable"); sys.exit(0)
for group in data.get("data", {}).get("groups", []):
    for rule in group.get("rules", []):
        if rule.get("name") == name:
            print(rule.get("state", "unknown")); sys.exit(0)
print("absent")
' "$1" 2>/dev/null || echo unavailable
}

# sink_has ALERTNAME -> exit 0 if a firing payload for that alert arrived
sink_has() {
  python3 - "$1" "$SINK_OUT" <<'PY'
import json, os, sys
name, path = sys.argv[1], sys.argv[2]
if not os.path.exists(path):
    sys.exit(1)
with open(path, encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        for alert in payload.get("alerts", []):
            if (alert.get("labels", {}).get("alertname") == name
                    and alert.get("status") == "firing"):
                sys.exit(0)
sys.exit(1)
PY
}

wait_for_states() {
  local want_regex="$1" timeout="$2" label="$3"
  local deadline=$(( $(date +%s) + timeout )) last_report=0

  while [ "$(date +%s)" -lt "$deadline" ]; do
    local all_ok=1 line=""
    for alert in "${ALERTS[@]}"; do
      local st
      st="$(state_of "$alert")"
      line+="$alert=$st "
      [[ "$st" =~ $want_regex ]] || all_ok=0
    done

    if [ "$all_ok" -eq 1 ]; then
      say "OK  $label — $line"
      return 0
    fi

    local now
    now=$(date +%s)
    if [ $(( now - last_report )) -ge 30 ]; then
      say "    waiting for $label ($(( deadline - now ))s left) — $line"
      last_report=$now
    fi
    sleep 5
  done

  say "TIMEOUT waiting for $label"
  return 1
}

# ── preflight ────────────────────────────────────────────────────────────────
say "Module 14 alert smoke test"

docker info >/dev/null 2>&1 || {
  echo "RESULT: FAIL — cannot talk to the Docker daemon. Re-run with sudo."
  exit 2
}
command -v python3 >/dev/null || { echo "RESULT: FAIL — python3 required"; exit 2; }
command -v curl >/dev/null || { echo "RESULT: FAIL — curl required"; exit 2; }

for port in 18080 19090 19093; do
  if (echo >/dev/tcp/127.0.0.1/$port) >/dev/null 2>&1; then
    echo "RESULT: FAIL — port $port is already in use; free it and re-run."
    exit 2
  fi
done

trap 'echo; say "Interrupted."; teardown; exit 130' INT TERM

# ── bring up ─────────────────────────────────────────────────────────────────
say "Cleaning up any previous run..."
docker compose -p "$PROJECT" -f "$COMPOSE_FILE" down -v --remove-orphans >/dev/null 2>&1

mkdir -p out
: >"$SINK_OUT"
chmod 777 out "$SINK_OUT" 2>/dev/null

say "Starting isolated stack (project: $PROJECT)..."
docker compose -p "$PROJECT" -f "$COMPOSE_FILE" up -d || fail "docker compose up failed"

# Alertmanager first: a bad alertmanager.yml makes it exit immediately, and there is
# no point spending 12 minutes on the `for:` durations only to fail at delivery.
say "Waiting for Alertmanager to accept its config..."
am_deadline=$(( $(date +%s) + 60 ))
until curl -sf --max-time 3 "$ALERTMANAGER/-/ready" >/dev/null 2>&1; do
  if [ "$(date +%s)" -ge "$am_deadline" ]; then
    echo "--- alertmanager logs ---"
    docker compose -p "$PROJECT" -f "$COMPOSE_FILE" logs --tail=30 alertmanager 2>&1
    fail "Alertmanager never became ready — its config was rejected (see logs above)"
  fi
  sleep 3
done
say "Alertmanager ready — config accepted."

say "Waiting for Prometheus to answer..."
up_deadline=$(( $(date +%s) + 90 ))
until curl -sf --max-time 3 "$PROM/-/ready" >/dev/null 2>&1; do
  [ "$(date +%s)" -lt "$up_deadline" ] || fail "Prometheus never became ready"
  sleep 3
done
say "Prometheus ready."

# Confirm the fixture is actually being scraped before trusting rule states.
say "Checking the fixture target is UP..."
scrape_deadline=$(( $(date +%s) + 60 ))
until curl -sf --max-time 5 "$PROM/api/v1/query?query=up%7Bjob%3D%22fixture%22%7D" 2>/dev/null \
      | grep -q '"value"'; do
  [ "$(date +%s)" -lt "$scrape_deadline" ] || fail "Prometheus never scraped the metrics fixture"
  sleep 3
done
say "Fixture target UP."

# ── phase 1: rules load and match the fixture series ─────────────────────────
say "Phase 1: both rules loaded and matching the synthetic series"
wait_for_states '^(pending|firing)$' "$PENDING_TIMEOUT" "pending/firing" \
  || fail "a rule never left inactive — its expression does not match the metric names the exporters emit"

# ── phase 2: honour the shipped for: durations ───────────────────────────────
say "Phase 2: both rules reach firing (for: 5m and 10m — this is the slow part)"
wait_for_states '^firing$' "$FIRING_TIMEOUT" "firing" \
  || fail "a rule stayed pending and never fired"

# ── phase 3: delivery actually happens ───────────────────────────────────────
say "Phase 3: Alertmanager delivers both alerts to the receiver"
sink_deadline=$(( $(date +%s) + SINK_TIMEOUT ))
while :; do
  missing=()
  for alert in "${ALERTS[@]}"; do
    sink_has "$alert" || missing+=("$alert")
  done
  [ ${#missing[@]} -eq 0 ] && break

  now=$(date +%s)
  [ "$now" -lt "$sink_deadline" ] \
    || fail "receiver never got: ${missing[*]} — alerts fired but delivery is broken"
  say "    waiting for delivery ($(( sink_deadline - now ))s left) — missing: ${missing[*]}"
  sleep 10
done
say "OK  receiver got both alerts"

# ── report ───────────────────────────────────────────────────────────────────
echo
echo "Delivered payloads:"
python3 - "$SINK_OUT" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        for alert in payload.get("alerts", []):
            labels = alert.get("labels", {})
            print("  %-20s %-8s severity=%-8s %s" % (
                labels.get("alertname"),
                alert.get("status"),
                labels.get("severity"),
                alert.get("annotations", {}).get("summary", ""),
            ))
PY

echo
echo "RESULT: PASS — GPUMemoryPressure and KafkaConsumerLag both fired and reached the receiver."
echo "Log: $(pwd)/$LOG"
teardown
exit 0
