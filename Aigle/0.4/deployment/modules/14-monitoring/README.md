# Monitoring & Observability

Prometheus + Grafana + Alertmanager + Loki/Promtail, plus node-exporter, cAdvisor,
redis-exporter, kafka-exporter, elasticsearch-exporter, DCGM exporter, and cluster metrics
exposed via OpenSearch's built-in `prometheus-exporter` plugin (see [17-hybrid-search](../17-hybrid-search/)).

## OpenSearch scrape password

OpenSearch's security plugin enforces HTTPS + auth on every REST path (including
`/_prometheus/metrics`), and Prometheus's scrape config format doesn't support environment
variable expansion (unlike docker-compose's `${VAR}`), so the password can't be injected
directly with `${OPENSEARCH_PASSWORD}` the way other jobs do. The workaround is
`basic_auth.password_file`, pointing at a password file mounted into the container that is
**not committed to git**:

```yaml
  - job_name: 'opensearch'
    basic_auth:
      username: admin
      password_file: /etc/prometheus/secrets/opensearch_password
```

This file (`deployment/modules/14-monitoring/secrets/opensearch_password`) is automatically
regenerated from `.env`'s `OPENSEARCH_PASSWORD` every time this module starts, by
`python build.py -m 14` (via `build.py`'s `ensure_opensearch_metrics_secret()`). The directory
itself is in `.gitignore`, so the password string never enters git history.

## Scrape targets

`config/prometheus/prometheus.yml` currently defines **16 scrape jobs**:

| # | `job_name` | Target | `metrics_path` |
|---|-----------|--------|----------------|
| 1 | `prometheus` | `localhost:9090` | `/metrics` (default) |
| 2 | `raptor-agent-protocol` | `raptor-agent-protocol:8030` | `/metrics` |
| 3 | `raptor-api-gateway` | `raptor-api-gateway:8012` | `/metrics` |
| 4 | `raptor-hybridsearch` | `raptor-hybridsearch-api:8000` | `/metrics` |
| 5 | `raptor-ai-lifecycle` | `raptor-ai-lifecycle-api:8010` | `/metrics` |
| 6 | `raptor-query-orchestrator` | `raptor-query-orchestrator:8000` | `/metrics` |
| 7 | `raptor-chat-service` | `raptor-chat-service:8021` | `/metrics` |
| 8 | `qdrant` | `raptor-qdrant:6333` | `/metrics` |
| 9 | `neo4j` | `raptor-neo4j:2004` | `/metrics` |
| 10 | `lakefs` | `raptor-lakefs:8000` | `/metrics` |
| 11 | `raptor-redis-exporter` | `raptor-redis-exporter:9121` | `/metrics` (default) |
| 12 | `kafka` | `raptor-kafka-exporter:9308` | `/metrics` (default) |
| 13 | `opensearch` | `raptor-opensearch-node1:9200` | `/_prometheus/metrics` (HTTPS + basic auth, see above) |
| 14 | `keycloak` | `raptor-keycloak:9000` | `/metrics` |
| 15 | `raptor-elasticsearch-exporter` | `raptor-elasticsearch-exporter:9114` | `/metrics` (default) |
| 16 | `dcgm-exporter` | `raptor-dcgm-exporter:9400` | `/metrics` |

Two things worth noting:

- **`node-exporter` and cAdvisor are deployed but nothing is scraping them.** Both containers
  are defined in `docker-compose.yml` (`raptor-node-exporter:9100`, `raptor-cadvisor:8080`), but
  `prometheus.yml` has **no** corresponding scrape job, so host-level and container-level
  metrics currently never make it into Prometheus. To collect them, just add a job for each.
- **`dcgm-exporter` sits under `profiles: [gpu]`** and doesn't start by default. Setting
  `ENABLE_GPU_MONITORING=true` in `.env` and deploying with `build.py -m 14` automatically adds
  the `--profile gpu` flag for you (no need to remember to pass it manually). On a machine
  without a GPU (or with this variable still false), this target showing DOWN and the
  `ServiceDown` alert firing for it are both expected — not a sign of something broken.

## Exporters

| Exporter | Target | Corresponding service |
|----------|--------|---------|
| node-exporter | `raptor-node-exporter:9100` | Host-level metrics (⚠️ not scraped) |
| cAdvisor | `raptor-cadvisor:8080` | Container-level metrics (⚠️ not scraped) |
| redis-exporter | `raptor-redis-exporter:9121` | `raptor-redis-standalone` (02-redis-cluster) |
| kafka-exporter | `raptor-kafka-exporter:9308` | `kafka-broker1/2/3` (05-kafka-cluster, overridable via `KAFKA_BROKER_1/2/3`) |
| elasticsearch-exporter | `raptor-elasticsearch-exporter:9114` | `opensearch-node1` (duplicates the native plugin below, kept as a second source) |
| DCGM exporter | `raptor-dcgm-exporter:9400` | GPU metrics (`--profile gpu`, GPU hosts only) |
| OpenSearch `prometheus-exporter` plugin | `raptor-opensearch-node1:9200/_prometheus/metrics` | `opensearch-node1/2` (17-hybrid-search, requires HTTPS + basic auth, see above) |
| Keycloak built-in metrics | `raptor-keycloak:9000/metrics` | `keycloak` (06-authentication, `KC_METRICS_ENABLED=true` is enabled by default, management port 9000 is not exposed externally) |

## Alert rules

A single rules file (`config/prometheus/alert_rules.yml`) is mounted into the Prometheus
container (the `prometheus` volumes in `docker-compose.yml`), containing two `groups:`. It was
once split into `alert_rules.yml` + `alert_rules_03.yml` (the `_03` name is a holdover from the
Raptor 0.3 era) — the latter was only listed in `prometheus.yml`'s `rule_files` but never
actually mounted, so that entire set of rules was silently never evaluated. Merging them into
one file means this can't happen again — there's only one `rule_files` entry, one volume mount.

### Group `raptor_basic`

| Alert | Condition | `for` | Severity |
|-------|------|-------|----------|
| `ServiceDown` | `up == 0` | 2m | critical |
| `HighErrorRate` | `sum(rate(http_requests_total{status=~"5.."}[5m])) by (job)` / total > 0.05 | 5m | warning |

### Group `raptor_extra_alerts`

| Alert | Condition | `for` | Severity | Metric source |
|-------|------|-------|----------|---------|
| `APILatencyHigh` | p95 `http_request_duration_seconds_bucket` > 2s | 5m | warning | FastAPI instrumentator |
| **`KafkaConsumerLag`** | `sum(kafka_consumergroup_lag) by (consumergroup, topic) > 1000` | 10m | warning | `danielqsj/kafka-exporter` |
| `PersonalIndexDLQ` | `sum(kafka_topic_partition_current_offset{topic="personal-index-requests-dlq"}) > 0` | 1m | critical | `danielqsj/kafka-exporter` (VIE01-190, Module 25) |
| `PersonalIndexConsumerLag` | `sum(kafka_consumergroup_lag{consumergroup="personal-db-service"}) by (topic) > 1000` | 10m | warning | `danielqsj/kafka-exporter` (Module 25) |
| **`GPUMemoryPressure`** | `(DCGM_FI_DEV_FB_USED / (DCGM_FI_DEV_FB_USED + DCGM_FI_DEV_FB_FREE)) * 100 > 90` | 5m | critical | `nvcr.io/nvidia/k8s/dcgm-exporter` |
| `QdrantSearchError` | Qdrant 5xx ratio > 1% | 5m | critical | Qdrant `/metrics` |
| `OpenSearchDown` | `up{job="opensearch"} == 0` | 2m | critical | OpenSearch plugin |
| `Neo4jDown` | `up{job="neo4j"} == 0` | 2m | critical | Neo4j `:2004/metrics` |

Details on the two most important rules:

**`KafkaConsumerLag`** — the metric name must be `kafka_consumergroup_lag`, with a
`consumergroup` label (not `kafka_consumer_group_lag` / `group` — those aren't names
`danielqsj/kafka-exporter` actually produces). `sum by (consumergroup, topic)` adds up lag
across every partition under that group.

**`GPUMemoryPressure`** — the DCGM exporter produces `DCGM_FI_DEV_*`, not `nvidia_smi_*`. The
`FB_USED` and `FB_FREE` series share exactly the same labels, so the binary operation lines up
directly. Note that which fields this exporter emits comes from the image's built-in
`default-counters.csv`, which isn't pinned in this repo, and the image tag is `:latest` — worth
confirming the fields still exist before changing the image version.

## Alert notification setup (Alertmanager)

`config/alertmanager/alertmanager.yml` originally routed every alert to `receiver: 'null'` (a
receiver with no notification channel at all) — the result was that rules evaluated normally,
Alertmanager grouped them normally, and then just dropped the notifications, **so nobody ever
received anything**. It's now been changed to a real `oncall` receiver:

- All alerts → `oncall`, `repeat_interval: 12h`
- `severity="critical"` takes a sub-route, also `oncall`, but with `repeat_interval: 4h`
- `inhibit_rules` are unchanged: on the same `alertname` + `job`, critical suppresses warning

**Something you need to do at deploy time: set `ALERT_WEBHOOK_URL` in `.env`.**

```
ALERT_WEBHOOK_URL=https://hooks.slack.com/services/XXX/YYY/ZZZ
```

`build.py run 14` (or `python build.py -m 14`) calls `ensure_alertmanager_webhook()` every time,
automatically regenerating `config/alertmanager/webhook_url` from this variable (gitignored —
don't edit this file by hand, since it will just get overwritten the next time `build.py` runs).
If it isn't set, a safe placeholder is written instead
(`http://raptor-alert-webhook.invalid/replace-me`) — Alertmanager still starts normally, it's
just that the resulting delivery errors show up in `docker logs raptor-alertmanager`, which is
far easier to debug than silently dropping them the way the original setup did. After changing
`.env` you need to `docker compose restart alertmanager` (or re-run `build.py -m 14`) for it to
take effect.

Why a `url_file` instead of writing the URL directly into `alertmanager.yml`: Alertmanager's
config file, like `prometheus.yml`, **doesn't support `${VAR}` environment variable expansion**,
so the URL can't be injected the usual docker-compose way — only via a file mounted into the
container. `ensure_alertmanager_webhook()` exists specifically to handle that, so all you need
to manage is `.env`, never this file directly. To switch to email instead, `alertmanager.yml`
has a commented-out `email_configs` template, which works with the `SMTP_*` variables already
present (currently empty) in `deployment/modules/.env`.

## Alert testing procedure

`test/alert_smoke_test.sh` verifies end-to-end that `GPUMemoryPressure` and `KafkaConsumerLag`
actually fire, and actually reach the receiver:

```text
synthetic metrics → Prometheus (with the real alert_rules.yml mounted)
                  → Alertmanager (with the real alertmanager.yml mounted)
                  → webhook sink (verifies it's actually received)
```

Run it:

```bash
sudo bash deployment/modules/14-monitoring/test/alert_smoke_test.sh
```

- It spins up a separate, **isolated** stack: project name `raptor-alerttest`, its own bridge
  network, host ports 18080 / 19090 / 19093 — it never touches the `raptor` network or any
  already-running container.
- The synthetic metrics (`test/fixtures/metrics`) use the exact metric and label names the real
  exporters produce. If a rule gets changed to a name the exporter doesn't actually emit, Phase
  1 will fail — that's precisely the point of this test.
- **`for:` uses the real production values (5m / 10m), not shortened for testing**, so a full
  run takes about 12 minutes.
- Cleans up automatically with `down -v` when finished; results are written to
  `test/alert_smoke_test.log`.

The three phases and what a failure in each means:

| Phase | What it checks | A failure means |
|-------|---------|---------|
| 1 | Both rules leave `inactive` within ~2 minutes (enter `pending`/`firing`) | The rule's expression doesn't match the metric name the exporter actually produces |
| 2 | Both rules enter `firing` once their `for:` expires | The rule is stuck in pending — a problem with the threshold or `for:` |
| 3 | The webhook sink actually receives both firing payloads | The rules fired, but Alertmanager's route / receiver is broken (e.g. pointed back at `null` again) |

On success, the last line reads:

```text
RESULT: PASS — GPUMemoryPressure and KafkaConsumerLag both fired and reached the receiver.
```

On failure, the log automatically includes the last 40 lines from the prometheus / alertmanager
/ fixture containers, plus the raw payload the sink received.
