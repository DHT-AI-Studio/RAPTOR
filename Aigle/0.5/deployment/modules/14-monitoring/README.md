# Monitoring & Observability

Prometheus + Grafana + Alertmanager + Loki/Promtail，另有 node-exporter、cAdvisor、
redis-exporter、kafka-exporter、elasticsearch-exporter、DCGM exporter，以及透過 OpenSearch
內建 `prometheus-exporter` plugin（見 [17-hybrid-search](../17-hybrid-search/)）暴露的叢集指標。

## OpenSearch scrape 密碼

OpenSearch 的 security plugin 對所有 REST 路徑（含 `/_prometheus/metrics`）
強制要求 HTTPS + auth，而 Prometheus 的 scrape config 格式不支援環境變數展開
（不像 docker-compose 的 `${VAR}`），所以沒辦法像其他 job 一樣直接用 `${OPENSEARCH_PASSWORD}`
注入密碼。解法是 `basic_auth.password_file`，指向一個掛載進 container、**不進 git**
的密碼檔：

```yaml
  - job_name: 'opensearch'
    basic_auth:
      username: admin
      password_file: /etc/prometheus/secrets/opensearch_password
```

這個檔案（`deployment/modules/14-monitoring/secrets/opensearch_password`）由
`python build.py -m 14` 在每次啟動這個 module 時，從 `.env` 的 `OPENSEARCH_PASSWORD`
自動重新產生（`build.py` 的 `ensure_opensearch_metrics_secret()`），目錄本身在
`.gitignore` 裡，密碼字串不會進 git 歷史。

## Scrape targets

`config/prometheus/prometheus.yml` 目前定義 **17 個 scrape job**：

| # | `job_name` | Target | `metrics_path` |
|---|-----------|--------|----------------|
| 1 | `prometheus` | `localhost:9090` | `/metrics`（預設） |
| 2 | `raptor-agent-protocol` | `raptor-agent-protocol:8030` | `/metrics` |
| 3 | `raptor-api-gateway` | `raptor-api-gateway:8012` | `/metrics` |
| 4 | `raptor-hybridsearch` | `raptor-hybridsearch-api:8000` | `/metrics` |
| 5 | `raptor-ai-lifecycle` | `raptor-ai-lifecycle-api:8010` | `/metrics` |
| 6 | `raptor-inference-service` | `raptor-inference-service:8035` | `/metrics`（Module 35） |
| 7 | `raptor-query-orchestrator` | `raptor-query-orchestrator:8000` | `/metrics` |
| 8 | `raptor-chat-service` | `raptor-chat-service:8021` | `/metrics` |
| 9 | `qdrant` | `raptor-qdrant:6333` | `/metrics` |
| 10 | `neo4j` | `raptor-neo4j:2004` | `/metrics` |
| 11 | `lakefs` | `raptor-lakefs:8000` | `/metrics` |
| 12 | `raptor-redis-exporter` | `raptor-redis-exporter:9121` | `/metrics`（預設） |
| 13 | `kafka` | `raptor-kafka-exporter:9308` | `/metrics`（預設） |
| 14 | `opensearch` | `raptor-opensearch-node1:9200` | `/_prometheus/metrics`（HTTPS + basic auth，見上方） |
| 15 | `keycloak` | `raptor-keycloak:9000` | `/metrics` |
| 16 | `raptor-elasticsearch-exporter` | `raptor-elasticsearch-exporter:9114` | `/metrics`（預設） |
| 17 | `dcgm-exporter` | `raptor-dcgm-exporter:9400` | `/metrics` |

兩個要注意的地方：

- **`node-exporter` 和 cAdvisor 有部署但沒有人在收。** 兩個 container 都定義在
  `docker-compose.yml`（`raptor-node-exporter:9100`、`raptor-cadvisor:8080`），但
  `prometheus.yml` 裡**沒有**對應的 scrape job，所以 host 層和 container 層的指標
  目前完全沒有進 Prometheus。要收的話各補一個 job 即可。
- **`dcgm-exporter` 掛在 `profiles: [gpu]` 底下**，預設不會啟動。在 `.env` 設
  `ENABLE_GPU_MONITORING=true` 之後用 `build.py -m 14` 部署，才會自動幫你加上
  `--profile gpu`（不用自己記得手動加這個 flag）。在沒有 GPU 的機器上（或這個
  變數還是 false）這個 target 顯示 DOWN、`ServiceDown` alert 對它 firing 都是
  預期行為，不代表有東西壞掉。

## Exporters

| Exporter | Target | 對應服務 |
|----------|--------|---------|
| node-exporter | `raptor-node-exporter:9100` | Host 層指標（⚠️ 未被 scrape） |
| cAdvisor | `raptor-cadvisor:8080` | Container 層指標（⚠️ 未被 scrape） |
| redis-exporter | `raptor-redis-exporter:9121` | `raptor-redis-standalone`（02-redis-cluster） |
| kafka-exporter | `raptor-kafka-exporter:9308` | `kafka-broker1/2/3`（05-kafka-cluster，可用 `KAFKA_BROKER_1/2/3` override） |
| elasticsearch-exporter | `raptor-elasticsearch-exporter:9114` | `opensearch-node1`（與下面的原生 plugin 重複，保留作第二來源） |
| DCGM exporter | `raptor-dcgm-exporter:9400` | GPU 指標（`--profile gpu`，僅 GPU 主機） |
| OpenSearch `prometheus-exporter` plugin | `raptor-opensearch-node1:9200/_prometheus/metrics` | `opensearch-node1/2`（17-hybrid-search，需 HTTPS + basic auth，見上方） |
| Keycloak 內建 metrics | `raptor-keycloak:9000/metrics` | `keycloak`（06-authentication，`KC_METRICS_ENABLED=true` 已預設開啟，management port 9000 未對外暴露） |

## Alert rules

一個 rule 檔（`config/prometheus/alert_rules.yml`）掛進 Prometheus container（`docker-compose.yml`
的 `prometheus` volumes），裡面有兩個 `groups:`。曾經拆成 `alert_rules.yml` +
`alert_rules_03.yml` 兩個檔案（`_03` 是 Raptor 0.3 時代留下的命名），後者只寫在
`prometheus.yml` 的 `rule_files` 卻沒有掛載，導致整組規則靜靜地從來沒有評估過。
併成一個檔案之後這個問題不會再發生——只有一個 `rule_files` 條目、一個 volume 掛載。

### group `raptor_basic`

| Alert | 條件 | `for` | Severity |
|-------|------|-------|----------|
| `ServiceDown` | `up == 0` | 2m | critical |
| `HighErrorRate` | `sum(rate(http_requests_total{status=~"5.."}[5m])) by (job)` / 總量 > 0.05 | 5m | warning |

### group `raptor_extra_alerts`

| Alert | 條件 | `for` | Severity | 指標來源 |
|-------|------|-------|----------|---------|
| `APILatencyHigh` | p95 `http_request_duration_seconds_bucket` > 2s | 5m | warning | FastAPI instrumentator |
| **`KafkaConsumerLag`** | `sum(kafka_consumergroup_lag) by (consumergroup, topic) > 1000` | 10m | warning | `danielqsj/kafka-exporter` |
| `PersonalIndexDLQ` | `sum(kafka_topic_partition_current_offset{topic="personal-index-requests-dlq"}) > 0` | 1m | critical | `danielqsj/kafka-exporter`（Module 25） |
| `PersonalIndexConsumerLag` | `sum(kafka_consumergroup_lag{consumergroup="personal-db-service"}) by (topic) > 1000` | 10m | warning | `danielqsj/kafka-exporter`（Module 25） |
| **`GPUMemoryPressure`** | `(DCGM_FI_DEV_FB_USED / (DCGM_FI_DEV_FB_USED + DCGM_FI_DEV_FB_FREE)) * 100 > 90` | 5m | critical | `nvcr.io/nvidia/k8s/dcgm-exporter` |
| `QdrantSearchError` | Qdrant 5xx 比例 > 1% | 5m | critical | Qdrant `/metrics` |
| `OpenSearchDown` | `up{job="opensearch"} == 0` | 2m | critical | OpenSearch plugin |
| `Neo4jDown` | `up{job="neo4j"} == 0` | 2m | critical | Neo4j `:2004/metrics` |

兩個重點規則的細節：

**`KafkaConsumerLag`** — 指標名必須是 `kafka_consumergroup_lag`，label 是 `consumergroup`
（不是 `kafka_consumer_group_lag` / `group`，那是 `danielqsj/kafka-exporter` 不會產生的名字）。
`sum by (consumergroup, topic)` 是把該 group 底下所有 partition 的 lag 加總。

**`GPUMemoryPressure`** — DCGM exporter 產生的是 `DCGM_FI_DEV_*`，不是 `nvidia_smi_*`。
`FB_USED` 和 `FB_FREE` 兩個 series 的 label 完全相同，所以二元運算可以直接對上。
注意這個 exporter 的啟用欄位來自 image 內建的 `default-counters.csv`，repo 裡沒有固定，
而 image tag 是 `:latest` — 換 image 版本前值得確認欄位還在。

## Alert 通知設定（Alertmanager）

`config/alertmanager/alertmanager.yml` 原本把所有 alert 都送到 `receiver: 'null'`
（一個沒有任何通知管道的 receiver），結果就是規則正常評估、Alertmanager 正常分組，
然後把通知直接丟掉，**沒有任何人會收到**。現在改成一個真的 `oncall` receiver：

- 所有 alert → `oncall`，`repeat_interval: 12h`
- `severity="critical"` 走 sub-route，一樣是 `oncall`，但 `repeat_interval: 4h`
- `inhibit_rules` 維持原樣：同一個 `alertname` + `job` 上，critical 會抑制 warning

**部署時要做的事：在 `.env` 設定 `ALERT_WEBHOOK_URL`。**

```
ALERT_WEBHOOK_URL=https://hooks.slack.com/services/XXX/YYY/ZZZ
```

`build.py run 14`（或 `python build.py -m 14`）每次都會呼叫
`ensure_alertmanager_webhook()`，從這個變數自動重新產生
`config/alertmanager/webhook_url`（gitignored，不要手動改這個檔案，改了也會在
下次 `build.py` 跑的時候被蓋掉）。沒設的話會寫入安全的 placeholder
（`http://raptor-alert-webhook.invalid/replace-me`），Alertmanager 仍會正常啟動，
只是送不出去的錯誤會出現在 `docker logs raptor-alertmanager`——這比原本靜悄悄丟掉
好除錯得多。改完 `.env` 之後要 `docker compose restart alertmanager`（或重新跑
`build.py -m 14`）才會生效。

為什麼是 `url_file` 而不是 alertmanager.yml 裡直接寫 URL：Alertmanager 的設定檔
和 `prometheus.yml` 一樣**不支援 `${VAR}` 環境變數展開**，所以 URL 沒辦法用一般
docker-compose 那種方式注入，只能透過掛載進容器的檔案——`ensure_alertmanager_webhook()`
就是專門做這件事的，讓你只需要管 `.env` 就好，不用碰這個檔案本身。
要改用 email 的話，`alertmanager.yml` 裡有一段註解好的 `email_configs` 範本，
搭配 `deployment/modules/.env` 既有（但目前是空的）的 `SMTP_*` 變數即可。

## Alert 測試流程

`test/alert_smoke_test.sh` 會端到端驗證 `GPUMemoryPressure` 和 `KafkaConsumerLag`
真的會燒起來、而且真的會送到 receiver：

```text
合成指標 → Prometheus（掛真正的 alert_rules.yml）
         → Alertmanager（掛真正的 alertmanager.yml）
         → webhook sink（驗證真的收到）
```

執行：

```bash
sudo bash deployment/modules/14-monitoring/test/alert_smoke_test.sh
```

- 會另外開一組**隔離**的 stack：project name `raptor-alerttest`、自己的 bridge network、
  host port 用 18080 / 19090 / 19093，不會碰到 `raptor` network 或任何正在跑的 container。
- 合成指標（`test/fixtures/metrics`）用的是 exporter 真實會產生的指標名與 label 名。
  規則如果被改成 exporter 不會產生的名字，Phase 1 就會失敗 — 這正是這個測試的目的。
- **`for:` 沿用正式值（5m / 10m），沒有為了測試縮短**，所以完整跑一次大約 12 分鐘。
- 結束會自動 `down -v` 清乾淨，結果寫在 `test/alert_smoke_test.log`。

三個階段和失敗時代表的意思：

| Phase | 檢查什麼 | 失敗代表 |
|-------|---------|---------|
| 1 | 兩條規則在 ~2 分鐘內離開 `inactive`（進到 `pending`/`firing`） | 規則的 expression 對不上 exporter 實際產生的指標名 |
| 2 | 兩條規則在 `for:` 到期後進到 `firing` | 規則卡在 pending，門檻或 `for:` 有問題 |
| 3 | webhook sink 真的收到兩個 firing payload | 規則有燒，但 Alertmanager 的 route / receiver 壞了（例如又被指回 `null`） |

通過時最後一行是：

```text
RESULT: PASS — GPUMemoryPressure and KafkaConsumerLag both fired and reached the receiver.
```

失敗時 log 會自動附上 prometheus / alertmanager / fixture 三個 container 的最後 40 行
和 sink 收到的原始 payload。

## Loki 日誌保留

`config/loki/loki-config.yaml` 預設保留 30 天：`compactor:` 區塊開啟 `retention_enabled`，
`limits_config.retention_period: 720h`。超過 30 天的 log chunk 由 Loki 的 compactor 永久刪除，無法復原。
這不是 Module 33 的 retention engine 處理的項目（它只管 PostgreSQL 表格和 Module 26 的 MemVID session 檔案）；
Module 33 的 `gdpr_retention_policies` 裡的 `loki_logs` 列（30 天）只是記錄這個政策。

```yaml
compactor:
  working_directory: /loki/compactor
  delete_request_store: filesystem
  retention_enabled: true

limits_config:
  ...
  retention_period: 720h   # 30 days
```

要改保留期：改這份設定檔的 `retention_period`，並同步改 `loki_logs` 列（`PUT /gdpr/retention/policy`）與根目錄 `.env`
的 `GD_LOKI_RETENTION_PERIOD`（純文件用途，沒有程式碼會自動套用）。改完後
`docker compose -f 14-monitoring/docker-compose.yml restart loki`（或整個 `python build.py -m 14`）套用。
