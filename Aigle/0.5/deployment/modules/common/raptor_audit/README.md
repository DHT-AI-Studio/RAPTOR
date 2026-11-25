# raptor_audit：共用 audit event library

guardrail、GDPR/CCPA、gRPC、audio 這幾個 initiative 共用同一份正規化、隱私最小化的 audit event schema。各模組都 import 這個 library，不要自行實作。

目前已採用：Module 23（guardrail-service）、Module 33（privacy-compliance-service：每個結束的 erasure job 寫一筆 `gdpr.erasure`；export、DSAR、更正、限制處理、反對處理、consent 變更、legal hold 各寫一筆自己的 event，見下方）。

## Schema

| 欄位 | 說明 |
|---|---|
| `event_id` | UUID |
| `timestamp` | ISO-8601（UTC） |
| `source_module` | 模組編號，例如 `23` |
| `source_service` | 服務名稱，例如 `guardrail-service` |
| `event_type` | `guardrail.decision`、`gdpr.erasure`、`grpc.call`、`audio.transcript`；Module 33 的隱私動作：`gdpr.export`、`gdpr.access`、`gdpr.rectification`、`gdpr.restriction`、`gdpr.objection`、`gdpr.consent`、`gdpr.legal_hold` |
| `actor_id_hash` | user id 的 SHA-256，**不存原始的 Keycloak sub** |
| `resource_id` | `request_id` 或 `session_id` |
| `outcome` | `allow`、`block`、`redact`、`erase`、`error` |
| `policy_version` | 判定時使用的 policy 版本 |
| `content_hash` | 受檢文字的 SHA-256，**不存原文** |
| `metadata` | 其他非敏感資訊（JSON） |
| `retention_days` | 預設 365 |

**隱私最小化**：
- `build_event()` 只把 `actor_id` 與 `content` 拿來算 hash，不會放進 event。
- `AuditEvent` 設定為 `extra="forbid"`，不能夾帶其他欄位。

`outcome` 沒有對應的值（例如 guardrail 的 `warn`），就記為 `allow`，並把原始判定寫在 `metadata.decision`。

Module 33 的隱私動作 event（`gdpr.erasure` 以外）同樣沿用這組 `outcome`：`allow` 表示「該動作已執行」，`error` 表示「執行失敗」；實際做了什麼寫在 `metadata.action`（例如 `grant`／`withdraw`、`restrict`／`lift`、`place`／`release`）。

## 寫入（PostgreSQL `audit_events`）

Jira 允許同步呼叫者用 PostgreSQL 表取代 Kafka topic，所以目前先寫 PostgreSQL。

- `AUDIT_EVENTS_DDL`：由採用的模組在啟動時執行，可重複執行。
- `insert_event(conn, event)`：在呼叫端自己的 connection 或 transaction 中寫入一筆，可以和模組自己的資料列放在同一個 transaction。
- `PostgresAuditWriter(get_pool).emit(event, fail_closed=False)`，失敗時依模式處理：
  - `fail_closed=False`：operational audit，**fail-open**，只記 warning，不中斷呼叫端。
  - `fail_closed=True`：erasure 確認用，**fail-closed**，拋出 `AuditWriteError`。

library 本身只依賴 pydantic。資料庫連線由呼叫端傳入，所以不需要 asyncpg。

## 模組怎麼採用（以 Module 23 為例）

1. **compose**：build context 改為 `deployment/modules/`：

   ```yaml
   build:
     context: ..
     dockerfile: NN-xxx/Dockerfile
   ```

2. **Dockerfile**：COPY 路徑都要加上模組目錄的前綴，並把 library 複製進 WORKDIR：

   ```dockerfile
   COPY common/raptor_audit ./raptor_audit/
   ```

3. **`pytest.ini`**：加上 `pythonpath = . ../common`，讓本機與 CI（在模組目錄執行 `pytest`）都能 `import raptor_audit`。
4. **啟動時**：執行 `AUDIT_EVENTS_DDL`。

library 的單元測試目前放在 `23-guardrail-service/tests/unit/test_raptor_audit.py`。CI 只會跑有變更的模組的 `tests/unit/`。

## 尚未完成（follow-up）

- Kafka sink：`audit_event` topic，`acks=all`。
- Module 34 改用這個 library。
- ~~Module 33 的其他隱私動作（consent 變更、export、restriction 等）需要新的 `event_type`。~~ 已完成：見上方 `event_type`。
- ~~Module 33 的 retention engine 依 `retention_days` 清理 `audit_events`。~~ 已完成：每日清理 Module 33 自己的與 `guardrails` 資料庫的 `audit_events`，依每一筆自己的 `retention_days`。
- CI：`common/` 的變更也要觸發採用這個 library 的模組的測試與 build。
