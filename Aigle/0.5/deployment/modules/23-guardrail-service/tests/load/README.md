# Load test：policy-aware check（GRD-X1）

驗收標準：**100 個並發請求，p95 ≤ 200 ms（包含 guard model 判定）**。

- 量測對象：`POST /guardrail/policy/decide` 與 `POST /api/v0.2/check/input`。兩者跑的是同一套判定。
- 內容會輪流送出三種：安全的、guard model 判定違規的，以及會被 regex 命中的。

這個測試**不在 CI 中執行**（CI 只跑 `tests/unit/`），需要一個已經載入 guard model 的 guardrail service。`locust` 沒有放進 `requirements.txt`，請另外安裝。

## 執行方式

從 raptor docker network 內執行，直接打容器的 8026 port，不經過 host 的 8023：

```bash
docker run --rm -it --network raptor \
  -v "$PWD/tests/load:/mnt/load" -w /mnt/load \
  locustio/locust:2.31.8 \
  -f locustfile.py --host http://raptor-guardrail-service:8026 \
  -u 100 -r 20 -t 2m --headless --csv /mnt/load/out
```

或在已安裝 `locust` 的環境中：

```bash
pip install locust==2.31.8
locust -f tests/load/locustfile.py --host http://raptor-guardrail-service:8026 \
  -u 100 -r 20 -t 2m --headless --csv tests/load/out
```

- `-u 100`：100 個並發使用者。
- `-r 20`：每秒新增 20 個使用者。
- `-t 2m`：持續 2 分鐘。

**注意**：請在測試環境執行，不要對正式服務施壓。

## 讀取結果

`out_stats.csv` 中每個端點各有一列，`95%` 欄位就是 p95（毫秒）。

- 兩個端點的 p95 都要記錄下來。
- 同時記下當時的環境：GPU、guard model（`GUARD_MODEL*`）、`GUARD_CONFLICT_STRATEGY`、active policy。
- 啟用多個 guard model 時，延遲以最慢的那個為準；結果附在 PR 說明中。

`out_*.csv` 是量測結果，不要 commit。
