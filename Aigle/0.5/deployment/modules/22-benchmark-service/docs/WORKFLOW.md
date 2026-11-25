# Benchmark Service — 訓練 → 服務 → 評分 → 比較 完整流程

Module 22（Benchmark Service）提供一套機制：**對 Raptor 的 AI 輸出定義「考卷」，量化評分，並客觀比較「改動前後」變好還是變壞。** 最完整的用途是搭配 Module 16 評估微調模型，形成 AutoResearch 式的閉環。

- Base URL：`http://localhost:8022/api/v1/benchmark`
- 互動式 API 文件：`http://localhost:8022/docs`
- 評審模型（`llm_judge` / pairwise 用）：`qwen3.5:9b`（透過 Module 07，可在 schema 用 `judge_model` 覆蓋）

---

## 1. 整體流程

```text
┌─ Module 16 訓練 ─────────────┐        ┌─ Module 22 Benchmark ───────────────────┐
│  submit 訓練 job              │        │  1. 上傳 schema（考卷：測試題 + 評分標準） │
│    → 微調 checkpoint          │        │  2. 對模型跑 run（config_override.model_path） │
│      (final_model_path,       │──┐     │  3. executor → Module 16 /inference/infer   │
│       HF 格式)                │  │     │       → output                              │
└──────────────────────────────┘  │     │  4. scorer 評分 → aggregate_score (0~1)     │
                                   │     │  5. /compare 比較兩次 run（算術 or pairwise）│
   模型放在共用 aiml NFS ─────────┘     └─────────────────────────────────────────────┘
   (/app/tmp/models/...)
```

**核心概念**：考卷（schema）固定不動，只換「被評的東西」（`config_override.model_path`），跑前後兩次，用 `/compare` 看差異。

---

## 2. 快速上手（評 / 比微調模型）

### Step 1 — 上傳一份 schema（只做一次，之後重用）

```bash
curl -X POST http://localhost:8022/api/v1/benchmark/schemas \
  -H "Content-Type: application/json" -d '{
  "name": "QA quality v1",
  "target_pipeline": "local_infer",
  "test_cases": [
    {"id":"q1","input":{"inputs":"What is the capital of France? Answer in one sentence."},
     "expected_keywords":["Paris"]},
    {"id":"q2","input":{"inputs":"Name the largest planet in our solar system."},
     "expected_keywords":["Jupiter"]}
  ],
  "scoring_schema": {"dimensions": [
    {"name":"keywords","weight":0.4,"method":"keyword_match"},
    {"name":"quality", "weight":0.4,"method":"llm_judge",
     "params":{"rubric":"Score 1-5: is the answer factually correct and clearly stated?"}},
    {"name":"latency", "weight":0.2,"method":"latency_threshold","params":{"max_ms":60000}}
  ]}
}'
# → {"id":"<schema_id>", ...}
```

### Step 2 — 對模型 A 跑一次（baseline，改動前）

```bash
curl -X POST http://localhost:8022/api/v1/benchmark/runs \
  -H "Content-Type: application/json" -d '{
  "schema_id":"<schema_id>",
  "config_override":{"model_path":"/app/tmp/models/<模型A>","max_new_tokens":48,"temperature":0.2}
}'
# → {"run_id":"<run_A>","status":"queued"}   （非同步，立即回）
```

### Step 3 — 看結果

```bash
curl http://localhost:8022/api/v1/benchmark/runs/<run_A>
# → aggregate_score、每維度分數、每題的 output
```

### Step 4 — 改動後（換模型 B / 重訓）再跑一次 → 比前後

```bash
# 對模型 B 跑 → 得 <run_B>，然後：
curl "http://localhost:8022/api/v1/benchmark/runs/<run_A>/compare/<run_B>"
# → delta_aggregate（B - A）、逐維度、逐題差異

# 更靈敏的相對比較（LLM 逐題頭對頭 + 位置去偏）：
curl "http://localhost:8022/api/v1/benchmark/runs/<run_A>/compare/<run_B>?pairwise=true"
```

---

## 3. Schema 結構

```yaml
name: "..."                    # 考卷名稱
version: "1.0"
target_pipeline: local_infer   # 見下方「pipeline」
target_url: null               # 可選：覆蓋預設服務 URL
judge_model: null              # 可選：覆蓋 llm_judge 的評審模型（預設 qwen3.5:9b）

test_cases:                    # 一組固定的測試題
  - id: q1
    input: { inputs: "..." }   # 送進 pipeline 的輸入（欄位依 pipeline 而定）
    expected_keywords: ["..."] # 可選，keyword_match 用
    expected_answer: "..."     # 可選，cosine_similarity / exact_match 用

scoring_schema:
  dimensions:                  # 各維度分數 × weight 加總 = aggregate（weight 總和須=1.0）
    - name: quality
      weight: 0.5
      method: llm_judge
      params: { rubric: "..." }
  aggregate: weighted_sum
  score_range: [1, 5]          # llm_judge 原始分正規化到 0~1 用
```

### `target_pipeline` 可選值

| pipeline | 打的服務 | input 欄位 | 備註 |
|----------|----------|-----------|------|
| `local_infer` | Module 16 `/inference/infer` | `{inputs}` | **評本地/微調模型**；`model_path` 由 `config_override` 帶入 |
| `chat` | chat-service `/chat` | `{message, user_id?}` | 對話 / RAG |
| `search` | hybridsearch `/search` | `{query, top_k?}` | 混合搜尋 |
| `rag` | agent-protocol `/query` | `{question}` | RAG pipeline |
| `classify` | query-orchestrator `/classify` | `{query}` | 意圖分類 |

---

## 4. 評分方法（`method`）

| 類別 | method | 需要的 params / 欄位 | 說明 |
|------|--------|---------------------|------|
| 詞彙 | `keyword_match` | `expected_keywords` | 命中比例 |
| 詞彙 | `contains_all` / `contains_any` | `expected_keywords` 或 `params.keywords` | 全中 / 任一 → 1.0 |
| 詞彙 | `exact_match` | `params.expected` 或 `expected_answer` | 完全相符（可設 case_sensitive / strip）|
| 數值 | `numeric_tolerance` | `params.expected`, `params.tolerance` | 抓輸出第一個數字比對 |
| 結構 | `regex_match` | `params.pattern` | 符合正則 → 1.0 |
| 語意 | `cosine_similarity` | `expected_answer` | 與參考答案的語意相似度（需 Module 07 embedding）|
| 質性 | `llm_judge` | `params.rubric` | 用自然語言 rubric 讓 LLM 打分（最通用）|
| 效能 | `latency_threshold` | `params.max_ms` | 回應時間 ≤ 門檻 → 1.0 |

> 要加新方法：在 `app/services/scoring/builtins.py` 用 `@register_scorer("name")` 註冊一個函式即可，schema 立刻能用（不用改 dispatch）。

**aggregate_score** = Σ(dimension.weight × 該維度 0~1 分數)，落在 `[0, 1]`。

---

## 5. 兩種比較模式（重要）

| 模式 | 怎麼比 | 成本 | 適用 |
|------|--------|------|------|
| 預設 `/compare` | 對兩次 run **已算好的分數**相減 | 便宜 | 差異明顯時 |
| `?pairwise=true` | 逐題把兩個 output **並排**讓 LLM 判誰好（A/B 交換各問一次去偏）| 較貴 | **抓小改動、絕對分飽和時** |

**為什麼要 pairwise**：LLM 對「絕對打分」不穩、對「相對比較」很準。當兩個模型都拿滿分時，算術相減 = 0（看不出差別），但 pairwise 並排比較能分辨誰更好。

### 實例（base vs 微調，本服務實測）

| | q1 輸出 | 算術 compare | pairwise |
|---|---|---|---|
| A（base）| `Paris` | \ | \ |
| B（微調）| `The capital of France is Paris.` | delta = **0**（兩者都滿分）| q1 **B 勝** |

→ 微調讓答案更完整，**絕對分抓不到（都封頂）、pairwise 抓到了**。做 before/after 時優先看 pairwise。

---

## 6. API 一覽

| Method | Path | 說明 |
|--------|------|------|
| `POST` | `/schemas` | 上傳 schema（JSON 或 `Content-Type: application/yaml`）|
| `GET` | `/schemas` | 列出所有 schema |
| `GET` | `/schemas/{id}` | 取單一 schema |
| `DELETE` | `/schemas/{id}` | 刪除（連同其 runs）|
| `POST` | `/runs` | 提交 run：`{schema_id, config_override?}` |
| `GET` | `/runs/{run_id}` | run 狀態 + 分數 + 每題 output |
| `GET` | `/schemas/{id}/runs` | 某 schema 的 run 歷史 |
| `GET` | `/runs/{a}/compare/{b}[?pairwise=true]` | 比較兩次 run |

---

## 7. 典型工作流總結

```text
（一次性）上傳 schema —— 定義「好」的標準，之後保持不變
    │
改動前 → 跑一次 → run_A（baseline）
    │
做改動（換模型 / 重訓 / 調 prompt / 調參數）
    │
改動後 → 跑同一份 schema → run_B
    │
/compare run_A vs run_B
    ├─ delta_aggregate > 0 → 變好
    ├─ delta_aggregate < 0 → 變壞
    └─ ?pairwise=true → 小改動更靈敏，看 B 勝率
```

**關鍵原則**：schema（考卷）必須固定、每次只改一個變因、差異要大於評分雜訊（尤其 `llm_judge` 有隨機性，微小差異優先用 pairwise 或多跑幾次）。

---

## 8. 相依 / 環境

| 需要 | 用途 |
|------|------|
| Module 02 Redis（standalone）| run 即時狀態 |
| Module 03 PostgreSQL（`benchmark` DB）| schema / run 歷史 |
| Module 07 AI Lifecycle | `llm_judge` / pairwise / `cosine_similarity` |
| Module 16 Training Service | `local_infer` 服務微調模型（GPU）|

啟動：`bash deploy.sh -m 22`（相依 02/03/07/13）。
