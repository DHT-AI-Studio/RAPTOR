# Benchmark Demo Guide — MLflow Run History & LLM-as-Judge

> Audience: whoever is giving a live demo of module 22 benchmark-service's two new features.
> Every command here has been live-tested on the local machine (`192.168.157.165`), and the data is still in v3.14 MLflow.
> Related requirement docs: [MLflow_Run_History.md](MLflow_Run_History.md), [LLM_JUDGEMENT_INTEGRATION.md](LLM_JUDGEMENT_INTEGRATION.md).
>
> **Runnable version**: `22-benchmark-service/tests/demo_benchmark_features.py` (uses `# %%`
> cell markers, so it can be opened directly as a notebook or pasted cell-by-cell into a .ipynb;
> `run_id` is passed automatically between variables). This document walks through the flow and
> acceptance checks with curl; the Python version additionally turns Demo 1's trend into a clearly
> rising curve using a "latency-threshold staircase" (see the Demo 1 notes below) — it screenshots
> better live.

---

## 0. Environment & prerequisites

The demo uses a benchmark-service dev instance mounted under Module 07's dev stack, with MLflow
pointed at **v3.14** (`aiml-v314-mlflow`); the LLM judge / chat pipeline hit the real raptor services.

| Component | Location | Notes |
|---|---|---|
| benchmark-service | `http://<host>:8023` | dev instance, API prefix `/api/v1` |
| MLflow UI (run history) | `http://<host>:5557` | v3.14.0, experiments named `benchmark_{schema_name}` |
| **Model under test** | `aiml-v314-api:9998` | Module 07 lifecycle-api, `qwen3-0.6B` (transformers) registered in the v314 registry; `target_pipeline=lifecycle_infer` |
| judge / embedding | `aiml-test-api:9997` | judge model `gemma3-12b` (via `BM_INFERENCE_URL`), deliberately kept on a separate endpoint from the model under test |
| chat pipeline | `raptor-chat-service:8021/api/v1` | Demo 2 only (best-effort), see §5 |

### 0.1 Starting the benchmark-service dev stack

```bash
cd deployment/modules/22-benchmark-service
docker compose -f docker-compose.aiml-test.dev.yml up -d --build
curl -s http://localhost:8023/health          # {"status":"ok"}
```

Compose has all the connections preset (postgres auto-creates the `benchmark` DB, redis is
included, multiple networks, `BM_MLFLOW_URL` / `BM_INFERENCE_URL` (judge) / `BM_LIFECYCLE_INFER_URL`
(model under test) / `BM_CHAT_URL` / `BM_JUDGE_MODEL`) — no extra configuration needed.

### 0.1a Model under test: the `lifecycle_infer` target pipeline

Benchmark's original target pipelines (chat/search/rag/classify/local_infer) are none of them
"directly test one registered model." To benchmark a model registered in Module 07, use the newly
added `target_pipeline=lifecycle_infer`: it hits `BM_LIFECYCLE_INFER_URL` (= aiml-v314-api:9998)'s
`/inference/infer`, with the model and engine specified by the run's `config_override` — so the
same schema can be re-run against a different model.

```jsonc
// schema
"target_pipeline": "lifecycle_infer",
"test_cases": [{"id": "tc1", "input": {"inputs": "What is the capital of France?"}}]
// when submitting a run (config_override)
{"model_name": "qwen3-0.6B", "engine": "transformers", "temperature": 0.7}
```

> `qwen3-0.6B` is a small 0.6B model with mediocre answer quality — perfect for letting the LLM
> judge show its discriminative power (a bad answer gets a low score). The judge is still
> `gemma3-12b`, on a separate endpoint (aiml-test-api:9997).

### 0.2 MLflow v3.14's security middleware (**required, or charts stay blank / runs fail to write**)

MLflow **3.5.0 onward** ships built-in DNS-rebinding protection middleware (CVE-2025-14279); v3.3.2
doesn't have this layer. It has **two independent** checks, both of which return 403:

| Check | Blocks what | Consequence if unset |
|---|---|---|
| `MLFLOW_SERVER_CORS_ALLOWED_ORIGINS` | Browser POSTs to `/ajax-api` (carrying an Origin header) — only localhost is allowed by default | Opening the UI via the host IP loads the page, **but the charts stay blank** (the `runs/search` call that fetches the data gets 403'd) |
| `MLFLOW_SERVER_ALLOWED_HOSTS` | The Host header on every request (**including the SDK's server-to-server calls**) — only localhost and bare private IP ranges are recognized by default, not container names | benchmark-service writing via `aiml-v314-mlflow:5557` gets 403'd — **the run still completes, but `mlflow_run_id` silently ends up null** |

Both variables have already been added to
[docker-compose.mlflow.v3.14.dev.yml](../docker-compose.mlflow.v3.14.dev.yml)
and [.env.v3.14.dev](../.env.v3.14.dev). If MLflow hasn't been started with them yet, run:

```bash
cd deployment/modules/07-ai-ml-services
docker compose --env-file .env.v3.14.dev -f docker-compose.mlflow.v3.14.dev.yml up -d mlflow
docker exec aiml-v314-mlflow env | grep MLFLOW_SERVER   # should show both variables
```

> Current values: `CORS_ALLOWED_ORIGINS=http://192.168.157.165:5557`, `ALLOWED_HOSTS=*`.
> `ALLOWED_HOSTS=*` is because this is an isolated internal test network with several container
> aliases connecting in; **a real external-facing environment should use an explicit host list
> instead**. See the `mlflow-v314-security-middleware` memory entry for details.

No need to actually open a browser — sending a simulated browser Origin header is enough to verify
whether the CORS layer allows the request through:

```bash
# Chart data source: use the host IP as Origin, simulating a browser POST to /ajax-api
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://localhost:5557/ajax-api/2.0/mlflow/runs/search \
  -H "Content-Type: application/json" -H "Origin: http://192.168.157.165:5557" -d '{"experiment_ids":["0"]}'
# → 200 (was 403 before CORS_ALLOWED_ORIGINS was set)
```

The most direct way to verify the Host-header layer (the SDK write path) is to run a benchmark run
and check whether `mlflow_run_id` gets written (see the Demo 1 acceptance check) — a non-null value
means it went through.

---

## Demo 1 — MLflow Run History: score trend chart

**Scenario**: a benchmark operator reruns the same schema repeatedly (simulating "tweak the prompt
/ swap the model, then retest"), and checks the MLflow UI to see whether the score improves across
iterations — that's this feature's ultimate value.

**This schema deliberately uses pure rule-based scoring** (`keyword_match` + `latency_threshold`),
with no LLM involved, so the scores are stable and predictable — a good baseline for a trend.

```bash
# 1. Upload the schema, note the returned id
curl -s -X POST http://localhost:8023/api/v1/benchmark/schemas \
  -H "Content-Type: application/json" -d '{
  "name": "mlflow-e2e", "version": "1.0", "target_pipeline": "chat",
  "test_cases": [
    {"id": "tc1", "input": {"message": "hello, how are you?", "user_id": "bench"}, "expected_keywords": ["hello"]},
    {"id": "tc2", "input": {"message": "what is 2+2?", "user_id": "bench"}, "expected_keywords": ["4"]}
  ],
  "scoring_schema": {"dimensions": [
    {"name": "keywords", "weight": 0.5, "method": "keyword_match"},
    {"name": "latency",  "weight": 0.5, "method": "latency_threshold", "max_ms": 30000}]}
}'

# 2. Fire off a few runs in a row (live, you can pretend to "tweak something" between each)
SCHEMA_ID=<id returned above>
for i in 1 2 3; do
  curl -s -X POST http://localhost:8023/api/v1/benchmark/runs \
    -H "Content-Type: application/json" -d "{\"schema_id\": \"$SCHEMA_ID\"}"
  echo; sleep 20
done
```

**Acceptance check**

1. `GET /api/v1/benchmark/schemas/{id}/runs` — every run carries `mlflow_run_id`:
   ```bash
   curl -s http://localhost:8023/api/v1/benchmark/schemas/$SCHEMA_ID/runs | python3 -m json.tool
   ```
2. Open `http://<host>:5557` → experiment **`benchmark_mlflow-e2e`** → select several runs →
   view the `aggregate_score` line chart. Each run carries three metrics — `aggregate_score`,
   `score_keywords`, `score_latency` — and four tags: `schema_id` / `schema_version` /
   `target_pipeline` / `test_case_count`.

> To show "a new point appearing live," just fire another run and refresh the UI.

---

## Demo 2 — Best-effort degradation (MLflow going down doesn't affect benchmark)

**Scenario**: verify the requirement that "If MLflow is unreachable, the service logs a warning
without failing the benchmark run" — the run must still complete, just without being recorded.

```bash
# 1. Stop MLflow, simulating the tracking server being unreachable
docker stop aiml-v314-mlflow

# 2. Rerun a run — still completes, just with mlflow_run_id as null
RUN_ID=$(curl -s -X POST http://localhost:8023/api/v1/benchmark/runs \
  -H "Content-Type: application/json" -d "{\"schema_id\": \"$SCHEMA_ID\"}" \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['run_id'])")
sleep 20
curl -s http://localhost:8023/api/v1/benchmark/runs/$RUN_ID \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print('status:', d['status'], '| mlflow_run_id:', d['mlflow_run_id'])"
# → status: completed | mlflow_run_id: None

# 3. The service logs a warning, not an error
docker logs benchmark-dev-service --tail 20 | grep "MLflow logging failed"

# 4. Restore
docker start aiml-v314-mlflow
```

**Key acceptance point**: `status` is still `completed`, the score is still computed, only
`mlflow_run_id` is null, and the log is at `WARNING` level. This proves MLflow logging is a
best-effort side path, not on benchmark's critical path.

---

## Demo 3 — LLM-as-Judge: subjective quality scoring

**Scenario**: the schema has nothing to keyword-match against, so an LLM scores the answer against
a rubric instead (e.g., "does it stay on topic?"). This is a dimension keyword/regex simply can't
cover.

```bash
# Target under test uses lifecycle_infer (qwen3-0.6B), judge uses gemma3-12b
curl -s -X POST http://localhost:8023/api/v1/benchmark/schemas \
  -H "Content-Type: application/json" -d '{
  "name": "llm-judge-demo", "version": "1.0", "target_pipeline": "lifecycle_infer",
  "test_cases": [
    {"id": "tc1", "input": {"inputs": "What is the capital of France?"}},
    {"id": "tc2", "input": {"inputs": "Explain photosynthesis in one sentence."}}
  ],
  "scoring_schema": {"dimensions": [
    {"name": "relevance", "weight": 1.0, "method": "llm_judge",
     "rubric": "Score 1-5: does the answer correctly and directly address the question?"}
  ], "score_range": [1, 5]},
  "judge_model": "gemma3-12b"
}'
# Note the id, and when submitting the run, pass config_override to pick the model under test:
#   -d '{"schema_id":"<id>","config_override":{"model_name":"qwen3-0.6B","engine":"transformers","temperature":0.7}}'
```

**Acceptance check**

```bash
curl -s http://localhost:8023/api/v1/benchmark/runs/<run_id> | python3 -m json.tool
```

- `scores_per_case[].output` is **the real answer from the model under test, qwen3-0.6B**;
- `per_dimension.relevance` is the judge model gemma3-12b's score, converted to [0, 1]. qwen3-0.6B
  is a small model whose answers are often incomplete or off-topic, so the judge assigns **a mix of
  high and low** scores (in testing: 1.0 for a correctly-answered question, 0.0 for an off-topic
  one) — which is exactly what shows off the judge's discriminative power, more convincing than a
  row of all-1.0s;
- MLflow experiment `benchmark_llm-judge-demo` shows a `score_relevance` metric.

> **The single most convincing line for this demo**: the subjective score the LLM assigned also
> flows into Demo 1's trend chart — tying the run-history and LLM-judge features together.
>
> Note: qwen3-0.6B uses `temperature=0.7`, so answers (and scores) will vary across repeated runs
> of the same question — that's expected behavior, not a bug. Lower the temperature for more
> stability (though transformers won't accept exactly 0).

---

## Demo 4 — Pairwise: two models head-to-head

**Scenario**: the same set of questions run against two different models, asking the LLM "which
model answered better" — the standard way to do an A/B model comparison. `compare` requires both
runs to belong to the **same schema**, so this uses `config_override` to swap models while keeping
the schema fixed. Models under test: `qwen3-0.6B` (weaker) vs. `gemma-3-1b-it` (1B instruct,
stronger).

```bash
# 1. Create a 3-question model-compare schema (with an llm_judge relevance dimension)
curl -s -X POST http://localhost:8023/api/v1/benchmark/schemas \
  -H "Content-Type: application/json" -d '{
  "name": "model-compare", "version": "1.0", "target_pipeline": "lifecycle_infer",
  "test_cases": [
    {"id": "q1", "input": {"inputs": "What is the capital of Japan?"}},
    {"id": "q2", "input": {"inputs": "What is 12 multiplied by 12?"}},
    {"id": "q3", "input": {"inputs": "List three planets in our solar system."}}
  ],
  "scoring_schema": {"dimensions": [
    {"name": "relevance", "weight": 1.0, "method": "llm_judge",
     "rubric": "Score 1-5: is the answer correct and does it directly address the question?"}
  ], "score_range": [1, 5]},
  "judge_model": "gemma3-12b"
}'
# 2. Submit two runs against the same schema, each with a different model (config_override):
#   run A: {"schema_id":"<id>","config_override":{"model_name":"qwen3-0.6B","engine":"transformers","temperature":0.7}}
#   run B: {"schema_id":"<id>","config_override":{"model_name":"gemma-3-1b-it","engine":"transformers","temperature":0.7}}
# 3. Pairwise comparison
curl -s "http://localhost:8023/api/v1/benchmark/runs/<run_a>/compare/<run_b>?pairwise=true" \
  | python3 -m json.tool
```

**Acceptance check**: pairwise's `winner` (A=run_a=qwen, B=run_b=gemma) per question, plus
`b_win_rate`; `delta_aggregate` is the absolute B−A score difference. In testing, gemma-3-1b-it won
(B won 2, tied 1, `b_win_rate≈0.67`, aggregate A=0.67 vs B=1.0) — a clear result. The `.py` version
also prints both models' per-question answers side by side, which reads most clearly live.

> Internally, positional bias is cancelled out with swap + agree (the same answer pair is asked in
> both orders, and only counts as a win if both agree) — see `run_manager._debiased_pairwise`. This
> is existing machinery; just demonstrate it in passing.

---

## 5. Pitfalls hit along the way (check this list before demoing on a new machine)

| Pitfall | Symptom | Fix |
|---|---|---|
| MLflow v3.14 security middleware | Blank charts / `mlflow_run_id` null | Set `MLFLOW_SERVER_CORS_ALLOWED_ORIGINS` + `MLFLOW_SERVER_ALLOWED_HOSTS` (§0.2) |
| Inference port | judge / embedding unreachable | `aiml-test-api`'s `PORT_AI_LIFECYCLE_API` env shows 8010, which is stale — it actually listens on **9997** |
| Judge model name | `ModelNotFoundError: qwen3.5:9b` | Module 07's registered name is `qwen3.5:9b-ollama`, not `qwen3.5:9b` |
| **A reasoning model as judge** | llm_judge scores come back **all 0** | `judge.py._infer()` hardcodes `max_length` to 16; the qwen3.5 family are reasoning models, so all 16 tokens get consumed by the invisible reasoning process, leaving `response` empty. Switched to the non-reasoning **`gemma3-12b`** (answers within a few tokens) |
| chat pipeline URL | `/chat` 404 or connection refused | `raptor-chat-service` listens on **8021** with route prefix `/api/v1`, so `BM_CHAT_URL=http://raptor-chat-service:8021/api/v1` |
| API path | `{"detail":"Not Found"}` | the whole benchmark API sits under the `/api/v1` prefix |
| The live network | dev service can't reach chat | the live raptor services are on the **`raptor0.3` bridge** network — no services live on the `raptor` overlay |
| **v314-api container GPU disconnect** | inference on the model under test throws `CUDA error: device busy/unavailable`; `nvidia-smi` inside the container reports `Failed to initialize NVML` | the host GPU is actually healthy. Usually a host-side event like `daemon-reload` reset the container's cgroup device permissions; `docker restart aiml-v314-api` remounts it |
| **transformers temperature** | `temperature (=0.0) has to be strictly positive` | ollama allows `temperature=0`, transformers doesn't; lifecycle_infer defaults to `0.7` — don't pass 0 |
| **best-effort demo target** | stopping MLflow also broke the model under test, scores dropped to zero | Module 07 uses MLflow as its model registry — Demo 2 was stopping that same MLflow. Switched Demo 2's target to chat (unrelated to that MLflow) so it cleanly demonstrates "only logging fails" |

All of the above is already configured in `docker-compose.aiml-test.dev.yml` — when demoing on a
different machine, just check off this table item by item.
