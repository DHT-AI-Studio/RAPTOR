# Benchmark Service — Train → Serve → Score → Compare, End to End

Module 22 (Benchmark Service) provides a mechanism to: **define a "test" against Raptor's AI output, score it quantitatively, and objectively compare whether "before vs. after a change" got better or worse.** Its fullest use is pairing with Module 16 to evaluate fine-tuned models, forming an AutoResearch-style closed loop.

- Base URL: `http://localhost:8022/api/v1/benchmark`
- Interactive API docs: `http://localhost:8022/docs`
- Judge model (used by `llm_judge` / pairwise): `qwen3.5:9b` (via Module 07; overridable per-schema with `judge_model`)

---

## 1. Overall Flow

```text
┌─ Module 16 Training ──────────┐        ┌─ Module 22 Benchmark ───────────────────┐
│  submit a training job        │        │  1. upload a schema (test cases + scoring rules) │
│    → fine-tuned checkpoint    │        │  2. run against a model (config_override.model_path) │
│      (final_model_path,       │──┐     │  3. executor → Module 16 /inference/infer   │
│       HF format)               │  │     │       → output                              │
└──────────────────────────────┘  │     │  4. scorer scores it → aggregate_score (0~1) │
                                   │     │  5. /compare compares two runs (arithmetic or pairwise) │
   the model lives on the shared  │     └─────────────────────────────────────────────┘
   aiml NFS (/app/tmp/models/...) ─┘
```

**Core idea**: the "test" (schema) stays fixed — only "what's being evaluated" (`config_override.model_path`) changes. Run it twice, before and after, and use `/compare` to see the difference.

---

## 2. Quick Start (evaluating / comparing fine-tuned models)

### Step 1 — Upload a schema (do this once, reuse afterward)

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

### Step 2 — Run against model A (baseline, before the change)

```bash
curl -X POST http://localhost:8022/api/v1/benchmark/runs \
  -H "Content-Type: application/json" -d '{
  "schema_id":"<schema_id>",
  "config_override":{"model_path":"/app/tmp/models/<model_A>","max_new_tokens":48,"temperature":0.2}
}'
# → {"run_id":"<run_A>","status":"queued"}   (asynchronous, returns immediately)
```

### Step 3 — Check the results

```bash
curl http://localhost:8022/api/v1/benchmark/runs/<run_A>
# → aggregate_score, per-dimension scores, per-question output
```

### Step 4 — After the change (switch to model B / retrain), run again → compare before/after

```bash
# run against model B → get <run_B>, then:
curl "http://localhost:8022/api/v1/benchmark/runs/<run_A>/compare/<run_B>"
# → delta_aggregate (B - A), per-dimension and per-question differences

# a more sensitive relative comparison (LLM head-to-head per question + position-bias correction):
curl "http://localhost:8022/api/v1/benchmark/runs/<run_A>/compare/<run_B>?pairwise=true"
```

---

## 3. Schema Structure

```yaml
name: "..."                    # test name
version: "1.0"
target_pipeline: local_infer   # see "pipeline" below
target_url: null               # optional: override the default service URL
judge_model: null              # optional: override llm_judge's judge model (default qwen3.5:9b)

test_cases:                    # a fixed set of test questions
  - id: q1
    input: { inputs: "..." }   # the input sent into the pipeline (fields depend on the pipeline)
    expected_keywords: ["..."] # optional, used by keyword_match
    expected_answer: "..."     # optional, used by cosine_similarity / exact_match

scoring_schema:
  dimensions:                  # each dimension's score × weight, summed = aggregate (weights must sum to 1.0)
    - name: quality
      weight: 0.5
      method: llm_judge
      params: { rubric: "..." }
  aggregate: weighted_sum
  score_range: [1, 5]          # used to normalize llm_judge's raw score to 0~1
```

### `target_pipeline` allowed values

| pipeline | service called | input fields | notes |
|----------|----------|-----------|------|
| `local_infer` | Module 16 `/inference/infer` | `{inputs}` | **evaluates a local/fine-tuned model**; `model_path` is passed via `config_override` |
| `chat` | chat-service `/chat` | `{message, user_id?}` | conversation / RAG |
| `search` | hybridsearch `/search` | `{query, top_k?}` | hybrid search |
| `rag` | agent-protocol `/query` | `{question}` | RAG pipeline |
| `classify` | query-orchestrator `/classify` | `{query}` | intent classification |

---

## 4. Scoring Methods (`method`)

| Category | method | Required params / fields | Description |
|------|--------|---------------------|------|
| Lexical | `keyword_match` | `expected_keywords` | Hit ratio |
| Lexical | `contains_all` / `contains_any` | `expected_keywords` or `params.keywords` | All match / any match → 1.0 |
| Lexical | `exact_match` | `params.expected` or `expected_answer` | Exact match (case_sensitive / strip configurable) |
| Numeric | `numeric_tolerance` | `params.expected`, `params.tolerance` | Extracts the first number in the output for comparison |
| Structural | `regex_match` | `params.pattern` | Matches the regex → 1.0 |
| Semantic | `cosine_similarity` | `expected_answer` | Semantic similarity to the reference answer (requires Module 07 embedding) |
| Qualitative | `llm_judge` | `params.rubric` | Uses a natural-language rubric to have an LLM assign a score (most general-purpose) |
| Performance | `latency_threshold` | `params.max_ms` | Response time ≤ threshold → 1.0 |

> To add a new method: register a function with `@register_scorer("name")` in `app/services/scoring/builtins.py` — it's immediately usable from a schema (no dispatch changes needed).

**aggregate_score** = Σ(dimension.weight × that dimension's 0~1 score), falling in `[0, 1]`.

---

## 5. Two Comparison Modes (important)

| Mode | How it compares | Cost | Best for |
|------|--------|------|------|
| Default `/compare` | Subtracts the **already-computed scores** of two runs | Cheap | When the difference is clear |
| `?pairwise=true` | Places both outputs for each question **side by side** and has an LLM judge which is better (asked once with A/B swapped to correct for position bias) | More expensive | **Catching small changes, when absolute scores are saturated** |

**Why pairwise matters**: an LLM is unreliable at "absolute scoring" but quite accurate at "relative comparison." When both models score full marks, an arithmetic subtraction = 0 (no visible difference), but a pairwise side-by-side comparison can still tell which is actually better.

### Example (base vs. fine-tuned, tested against this service)

| | q1 output | arithmetic compare | pairwise |
|---|---|---|---|
| A (base) | `Paris` | \ | \ |
| B (fine-tuned) | `The capital of France is Paris.` | delta = **0** (both score full marks) | q1 **B wins** |

→ The fine-tune made the answer more complete, **which the absolute score couldn't catch (both capped), but pairwise did.** Prefer pairwise when doing before/after comparisons.

---

## 6. API Overview

| Method | Path | Description |
|--------|------|------|
| `POST` | `/schemas` | Upload a schema (JSON or `Content-Type: application/yaml`) |
| `GET` | `/schemas` | List all schemas |
| `GET` | `/schemas/{id}` | Get a single schema |
| `DELETE` | `/schemas/{id}` | Delete (along with its runs) |
| `POST` | `/runs` | Submit a run: `{schema_id, config_override?}` |
| `GET` | `/runs/{run_id}` | Run status + scores + per-question output |
| `GET` | `/schemas/{id}/runs` | Run history for a schema |
| `GET` | `/runs/{a}/compare/{b}[?pairwise=true]` | Compare two runs |

---

## 7. Typical Workflow Summary

```text
(one-time) upload a schema — defines the standard for "good," stays unchanged afterward
    │
before the change → run once → run_A (baseline)
    │
make the change (switch models / retrain / adjust the prompt / tune parameters)
    │
after the change → run the same schema → run_B
    │
/compare run_A vs run_B
    ├─ delta_aggregate > 0 → got better
    ├─ delta_aggregate < 0 → got worse
    └─ ?pairwise=true → more sensitive to small changes, look at B's win rate
```

**Key principle**: the schema (the test) must stay fixed, change only one variable at a time, and the difference must exceed scoring noise (in particular, `llm_judge` has some randomness — for small differences, prefer pairwise or run it multiple times).

---

## 8. Dependencies / Environment

| Requires | Purpose |
|------|------|
| Module 02 Redis (standalone) | Real-time run status |
| Module 03 PostgreSQL (`benchmark` DB) | Schema / run history |
| Module 07 AI Lifecycle | `llm_judge` / pairwise / `cosine_similarity` |
| Module 16 Training Service | Serves fine-tuned models for `local_infer` (GPU) |

Start with: `bash deploy.sh -m 22` (depends on 02/03/07/13).
