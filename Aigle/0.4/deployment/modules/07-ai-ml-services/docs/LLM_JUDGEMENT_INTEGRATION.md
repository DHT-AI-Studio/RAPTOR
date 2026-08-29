# LLM Judgement Integration (BM-6) — Requirements & Implementation Status

> Status baseline: `llm-judge-integration-george` branch (ModelLifecycle-george + merged origin/benchmark-cing @ 828fb7f, 2026-07-09).
> Implementation location: `deployment/modules/22-benchmark-service/`.

## Description

As a benchmark operator, I want the scoring engine to use an LLM to evaluate subjective quality dimensions (relevance, fluency, accuracy) via a user-defined rubric, so that benchmark runs can assess dimensions that keyword or regex matching cannot capture.

## Status overview

| # | Acceptance criterion | Status |
|---|---|---|
| 1 | judge.py provides evaluate() | ✅ Ruled and closed (keep float, no dict/class; deviation accepted) |
| 2 | Calls the judge model via Module 07 `/inference/infer` | ✅ Done |
| 3 | Score parsing and normalization | ✅ Done |
| 4 | On parse failure, return 0.0 and log the reason | ✅ Ruled and closed (returns 0.0 ✓; reason goes to the service log, not returned as an explanation) |
| 5 | scorer routes `llm_judge` / `cosine_similarity` | ✅ Done (pluggable registry — more general than required) |
| 6 | embedding cosine similarity (bge-m3, 1024-dim) | ✅ Done (via Module 07's `embedding` task) |
| 7 | judge timeout configurable, raises ScoringError on timeout | ✅ Ruled and done (timeout returns 0.0 instead of raising; `BM_JUDGE_TIMEOUT_S` alias added) |
| 8 | Unit tests (mocked Module 07) | ✅ Done (`tests/test_judge.py` + `tests/test_scorer.py`) |

---

## Acceptance Criteria (marked one by one)

### 1. ✅ `services/judge.py` contains `LLMJudge.evaluate(rubric, output, expected_answer)` → ~~`{raw_score, normalised_score, explanation}`~~ → float

**Current state**: `app/services/judge.py` exists, but as a **module-level async function** rather than an `LLMJudge` class:

```python
await judge.evaluate(rubric, output, expected_answer, score_range, model) -> float  # normalised [0, 1]
```

The return value is only the normalised score (float) — there's no `raw_score` / `explanation` field. It also provides, beyond what was originally required:

- `pairwise(task_input, output_a, output_b)` → `"A" | "B" | "TIE"` (used by /compare; the caller swaps sides to cancel out positional bias)
- `complete(prompt, ...)` (a shared general-purpose completion used by the autotune proposer/planner)
- `embed(text)` (used by the cosine_similarity scorer)

**Ruling (2026-07-10)**: **keep float, accept the deviation from the original requirement.** Rationale: every consumer of the score is an automated pipeline (weighted aggregation, leaderboard, autotune proposer) — nothing reads the explanation; a "return only an integer, max_length=16" design keeps every judge call fast and its parsing reliable (a single autotune iteration can fire hundreds of judge calls); parse errors are already logged to the service log (see item 4). If per-case commentary is needed later, the direction is a schema-level per-dimension opt-in (`params: {"explain": true}` requests commentary and stores it in `scores_per_case`) — not a wholesale switch back to a dict.

### 2. ✅ Calls `POST http://raptor-ai-lifecycle-api:8010/inference/infer`, model from schema field `judge_model` (configurable via `BM_JUDGE_MODEL`)

**Current state**: Done. The implementation calls with `task: "text-generation"` + `engine: "ollama"` — the original requirement's `task: "text-generation-ollama"` is a legacy name in the current Module 07 and gets auto-canonicalized to `text-generation` (see `07-ai-ml-services/src/inference/spec.py`); the implementation uses the new canonical name directly.

- The schema can carry `judge_model` to override the judge model for a single run ✅
- The `BM_JUDGE_MODEL` env var changes the default ✅
- **Deviation**: the default model is `qwen3.5:9b`, not the originally-specified `qwen3.5:0.8b` (0.8b wasn't stable enough at scoring — deliberately upgraded; use `BM_JUDGE_MODEL` to revert if resources are tight)

### 3. ✅ Prompt requires a numeric score; parser extracts the first number and maps it to [0, 1]

**Current state**: Done. The prompt requires "ONLY a single integer between lo and hi" (`max_length=16`, `temperature=0.0`); the parser extracts the first int/float in the response via regex, with the normalization formula:

```
normalised = (raw - lo) / (hi - lo)    # score_range defaults to (1, 5)
```

When `lo = 1`, this is equivalent to the original requirement's `(raw - 1) / (score_range[1] - 1)`, but more general (supports any lower bound), and the result is clamped to [0, 1].

**Difference from the original requirement**: the prompt asks only for a number, with no accompanying commentary (explanation) — the same gap as item 1. The current design deliberately favors small outputs and fast parsing.

### 4. ✅ If the LLM response cannot be parsed, `normalised_score` defaults to 0.0 and ~~explanation~~ the service log records the parse error

**Current state**: On parse failure (or a failure of the judge call itself), returns `0.0` ✅, and logs the raw response/exception via `logger.warning` ✅. Per the item-1 ruling (keep float), the parse error goes to the service log rather than a return field — during debugging, check `raptor-benchmark-service`'s logs to see the judge's raw response.

### 5. ✅ `scorer.py` routes `method: "llm_judge"` and `method: "cosine_similarity"`

**Current state**: Done, and more general than required. `app/services/scorer.py` doesn't hardcode routing — it dispatches through `app/services/scoring/registry.py`'s `@register_scorer("name")`; both `llm_judge` and `cosine_similarity` are registered in `app/services/scoring/builtins.py`, alongside built-in scorers `keyword_match`, `contains_all/any`, `exact_match`, `regex_match`, `numeric_tolerance`, `latency_threshold`. Per-dimension scores are weighted and aggregated into an [0, 1] aggregate.

### 6. ✅ `embed_and_compare(text_a, text_b)` — cosine similarity via Module 07 embedding task with `bge-m3`, numpy dot on normalised 1024-dim vectors

**Current state**: Done. `judge.embed_and_compare(text_a, text_b)` fetches both vectors with a **single batched call** to Module 07's `/inference/infer` (`task: "embedding"` — the canonical name is `embedding`, not the originally-specified `text-embedding`; legacy names like `feature-extraction` auto-canonicalize), computes cosine via numpy dot and clamps to [0, 1]; the `cosine_similarity` scorer calls it directly. The single-text `judge.embed(text)` is also retained.

To support this, Module 07's **Ollama adapter** (`src/inference/adapters/ollama.py`) was extended: when `task_family == "embedding"`, it hits the daemon's `/api/embed` (batched), with the response shape aligned to the HF `EmbeddingHandler`'s (`{"embeddings": [...], "metadata": {...}}`) so the caller doesn't need to know which runtime it's talking to.

**Deployment prerequisite**: `bge-m3` must first be registered into Module 07's registry:

```
POST /models/register_ollama  {"model_name": "bge-m3", "task": "embedding"}
```

The old direct-to-Ollama path (`BM_OLLAMA_URL`) has been removed.

### 7. ✅ LLM judge timeout controlled by `BM_JUDGE_TIMEOUT_S` (default 30 s); ~~raises `ScoringError` on timeout~~ → returns 0.0

**Current state**: Timeout is configurable, defaulting to 30 s ✅; `BM_JUDGE_TIMEOUT_S` has been added as an `AliasChoices` alias for `BM_INFERENCE_TIMEOUT` ✅.

**Ruling (2026-07-10)**: on timeout / any judge failure, **return 0.0 and log a warning, do not raise `ScoringError`** — a single judge failure shouldn't abort the entire benchmark run (the autotune orchestrator depends on this behavior; one candidate's run crashing would ruin the whole iteration). This ruling supersedes the original requirement's raise behavior, and is pinned down by `tests/test_judge.py::test_evaluate_timeout_is_zero`.

### 8. ✅ Unit tests: mocked Module 07 → verify score parsing for valid score, non-numeric response, and timeout

**Current state**: Done. `tests/test_judge.py` mocks Module 07's response at the httpx transport layer, covering:
- A valid integer/a float embedded in prose → correct normalised score
- A non-numeric response → 0.0
- Timeout (`httpx.TimeoutException`) → 0.0 (per the item-7 ruling)
- Out-of-range score clamping, custom score_range
- Embedding: goes through `task: "embedding"`, batched two-text call, cosine computation, transport errors propagate up (degraded to 0.0 by the scorer)

`tests/test_scorer.py` continues with scorer-layer tests (mocking `judge.evaluate` / `judge.embed_and_compare`); `tests/test_scoring_registry.py` covers the registry. All 92 tests pass.

---

## Subtasks

- [x] `app/services/judge.py` — evaluate() and embedding (implemented as module-level functions, not an `LLMJudge` class; see item 1)
- [x] Extend `app/services/scorer.py` to route `llm_judge` and `cosine_similarity` (implemented via the scorer registry)
- [x] Add `BM_JUDGE_MODEL` and timeout to `app/core/config.py` (timeout is `BM_INFERENCE_TIMEOUT`, aliased as `BM_JUDGE_TIMEOUT_S`)
- [x] `judge.embed()` / `embed_and_compare()` switched to Module 07's `task: "embedding"` (replacing the direct Ollama connection; Module 07's Ollama adapter now supports embedding too)
- [x] Judge-layer unit tests (valid / non-numeric / timeout, mocked httpx) — `tests/test_judge.py`
- [x] `BM_JUDGE_TIMEOUT_S` env var alias
- [x] Ruling (2026-07-10): timeout behavior = return 0.0 (do not raise `ScoringError`)
- [x] Ruling (2026-07-10): evaluate's return shape = keep float (deviation accepted; revisit as a per-dimension opt-in explain if a real need arises, see item 1)
- [ ] Deployment: register `bge-m3` into Module 07 (`POST /models/register_ollama`, task="embedding")

## Related files

| File | Contents |
|---|---|
| `22-benchmark-service/app/services/judge.py` | evaluate / pairwise / complete / embed / embed_and_compare (the single point where all Module 07 calls are made) |
| `22-benchmark-service/app/services/scorer.py` | per-dimension dispatch + weighted aggregation |
| `22-benchmark-service/app/services/scoring/builtins.py` | built-in scorers: `llm_judge`, `cosine_similarity`, etc. |
| `22-benchmark-service/app/services/scoring/registry.py` | `@register_scorer` pluggable registration mechanism |
| `22-benchmark-service/app/core/config.py` | settings: `BM_JUDGE_MODEL`, `BM_INFERENCE_TIMEOUT` (aliased `BM_JUDGE_TIMEOUT_S`), `BM_EMBED_MODEL`, etc. |
| `22-benchmark-service/tests/test_judge.py` | judge-layer tests (mocked httpx: score-parsing edge cases + the embedding path) |
| `22-benchmark-service/tests/test_scorer.py` | scorer-layer tests (judge mocked out) |
| `07-ai-ml-services/src/inference/adapters/ollama.py` | Ollama adapter (embedding task → daemon `/api/embed`) |
| `07-ai-ml-services/src/inference/spec.py` | task canonicalization mapping (`text-generation-ollama` → `text-generation`), `embedding` task family |
