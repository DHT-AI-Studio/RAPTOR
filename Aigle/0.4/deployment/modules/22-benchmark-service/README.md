# Module 22 — benchmark-service

Auto-Judge benchmarking: define a "marking schema" (test cases + scoring rules) for
any Raptor AI output, score it quantitatively, and compare before/after a change to
tell objectively whether things got better or worse.

Its fullest use is closing the loop with Module 16: fine-tune a model → benchmark it
against the baseline → let AutoTune propose the next set of hyper-parameters.

- Base URL: `http://localhost:8022/api/v1`
- Interactive API docs: `http://localhost:8022/docs`
- Judge model (for `llm_judge` / pairwise): `qwen3.5:9b` via Module 07,
  overridable per schema with `judge_model`

**→ [WORKFLOW.md](WORKFLOW.md) is the full guide** (end-to-end walkthrough, schema
structure, every scoring method, comparison modes, and worked examples).

## What it does

**Score.** A schema fixes the "exam paper"; only the thing under test changes. Runs
are asynchronous — `POST /runs` returns immediately with a `run_id`, and the executor
dispatches each test case to the target pipeline (`local_infer`, `chat`, `search`,
`rag`, `classify`), scores every dimension, and aggregates to a single `[0, 1]` score.

Scorers available out of the box: `keyword_match`, `contains_all` / `contains_any`,
`exact_match`, `numeric_tolerance`, `regex_match`, `cosine_similarity` (via Module 07
embeddings), `llm_judge` (natural-language rubric), and `latency_threshold`. Add your
own with `@register_scorer("name")` in `app/services/scoring/builtins.py` — no
dispatch code to touch.

All six pipelines are live-verified against the real deployment: `chat`,
`search`, `rag`, `classify`, `lifecycle_infer` work; `local_infer` (and
`/optimize`'s AutoTune loop) need Module 16, not deployed here. `search`
and `rag` both hit Module 25's per-branch ArcadeDB, so they need a
`branch_id` — a test case's own `input.branch_id`/`input.user_id`, or
whoever submitted the run (auto-injected by Module 13 from the caller's
JWT) — see [API_REFERENCE.md](../../../API_REFERENCE.md#benchmark).

**Compare.** `GET /runs/{a}/compare/{b}` subtracts the two runs' scores. With
`?pairwise=true` an LLM instead judges the two outputs side by side, per test case,
asking both A/B orderings to cancel position bias. Pairwise is what catches small
improvements once absolute scores saturate at the ceiling.

**AutoTune** (`/optimize`). Give it a natural-language goal; the planner grounds a
search space against the real dataset, you confirm the plan, and the orchestrator
loops train → score → propose over Module 16, keeping a held-out set to guard against
overfitting the benchmark.

**MLflow history.** Completed runs log their aggregate and per-dimension scores to
Module 07's MLflow tracking server, so score curves per schema are visible in the
MLflow UI. This is strictly best effort — MLflow being down never fails a benchmark.

## API summary

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/benchmark/schemas` | Upload a schema (JSON, or YAML via `Content-Type: application/yaml`) |
| `GET` | `/benchmark/schemas` | List schemas (paginated) |
| `GET` / `DELETE` | `/benchmark/schemas/{id}` | Fetch / delete a schema (delete cascades to its runs) |
| `POST` | `/benchmark/runs` | Submit a run: `{schema_id, config_override?}` → `202` + `run_id` |
| `GET` | `/benchmark/runs/{run_id}` | Run status, scores, per-case outputs |
| `GET` | `/benchmark/schemas/{id}/runs` | Run history for a schema |
| `GET` | `/benchmark/schemas/{id}/leaderboard` | Ranked runs for a schema |
| `GET` | `/benchmark/runs/{a}/compare/{b}` | Compare two runs; `?pairwise=true` for head-to-head |
| `POST` | `/optimize` | Start an AutoTune experiment from a natural-language goal |
| `GET` | `/optimize` / `/optimize/{id}` | List experiments / status, best result, history |
| `GET` | `/optimize/{id}/plan` | The planner's grounded plan, awaiting confirmation |
| `POST` | `/optimize/{id}/confirm` / `/stop` | Approve the plan / stop the loop |

## Dependencies

| Module | Used for |
|--------|----------|
| 02 Redis (standalone) | live run state |
| 03 PostgreSQL (`benchmark` DB, created by `03-database/init/postgresql/001_init.sql`) | schemas, run history |
| 07 AI Lifecycle | `llm_judge`, pairwise comparison, `cosine_similarity` embeddings, MLflow tracking |
| 16 Training Service | serving fine-tuned models for `local_infer`, and training during AutoTune |

Start with `bash deploy.sh -m 22`. Copy `.env.example` to `.env` first — shared infra
credentials (`POSTGRES_*`, `REDIS_*`) are inherited from `../.env`; benchmark-specific
settings use the `BM_` prefix.

## Development

```bash
pytest                 # unit tests (asyncio_mode=auto, see pytest.ini)
python demo.py         # interactive AutoTune demo client, stdlib only
```

`tests/demo_benchmark_features.ipynb` walks through the scoring and comparison
features against a running service.
