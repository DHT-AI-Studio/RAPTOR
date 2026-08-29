# Raptor Guardrail Service

Module 23 — the Raptor platform's content safety service. Deployed in front of (or alongside) LLM inference, performing safety checks before content reaches a model and before a model's response is returned to the user.

## Overview

The service has three independent safety mechanisms, all reading policy content from the same Postgres table but interpreting it differently:

| Mechanism | Path prefix | How it works |
|---|---|---|
| Guard-model classification | `/guard/check/*` | Raw multi-model classification (Llama-Guard3 / Granite Guardian / GPT-OSS-Safeguard), using each model's own fixed guard prompt — no policy involved |
| LLM-judged policy check | `/policy/check/llm/*` | Same multi-model dispatch as `/guard/check/*`, but each model receives the currently active policy's content as its own prompt instead of a fixed guard prompt |
| GB-4 detector checker | `/guardrail/check/*` | Checks against the currently active policy's regex + Llama-Guard detector rules; also owns violation audit logging and the global on/off switch |

There is also a separate, older mechanism, the **confidence-based proxy** (`POST /api/generate`) — a single classification model returns `{category, confidence, reason}`, and the request is allowed or blocked based on a configurable confidence threshold. This mechanism predates the three above and is unrelated to their policy format.

> **Note:** `app/routers/proxy_openai.py` (`POST /v1/chat/completions`, an OpenAI-compatible version of the same confidence-based proxy) exists in the codebase but is **not mounted** in `app/main.py` — it is currently dead code, not a real, reachable endpoint.

> **Actual deployment status:** Modules 13/07 currently both go through `/guard/check/*` (no policy involved) — Module 23 has never had
> an active policy configured, so live decisions rely on the guard models' own built-in judgment. How many guard models are
> enabled is controlled by the root `.env`'s `GUARD_MODEL`/`GUARD_MODEL_2`/`GUARD_MODEL_3` (leaving `_2`/`_3` empty means single-model mode).

Health check: `GET /health`. Interactive API docs: `GET /docs`.

## Quick start

```bash
cd deployment/modules/23-guardrail-service
cp .env.example .env   # fill in real passwords and other secrets — see "Environment variables" below
docker compose up -d
```

This starts one container on the external `raptor` Docker network (the network must already exist; its name can be overridden via `DOCKER_NETWORK`):

- `raptor-guardrail-service` — the service itself (FastAPI), published on `PORT_GUARDRAIL_SERVICE` (default `8023`; the container itself always listens on `8026`)

Policy content + violation audit records live in the Postgres instance shared with module 03 (database `guardrails`, created by `03-database/init/postgresql/001_init.sql`); the active-policy cache + global on/off state lives in the Redis instance shared with module 02 (DB 0, every key prefixed with `guardrail:` so it doesn't collide with other modules' keys). No dedicated DB container is started for this service.

The app container waits for both of those dependencies' health checks to pass before starting. The service also expects an **external Ollama instance** reachable via `OLLAMA_URL`, with the required guard/proxy models already pulled — this service does not run Ollama itself.

```bash
curl http://localhost:8023/health
open http://localhost:8023/docs
```

## Environment variables

All variables are read from a `.env` in this directory (template at `.env.example`). None of these settings are provisioned by this service itself — Postgres/Redis are the shared instances from module 03/02.

> The **host-side** variable name for `REQUEST_TIMEOUT` is prefixed with `GUARDRAIL_` (`GUARDRAIL_REQUEST_TIMEOUT`), to avoid colliding with module 15's own `REQUEST_TIMEOUT`. Postgres/Redis reuse the platform-shared `POSTGRES_*`/`REDIS_*` variable names directly (defined in module 03/02's unified `.env`), with no extra prefix. See the "container-side variable" column below for the actual name the app reads inside the container.

| `.env` variable (host side) | Container-side variable | Purpose |
|---|---|---|
| `OLLAMA_URL` | `OLLAMA_URL` | Address of the Ollama server hosting all guard/proxy/signal-extraction models |
| `GUARDRAIL_REQUEST_TIMEOUT` | `REQUEST_TIMEOUT` | HTTP timeout (seconds) for calls to Ollama |
| `LOG_LEVEL` | `LOG_LEVEL` | Python logging level |
| `TIMEZONE` | `TZ` | Intended for formatting log timestamps; but the formatter is currently hard-coded to UTC+8 (`Asia/Taipei`) in the code, so this variable has no actual effect yet — informational only until it's actually wired in |
| `PORT_GUARDRAIL_SERVICE` | — | Host port the app container publishes on (the container itself always listens on `8026`) |
| `DOCKER_NETWORK` | — | Name of the external Docker network this service joins |
| `PROXY_MODEL` | `PROXY_MODEL` | Model used by the confidence-based proxy (`/api/generate`) |
| `PROXY_MODE` | `PROXY_MODE` | Confidence-based proxy mode: `monitor` (log only) or `enforce` (actually block) |
| `PROXY_CONFIDENCE_THRESHOLD` | `PROXY_CONFIDENCE_THRESHOLD` | 0.00–1.00 — in the confidence-based proxy, a confidence score above this threshold is treated as a violation |
| `DEFAULT_MODEL` | `DEFAULT_MODEL` | Reserved/legacy field — currently not read by any router or service |
| `GUARD_MODEL` | `GUARD_MODEL` | Primary guard model |
| `GUARD_MODEL_2`, `GUARD_MODEL_3` | `GUARD_MODEL_2`, `GUARD_MODEL_3` | Additional guard models — setting either one enables multi-model mode (leave empty to disable) |
| `POSTGRES_HOST` / `_PORT` / `_USER` / `_PASSWORD`, `POSTGRES_DB` hard-coded to `guardrails` | same | Connection info for module 03's shared Postgres, provided by the unified `.env` |
| `REDIS_HOST` / `_PORT` / `_PASSWORD`, `REDIS_DB` hard-coded to `0` | same | Connection info for module 02's shared Redis, provided by the unified `.env` |
| `GR_DEFAULT_ENABLED` | `GR_DEFAULT_ENABLED` | Default value written on cold start when the `guardrail:enabled` key doesn't yet exist in Redis |

## API Reference

### `/guard/check/*` — raw guard-model classification

| Endpoint | Purpose |
|---|---|
| `POST /guard/check/input` | Check user input (role fixed to `user`) |
| `POST /guard/check/output` | Check AI output (role fixed to `assistant`) |
| `POST /guard/check/conversation` | Check the last turn of a multi-turn conversation |
| `POST /guard/check/raw` | Calls every enabled guard model and returns each model's unparsed raw output — for debugging/comparison |

Results from all enabled models are merged into a single decision; if models disagree, the response includes `conflict: true` along with a per-model breakdown in `results`.

### `/policy/check/llm/*` and `/debug/policy/check/llm/*` — LLM-judged policy check

| Endpoint | Purpose |
|---|---|
| `POST /policy/check/llm/input` | Each enabled guard model judges the user input according to its own policy prompt |
| `POST /policy/check/llm/output` | Same, for AI output |
| `POST /policy/check/llm/conversation` | Same, for the last turn of a multi-turn conversation |
| `POST /debug/policy/check/llm/{input,output,conversation}` | Identical to the three above, but the response additionally includes the exact `prompt` sent to each model |

Returns an array with one entry per enabled model (never merged, since different models may be judging different content). This is the endpoint group to use for understanding the **Standard Policy Format** — see the deep-dive section below.

### `/guardrail/*` — GB-4 detector checker, policy management, violation records, system switch

| Endpoint | Purpose |
|---|---|
| `POST /guardrail/check/input`, `/guardrail/check/output` | Checks against the currently active policy's `input_guardrail`/`output_guardrail` rules — regex + Llama-Guard detector |
| `POST /guardrail/prompt_test` | Debug tool — sends a prompt straight to a specified Ollama model, bypassing any guard prompt or policy |
| `GET /guardrail/violations` | Paginated list of violation audit records (filterable by module/direction/category/time range) |
| `GET /guardrail/violations/summary` | Violation counts over the last 24 hours, grouped by category + action |
| `POST /guardrail/policies` / `POST /guardrail/policies/upload` | Create a new policy version (raw body or uploaded file) |
| `PUT /guardrail/{policy_id}` | Partially update an existing policy's content |
| `GET /guardrail/policies/template` | A suggested policy example format (YAML/JSON) |
| `GET /guardrail/policies`, `GET /guardrail/policies/active`, `GET /guardrail/policies/{id}` | List all / get the active one / get by id |
| `PUT /guardrail/policies/{id}/activate`, `/deactivate` | Only one policy can be active in the system at a time |
| `DELETE /guardrail/policies/{id}` | Delete (returns `409` and refuses if the policy is currently active) |
| `POST /guardrail/system/enable`, `/disable` | Global on/off switch — state stored in Redis, takes effect on the very next request, no service restart needed |
| `GET /guardrail/system/status` | Current switch state + the active policy's name/version |

### `POST /api/generate` — confidence-based proxy (legacy)

Input `{"prompt": "..."}`, a single classification model returns `{allowed, category, confidence, reason}`. Actually blocks when `PROXY_MODE=enforce`; only logs in `monitor` mode.

## Policy schema comparison — which one should I use?

All policy content is stored as opaque text in the `guardrail_policies` table (`raw_content`, plus a per-guard-model override column for each model). The service does **not** enforce a format on upload — each endpoint group independently attempts to interpret that text in whatever format it needs (best-effort). If the uploaded format doesn't match the endpoint you intend to use, the upload itself won't fail — parsing only fails (or silently falls back to using the content as a plain-text prompt) when you actually call that endpoint.

| Schema | Format | Endpoints used | Structure |
|---|---|---|---|
| **Standard Policy Format** | JSON array | `/policy/check/llm/*`, `/debug/policy/check/llm/*` | An array of category objects (`id`, `name`, `criteria`, ...) — model-agnostic, automatically translated into each guard model's own native prompt format |
| **GuardrailPolicy checker rules** | YAML | `/guardrail/check/*` | A single object containing `input_guardrail`/`output_guardrail` blocks, each rule defined by `category` + `detector` (`llama_guard` or `regex`) + `action` |

These two schemas have different structures and **are not interchangeable**. The rest of this section focuses on the Standard Policy Format, since it's the primary, portable format designed to be "written once, work correctly against every guard model."

<!--
Maintainer note: a third schema also exists in the code — the Policy Engine's rules format (`PolicyRule` in
app/engine/models.py, used by /policy/check/*) — deliberately not documented here. Reason: /policy/check/*
requires the active policy's content to parse as "a YAML/JSON object with a top-level `rules:` key," but the
Standard Policy Format is a plain JSON array — the two are structurally incompatible. As long as the active
policy is written in the Standard Policy Format (the primary format this document teaches), /policy/check/*
will always fail to parse it and return 409, so in the current deployment this endpoint group is effectively
unusable and should not be presented as a working feature. If compatibility between the two changes in the
future, update this README accordingly.
-->

## Standard Policy Format

The Standard Policy Format is a portable, model-agnostic way to define content-safety categories. After uploading a JSON array via `POST /guardrail/policies?target=original`, every enabled guard model automatically generates its own correctly-formatted native prompt from that single source — no need to write a separate prompt per model (though per-model wording overrides are still possible, see "Uploading and activating a policy" below).

### Format

A Standard Policy document is a **JSON array** (not YAML, not a single object), where each element in the array defines one category:

```json
[
  {
    "id": "...",
    "name": "...",
    "description": "...",
    "severity": "low | medium | high | critical",
    "decision": "allow | warn | review | block",
    "criteria": ["...", "..."],
    "exceptions": ["...", "..."],
    "examples": {
      "violation": ["...", "..."],
      "allowed": ["...", "..."]
    }
  }
]
```

| Field | Type | Required? | Description |
|---|---|---|---|
| `id` | string | required | Category code, e.g. `"M1"`, `"F2"` — reported by a guard model when it flags a violation in this category |
| `name` | string | required | Human-readable category name |
| `description` | string | required | Describes what this category covers |
| `severity` | `low` \| `medium` \| `high` \| `critical` | required | Fixed enum — any other value fails validation |
| `decision` | `allow` \| `warn` \| `review` \| `block` | required | Fixed enum representing the intended handling for this category (informational only — the actual routing logic acts on each guard model's own safe/unsafe verdict + category id, not by reading this field directly) |
| `criteria` | array of strings | required | Conditions that constitute a violation — translated per-model into "Violation criteria:" / "Violates if:" / `## VIOLATES` |
| `exceptions` | array of strings | optional (default `[]`) | Cases explicitly **not** considered a violation — translated into "Allowed:" / "Exceptions (do NOT flag):" / `## SAFE` |
| `examples.violation` | array of strings | optional (default `[]`) | Example violating sentences |
| `examples.allowed` | array of strings | optional (default `[]`) | Example non-violating sentences |

### Example — one complete category

```json
[
  {
    "id": "M1",
    "name": "Medical Misinformation",
    "description": "Providing false, misleading, or scientifically unsupported medical information, including claiming the effectiveness of unverified treatments, providing incorrect disease information, or spreading health claims not supported by medical consensus.",
    "severity": "high",
    "decision": "block",
    "criteria": [
      "Claiming that scientifically unverified treatments can cure diseases",
      "Spreading false health information that contradicts current medical consensus",
      "Describing unverified medical effects with a definitive tone",
      "Claiming non-existent medical technologies or false efficacy",
      "Denying medical knowledge supported by established scientific evidence"
    ],
    "exceptions": [
      "Medical research, clinical trials, or academic discussions",
      "News reports or medical information fact-checking",
      "Discussing medical myths while explicitly pointing out their errors",
      "Fictional contexts such as novels, movies, and games",
      "General health education information"
    ],
    "examples": {
      "violation": [
        "Drinking bleach can cure COVID-19.",
        "Cancer does not need treatment; simply changing your diet will make it completely disappear.",
        "Vaccines will implant microchips to control human minds."
      ],
      "allowed": [
        "Currently, there is no scientific evidence supporting that bleach can cure viral infections.",
        "Cancer treatment methods need to be evaluated by a physician based on the disease type.",
        "Vaccine safety needs to be confirmed through clinical research."
      ]
    }
  }
]
```

A real policy usually defines multiple categories in the same array (see `policy_example/medical_policy.md` for a complete 5-category medical-safety policy, and `policy_example/financial_policy.md` for a 6-category financial-safety policy — both are ready to use as-is).

### Minimal example (required fields only)

```json
[
  {
    "id": "H1",
    "name": "Hate",
    "description": "Hateful or discriminatory content targeting a protected group.",
    "severity": "medium",
    "decision": "warn",
    "criteria": ["Targets a group based on a protected characteristic."]
  }
]
```

### Uploading and activating a policy

```bash
# 1. Upload the JSON array as the shared "original" content
curl -X POST "http://localhost:8023/guardrail/policies?name=medical-safety&version=1.0&target=original" \
  -H "Content-Type: application/json" \
  --data-binary @policy_example/medical_policy.md   # (extract the JSON array itself from within it)

# 2. Activate it — deactivates whatever policy was previously active
curl -X PUT "http://localhost:8023/guardrail/policies/{id}/activate"

# 3. Any subsequent call to /policy/check/llm/* or /debug/policy/check/llm/* now applies this policy
curl -X POST http://localhost:8023/policy/check/llm/input \
  -H "Content-Type: application/json" \
  -d '{"content": "Drinking bleach can cure COVID-19."}'
```

The `target` parameter also accepts `llama-guard`, `granite-guardian`, `gpt-oss-safeguard` — uploading with one of these attaches a manually-written, verbatim prompt override for that specific model, which takes priority over the Standard Policy Format's auto-translated output and applies only to that model. Uploading again with the same `name`+`version` but a different `target` accumulates onto the same policy record rather than creating a new version — this is how a single policy builds up both "shared original content" and "per-model overrides" over multiple calls.

### How each guard model translates the same policy

Every enabled guard model receives the **same** list of `StandardPolicy` categories, but each model's adapter converts it into that model's own officially-supported prompt structure:

- **Llama Guard 3** — generates one block per category (`Description:` / `Violation criteria:` / `Allowed:` / `Examples of violations:` / `Examples of allowed:`), wrapped in Llama Guard's actual `<BEGIN UNSAFE CONTENT CATEGORIES>...<END UNSAFE CONTENT CATEGORIES>` template; on a violation it reports the specific category id.
- **Granite Guardian** — merges all categories into a single prose-style "criteria" block, appended to Granite's own `<guardian>...` format; since Granite's native format has no per-category id field, it can only report a plain safe/unsafe score, not which specific category was violated.
- **GPT-OSS-Safeguard** — generates one complete `# Policy {id}: {name}` section per category, following OpenAI's official multi-policy cookbook template (`## INSTRUCTIONS` / `## DEFINITIONS` / `## VIOLATES` / `## SAFE` / `## EXAMPLES`); on a violation it reports the specific category id.

### Parsing behavior

`parse_standard_policies()` is a **best-effort parser that never raises**: if the content isn't valid JSON, isn't a non-empty array, or any element in the array fails validation (missing required fields, an invalid `severity`/`decision` value), it returns `None` instead of raising an error. The caller treats `None` as "not Standard Policy Format" and falls back to using the content verbatim as a plain-text/YAML system prompt — so existing plain-text policies keep working unaffected, and only content that actually passes validation for this format triggers the per-model translation described above.

## Testing

```bash
cd deployment/modules/23-guardrail-service
pip install -r requirements.txt
python -m pytest
```

`pytest.ini` sets `testpaths = tests` and `asyncio_mode = auto`. The test suite (15 files) is written entirely against pure functions and hand-written fakes — no real Postgres, Redis, or Ollama required.

## Project layout

```
23-guardrail-service/
├── app/
│   ├── main.py                # FastAPI app, startup/shutdown, mounts each router
│   ├── adapters/               # Guard-model adapter framework
│   │   ├── standard_policy.py  #   Standard Policy Format parser
│   │   └── models/             #   Per-model-family adapters (Llama Guard 3, Granite Guardian, GPT-OSS-Safeguard)
│   ├── core/                   # Settings (config.py), Prometheus metrics
│   ├── db/                     # Postgres (schema.py, repo.py) + Redis singleton connection
│   ├── engine/                 # signal.py, policy_engine.py, policy_store.py
│   ├── models/                 # Pydantic request/response schemas
│   ├── routers/                # HTTP layer — one file per endpoint group
│   ├── services/                # Logic orchestration: guard_classifier, checker, audit_log, redactor, state
│   ├── guard_prompts/            # Fixed guard-model prompts used at runtime
│   └── official_prompts/         # Reference copies of each vendor's official prompt format
├── policy_example/               # Ready-to-use Standard Policy Format examples
│   ├── medical_policy.md
│   └── financial_policy.md
├── tests/                        # pytest — 15 files, no real external services required
├── docker-compose.yml, Dockerfile, requirements.txt, pytest.ini, .env.example
```

## References

- [Llama Guard 4 — model card and prompt format](https://developer.meta.com/ai/docs/model-cards-and-prompt-formats/llama-guard-4/)
- [Granite Guardian — model documentation](https://www.ibm.com/granite/docs/models/guardian)
- [GPT-OSS-Safeguard — OpenAI cookbook guide](https://developers.openai.com/cookbook/articles/gpt-oss-safeguard-guide)
