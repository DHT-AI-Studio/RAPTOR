# Document Processing Module

GPU-isolated document pipeline: VLM/OCR analysis, LLM summary, PII-scanned
personal-index publishing — Kafka-driven. Runs on the shared `raptor/media-worker`
image (built by [08-media-worker](../08-media-worker/)) — this module has no
`build:` of its own; `WORKER_TYPE` selects which service `main.py` dispatches to
inside the container.

## Services

| Container | `WORKER_TYPE` | GPU | Role |
|---|---|---|---|
| `raptor-document-kafka-init` | — | — | One-shot: creates every topic below, then exits |
| `raptor-document-orchestrator` | `document_orchestrator` | no | Consumes `document-processing-requests`; fans out analysis/summary/indexer requests |
| `raptor-document-analysis` | `document_analysis` | yes | InternVL VLM/OCR content extraction (PDF, images, etc.) |
| `raptor-document-summary` | `document_summary` | no | LLM summary + entity extraction via Module 07 (Ollama) |
| `raptor-document-indexer` | `document_indexer` | no | PII scan (Module 23) then publishes to the personal index (Module 25); also still emits to `opensearch-index-requests` for Module 17's (deprecated) OpenSearch index |
| `raptor-document-cleanup` | — | — | `busybox` sidecar: deletes `/tmp/media_processing/document_processing/*` older than 24h, hourly |
| `raptor-document-gpu-watchdog` | — | — | Restarts `document-analysis` if its `torch.cuda.is_available()` healthcheck goes unhealthy |

No container is named `aigle-*`, has a static IP, or resolves at `*.aigle.local` —
every service is reachable only by its docker-network alias (`raptor-document-*`),
same convention as the rest of this repo.

A fifth worker directory, `document_graph_service`, still exists in
`worker/services/` but is **retired** (temporal plan P12) and not wired into
`08-media-worker`'s dispatch table — it never received real chunks, and its job
(building entities/temporal facts from document content) is now done by Module 25's
own temporal job instead. It is not a service you deploy.

## Kafka topics

Created once by `document-kafka-init-topics`:

```
document-processing-requests ──▶ document-orchestrator ──┬─▶ document-analysis-requests ──▶ document-analysis ──▶ document-analysis-results
                                                           ├─▶ document-summary-requests  ──▶ document-summary  ──▶ document-summary-results
                                                           └─▶ document-indexer-requests  ──▶ document-indexer  ──┬─▶ document-indexer-results
                                                                                                                    └─▶ opensearch-index-requests (Module 17, deprecated)
```

DLQs (`document-processing-dlq`, `document-analysis-dlq`, `document-summary-dlq`,
`document-indexer-dlq`) are written to, never consumed — see the repo-wide known
limitation in [`docs/CHANGELOG.md`](../../../docs/CHANGELOG.md).

## PII scan at ingestion

`document-indexer` asks Module 23 (`POST {GUARDRAIL_URL}/guardrail/policy/decide`)
whether each chunk contains PII before personal indexing —
`M12_PII_SCAN_ENABLED=true` by default, only actually runs when `GUARDRAIL_URL` is
set (normally inherited from root `.env`). Fail-open: a scan slower than
`M12_PII_SCAN_TIMEOUT_MS` (default 1000ms) is skipped, never blocks indexing.

## GPU

`document-analysis` reserves one GPU (`DOCUMENT_ANALYSIS_GPU`, default `0`) for its
VLM (`DOCUMENT_ANALYSIS_VLM_MODEL`/`_REVISION`, pinned). A
`torch.cuda.is_available()` healthcheck backs `document-gpu-watchdog`, same pattern
as module 16's training-watchdog.

## Key environment variables

| Variable | Default | Used by |
|---|---|---|
| `DOCUMENT_ANALYSIS_VLM_MODEL` | `OpenGVLab/InternVL3_5-1B` | `document-analysis` |
| `DOCUMENT_ANALYSIS_MAX_CHUNK_TOKENS` | `400` | `document-analysis` |
| `DOCUMENT_ANALYSIS_MAX_WORKERS` | `8` | `document-analysis` |
| `GUARDRAIL_URL` | `http://raptor-guardrail-service:8026` | `document-indexer` (PII scan) |
| `M12_PII_SCAN_ENABLED` | `true` | `document-indexer` |
| `M12_PII_SCAN_TIMEOUT_MS` | `1000` | `document-indexer` |
| `INFERENCE_MODEL` | `gemma4:12b` | `document-summary` (Module 07) |

## Deploy

```bash
python build.py -m 12
```
