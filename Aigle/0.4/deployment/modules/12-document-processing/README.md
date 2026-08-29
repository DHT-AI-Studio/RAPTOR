# Module 12 — document-processing

GPU worker pipeline for documents (PDF/Office/CSV/HTML/text): layout analysis, VLM image captioning within documents, LLM summarization, and graph extraction, triggered by Kafka events from asset upload and indexed into Module 25's per-user ArcadeDB store. Runs on the shared `raptor/media-worker:0.4` base image (built by Module 08).

**Key dependencies:** 01 (NFS), 02 (Redis), 04 (asset management — presigned source URLs), 05 (Kafka), 07 (AI Lifecycle API), 08 (build-time base image), 25 (final index, via Kafka)

## Services

All run as separate containers from the same `worker/` image, selected by `WORKER_TYPE`:

| Service | `WORKER_TYPE` | Port | Role |
| --- | --- | --- | --- |
| `document-orchestrator` | `document_orchestrator` | — | Coordinates the pipeline stages for one upload |
| `document-analysis` | `document_analysis` | `PORT_DOCUMENT_SYNC_API` (default `8020`) | Format detection + layout/table extraction + VLM image captioning (`DOCUMENT_ANALYSIS_VLM_MODEL`, default `OpenGVLab/InternVL3_5-1B`); also exposes a synchronous `/analyze` used directly by other modules (e.g. Module 29's planned integration, see its `Module_Overlap_Analysis.md`) |
| `document-summary` | `document_summary` | — | LLM-generated summary via Module 07 (`INFERENCE_MODEL`) |
| `document-indexer` | `document_indexer` | — | Publishes chunks into Module 25 (ArcadeDB) — the old Module 17 (`HYBRID_SEARCH_INGEST_URL`) ingest path is commented out in `kafka_handler.py`, kept only for rollback |
| `document-graph` | `document_graph` | — | Entity/relationship graph extraction |
| `document-cleanup` | — | — | Removes temporary media after processing completes |
| `gpu-watchdog` | — | — | Restarts a worker if its container reports CUDA/NVML as stale |

## Quick start

```bash
cd deployment/modules/12-document-processing
cp .env.example .env
docker compose up -d
```

Full request/response schemas for the sync API above: `GET /docs` on the running service, or `API_REFERENCE.md`'s [Search](../../../API_REFERENCE.md#search) section for how indexed documents end up searchable.
