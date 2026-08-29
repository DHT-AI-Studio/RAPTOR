# Module 10 — image-processing

GPU worker pipeline for images: InternVL-based visual description, triggered by Kafka events from asset upload and indexed into Module 25's per-user ArcadeDB store. Runs on the shared `raptor/media-worker:0.4` base image (built by Module 08).

**Key dependencies:** 01 (NFS), 02 (Redis), 04 (asset management — presigned source URLs), 05 (Kafka), 07 (AI Lifecycle API), 08 (build-time base image), 25 (final index, via Kafka)

## Services

All run as separate containers from the same `worker/` image, selected by `WORKER_TYPE`:

| Service | `WORKER_TYPE` | Port | Role |
| --- | --- | --- | --- |
| `image-orchestrator` | `image_orchestrator` | — | Coordinates the pipeline stages for one upload |
| `image-processing` | `image_processing` | `PORT_IMAGE_SYNC_API` (default `8018`) | InternVL visual description (`VLM_MODEL_PATH`, default `OpenGVLab/InternVL3_5-1B`) |
| `image-indexer` | `image_indexer` | — | Publishes chunks into Module 25 (ArcadeDB) — the old Module 17 (`HYBRID_SEARCH_INGEST_URL`) ingest path is commented out in `kafka_handler.py`, kept only for rollback |
| `image-cleanup` | — | — | Removes temporary media after processing completes |
| `gpu-watchdog` | — | — | Restarts a worker if its container reports CUDA/NVML as stale |

## Quick start

```bash
cd deployment/modules/10-image-processing
cp .env.example .env
docker compose up -d
```

Full request/response schemas for the sync API above: `GET /docs` on the running service, or `API_REFERENCE.md`'s [Search](../../../API_REFERENCE.md#search) section for how indexed images end up searchable.
