# Module 11 — video-processing

GPU worker pipeline for video: chunking, frame description (InternVL), OCR (PaddleOCR), contextual summarization, and graph extraction, triggered by Kafka events from asset upload and indexed into Module 25's per-user ArcadeDB store. Runs on the shared `raptor/media-worker:0.4` base image (built by Module 08) — the highest-VRAM module in the platform (see `BUILD.md` §1 and [issue #17](https://github.com/DHT-AI-Studio/RAPTOR/issues/17) for current VRAM guidance/status).

**Key dependencies:** 01 (NFS), 02 (Redis), 04 (asset management — presigned source URLs), 05 (Kafka), 07 (AI Lifecycle API), 08 (build-time base image), 09 (`audio-recognizer`, for synced transcription), 25 (final index, via Kafka)

## Services

All run as separate containers from the same `worker/` image, selected by `WORKER_TYPE`:

| Service | `WORKER_TYPE` | Port | Role |
| --- | --- | --- | --- |
| `video-orchestrator` | `video_orchestrator` | — | Coordinates the pipeline stages for one upload |
| `video-chunking` | `video_chunking` | — | Splits video into time-coded chunks |
| `video-ocr-frame` | `video_ocr_frame` | `PORT_VIDEO_OCR_SYNC` (default `8032`) | On-screen text extraction (PaddleOCR, `OCR_DET_MODEL`/`OCR_REC_MODEL`) |
| `video-frame-description` | `video_frame_description` | `PORT_VIDEO_FRAME_SYNC_API` (default `8031`) | Per-frame visual description (`VLM_MODEL_PATH`, default `OpenGVLab/InternVL3_5-1B`) |
| `video-analysis` | `video_analysis` | — | Aggregates chunking/OCR/frame-description/ASR (via Module 09) output |
| `video-summary` | `video_summary` | — | LLM-generated summary via Module 07 (`INFERENCE_MODEL`) |
| `video-contextualize` | `video_contextualize` | — | Contextual chunking (per-chunk LLM-generated context) |
| `video-indexer` | `video_indexer` | — | Publishes chunks into Module 25 (ArcadeDB) — the old Module 17 (`HYBRID_SEARCH_INGEST_URL`) ingest path is commented out in `kafka_handler.py`, kept only for rollback |
| `video-graph` | `video_graph` | — | Entity/relationship graph extraction |
| `video-cleanup` | — | — | Removes temporary media after processing completes |
| `gpu-watchdog` | — | — | Restarts a worker if its container reports CUDA/NVML as stale |

## Quick start

```bash
cd deployment/modules/11-video-processing
cp .env.example .env
docker compose up -d
```

Full request/response schemas for the sync APIs above: `GET /docs` on the running service, or `API_REFERENCE.md`'s [Video Search](../../../API_REFERENCE.md#video-search) section for how indexed video ends up searchable.
