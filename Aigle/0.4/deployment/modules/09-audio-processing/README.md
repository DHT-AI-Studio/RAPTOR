# Module 09 — audio-processing

GPU worker pipeline for audio: WhisperX transcription/diarization, classification, text-to-speech, and LLM-generated summaries, triggered by Kafka events from asset upload and indexed into Module 25's per-user ArcadeDB store. Runs on the shared `raptor/media-worker:0.4` base image (built by Module 08).

**Key dependencies:** 01 (NFS), 02 (Redis), 04 (asset management — presigned source URLs), 05 (Kafka), 07 (AI Lifecycle API — summary LLM), 08 (build-time base image), 25 (final index, via Kafka)

## Services

All run as separate containers from the same `worker/` image, selected by `WORKER_TYPE`:

| Service | `WORKER_TYPE` | Port | Role |
| --- | --- | --- | --- |
| `audio-orchestrator` | `audio_orchestrator` | — | Coordinates the pipeline stages for one upload |
| `audio-recognizer` | `audio_recognizer` | `PORT_AUDIO_SYNC_API` (default `8019`) | WhisperX speech-to-text (`WHISPER_MODEL=large-v3`) |
| `audio-diarization` | `audio_diarization` | `PORT_AUDIO_DIARIZATION_SYNC` (default `8029`) | Speaker diarization |
| `audio-classifier` | `audio_classifier` | `PORT_AUDIO_CLASSIFIER_SYNC` (default `8039`) | Audio event/type classification (PANNs) |
| `audio-analysis` | `audio_analysis` | — | Aggregates recognizer/diarization/classifier output |
| `audio-summary` | `audio_summary` | — | LLM-generated summary via Module 07 (`INFERENCE_MODEL`) |
| `audio-tts` | `audio_tts` | — | Text-to-speech |
| `audio-indexer` | `audio_indexer` | — | Publishes chunks into Module 25 (ArcadeDB) — the old Module 17 (`HYBRID_SEARCH_INGEST_URL`) ingest path is commented out in `kafka_handler.py`, kept only for rollback |
| `audio-cleanup` | — | — | Removes temporary media after processing completes |
| `gpu-watchdog` | — | — | Restarts a worker if its container reports CUDA/NVML as stale |

## Quick start

```bash
cd deployment/modules/09-audio-processing
cp .env.example .env
docker compose up -d
```

Full request/response schemas for the sync APIs above: `GET /docs` on the running service, or `API_REFERENCE.md`'s [Search](../../../API_REFERENCE.md#search) section for how indexed audio ends up searchable.
