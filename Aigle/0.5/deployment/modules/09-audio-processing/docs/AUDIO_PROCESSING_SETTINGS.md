# Audio Processing Module — Settings Reference

Architecture and service list: [../README.md](../README.md). This file is just the
`.env` reference, grouped the same way as `.env.example`.

## Kafka

| Variable | Default |
|---|---|
| `KAFKA_BOOTSTRAP_SERVERS` | `kafka-broker1:9092,kafka-broker2:9092,kafka-broker3:9092` |

## Redis

| Variable | Default |
|---|---|
| `REDIS_HOST` | `raptor-redis-standalone` |
| `REDIS_PORT` | `6379` |
| `REDIS_PASSWORD` | — (required) |

## Downstream services

| Variable | Default | Used by |
|---|---|---|
| `HYBRID_SEARCH_INGEST_URL` | `http://raptor-hybridsearch-api:8000/api/v1/ingest/file` | shared worker env |
| `ASSET_MANAGEMENT_URL` | `http://raptor-asset-management:8000` | `audio-orchestrator` |
| `INFERENCE_URL` | `http://raptor-ai-lifecycle-api:8010` | `audio-summary` (Module 07); also read by `audio-tts`, see below |
| `INFERENCE_MODEL` | `gemma4:12b` | `audio-summary` |
| `INFERENCE_THINK` | `false` | `audio-summary` |
| `GUARDRAIL_URL` | — | `audio-indexer` (moderation scan; empty skips it) |

`AI_MODEL_LIFECYCLE_URL` in `.env.example` is a dead leftover — `audio-tts`'s own
code (`tts_client.py`) doesn't read it; confirmed consistent with `docs/BUILD.md`'s
own note about this exact variable.

`audio-tts`'s `tts_client.py` POSTs directly to whatever `INFERENCE_URL` resolves
to, with no path appended — its own code-level default is
`http://raptor-ai-lifecycle-api:8010/inference/tts` (the full endpoint), but
`worker-common`'s shared `INFERENCE_URL` default in `docker-compose.yml` is just
the base URL (`http://raptor-ai-lifecycle-api:8010`, no `/inference/tts`), and
that's what docker-compose actually injects for every worker including this one.
Moot in practice: `audio-tts` is orphaned (see ../README.md's Services table) —
nothing in the repo produces to `audio-tts-requests`, so this code path never
runs even if the `tts` profile is started by hand.

## Models

| Variable | Default | Used by |
|---|---|---|
| `HF_TOKEN` | — (required) | `audio-diarization` — `pyannote/speaker-diarization-3.1` needs HF token + model access |
| `WHISPER_MODEL` | `large-v3` | `audio-recognizer`, `audio-diarization` |
| `HF_CACHE_PATH` | `/path/to/huggingface/cache` | shared worker env, mounted at `/hf_cache` |
| `PANNS_CACHE_PATH` | `/path/to/panns/cache` | `audio-classifier` only |

## GPU allocation

| Variable | Default |
|---|---|
| `AUDIO_RECOGNIZER_GPU` | `0` |
| `AUDIO_RECOGNIZER_GPU_COUNT` | `1` |
| `AUDIO_RECOGNIZER_MEMORY_LIMIT` | `12g` |
| `AUDIO_RECOGNIZER_CPU_LIMIT` | `4` |
| `AUDIO_DIARIZATION_GPU` | `0` |
| `AUDIO_DIARIZATION_GPU_COUNT` | `1` |
| `AUDIO_DIARIZATION_MEMORY_LIMIT` | `12g` |
| `AUDIO_DIARIZATION_CPU_LIMIT` | `4` |
| `AUDIO_CLASSIFIER_GPU` | `2` |
| `AUDIO_CLASSIFIER_GPU_COUNT` | `1` |
| `AUDIO_CLASSIFIER_MEMORY_LIMIT` | `8g` |
| `AUDIO_CLASSIFIER_CPU_LIMIT` | `2` |

Assign different device IDs across `*_GPU` vars on a multi-GPU host to avoid VRAM
contention — all three default to different values already (`0`/`0`/`2`) except
recognizer and diarization, which share GPU 0 by default.

## CPU-only services

| Variable | Default |
|---|---|
| `AUDIO_ORCHESTRATOR_MEMORY_LIMIT` / `_CPU_LIMIT` | `2g` / `1` |
| `AUDIO_ANALYSIS_MEMORY_LIMIT` / `_CPU_LIMIT` | `4g` / `2` |
| `AUDIO_SUMMARY_MEMORY_LIMIT` / `_CPU_LIMIT` | `4g` / `2` |
| `AUDIO_INDEXER_MEMORY_LIMIT` / `_CPU_LIMIT` | `2g` / `1` |
| `AUDIO_TTS_MEMORY_LIMIT` / `_CPU_LIMIT` | `2g` / `1` |

## Synchronous API ports

| Variable | Default | Service |
|---|---|---|
| `PORT_AUDIO_SYNC_API` | `8019` | `audio-recognizer` |
| `PORT_AUDIO_DIARIZATION_SYNC` | `8029` | `audio-diarization` |
| `PORT_AUDIO_CLASSIFIER_SYNC` | `8039` | `audio-classifier` |

## Shared storage

| Variable | Default |
|---|---|
| `NFS_SERVER` | — (required) |
| `PORT_NFS` | `2049` |
| `NFS_MEDIA_TMP_PATH` | `/media/processing` |

## Other

| Variable | Default |
|---|---|
| `TIMEZONE` | `Asia/Taipei` |
