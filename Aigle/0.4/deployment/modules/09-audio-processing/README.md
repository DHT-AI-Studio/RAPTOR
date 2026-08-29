# Audio Processing Module

## Overview

The Audio Processing Module consolidates all STT (Speech-to-Text) and audio processing services that require GPU resources. This module extracts GPU-intensive audio processing services from the media-workers module to provide better resource isolation and management.

## Services

### 1. Audio Recognizer Service (STT)
- **Container**: `aigle-audio-recognizer`
- **Hostname**: `audio-recognizer.aigle.local`
- **IP Address**: `172.30.12.1` (configurable via `AUDIO_RECOGNIZER_IP`)
- **Model**: WhisperX `large-v3` (STT - Speech-to-Text)
- **GPU**: Required (1 GPU, configurable)
- **Memory**: 8GB (configurable)
- **CPU**: 2 cores (configurable)
- **Kafka Topic**: `audio-recognition`
- **Purpose**: Transcribes audio to text using WhisperX STT model

### 2. Audio Diarization Service (STT-related)
- **Container**: `aigle-audio-diarization`
- **Hostname**: `audio-diarization.aigle.local`
- **IP Address**: `172.30.12.2` (configurable via `AUDIO_DIARIZATION_IP`)
- **Model**: WhisperX DiarizationPipeline (uses WhisperX)
- **GPU**: Required (1 GPU, configurable)
- **Memory**: 8GB (configurable)
- **CPU**: 2 cores (configurable)
- **Kafka Topic**: `audio-diarization`
- **Purpose**: Identifies and separates speakers in audio using WhisperX

### 3. Audio Classifier Service
- **Container**: `aigle-audio-classifier`
- **Hostname**: `audio-classifier.aigle.local`
- **IP Address**: `172.30.12.3` (configurable via `AUDIO_CLASSIFIER_IP`)
- **Model**: PANNs (Audio Tagging and Sound Event Detection)
- **GPU**: Required (1 GPU, configurable)
- **Memory**: 4GB (configurable)
- **CPU**: 2 cores (configurable)
- **Kafka Topic**: `audio-classification`
- **Purpose**: Classifies audio content and detects sound events

### 4. Audio Summary Service
- **Container**: `aigle-audio-summary`
- **Hostname**: `audio-summary.aigle.local`
- **IP Address**: `172.30.12.4` (configurable via `AUDIO_SUMMARY_IP`)
- **Model**: LLM (via Ollama, default: `qwenforsummary`)
- **GPU**: Required (1 GPU, configurable)
- **Memory**: 8GB (configurable)
- **CPU**: 2 cores (configurable)
- **Kafka Topic**: `audio-summary`
- **Purpose**: Generates summaries of audio content using LLM

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Kafka Cluster                              │
│  Topics: audio-recognition, audio-diarization,          │
│          audio-classification, audio-summary            │
└──────────────┬──────────────────────────────────────────┘
               │
               │ Messages
               │
┌──────────────▼──────────────────────────────────────────┐
│         Audio Processing Module                         │
│  ┌────────────────────┐  ┌────────────────────┐       │
│  │ audio-recognizer   │  │ audio-diarization  │       │
│  │ (WhisperX STT)     │  │ (WhisperX)         │       │
│  │ GPU: 1             │  │ GPU: 1             │       │
│  │ IP: 172.30.12.1    │  │ IP: 172.30.12.2    │       │
│  └────────────────────┘  └────────────────────┘       │
│  ┌────────────────────┐  ┌────────────────────┐       │
│  │ audio-classifier   │  │ audio-summary      │       │
│  │ (PANNs)            │  │ (LLM)              │       │
│  │ GPU: 1             │  │ GPU: 1             │       │
│  │ IP: 172.30.12.3    │  │ IP: 172.30.12.4    │       │
│  └────────────────────┘  └────────────────────┘       │
└─────────────────────────────────────────────────────────┘
```

## Models Used

### WhisperX (STT)
- **Model**: `large-v3`
- **Used By**: `audio-recognizer` and `audio-diarization`
- **Purpose**: Speech-to-text transcription and speaker diarization
- **GPU Memory**: ~4-8GB per GPU

### PANNs
- **Model**: Audio Tagging and Sound Event Detection
- **Used By**: `audio-classifier`
- **Purpose**: Audio classification and sound event detection
- **GPU Memory**: ~2-4GB per GPU

### LLM (via Ollama)
- **Model**: `qwenforsummary` (default)
- **Used By**: `audio-summary`
- **Purpose**: Audio content summarization
- **GPU Memory**: ~4-8GB per GPU

## Configuration

### Environment Variables

#### Audio Recognizer Service (STT)
```bash
AUDIO_RECOGNIZER_IP=172.30.12.1
AUDIO_RECOGNIZER_MODEL=large-v3
AUDIO_RECOGNIZER_BATCH_SIZE=16
AUDIO_RECOGNIZER_GPU=0
AUDIO_RECOGNIZER_MEMORY_LIMIT=8g
AUDIO_RECOGNIZER_CPU_LIMIT=2
AUDIO_RECOGNIZER_GPU_COUNT=1
```

#### Audio Diarization Service
```bash
AUDIO_DIARIZATION_IP=172.30.12.2
AUDIO_DIARIZATION_GPU=0
AUDIO_DIARIZATION_MIN_SPEAKERS=1
AUDIO_DIARIZATION_MAX_SPEAKERS=10
AUDIO_DIARIZATION_MEMORY_LIMIT=8g
AUDIO_DIARIZATION_CPU_LIMIT=2
AUDIO_DIARIZATION_GPU_COUNT=1
```

#### Audio Classifier Service
```bash
AUDIO_CLASSIFIER_IP=172.30.12.3
AUDIO_CLASSIFIER_GPU=0
AUDIO_CLASSIFIER_TOP_K=5
AUDIO_CLASSIFIER_SEGMENT_LENGTH=30.0
AUDIO_CLASSIFIER_MEMORY_LIMIT=4g
AUDIO_CLASSIFIER_CPU_LIMIT=2
AUDIO_CLASSIFIER_GPU_COUNT=1
```

#### Audio Summary Service
```bash
AUDIO_SUMMARY_IP=172.30.12.4
AUDIO_SUMMARY_GPU=0
AUDIO_SUMMARY_MODEL=qwenforsummary
AUDIO_SUMMARY_MEMORY_LIMIT=8g
AUDIO_SUMMARY_CPU_LIMIT=2
AUDIO_SUMMARY_GPU_COUNT=1
```

## Network Configuration

- **Network**: `aigle-network` (external)
- **Subnet**: `172.30.12.0/24`
- **IP Range**: `172.30.12.1 - 172.30.12.254`

## Dependencies

- **Kafka Cluster**: Required for message queue
- **Redis**: Required for caching
- **Qdrant**: Required for vector storage
- **SeaweedFS**: Required for object storage
- **GPU**: NVIDIA GPU with CUDA support required
- **HuggingFace Token**: Required for `audio-diarization` (pyannote.audio)

## HuggingFace Model License (One-time Setup)

The pyannote models used by `audio-diarization` are **gated models**. Having a HuggingFace token alone is not enough — the account associated with the token must explicitly accept the license on the HuggingFace website before the models can be downloaded.

**Complete the following steps once before first deployment:**

1. Visit https://huggingface.co/pyannote/speaker-diarization-3.1 → log in → click **Agree and access repository**
2. Visit https://huggingface.co/pyannote/segmentation-3.0 → log in → click **Agree and access repository**

After accepting both licenses, restart the `raptor-audio-diarization` container. The models will be downloaded and cached to `HF_CACHE_PATH` — subsequent restarts will not re-download.

If the licenses have not been accepted, the container will crash on startup with:
```
Could not download 'pyannote/speaker-diarization-3.1' pipeline.
AttributeError: 'NoneType' object has no attribute 'to'
```

## Deployment

```bash
cd /opt/dht/apps/raptor/deployment/modules/12-audio-processing
docker compose up -d
```

## Monitoring

Check service status:
```bash
docker ps | grep audio-processing
docker logs aigle-audio-recognizer
docker logs aigle-audio-diarization
docker logs aigle-audio-classifier
docker logs aigle-audio-summary
```

## Migration Notes

These services were previously part of the `07-media-workers` module:
- `audio-recognizer`: Moved from `172.30.7.15` to `172.30.12.1`
- `audio-diarization`: Moved from `172.30.7.14` to `172.30.12.2`
- `audio-classifier`: Moved from `172.30.7.13` to `172.30.12.3`
- `audio-summary`: Moved from `172.30.7.16` to `172.30.12.4`

Update any hardcoded IP addresses or service references when migrating.
