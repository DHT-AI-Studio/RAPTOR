#!/usr/bin/env python3
"""
Audio Processing Worker Dispatcher
Reads WORKER_TYPE env var and routes to the appropriate service.

Valid WORKER_TYPE values:
  audio_orchestrator  - Receives audio upload events, fans out to workers
  audio_analysis      - Audio preprocessing / segmentation (CPU)
  audio_classifier    - Audio tagging via panns_inference (GPU)
  audio_diarization   - Speaker diarization via whisperx (GPU)
  audio_recognizer    - ASR via whisperx (GPU)
  audio_summary       - LLM summarization via Ollama (CPU)
  audio_indexer   - Ingests audio vectors to hybrid search
  audio_tts           - Text-to-speech via VibeVoice API (CPU)
"""

import os
import sys
import asyncio
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SERVICE_MAP = {
    "audio_orchestrator": "audio_orchestrator_service",
    "audio_analysis":     "audio_analysis_service",
    "audio_classifier":   "audio_classifier_service",
    "audio_diarization":  "audio_diarization_service",
    "audio_recognizer":   "audio_recognizer_service",
    "audio_summary":      "audio_summary_service",
    "audio_indexer":      "audio_indexer_service",
    "audio_tts":          "audio_tts_service",
}


def run():
    worker_type = os.environ.get("WORKER_TYPE", "").strip()

    if not worker_type:
        logger.error("WORKER_TYPE environment variable is not set")
        sys.exit(1)

    if worker_type not in SERVICE_MAP:
        logger.error(
            f"Unknown WORKER_TYPE: '{worker_type}'. "
            f"Valid values: {list(SERVICE_MAP.keys())}"
        )
        sys.exit(1)

    service_dir = os.path.join("/app/services", SERVICE_MAP[worker_type])
    if not os.path.isdir(service_dir):
        logger.error(f"Service directory not found: {service_dir}")
        sys.exit(1)

    sys.path.insert(0, service_dir)
    logger.info(f"Starting worker '{worker_type}' from {service_dir}")

    from main import main as service_main  # noqa: PLC0415
    asyncio.run(service_main())


if __name__ == "__main__":
    run()
