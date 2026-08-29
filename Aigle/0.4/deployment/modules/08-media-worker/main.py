#!/usr/bin/env python3
"""
Media Processing Worker Dispatcher
Reads WORKER_TYPE env var and routes to the appropriate service.

Audio pipeline:
  audio_orchestrator  audio_analysis    audio_classifier  audio_diarization
  audio_recognizer    audio_summary     audio_indexer     audio_tts

Image pipeline:
  image_orchestrator  image_processing  image_indexer

Video pipeline:
  video_orchestrator  video_chunking        video_analysis   video_scene_detection
  video_ocr_frame     video_frame_description  video_summary  video_contextualize  video_indexer  video_graph

Document pipeline:
  document_orchestrator  document_analysis  document_summary
  document_indexer       document_graph
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
    # Audio
    "audio_orchestrator":      "audio_orchestrator_service",
    "audio_analysis":          "audio_analysis_service",
    "audio_classifier":        "audio_classifier_service",
    "audio_diarization":       "audio_diarization_service",
    "audio_recognizer":        "audio_recognizer_service",
    "audio_summary":           "audio_summary_service",
    "audio_indexer":           "audio_indexer_service",
    "audio_tts":               "audio_tts_service",
    # Image
    "image_orchestrator":      "image_orchestrator_service",
    "image_processing":        "image_processing_service",
    "image_indexer":           "image_indexer_service",
    # Video
    "video_orchestrator":      "video_orchestrator_service",
    "video_chunking":          "video_chunking_service",
    "video_analysis":          "video_analysis_service",
    "video_scene_detection":   "video_scene_detection_service",
    "video_ocr_frame":         "video_ocr_frame_service",
    "video_frame_description": "video_frame_description_service",
    "video_summary":           "video_summary_service",
    "video_contextualize":     "video_contextualize_service",
    "video_indexer":           "video_indexer_service",
    "video_graph":             "video_graph_service",
    # Document
    "document_orchestrator":   "document_orchestrator_service",
    "document_analysis":       "document_analysis_service",
    "document_summary":        "document_summary_service",
    "document_indexer":        "document_indexer_service",
    "document_graph":          "document_graph_service",
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
