#!/usr/bin/env python3
"""
Vision Processing Worker Dispatcher
Reads WORKER_TYPE env var and routes to the appropriate service.

Valid WORKER_TYPE values:
  image_orchestrator        - Receives image upload events, fans out
  image_processing          - VLM description + OCR (GPU: InternVL)
  image_indexer             - Ingests image vectors to hybrid search

  video_orchestrator        - Receives video upload events, fans out
  video_chunking            - Splits video into time-based chunks (ffmpeg)
  video_analysis            - Frame extraction, coordinates parallel workers
  video_scene_detection     - Scene boundary detection (cv2)
  video_ocr_frame           - OCR on video frames (GPU: PaddleOCR)
  video_frame_description   - VLM description of frames (GPU: InternVL)
  video_summary             - LLM summarization via Ollama
  video_contextualize       - Contextual RAG prefix generation via LLM
  video_indexer             - Persists video vectors to hybrid search
  video_graph               - Forwards indexed records to graph-service (module 20)
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
    "image_orchestrator":      "image_orchestrator_service",
    "image_processing":        "image_processing_service",
    "image_indexer":           "image_indexer_service",
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
