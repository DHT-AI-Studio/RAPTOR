# services/video_ocr_frame_service/main.py

import asyncio
import logging
import os
import sys
import uvicorn

from kafka_handler import OCRFrameKafkaHandler
from ocr_frame import VideoOCRFrameProcessor
from config import LOG_LEVEL, LOG_FILE
import sync_api

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE)
    ]
)

logger = logging.getLogger(__name__)

SYNC_PORT = int(os.getenv("PORT_VIDEO_OCR_SYNC", "8032"))


async def main():
    logger.info("Starting Video OCR Frame Service with Sync API...")
    logger.info(f"Sync API Port: {SYNC_PORT}")

    # Persistent OCR processor for sync API (cleanup_after=False keeps engine alive).
    # Kafka handler creates its own per-request processors independently.
    ocr_processor = VideoOCRFrameProcessor(
        output_dir="/tmp/media_processing/video_processing/ocr_sync_frames"
    )

    sync_api.init(ocr_processor)
    kafka_handler = OCRFrameKafkaHandler()

    server_config = uvicorn.Config(
        sync_api.app, host="0.0.0.0", port=SYNC_PORT, log_level="info"
    )
    server = uvicorn.Server(server_config)

    try:
        await asyncio.gather(
            kafka_handler.start_consumer(),
            server.serve(),
        )
    except KeyboardInterrupt:
        logger.info("Shutting down Video OCR Frame Service...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
