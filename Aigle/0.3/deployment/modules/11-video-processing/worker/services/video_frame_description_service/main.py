# services/video_frame_description_service/main.py

import asyncio
import concurrent.futures
import logging
import sys
import os
import uvicorn

from config import LOG_LEVEL, LOG_FILE

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE)
    ]
)

logger = logging.getLogger(__name__)

SYNC_API_PORT = int(os.getenv("VIDEO_FRAME_SYNC_API_PORT", "8021"))
INFERENCE_URL = os.getenv("INFERENCE_URL", "http://raptor-ai-lifecycle-api:8010")
INFERENCE_MODEL = os.getenv("INFERENCE_MODEL", "qwen3.5:9b")
SUMMARY_MAX_TOKENS = int(os.getenv("SUMMARY_MAX_TOKENS", "1600"))
SUMMARY_MAX_LENGTH = int(os.getenv("SUMMARY_MAX_LENGTH", "300"))

# Load VideoSummary from sibling service directory (same worker image, different service path)
_services_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(_services_dir, "video_summary_service"))

from kafka_handler import FrameDescriptionKafkaHandler
from frame_description_processor import FrameDescriptionProcessor
from video_summary import VideoSummary
import sync_api


async def main():
    logger.info("Starting Video Frame Description Service with Sync API...")
    logger.info(f"Sync API Port: {SYNC_API_PORT}")

    processor = FrameDescriptionProcessor()
    summarizer = VideoSummary(
        model_name=INFERENCE_MODEL,
        inference_url=INFERENCE_URL,
        max_tokens_per_batch=SUMMARY_MAX_TOKENS,
        max_summary_length=SUMMARY_MAX_LENGTH,
    )

    # max_workers=1 serializes GPU inference across Kafka consumer and sync API
    thread_pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="VideoFrameDesc"
    )

    sync_api.init(processor, summarizer, thread_pool)
    kafka_handler = FrameDescriptionKafkaHandler(processor=processor)

    server_config = uvicorn.Config(
        sync_api.app, host="0.0.0.0", port=SYNC_API_PORT, log_level="info"
    )
    server = uvicorn.Server(server_config)

    try:
        await asyncio.gather(
            kafka_handler.start_consumer(),
            server.serve(),
        )
    except KeyboardInterrupt:
        logger.info("Shutting down Video Frame Description Service...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise
    finally:
        thread_pool.shutdown(wait=False)


if __name__ == "__main__":
    asyncio.run(main())
