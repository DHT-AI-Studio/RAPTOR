# services/video_chunking_service/main.py

import asyncio
import logging
import sys
from kafka_handler import VideoChunkingKafkaHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("video_chunking_service.log")],
)
logger = logging.getLogger(__name__)


async def main():
    logger.info("Starting Video Chunking Service...")
    handler = VideoChunkingKafkaHandler()
    try:
        await handler.start_consumer()
    except KeyboardInterrupt:
        logger.info("Shutting down Video Chunking Service...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
