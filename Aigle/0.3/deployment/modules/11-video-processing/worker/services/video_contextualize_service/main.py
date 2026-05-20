# services/video_contextualize_service/main.py

import asyncio
import logging
import sys

from kafka_handler import VideoContextualizeKafkaHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("video_contextualize_service.log"),
    ],
)

logger = logging.getLogger(__name__)


async def main():
    logger.info("Starting Video Contextualize Service...")
    handler = VideoContextualizeKafkaHandler()
    try:
        await handler.start_consumer()
    except KeyboardInterrupt:
        logger.info("Shutting down Video Contextualize Service...")
    except Exception as exc:
        logger.error("Unexpected error: %s", exc)
        raise


if __name__ == "__main__":
    asyncio.run(main())
