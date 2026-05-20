# services/video_graph_service/main.py

import asyncio
import logging
import sys
from kafka_handler import VideoGraphKafkaHandler
import config

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


async def main():
    logger.info("Starting Video Graph Service...")
    handler = VideoGraphKafkaHandler()
    try:
        await handler.start()
    except KeyboardInterrupt:
        logger.info("Shutting down Video Graph Service...")


if __name__ == "__main__":
    asyncio.run(main())
