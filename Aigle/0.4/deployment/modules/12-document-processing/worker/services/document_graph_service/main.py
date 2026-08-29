# services/document_graph_service/main.py

import asyncio
import logging
import sys
from kafka_handler import DocumentGraphKafkaHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("document_graph_service.log"),
    ],
)
logger = logging.getLogger(__name__)


async def main():
    logger.info("Starting Document Graph Service...")
    handler = DocumentGraphKafkaHandler()
    try:
        await handler.start_consumer()
    except KeyboardInterrupt:
        logger.info("Shutting down Document Graph Service...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
