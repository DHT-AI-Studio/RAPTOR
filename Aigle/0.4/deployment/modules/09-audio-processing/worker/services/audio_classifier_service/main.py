# services/audio_classifier_service/main.py

import asyncio
import concurrent.futures
import logging
import os
import sys
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('audio_classifier_service.log')
    ]
)

logger = logging.getLogger(__name__)

SYNC_API_PORT = int(os.getenv("PORT_AUDIO_CLASSIFIER_SYNC", "8039"))

from audio_classification import AudioClassificationClient
from kafka_handler import AudioClassifierKafkaHandler
import sync_api


async def main():
    logger.info("Starting Audio Classifier Service (Kafka + Sync API)...")
    logger.info(f"Sync API Port: {SYNC_API_PORT}")

    classification_client = AudioClassificationClient()

    # max_workers=1 serializes GPU calls across Kafka consumer and sync API
    thread_pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="AudioClassifier"
    )

    sync_api.init(classification_client, thread_pool)
    kafka_handler = AudioClassifierKafkaHandler(
        classification_client=classification_client,
        thread_pool=thread_pool,
    )

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
        logger.info("Shutting down Audio Classifier Service...")
    finally:
        thread_pool.shutdown(wait=False)


if __name__ == "__main__":
    asyncio.run(main())
