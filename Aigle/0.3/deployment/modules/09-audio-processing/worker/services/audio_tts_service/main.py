# services/audio_tts_service/main.py

import asyncio
import logging
import sys
from kafka_handler import AudioTTSKafkaHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("audio_tts_service.log")],
)
logger = logging.getLogger(__name__)


async def main():
    logger.info("Starting Audio TTS Service...")
    handler = AudioTTSKafkaHandler()
    try:
        await handler.start_consumer()
    except KeyboardInterrupt:
        logger.info("Shutting down Audio TTS Service...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
