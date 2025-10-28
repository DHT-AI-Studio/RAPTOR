# services/audio_save2qdrant_service/main.py

import asyncio
import logging
import sys
from kafka_handler import AudioSave2QdrantKafkaHandler
import config

# 配置日誌
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('audio_save2qdrant_service.log')
    ]
)

logger = logging.getLogger(__name__)

async def main():
    """主程式入口"""
    logger.info("Starting Audio Save2Qdrant Service...")
    
    kafka_handler = AudioSave2QdrantKafkaHandler()
    
    try:
        await kafka_handler.start_consumer()
    except KeyboardInterrupt:
        logger.info("Shutting down Audio Save2Qdrant Service...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
