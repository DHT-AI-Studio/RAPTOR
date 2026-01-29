# services/audio_orchestrator_service/main.py

import asyncio
import logging
import sys
from kafka_handler import AudioOrchestratorKafkaHandler
from config import LOG_LEVEL, LOG_FILE

# 配置日誌
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE)
    ]
)

logger = logging.getLogger(__name__)

async def main():
    """主程式入口"""
    logger.info("Starting Audio Orchestrator Service...")
    
    kafka_handler = AudioOrchestratorKafkaHandler()
    
    try:
        await kafka_handler.start_consumer()
    except KeyboardInterrupt:
        logger.info("Shutting down Audio Orchestrator Service...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
