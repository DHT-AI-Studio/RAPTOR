# services/video_orchestrator_service/main.py

import asyncio
import logging
import sys
from kafka_handler import VideoOrchestratorKafkaHandler
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
    logger.info("Starting Video Orchestrator Service...")
    
    kafka_handler = VideoOrchestratorKafkaHandler()
    
    try:
        await kafka_handler.start_consumer()
    except KeyboardInterrupt:
        logger.info("Shutting down Video Orchestrator Service...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
