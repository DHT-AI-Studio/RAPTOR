# services/video_summary_service/main.py

import asyncio
import logging
import sys
from kafka_handler import VideoSummaryKafkaHandler

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('video_summary_service.log')
    ]
)

logger = logging.getLogger(__name__)

async def main():
    """主程式入口"""
    logger.info("Starting Video Summary Service with Standard Message Format...")
    
    kafka_handler = VideoSummaryKafkaHandler()
    
    try:
        await kafka_handler.start_consumer()
    except KeyboardInterrupt:
        logger.info("Shutting down Video Summary Service...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
