# services/document_orchestrator_service/main.py

import asyncio
import logging
import sys
from kafka_handler import DocumentOrchestratorKafkaHandler

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('document_orchestrator_service.log')
    ]
)

logger = logging.getLogger(__name__)

async def main():
    """主程式入口"""
    logger.info("Starting Document Orchestrator Service...")
    
    kafka_handler = DocumentOrchestratorKafkaHandler()
    
    try:
        await kafka_handler.start_consumer()
    except KeyboardInterrupt:
        logger.info("Shutting down Document Orchestrator Service...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
