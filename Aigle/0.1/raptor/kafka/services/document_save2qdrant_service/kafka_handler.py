# services/document_save2qdrant_service/kafka_handler.py

import asyncio
import json
import logging
import aiohttp
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from typing import Dict, Any
import config
from message_utils import create_response_message
from dotenv import load_dotenv
import os
# 計算上層資料夾的路徑
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 構建 .env 檔案的完整路徑
dotenv_path = os.path.join(parent_dir, ".env")

# 載入上層資料夾的 .env 檔案
load_dotenv(dotenv_path)
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
QDRANT_API_URL = os.getenv("QDRANT_API_URL")
logger = logging.getLogger(__name__)

class DocumentSave2QdrantKafkaHandler:
    def __init__(self):
        self.consumer = None
        self.producer = None
        
    async def start_consumer(self):
        """啟動 Kafka consumer"""
        self.consumer = AIOKafkaConsumer(
            config.KAFKA_TOPIC_SAVE_REQUEST,
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            group_id=config.KAFKA_GROUP_ID,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='latest',
            enable_auto_commit=True
        )
        
        self.producer = AIOKafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        
        await self.consumer.start()
        await self.producer.start()
        logger.info(f"Kafka consumer started, listening to {config.KAFKA_TOPIC_SAVE_REQUEST}")
        
        try:
            async for message in self.consumer:
                await self.process_message(message.value)
        finally:
            await self.consumer.stop()
            await self.producer.stop()
    
    async def process_message(self, message: Dict[str, Any]):
        """處理接收到的訊息"""
        try:
            logger.info(f"Received message: {message.get('message_id')}")
            
            payload = message.get("payload", {})
            parameters = payload.get("parameters", {})
            summary_result_path = parameters.get("summary_result_path")
            
            if not summary_result_path:
                raise ValueError("Missing summary_result_path in parameters")
            
            # 調用 Qdrant API
            results = await self.save_to_qdrant(summary_result_path)
            
            # 發送成功回應
            response = create_response_message(
                message,
                status="success",
                results=results
            )
            
            await self.send_response(response)
            logger.info(f"Successfully processed message: {message.get('message_id')}")
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            
            # 發送錯誤回應
            response = create_response_message(
                message,
                status="failed",
                error=str(e)
            )
            
            await self.send_response(response)
    
    async def save_to_qdrant(self, file_path: str) -> Dict[str, Any]:
        """調用 Qdrant API 保存數據"""
        try:
            async with aiohttp.ClientSession() as session:
                with open(file_path, 'rb') as f:
                    data = aiohttp.FormData()
                    data.add_field('file',
                                   f,
                                   filename=file_path.split('/')[-1],
                                   content_type='application/json')
                    
                    async with session.post(QDRANT_API_URL, data=data) as resp:
                        if resp.status != 200:
                            raise Exception(f"Qdrant API error: {resp.status}")
                        
                        result = await resp.json()
                        logger.info(f"Qdrant API response: {result}")
                        return result.get("results", {})
                        
        except Exception as e:
            logger.error(f"Error calling Qdrant API: {e}")
            raise
    
    async def send_response(self, response: Dict[str, Any]):
        """發送回應到 Kafka"""
        try:
            await self.producer.send_and_wait(
                config.KAFKA_TOPIC_SAVE_RESULT,
                response
            )
            logger.info(f"Response sent: {response.get('message_id')}")
        except Exception as e:
            logger.error(f"Error sending response: {e}")
            raise
