# services/document_orchestrator_service/redis_manager.py

import json
import logging
from typing import Dict, Any, Optional
import redis
from config import (
    #REDIS_HOST,
    #REDIS_PORT,
    REDIS_DB,
    REDIS_KEY_PREFIX,
    REDIS_KEY_TTL
)
from dotenv import load_dotenv
import os
# 計算上層資料夾的路徑
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 構建 .env 檔案的完整路徑
dotenv_path = os.path.join(parent_dir, ".env")

# 載入上層資料夾的 .env 檔案
load_dotenv(dotenv_path)
REDIS_HOST = os.getenv("REDIS_HOST",)
REDIS_PORT = int(os.getenv("REDIS_PORT"))
logger = logging.getLogger(__name__)

class RedisStateManager:
    def __init__(self):
        self.client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_keepalive=True,
            health_check_interval=30
        )
        self.key_prefix = REDIS_KEY_PREFIX
        self.key_ttl = REDIS_KEY_TTL
        
        # 測試連接
        try:
            self.client.ping()
            logger.info(f"Redis connected successfully at {REDIS_HOST}:{REDIS_PORT}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def _make_key(self, correlation_id: str) -> str:
        """生成 Redis key"""
        return f"{self.key_prefix}{correlation_id}"
    
    def set_state(self, correlation_id: str, state: Dict[str, Any]) -> bool:
        """設置處理狀態"""
        try:
            key = self._make_key(correlation_id)
            # 將 state 序列化為 JSON
            value = json.dumps(state, ensure_ascii=False, default=str)
            self.client.setex(key, self.key_ttl, value)
            logger.debug(f"State saved to Redis: {correlation_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to set state in Redis: {e}")
            return False
    
    def get_state(self, correlation_id: str) -> Optional[Dict[str, Any]]:
        """獲取處理狀態"""
        try:
            key = self._make_key(correlation_id)
            value = self.client.get(key)
            if value:
                state = json.loads(value)
                logger.debug(f"State retrieved from Redis: {correlation_id}")
                return state
            return None
        except Exception as e:
            logger.error(f"Failed to get state from Redis: {e}")
            return None
    
    def update_state(self, correlation_id: str, updates: Dict[str, Any]) -> bool:
        """更新處理狀態"""
        try:
            state = self.get_state(correlation_id)
            if state is None:
                logger.warning(f"State not found for update: {correlation_id}")
                return False
            
            # 更新狀態
            state.update(updates)
            return self.set_state(correlation_id, state)
        except Exception as e:
            logger.error(f"Failed to update state in Redis: {e}")
            return False
    
    def delete_state(self, correlation_id: str) -> bool:
        """刪除處理狀態"""
        try:
            key = self._make_key(correlation_id)
            self.client.delete(key)
            logger.debug(f"State deleted from Redis: {correlation_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete state from Redis: {e}")
            return False
    
    def exists(self, correlation_id: str) -> bool:
        """檢查狀態是否存在"""
        try:
            key = self._make_key(correlation_id)
            return self.client.exists(key) > 0
        except Exception as e:
            logger.error(f"Failed to check state existence in Redis: {e}")
            return False
    
    def extend_ttl(self, correlation_id: str, ttl: int = None) -> bool:
        """延長 TTL"""
        try:
            key = self._make_key(correlation_id)
            ttl = ttl or self.key_ttl
            self.client.expire(key, ttl)
            return True
        except Exception as e:
            logger.error(f"Failed to extend TTL in Redis: {e}")
            return False
    
    def close(self):
        """關閉 Redis 連接"""
        try:
            self.client.close()
            logger.info("Redis connection closed")
        except Exception as e:
            logger.error(f"Error closing Redis connection: {e}")
