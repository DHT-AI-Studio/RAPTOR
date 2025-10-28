# src/inference/cache.py
"""
模型緩存管理器 - 管理模型實例的緩存和生命週期

提供智能的模型緩存機制，包括：
- LRU 緩存策略
- 內存使用監控
- 自動清理機制
- 緩存統計和監控
"""

import logging
import time
import threading
from typing import Dict, Any, Optional, Tuple
from collections import OrderedDict
import gc

# GPU 管理器導入
try:
    from ..core.gpu_manager import gpu_manager
except ImportError:
    try:
        from src.core.gpu_manager import gpu_manager
    except ImportError:
        gpu_manager = None

logger = logging.getLogger(__name__)

class ModelCache:
    """
    模型緩存管理器
    
    實現 LRU 緩存策略，自動管理模型實例的生命週期。
    監控內存使用，在需要時自動清理緩存。
    """
    
    def __init__(self, max_cache_size: int = 5, max_memory_gb: float = 8.0):
        """
        初始化模型緩存
        
        Args:
            max_cache_size (int): 最大緩存模型數量
            max_memory_gb (float): 最大內存使用量（GB）
        """
        self.max_cache_size = max_cache_size
        self.max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024
        
        # 使用 OrderedDict 實現 LRU
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._cache_lock = threading.RLock()
        
        # 緩存統計
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_requests': 0
        }
        
        logger.info(f"模型緩存初始化 - 最大緩存: {max_cache_size}, 最大內存: {max_memory_gb}GB")
    
    def get(self, model_key: str) -> Optional[Any]:
        """
        從緩存獲取模型
        
        Args:
            model_key (str): 模型鍵值
            
        Returns:
            Optional[Any]: 模型實例，如果不存在則返回 None
        """
        with self._cache_lock:
            self._stats['total_requests'] += 1
            
            if model_key in self._cache:
                # 移動到最後（最近使用）
                cache_entry = self._cache.pop(model_key)
                self._cache[model_key] = cache_entry
                
                # 更新訪問時間
                cache_entry['last_accessed'] = time.time()
                cache_entry['access_count'] += 1
                
                self._stats['hits'] += 1
                logger.debug(f"緩存命中: {model_key}")
                return cache_entry['model']
            else:
                self._stats['misses'] += 1
                logger.debug(f"緩存未命中: {model_key}")
                return None
    
    def put(self, model_key: str, model: Any, metadata: Optional[Dict[str, Any]] = None):
        """
        將模型放入緩存
        
        Args:
            model_key (str): 模型鍵值
            model (Any): 模型實例
            metadata (Dict[str, Any], optional): 模型元數據
        """
        with self._cache_lock:
            current_time = time.time()
            
            # 創建緩存條目
            cache_entry = {
                'model': model,
                'created_time': current_time,
                'last_accessed': current_time,
                'access_count': 1,
                'memory_usage': self._estimate_model_memory(model),
                'metadata': metadata or {}
            }
            
            # 如果已存在，更新
            if model_key in self._cache:
                self._cache.pop(model_key)
            
            # 檢查是否需要清理空間
            self._make_space_if_needed(cache_entry['memory_usage'])
            
            # 添加到緩存
            self._cache[model_key] = cache_entry
            
            logger.debug(f"模型已緩存: {model_key}, 內存使用: {cache_entry['memory_usage']:.2f}MB")
    
    def remove(self, model_key: str) -> bool:
        """
        從緩存移除模型
        
        Args:
            model_key (str): 模型鍵值
            
        Returns:
            bool: 是否成功移除
        """
        with self._cache_lock:
            if model_key in self._cache:
                cache_entry = self._cache.pop(model_key)
                self._cleanup_model(cache_entry['model'])
                logger.debug(f"模型已從緩存移除: {model_key}")
                return True
            else:
                logger.debug(f"模型不在緩存中: {model_key}")
                return False
    
    def clear(self):
        """清空所有緩存"""
        with self._cache_lock:
            for model_key, cache_entry in self._cache.items():
                self._cleanup_model(cache_entry['model'])
            
            self._cache.clear()
            logger.info("模型緩存已清空")
    
    def _make_space_if_needed(self, required_memory: float):
        """
        如果需要，清理空間
        
        Args:
            required_memory (float): 需要的內存量（字節）
        """
        # 檢查緩存數量限制
        while len(self._cache) >= self.max_cache_size:
            self._evict_oldest()
        
        # 檢查內存限制
        current_memory = self._get_total_cache_memory()
        while current_memory + required_memory > self.max_memory_bytes and self._cache:
            evicted_memory = self._evict_oldest()
            current_memory -= evicted_memory
    
    def _evict_oldest(self) -> float:
        """
        清除最老的模型
        
        Returns:
            float: 被清除模型的內存使用量
        """
        if not self._cache:
            return 0.0
        
        # 獲取最老的條目（LRU）
        oldest_key, oldest_entry = next(iter(self._cache.items()))
        self._cache.pop(oldest_key)
        
        memory_freed = oldest_entry['memory_usage']
        self._cleanup_model(oldest_entry['model'])
        
        self._stats['evictions'] += 1
        logger.debug(f"清除最老模型: {oldest_key}, 釋放內存: {memory_freed:.2f}MB")
        
        return memory_freed
    
    def _cleanup_model(self, model: Any):
        """
        清理模型實例
        
        Args:
            model (Any): 模型實例
        """
        try:
            # 如果模型有特定的清理方法，調用它
            if hasattr(model, 'cleanup'):
                model.cleanup()
            elif hasattr(model, 'cpu'):
                # PyTorch 模型移動到 CPU
                model.cpu()
            
            # 觸發垃圾回收
            del model
            gc.collect()
            
            # 如果有 GPU 管理器，清理 GPU 緩存
            if gpu_manager:
                gpu_manager.clear_cache()
                
        except Exception as e:
            logger.warning(f"模型清理時出現警告: {e}")
    
    def _estimate_model_memory(self, model: Any) -> float:
        """
        估算模型內存使用量
        
        Args:
            model (Any): 模型實例
            
        Returns:
            float: 估算的內存使用量（字節）
        """
        try:
            # 嘗試 PyTorch 模型
            if hasattr(model, 'parameters'):
                total_params = sum(p.numel() * p.element_size() for p in model.parameters())
                return total_params
            
            # 嘗試獲取物件大小
            import sys
            return sys.getsizeof(model)
            
        except Exception as e:
            logger.debug(f"無法估算模型內存使用量: {e}")
            # 返回默認估算值（500MB）
            return 500 * 1024 * 1024
    
    def _get_total_cache_memory(self) -> float:
        """
        獲取緩存總內存使用量
        
        Returns:
            float: 總內存使用量（字節）
        """
        return sum(entry['memory_usage'] for entry in self._cache.values())
    
    def get_stats(self) -> Dict[str, Any]:
        """
        獲取緩存統計信息
        
        Returns:
            Dict[str, Any]: 統計信息
        """
        with self._cache_lock:
            hit_rate = (
                self._stats['hits'] / max(self._stats['total_requests'], 1)
            )
            
            total_memory_mb = self._get_total_cache_memory() / (1024 * 1024)
            
            return {
                'cache_size': len(self._cache),
                'max_cache_size': self.max_cache_size,
                'hit_rate': hit_rate,
                'total_memory_mb': total_memory_mb,
                'max_memory_mb': self.max_memory_bytes / (1024 * 1024),
                **self._stats
            }
    
    def get_cached_models(self) -> Dict[str, Dict[str, Any]]:
        """
        獲取緩存中的模型信息
        
        Returns:
            Dict[str, Dict[str, Any]]: 模型信息
        """
        with self._cache_lock:
            result = {}
            for model_key, cache_entry in self._cache.items():
                result[model_key] = {
                    'created_time': cache_entry['created_time'],
                    'last_accessed': cache_entry['last_accessed'],
                    'access_count': cache_entry['access_count'],
                    'memory_usage_mb': cache_entry['memory_usage'] / (1024 * 1024),
                    'metadata': cache_entry['metadata']
                }
            return result
    
    def resize_cache(self, new_max_size: int):
        """
        調整緩存大小
        
        Args:
            new_max_size (int): 新的最大緩存大小
        """
        with self._cache_lock:
            self.max_cache_size = new_max_size
            
            # 如果當前緩存超過新限制，清理多餘的模型
            while len(self._cache) > new_max_size:
                self._evict_oldest()
            
            logger.info(f"緩存大小已調整為: {new_max_size}")
    
    def set_memory_limit(self, max_memory_gb: float):
        """
        設置內存限制
        
        Args:
            max_memory_gb (float): 最大內存使用量（GB）
        """
        with self._cache_lock:
            self.max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024
            
            # 檢查當前內存使用是否超限
            current_memory = self._get_total_cache_memory()
            while current_memory > self.max_memory_bytes and self._cache:
                evicted_memory = self._evict_oldest()
                current_memory -= evicted_memory
            
            logger.info(f"內存限制已調整為: {max_memory_gb}GB")