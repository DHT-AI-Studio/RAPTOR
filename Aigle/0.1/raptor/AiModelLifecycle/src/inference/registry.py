# src/inference/registry.py
"""
模型註冊表 - 管理模型和處理器的註冊

提供裝飾器和註冊機制，允許動態註冊新的模型處理器和引擎。
支持插件式擴展，便於添加自定義模型。
"""

import logging
from typing import Dict, Any, Type, Callable, Optional
from functools import wraps

logger = logging.getLogger(__name__)

class ModelRegistry:
    """
    模型註冊表
    
    管理模型處理器和引擎的註冊，提供統一的註冊和查找機制。
    支持裝飾器式註冊和手動註冊。
    """
    
    _instance: Optional['ModelRegistry'] = None
    
    def __new__(cls):
        """單例模式實現"""
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """初始化註冊表"""
        if self._initialized:
            return
            
        # 模型處理器註冊表: {(task, model_type): handler_class}
        self._model_handlers: Dict[tuple, Type] = {}
        
        # 引擎註冊表: {engine_name: engine_class}
        self._engines: Dict[str, Type] = {}
        
        # 元數據註冊表
        self._handler_metadata: Dict[tuple, Dict[str, Any]] = {}
        self._engine_metadata: Dict[str, Dict[str, Any]] = {}
        
        self._initialized = True
        logger.info("模型註冊表初始化完成")
    
    def register_model_handler(self, 
                             task: str, 
                             model_type: str = "default",
                             **metadata) -> Callable:
        """
        註冊模型處理器的裝飾器
        
        Args:
            task (str): 任務類型
            model_type (str): 模型類型，默認為 "default"
            **metadata: 額外的元數據
            
        Returns:
            Callable: 裝飾器函數
            
        Usage:
            @register_model_handler("vlm", "llava")
            class LlavaHandler(BaseModelHandler):
                pass
        """
        def decorator(handler_class: Type) -> Type:
            key = (task, model_type)
            self._model_handlers[key] = handler_class
            self._handler_metadata[key] = {
                'task': task,
                'model_type': model_type,
                'class_name': handler_class.__name__,
                **metadata
            }
            
            logger.info(f"註冊模型處理器: {task}/{model_type} -> {handler_class.__name__}")
            return handler_class
        
        return decorator
    
    def register_engine(self, engine_name: str, **metadata) -> Callable:
        """
        註冊引擎的裝飾器
        
        Args:
            engine_name (str): 引擎名稱
            **metadata: 額外的元數據
            
        Returns:
            Callable: 裝飾器函數
            
        Usage:
            @register_engine("vllm")
            class VLLMEngine(BaseEngine):
                pass
        """
        def decorator(engine_class: Type) -> Type:
            self._engines[engine_name] = engine_class
            self._engine_metadata[engine_name] = {
                'engine_name': engine_name,
                'class_name': engine_class.__name__,
                **metadata
            }
            
            logger.info(f"註冊引擎: {engine_name} -> {engine_class.__name__}")
            return engine_class
        
        return decorator
    
    def get_model_handler(self, task: str, model_name: str) -> Type:
        """
        獲取模型處理器類
        
        Args:
            task (str): 任務類型
            model_name (str): 模型名稱
            
        Returns:
            Type: 模型處理器類
            
        Raises:
            KeyError: 找不到對應的處理器
        """
        # 首先嘗試精確匹配模型名稱
        model_type = self._infer_model_type(model_name)
        key = (task, model_type)
        
        if key in self._model_handlers:
            logger.debug(f"找到精確匹配的處理器: {task}/{model_type}")
            return self._model_handlers[key]
        
        # 回退到默認處理器
        default_key = (task, "default")
        if default_key in self._model_handlers:
            logger.debug(f"使用默認處理器: {task}/default")
            return self._model_handlers[default_key]
        
        # 如果都找不到，拋出異常
        available_keys = list(self._model_handlers.keys())
        raise KeyError(
            f"找不到任務 '{task}' 和模型 '{model_name}' 的處理器。"
            f"可用的處理器: {available_keys}"
        )
    
    def get_engine(self, engine_name: str) -> Type:
        """
        獲取引擎類
        
        Args:
            engine_name (str): 引擎名稱
            
        Returns:
            Type: 引擎類
            
        Raises:
            KeyError: 找不到對應的引擎
        """
        if engine_name not in self._engines:
            available_engines = list(self._engines.keys())
            raise KeyError(
                f"找不到引擎 '{engine_name}'。可用的引擎: {available_engines}"
            )
        
        return self._engines[engine_name]
    
    def _infer_model_type(self, model_name: str) -> str:
        """
        從模型名稱推斷模型類型
        
        Args:
            model_name (str): 模型名稱
            
        Returns:
            str: 推斷的模型類型
        """
        model_name_lower = model_name.lower()
        
        # 定義模型類型推斷規則
        type_rules = {
            'llava': ['llava'],
            'blip': ['blip'],
            'whisper': ['whisper'],
            'wav2vec': ['wav2vec'],
            'trocr': ['trocr'],
            'paddleocr': ['paddle'],
            'layoutlm': ['layout'],
            'donut': ['donut'],
            'bert': ['bert'],
            'gpt': ['gpt'],
            'llama': ['llama'],
            'mistral': ['mistral'],
            'qwen': ['qwen']
        }
        
        for model_type, keywords in type_rules.items():
            if any(keyword in model_name_lower for keyword in keywords):
                return model_type
        
        return "default"
    
    def list_registered_handlers(self) -> Dict[tuple, Dict[str, Any]]:
        """
        列出所有註冊的模型處理器
        
        Returns:
            Dict[tuple, Dict[str, Any]]: 處理器信息
        """
        return self._handler_metadata.copy()
    
    def list_registered_engines(self) -> Dict[str, Dict[str, Any]]:
        """
        列出所有註冊的引擎
        
        Returns:
            Dict[str, Dict[str, Any]]: 引擎信息
        """
        return self._engine_metadata.copy()
    
    def register_handler_manually(self, task: str, model_type: str, handler_class: Type, **metadata):
        """
        手動註冊模型處理器
        
        Args:
            task (str): 任務類型
            model_type (str): 模型類型
            handler_class (Type): 處理器類
            **metadata: 額外元數據
        """
        key = (task, model_type)
        self._model_handlers[key] = handler_class
        self._handler_metadata[key] = {
            'task': task,
            'model_type': model_type,
            'class_name': handler_class.__name__,
            **metadata
        }
        
        logger.info(f"手動註冊模型處理器: {task}/{model_type} -> {handler_class.__name__}")
    
    def register_engine_manually(self, engine_name: str, engine_class: Type, **metadata):
        """
        手動註冊引擎
        
        Args:
            engine_name (str): 引擎名稱
            engine_class (Type): 引擎類
            **metadata: 額外元數據
        """
        self._engines[engine_name] = engine_class
        self._engine_metadata[engine_name] = {
            'engine_name': engine_name,
            'class_name': engine_class.__name__,
            **metadata
        }
        
        logger.info(f"手動註冊引擎: {engine_name} -> {engine_class.__name__}")
    
    def unregister_handler(self, task: str, model_type: str = "default"):
        """
        取消註冊模型處理器
        
        Args:
            task (str): 任務類型
            model_type (str): 模型類型
        """
        key = (task, model_type)
        if key in self._model_handlers:
            del self._model_handlers[key]
            del self._handler_metadata[key]
            logger.info(f"取消註冊模型處理器: {task}/{model_type}")
        else:
            logger.warning(f"處理器未註冊: {task}/{model_type}")
    
    def unregister_engine(self, engine_name: str):
        """
        取消註冊引擎
        
        Args:
            engine_name (str): 引擎名稱
        """
        if engine_name in self._engines:
            del self._engines[engine_name]
            del self._engine_metadata[engine_name]
            logger.info(f"取消註冊引擎: {engine_name}")
        else:
            logger.warning(f"引擎未註冊: {engine_name}")

# 創建全局註冊表實例
model_registry = ModelRegistry()

# 提供便捷的裝飾器
register_model_handler = model_registry.register_model_handler
register_engine = model_registry.register_engine