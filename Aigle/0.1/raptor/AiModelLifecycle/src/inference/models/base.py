# src/inference/models/base.py
"""
模型處理器基類

定義所有模型處理器必須實現的接口。
負責數據的預處理和後處理，以及模型特定的邏輯。
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

class BaseModelHandler(ABC):
    """
    模型處理器基類
    
    所有模型處理器都必須繼承此類並實現其抽象方法。
    負責處理模型的輸入預處理和輸出後處理。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化處理器
        
        Args:
            config (Dict[str, Any], optional): 處理器配置
        """
        self.config = config or {}
        self.name = self.__class__.__name__
        logger.debug(f"初始化模型處理器: {self.name}")
    
    @abstractmethod
    def preprocess(self, data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """
        預處理輸入數據
        
        Args:
            data (Dict[str, Any]): 原始輸入數據
            options (Dict[str, Any]): 處理選項
            
        Returns:
            Dict[str, Any]: 預處理後的數據
            
        Raises:
            ValueError: 輸入數據無效
            Exception: 預處理失敗
        """
        pass
    
    @abstractmethod
    def postprocess(self, result: Any, options: Dict[str, Any]) -> Any:
        """
        後處理推理結果
        
        Args:
            result (Any): 原始推理結果
            options (Dict[str, Any]): 處理選項
            
        Returns:
            Any: 後處理後的結果
            
        Raises:
            Exception: 後處理失敗
        """
        pass
    
    def validate_input(self, data: Dict[str, Any], required_keys: list) -> bool:
        """
        驗證輸入數據
        
        Args:
            data (Dict[str, Any]): 輸入數據
            required_keys (list): 必需的鍵列表
            
        Returns:
            bool: 是否通過驗證
            
        Raises:
            ValueError: 驗證失敗
        """
        if not isinstance(data, dict):
            raise ValueError("輸入數據必須是字典類型")
        
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            raise ValueError(f"缺少必需的輸入鍵: {missing_keys}")
        
        return True
    
    def get_info(self) -> Dict[str, Any]:
        """
        獲取處理器信息
        
        Returns:
            Dict[str, Any]: 處理器信息
        """
        return {
            'handler_name': self.name,
            'config': self.config,
            'supported_formats': getattr(self, 'supported_formats', []),
            'description': getattr(self, 'description', '')
        }
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"{self.name}(config={self.config})"
    
    def __repr__(self) -> str:
        """詳細字符串表示"""
        return f"{self.name}(config={self.config})"