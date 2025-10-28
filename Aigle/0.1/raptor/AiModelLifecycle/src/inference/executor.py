# src/inference/executor.py
"""
模型執行器 - 負責實際的模型推理執行

結合引擎和模型處理器，執行完整的推理流程，包括：
- 模型加載和管理
- 數據預處理
- 推理執行
- 結果後處理
"""

import logging
import time
from typing import Dict, Any, Optional

# 核心依賴導入
try:
    from ..core.model_manager import model_manager
except ImportError:
    try:
        from src.core.model_manager import model_manager
    except ImportError:
        model_manager = None

logger = logging.getLogger(__name__)

class ModelExecutor:
    """
    模型執行器
    
    結合引擎和模型處理器，提供完整的推理執行功能。
    負責協調模型加載、數據處理和推理執行。
    """
    
    def __init__(self, engine, model_handler):
        """
        初始化模型執行器
        
        Args:
            engine: 推理引擎實例
            model_handler: 模型處理器實例
        """
        self.engine = engine
        self.model_handler = model_handler
        self._loaded_models: Dict[str, Any] = {}
        
        logger.debug(f"創建模型執行器: {engine.__class__.__name__} + {model_handler.__class__.__name__}")
    
    def execute(self, model_name: str, data: Dict[str, Any], options: Dict[str, Any]) -> Any:
        """
        執行推理
        
        Args:
            model_name (str): 模型名稱
            data (Dict[str, Any]): 輸入數據
            options (Dict[str, Any]): 推理選項
            
        Returns:
            Any: 推理結果
            
        Raises:
            Exception: 推理過程中的各種異常
        """
        try:
            start_time = time.time()
            logger.debug(f"開始執行推理: {model_name}")
            
            # 步驟 1: 確保模型已加載
            model = self._get_or_load_model(model_name)
            
            # 步驟 2: 數據預處理
            processed_data = self.model_handler.preprocess(data, options)
            logger.debug("數據預處理完成")
            
            # 步驟 3: 執行推理
            raw_result = self.engine.infer(model, processed_data, options)
            logger.debug("推理執行完成")
            
            # 步驟 4: 結果後處理
            final_result = self.model_handler.postprocess(raw_result, options)
            
            execution_time = time.time() - start_time
            logger.debug(f"推理執行完成，用時: {execution_time:.2f}秒")
            
            return final_result
            
        except Exception as e:
            logger.error(f"推理執行失敗: {e}")
            raise
    
    def _get_or_load_model(self, model_name: str) -> Any:
        """
        獲取或加載模型
        
        Args:
            model_name (str): 模型名稱
            
        Returns:
            Any: 模型實例
        """
        if model_name not in self._loaded_models:
            logger.info(f"加載模型: {model_name}")
            
            try:
                # 使用引擎加載模型
                model = self.engine.load_model(model_name)
                self._loaded_models[model_name] = model
                logger.info(f"模型加載成功: {model_name}")
                
            except Exception as e:
                logger.error(f"模型加載失敗: {model_name}, 錯誤: {e}")
                raise
        else:
            logger.debug(f"重用已加載的模型: {model_name}")
            
        return self._loaded_models[model_name]
    
    def unload_model(self, model_name: str) -> bool:
        """
        卸載模型
        
        Args:
            model_name (str): 模型名稱
            
        Returns:
            bool: 是否成功卸載
        """
        if model_name in self._loaded_models:
            try:
                # 如果引擎支持顯式卸載，調用引擎的卸載方法
                if hasattr(self.engine, 'unload_model'):
                    self.engine.unload_model(self._loaded_models[model_name])
                
                del self._loaded_models[model_name]
                logger.info(f"模型已卸載: {model_name}")
                return True
                
            except Exception as e:
                logger.error(f"模型卸載失敗: {model_name}, 錯誤: {e}")
                return False
        else:
            logger.warning(f"模型未加載，無需卸載: {model_name}")
            return False
    
    def get_loaded_models(self) -> list:
        """
        獲取已加載的模型列表
        
        Returns:
            list: 已加載的模型名稱列表
        """
        return list(self._loaded_models.keys())
    
    def clear_models(self):
        """清理所有已加載的模型"""
        model_names = list(self._loaded_models.keys())
        for model_name in model_names:
            self.unload_model(model_name)
        
        logger.info("所有模型已清理")
    
    def get_executor_info(self) -> Dict[str, Any]:
        """
        獲取執行器信息
        
        Returns:
            Dict[str, Any]: 執行器信息
        """
        return {
            'engine_type': self.engine.__class__.__name__,
            'model_handler_type': self.model_handler.__class__.__name__,
            'loaded_models': self.get_loaded_models(),
            'engine_info': getattr(self.engine, 'get_info', lambda: {})(),
            'handler_info': getattr(self.model_handler, 'get_info', lambda: {})()
        }