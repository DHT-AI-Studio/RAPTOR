# src/inference/router.py
"""
任務路由器 - 負責將任務路由到相應的執行器

根據任務類型、引擎類型和模型名稱，選擇合適的執行器進行推理。
提供直接的路由機制。
"""

import logging
from typing import Dict, Any, Optional

# 模組內部導入
from .executor import ModelExecutor
from .engines.ollama import OllamaEngine
from .engines.transformers import TransformersEngine
from .models import get_model_handler

logger = logging.getLogger(__name__)

class TaskRouter:
    """
    任務路由器
    
    負責根據任務類型和引擎類型選擇合適的執行器。
    管理引擎實例的創建和重用。
    """
    
    def __init__(self):
        """初始化任務路由器"""
        # 引擎實例緩存
        self._engines: Dict[str, Any] = {}
        
        # 執行器緩存: {(task, engine): executor}
        self._executors: Dict[tuple, ModelExecutor] = {}
        
        # 任務與引擎的映射關係
        self._task_engine_mapping = {
            # 文本生成任務 - 細分為 Ollama 和 HuggingFace
            'text-generation-ollama': {
                'ollama': OllamaEngine
            },
            'text-generation-hf': {
                'transformers': TransformersEngine
            },
            # 向下兼容：通用文本生成任務
            'text-generation': {
                'ollama': OllamaEngine,
                'transformers': TransformersEngine
            },
            # 視覺語言模型
            'vlm': {
                'transformers': TransformersEngine
            },
            # 語音識別任務 - 細分
            'asr-hf': {
                'transformers': TransformersEngine
            },
            'asr': {
                'transformers': TransformersEngine
            },
            # 語音活動檢測
            'vad-hf': {
                'transformers': TransformersEngine
            },
            # 光學字符識別 - 細分
            'ocr-hf': {
                'transformers': TransformersEngine
            },
            'ocr': {
                'transformers': TransformersEngine
            },
            # 音頻分類
            'audio-classification': {
                'transformers': TransformersEngine
            },
            # 視頻分析
            'video-analysis': {
                'transformers': TransformersEngine
            },
            # 場景檢測
            'scene-detection': {
                'transformers': TransformersEngine
            },
            # 文檔分析
            'document-analysis': {
                'transformers': TransformersEngine
            },
            # 圖像標題生成
            'image-captioning': {
                'transformers': TransformersEngine
            },
            # 視頻摘要
            'video-summary': {
                'transformers': TransformersEngine
            },
            # 音頻轉錄
            'audio-transcription': {
                'transformers': TransformersEngine
            }
        }
        
        self._initialized = True
        logger.info("任務路由器初始化完成 (v2.1.1)")
    
    def route(self, task: str, engine: str, model_name: str) -> ModelExecutor:
        """
        路由到相應的執行器
        
        Args:
            task (str): 任務類型
            engine (str): 引擎類型
            model_name (str): 模型名稱
            
        Returns:
            ModelExecutor: 模型執行器實例
            
        Raises:
            ValueError: 不支持的任務或引擎組合
        """
        try:
            logger.debug(f"路由任務: {task} -> {engine} -> {model_name}")
            
            # 檢查任務類型是否支持
            if task not in self._task_engine_mapping:
                raise ValueError(f"不支持的任務類型: {task}")
            
            # 檢查引擎類型是否支持該任務
            task_engines = self._task_engine_mapping[task]
            if engine not in task_engines:
                supported_engines = list(task_engines.keys())
                raise ValueError(
                    f"任務 '{task}' 不支持引擎 '{engine}'，支持的引擎: {supported_engines}"
                )
            
            # 檢查是否已有緩存的執行器
            executor_key = (task, engine)
            if executor_key in self._executors:
                logger.debug(f"重用已緩存的執行器: {task} -> {engine}")
                return self._executors[executor_key]
            
            # 獲取或創建引擎實例
            engine_instance = self._get_or_create_engine(engine, task_engines[engine])
            
            # 獲取模型處理器
            model_handler = get_model_handler(task, model_name)
            
            # 創建並緩存執行器
            executor = ModelExecutor(engine_instance, model_handler)
            self._executors[executor_key] = executor
            
            logger.debug(f"成功創建並緩存執行器: {task} -> {engine}")
            return executor
            
        except Exception as e:
            logger.error(f"路由失敗: {e}")
            raise
    
    def _get_or_create_engine(self, engine_type: str, engine_class) -> Any:
        """
        獲取或創建引擎實例
        
        Args:
            engine_type (str): 引擎類型
            engine_class: 引擎類
            
        Returns:
            Any: 引擎實例
        """
        if engine_type not in self._engines:
            logger.debug(f"創建新的引擎實例: {engine_type}")
            # 為 Ollama engine 傳遞配置
            if engine_type == 'ollama':
                try:
                    from ..core.config import config as app_config
                    ollama_config = app_config.get_config("ollama") or {}
                    logger.debug(f"Ollama 引擎配置: {ollama_config}")
                    self._engines[engine_type] = engine_class(config=ollama_config)
                except Exception as e:
                    logger.warning(f"無法加載 Ollama 配置: {e}，使用默認配置")
                    self._engines[engine_type] = engine_class()
            else:
                self._engines[engine_type] = engine_class()
        else:
            logger.debug(f"重用現有引擎實例: {engine_type}")
            
        return self._engines[engine_type]
    
    def get_supported_combinations(self) -> Dict[str, Any]:
        """
        獲取支持的任務和引擎組合
        
        Returns:
            Dict[str, Any]: 支持的組合配置
        """
        return {
            task: {
                'engines': list(engines.keys()),
                'description': self._get_task_description(task)
            }
            for task, engines in self._task_engine_mapping.items()
        }
    
    def _get_task_description(self, task: str) -> str:
        """
        獲取任務描述
        
        Args:
            task (str): 任務類型
            
        Returns:
            str: 任務描述
        """
        descriptions = {
            'text-generation-ollama': '文本生成任務 (Ollama 引擎)',
            'text-generation-hf': '文本生成任務 (HuggingFace 引擎)',
            'text-generation': '文本生成任務（通用）',
            'vlm': '視覺語言模型任務，結合圖像和文本進行理解和生成',
            'asr-hf': '自動語音識別任務 (HuggingFace)',
            'asr': '自動語音識別任務（通用）',
            'vad-hf': '語音活動檢測任務 (HuggingFace)',
            'ocr-hf': '光學字符識別任務 (HuggingFace)',
            'ocr': '光學字符識別任務（通用）',
            'audio-classification': '音頻分類任務，對音頻內容進行分類',
            'video-analysis': '視頻分析任務，分析視頻內容和場景',
            'scene-detection': '場景檢測任務，檢測視頻中的場景變化',
            'document-analysis': '文檔分析任務，理解和提取文檔結構與內容',
            'image-captioning': '圖像標題生成任務，為圖像生成描述文本',
            'video-summary': '視頻摘要任務，生成視頻內容摘要',
            'audio-transcription': '音頻轉錄任務，將音頻轉換為文本'
        }
        return descriptions.get(task, f'{task} 任務')
    
    def clear_engines(self):
        """清理引擎和執行器緩存"""
        # 清理執行器中的模型
        for executor in self._executors.values():
            executor.clear_models()
        
        self._executors.clear()
        self._engines.clear()
        logger.info("引擎和執行器緩存已清理")