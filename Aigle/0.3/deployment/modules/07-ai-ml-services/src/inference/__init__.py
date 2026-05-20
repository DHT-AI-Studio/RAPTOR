# src/inference/__init__.py
"""
重構後的推理模組

提供簡化、統一的推理接口，支持多種AI任務和引擎。

主要組件:
- InferenceManager: 統一推理管理器
- TaskRouter: 任務路由器  
- ModelExecutor: 模型執行器
- ModelRegistry: 模型註冊表
- ModelCache: 模型緩存管理器

支持的任務:
- text-generation: 文本生成
- vlm: 視覺語言模型
- asr: 自動語音識別
- ocr: 光學字符識別
- audio-classification: 音頻分類
- video-analysis: 視頻分析
- document-analysis: 文檔分析

支持的引擎:
- ollama: 用於文本生成
- transformers: 用於所有任務類型
"""

from .manager import InferenceManager, inference_manager
from .router import TaskRouter
from .executor import ModelExecutor
from .registry import ModelRegistry, model_registry
from .cache import ModelCache

# 引用新的API
from ..api.inference_api import router as inference_router

__version__ = "2.0.0"
__all__ = [
    'InferenceManager',
    'inference_manager', 
    'TaskRouter',
    'ModelExecutor',
    'ModelRegistry',
    'model_registry',
    'ModelCache',
    'inference_router'
]