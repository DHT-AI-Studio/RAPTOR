# src/inference/models/__init__.py
"""
模型處理器模組

提供不同任務類型的模型處理器，負責：
- 數據預處理
- 結果後處理
- 模型特定的邏輯處理
"""

from .base import BaseModelHandler
from .text_generation import TextGenerationHandler
from .vlm import VLMHandler
from .asr import ASRHandler
from .ocr import OCRHandler
from .audio_classification import AudioClassificationHandler
from .video_analysis import VideoAnalysisHandler
from .document_analysis import DocumentAnalysisHandler

# 註冊處理器
from ..registry import model_registry

# 註冊默認處理器

# 文本生成任務（包括細分類型）
model_registry.register_handler_manually('text-generation', 'default', TextGenerationHandler)
model_registry.register_handler_manually('text-generation-ollama', 'default', TextGenerationHandler)
model_registry.register_handler_manually('text-generation-hf', 'default', TextGenerationHandler)

# 視覺語言模型
model_registry.register_handler_manually('vlm', 'default', VLMHandler)

# 語音識別任務（包括細分類型）
model_registry.register_handler_manually('asr', 'default', ASRHandler)
model_registry.register_handler_manually('asr-hf', 'default', ASRHandler)
model_registry.register_handler_manually('vad-hf', 'default', ASRHandler)

# OCR 任務（包括細分類型）
model_registry.register_handler_manually('ocr', 'default', OCRHandler)
model_registry.register_handler_manually('ocr-hf', 'default', OCRHandler)

# 音頻相關任務
model_registry.register_handler_manually('audio-classification', 'default', AudioClassificationHandler)
model_registry.register_handler_manually('audio-transcription', 'default', ASRHandler)

# 視頻相關任務
model_registry.register_handler_manually('video-analysis', 'default', VideoAnalysisHandler)
model_registry.register_handler_manually('scene-detection', 'default', VideoAnalysisHandler)
model_registry.register_handler_manually('video-summary', 'default', VideoAnalysisHandler)

# 文檔和圖像任務
model_registry.register_handler_manually('document-analysis', 'default', DocumentAnalysisHandler)
model_registry.register_handler_manually('image-captioning', 'default', VLMHandler)

def get_model_handler(task: str, model_name: str) -> BaseModelHandler:
    """
    獲取模型處理器實例
    
    Args:
        task (str): 任務類型
        model_name (str): 模型名稱
        
    Returns:
        BaseModelHandler: 處理器實例
    """
    handler_class = model_registry.get_model_handler(task, model_name)
    return handler_class()

__all__ = [
    'BaseModelHandler',
    'TextGenerationHandler', 
    'VLMHandler',
    'ASRHandler',
    'OCRHandler',
    'AudioClassificationHandler',
    'VideoAnalysisHandler',
    'DocumentAnalysisHandler',
    'get_model_handler'
]