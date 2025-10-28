# src/inference/engines/__init__.py
"""
推理引擎模組

提供不同的推理引擎實現：
- OllamaEngine: Ollama 服務引擎
- TransformersEngine: HuggingFace Transformers 引擎
"""

from .base import BaseEngine
from .ollama import OllamaEngine  
from .transformers import TransformersEngine

__all__ = ['BaseEngine', 'OllamaEngine', 'TransformersEngine']
