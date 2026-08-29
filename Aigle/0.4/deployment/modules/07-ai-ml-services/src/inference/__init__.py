# src/inference/__init__.py
"""
推理模組 — spec-driven 二層架構

主要組件:
    - ModelSpec       規格資料類，從 MLflow tag 解析
    - InferenceService 單一進入點（取代舊的 manager + router + executor）
    - BaseAdapter     runtime 抽象（OllamaAdapter / HFTransformersAdapter）
    - BaseHandler     task-family handler（VLM / ASR / OCR / ...）
"""

from .spec import ModelSpec, canonicalize_task, RUNTIME_OLLAMA, RUNTIME_HF, TASK_FAMILIES
from .service import InferenceService, inference_service
from .adapters import BaseAdapter, get_adapter

__version__ = "3.0.0"

__all__ = [
    "ModelSpec",
    "canonicalize_task",
    "RUNTIME_OLLAMA",
    "RUNTIME_HF",
    "TASK_FAMILIES",
    "InferenceService",
    "inference_service",
    "BaseAdapter",
    "get_adapter",
]
