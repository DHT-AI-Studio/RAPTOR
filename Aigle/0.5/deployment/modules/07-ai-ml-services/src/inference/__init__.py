# src/inference/__init__.py
"""
推理模組 — spec-driven 二層架構

主要組件:
    - ModelSpec       規格資料類，從 MLflow tag 解析
    - InferenceService 單一進入點（取代舊的 manager + router + executor）
    - BaseAdapter     runtime 抽象（OllamaAdapter / HFTransformersAdapter）
    - BaseHandler     task-family handler（VLM / ASR / OCR / ...）
"""

from .adapters import BaseAdapter, get_adapter
from .service import InferenceService, inference_service
from .spec import (
    RUNTIME_HF,
    RUNTIME_OLLAMA,
    TASK_FAMILIES,
    ModelSpec,
    canonicalize_task,
)

__version__ = "3.0.0"

__all__ = [
    "RUNTIME_HF",
    "RUNTIME_OLLAMA",
    "TASK_FAMILIES",
    "BaseAdapter",
    "InferenceService",
    "ModelSpec",
    "canonicalize_task",
    "get_adapter",
    "inference_service",
]
