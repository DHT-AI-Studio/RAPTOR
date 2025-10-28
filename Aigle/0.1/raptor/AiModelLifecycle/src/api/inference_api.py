# src/api/inference_api_new.py
"""
重構後的推理API - 簡化版本

提供統一的推理接口，支持所有任務類型和引擎。
基於新的推理管理器實現。
"""

import logging
import time
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

# 導入新的推理管理器
from ..inference.manager import inference_manager

logger = logging.getLogger(__name__)

# 創建路由器
router = APIRouter(
    prefix="/inference",
    tags=["推理 (Inference)"]
)

# ===== 請求模型定義 =====

class InferenceRequest(BaseModel):
    """統一推理請求模型"""
    task: str = Field(
        description="任務類型",
        examples=["text-generation", "vlm", "asr", "ocr", "audio-classification", "video-analysis", "document-analysis"]
    )
    engine: str = Field(
        description="引擎類型",
        examples=["ollama", "transformers"]
    )
    model_name: str = Field(
        description="模型名稱",
        examples=["llama2-7b", "gpt-3.5-turbo", "llava-1.5-7b"]
    )
    data: Dict[str, Any] = Field(
        description="輸入數據，格式根據任務類型而異",
        examples=[
            {"inputs": "你好，請介紹一下自己"},
            {"image": "base64_image_data", "prompt": "描述這張圖片"},
            {"audio": "/path/to/audio.wav"}
        ]
    )
    options: Optional[Dict[str, Any]] = Field(
        default={},
        description="推理選項",
        examples=[{"max_length": 100, "temperature": 0.7}]
    )

class HealthCheckResponse(BaseModel):
    """健康檢查響應模型"""
    status: str
    components: Dict[str, bool]
    stats: Dict[str, Any]
    timestamp: float

class SupportedTasksResponse(BaseModel):
    """支持的任務響應模型"""
    tasks: Dict[str, Dict[str, Any]]
    total_tasks: int

class StatsResponse(BaseModel):
    """統計信息響應模型"""
    stats: Dict[str, Any]
    timestamp: float

# ===== API 端點 =====

@router.post("/infer", summary="統一推理接口")
def unified_inference(request: InferenceRequest):
    """
    統一推理接口 - 支持所有任務類型和引擎組合
    
    支持所有任務類型和引擎組合：
    
    **任務類型:**
    - text-generation-ollama: 文本生成（Ollama 引擎專用）
    - text-generation-hf: 文本生成（HuggingFace 引擎專用）
    - text-generation: 文本生成（通用，向下兼容）
    - vlm: 視覺語言模型（僅支持 transformers）
    - asr-hf: 自動語音識別（HuggingFace）
    - asr: 自動語音識別（通用）
    - vad-hf: 語音活動檢測（HuggingFace）
    - ocr-hf: 光學字符識別（HuggingFace）
    - ocr: 光學字符識別（通用）
    - audio-classification: 音頻分類
    - video-analysis: 視頻分析
    - scene-detection: 場景檢測
    - document-analysis: 文檔分析
    - image-captioning: 圖像標題生成
    - video-summary: 視頻摘要
    - audio-transcription: 音頻轉錄
    
    **引擎類型:**
    - ollama: 適用於 text-generation-ollama, text-generation 任務
    - transformers: 適用於所有其他任務類型
    
    **數據格式:**
    - text-generation*: {"inputs": "輸入文本"}
    - vlm: {"image": "base64圖像或路徑", "prompt": "提示詞"}
    - asr*/vad-hf/audio-*: {"audio": "音頻文件路徑"}
    - ocr*/image-*: {"image": "圖像文件路徑"}
    - video-*/scene-*: {"video": "視頻文件路徑"}
    - document-*: {"document": "文檔文件路徑"}
    
    **使用範例:**
    ```json
    {
      "task": "text-generation",
      "engine": "ollama", 
      "model_name": "llama2-7b",
      "data": {"inputs": "請寫一首關於春天的詩"},
      "options": {"max_length": 200, "temperature": 0.8}
    }
    ```
    
    **返回值:**
    - success: 是否成功
    - result: 推理結果（格式根據任務類型而異）
    - task: 執行的任務類型
    - engine: 使用的引擎
    - model_name: 使用的模型
    - processing_time: 處理時間（秒）
    - timestamp: 時間戳
    """
    try:
        start_time = time.time()
        logger.info(f"收到推理請求: {request.task} - {request.engine} - {request.model_name}")
        
        # 執行推理
        result = inference_manager.infer(
            task=request.task,
            engine=request.engine,
            model_name=request.model_name,
            data=request.data,
            options=request.options
        )
        
        # 添加API層的元數據
        result['api_processing_time'] = time.time() - start_time
        result['request_id'] = id(request)
        
        logger.info(f"推理完成: {request.task} - 用時 {result['processing_time']:.2f}秒")
        return result
        
    except ValueError as e:
        logger.error(f"推理參數錯誤: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"參數錯誤: {str(e)}"
        )
    except Exception as e:
        logger.error(f"推理執行失敗: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"推理執行失敗: {str(e)}"
        )

@router.get("/health", response_model=HealthCheckResponse, summary="健康檢查")
def health_check():
    """
    推理系統健康檢查
    
    檢查推理管理器和相關組件的狀態。
    """
    try:
        health_info = inference_manager.health_check()
        return health_info
    except Exception as e:
        logger.error(f"健康檢查失敗: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"健康檢查失敗: {str(e)}"
        )

@router.get("/supported-tasks", response_model=SupportedTasksResponse, summary="獲取支持的任務")
def get_supported_tasks():
    """
    獲取支持的任務類型和對應的引擎
    
    返回所有支持的任務配置信息。
    """
    try:
        tasks = inference_manager.get_supported_tasks()
        return {
            "tasks": tasks,
            "total_tasks": len(tasks)
        }
    except Exception as e:
        logger.error(f"獲取支持任務失敗: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"獲取支持任務失敗: {str(e)}"
        )

@router.get("/stats", response_model=StatsResponse, summary="獲取統計信息")
def get_stats():
    """
    獲取推理系統統計信息
    
    包括推理次數、成功率、緩存命中率等。
    """
    try:
        stats = inference_manager.get_stats()
        return {
            "stats": stats,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"獲取統計信息失敗: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"獲取統計信息失敗: {str(e)}"
        )

@router.post("/clear-cache", summary="清理緩存")
def clear_cache():
    """
    清理推理系統緩存
    
    清理所有模型緩存和相關資源。
    """
    try:
        inference_manager.clear_cache()
        return {
            "success": True,
            "message": "緩存已清理",
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"清理緩存失敗: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"清理緩存失敗: {str(e)}"
        )

# ===== 向下兼容的端點 =====

@router.post("/infer_fixed", summary="固定模型推理（向下兼容）")
def infer_fixed_compatibility(request: InferenceRequest):
    """
    向下兼容的固定模型推理端點
    
    重定向到統一推理接口。
    """
    logger.info("使用向下兼容的固定推理端點，重定向到統一接口")
    return unified_inference(request)

@router.post("/infer_multimodal", summary="多模態推理（向下兼容）") 
def infer_multimodal_compatibility(request: InferenceRequest):
    """
    向下兼容的多模態推理端點
    
    重定向到統一推理接口。
    """
    logger.info("使用向下兼容的多模態推理端點，重定向到統一接口")
    return unified_inference(request)

# ===== 使用示例生成端點 =====

@router.get("/examples", summary="獲取使用示例")
def get_usage_examples():
    """
    獲取各種任務類型的使用示例
    
    提供每種任務類型的請求格式示例。
    """
    examples = {
        "text-generation-ollama": {
            "description": "使用 Ollama 進行文本生成",
            "request": {
                "task": "text-generation",
                "engine": "ollama",
                "model_name": "llama2-7b",
                "data": {"inputs": "請寫一首關於春天的詩"},
                "options": {"max_length": 200, "temperature": 0.8}
            }
        },
        "text-generation-transformers": {
            "description": "使用 Transformers 進行文本生成",
            "request": {
                "task": "text-generation",
                "engine": "transformers",
                "model_name": "gpt2-medium",
                "data": {"inputs": "人工智能的未來發展"},
                "options": {"max_length": 150, "temperature": 0.7}
            }
        },
        "vlm": {
            "description": "視覺語言模型 - 圖像理解",
            "request": {
                "task": "vlm",
                "engine": "transformers",
                "model_name": "llava-1.5-7b",
                "data": {
                    "image": "base64_encoded_image_data",
                    "prompt": "請詳細描述這張圖片的內容"
                },
                "options": {"max_length": 256, "temperature": 0.7}
            }
        },
        "asr": {
            "description": "自動語音識別",
            "request": {
                "task": "asr",
                "engine": "transformers",
                "model_name": "whisper-large",
                "data": {"audio": "/path/to/audio.wav"},
                "options": {}
            }
        },
        "ocr": {
            "description": "光學字符識別",
            "request": {
                "task": "ocr",
                "engine": "transformers",
                "model_name": "trocr-large",
                "data": {"image": "/path/to/image.jpg"},
                "options": {}
            }
        },
        "audio-classification": {
            "description": "音頻分類",
            "request": {
                "task": "audio-classification",
                "engine": "transformers",
                "model_name": "ast-finetuned",
                "data": {"audio": "/path/to/audio.wav"},
                "options": {"top_k": 5}
            }
        }
    }
    
    return {
        "examples": examples,
        "total_examples": len(examples),
        "note": "所有示例都使用統一的 /inference/infer 端點"
    }