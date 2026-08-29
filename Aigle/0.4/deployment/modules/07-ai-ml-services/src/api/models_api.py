# src/api/models_api.py
from fastapi import APIRouter, Body, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
from ..core.model_manager import model_manager
import os
import logging

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/models",
    tags=["模型管理 (Model Management)"]
)

# --- Pydantic Models for Requests ---
class ModelSource(str, Enum):
    huggingface = "huggingface"
    ollama = "ollama"


class DownloadRequest(BaseModel):
    model_source: ModelSource = Field(..., description="模型來源")
    model_name: str = Field(..., description="模型名稱，例如 'google_gemma-2b' 或 'llama3'")

class UploadRequest(BaseModel):
    repo_name: str = Field(..., description="目標 lakeFS 儲存庫名稱")
    local_model_name: str = Field(..., description="在本地暫存區的模型資料夾名稱")
    branch_name: str = Field("main", description="目標 lakeFS 分支")

class RegisterLakeFSRequest(BaseModel):
    """註冊 lakeFS 中的模型到 MLflow。

    注意: backend 方法 register_lakefs_to_mlflow 需要 engine 與 model_params，因此此處提供相對應欄位。
    """
    registered_name: str = Field(..., description="在 MLflow 中註冊的名稱", example="my-text-model")
    task: str = Field(..., description="模型任務類型 (需與 inference.yaml 中一致)", example="text-generation-hf")
    engine: str = Field(..., description="推理引擎 (transformers, vllm, custom 等)", example="transformers")
    model_params: float = Field(..., description="模型參數量(單位:B(十億)，例如 7B 填 7)", example=7)
    lakefs_repo: str = Field(..., description="來源 lakeFS repository 名稱", example="my-model-repo")
    commit_id: Optional[str] = Field(None, description="指定 commit ID，不填則使用 lakefs_branch 最新提交")
    lakefs_branch: Optional[str] = Field("main", description="當未提供 commit_id 時使用該分支最新 commit", example="main")
    version_description: Optional[str] = Field("", description="版本描述", example="First production-ready version")
    stage: Optional[str] = Field(None, description="註冊後直接轉換到的階段 (production, staging, archived, none)", example="production")
    quantization: Optional[str] = Field(None, description="量化類型，可選值: '4bit', '8bit', None (無量化)", example="4bit")
    model_class: Optional[str] = Field(
        None,
        description="transformers 模型 class 名稱（如 'Qwen2_5_VLForConditionalGeneration'、"
                    "'AutoModelForCausalLM'）。若不設 pipeline_task 則此欄為必填。",
        example="AutoModelForCausalLM",
    )
    processor_class: Optional[str] = Field(
        None,
        description="processor / tokenizer class（如 'AutoProcessor', 'AutoTokenizer'）。",
        example="AutoTokenizer",
    )
    pipeline_task: Optional[str] = Field(
        None,
        description="若設了，推理走 transformers.pipeline 捷徑（如 'text-generation', "
                    "'automatic-speech-recognition'）。設了之後 model_class 可省略。",
        example="text-generation",
    )
    trust_remote_code: Optional[bool] = Field(True, description="是否信任模型原始碼，預設 True")
    torch_dtype: Optional[str] = Field(
        None,
        description="載入精度（'fp16' / 'bf16' / 'fp32' / 'auto'）。InternVL 等模型建議 'bf16'。",
        example="bf16",
    )
    custom_handler: Optional[str] = Field(
        None,
        description="自訂推理 handler（dotted 路徑，如 'pkg.mod:Class'，或內建短名如 'internvl'），用於非標準前後處理的模型。",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "registered_name": "gemma-270m-it",
                "task": "text-generation-hf",
                "engine": "transformers",
                "model_params": 0.27,
                "lakefs_repo": "gemma-270m-it",
                "commit_id": None,
                "lakefs_branch": "main",
                "version_description": "Initial registration",
                "stage": "staging",
                "quantization": "4bit",
                "model_class": "AutoModelForCausalLM",
                "processor_class": "AutoTokenizer",
                "pipeline_task": None,
                "trust_remote_code": True
            }
        }

class RegisterOllamaRequest(BaseModel):
    local_model_name: str = Field(..., description="本地已存在的 Ollama 模型完整名稱 (例如 llama3:latest)", example="llama3:latest")
    task: str = Field("text-generation-ollama", description="模型任務類型 (需與 inference.yaml 中定義)", example="text-generation-ollama")
    model_params: Optional[float] = Field(None, description="模型參數量(十億)，不填則自動從 ollama metadata 估算")
    registered_name: Optional[str] = Field(None, description="在 MLflow 中註冊的名稱，若為 None 則使用 local_model_name")
    version_description: Optional[str] = Field("", description="版本描述")
    stage: Optional[str] = Field(None, description="註冊後直接轉換到的階段 (production, staging, archived, none)")
    quantization: Optional[str] = Field(None, description="量化類型，可選值: '4bit', '8bit', None (使用模型本身的量化級別)", example="4bit")

    class Config:
        json_schema_extra = {
            "example": {
                "local_model_name": "llama3:latest",
                "task": "text-generation-ollama",
                "model_params": 8,
                "registered_name": "llama3-local",
                "version_description": "Register llama3 local model",
                "stage": "staging",
                "quantization": "4bit"
            }
        }

class TransitionStageRequest(BaseModel):
    model_name: str = Field(..., description="在 MLflow 中註冊的模型名稱")
    version: str = Field(..., description="要轉換階段的模型版本號")
    stage: str = Field(..., description="目標階段 (production, staging, archived, none)")
    archive_existing_versions: Optional[bool] = Field(False, description="是否將目標階段中現有的其他版本封存")

# --- API Endpoints ---

@router.post("/download", summary="從來源下載模型")
def download_model_endpoint(
    model_source: Optional[ModelSource] = Query(None, description="模型來源"),
    model_name: Optional[str] = Query(None, description="模型名稱，例如 'google/gemma-3-270m-it' 或 'llama3:latest'"),
    request: Optional[DownloadRequest] = Body(None)
):
    """
    從指定的來源下載模型到本地暫存區
    
    **輸入參數說明：**
    - **model_source**: 模型來源，可選值：
      - "huggingface": 從 HuggingFace Hub 下載
      - "ollama": 從 Ollama 拉取模型
    - **model_name**: 模型名稱
      - HuggingFace: 如 "google/gemma-3-270m-it"
      - Ollama: 如 "llama3:latest"
    
    **使用範例：**
    ```json
    {
      "model_source": "huggingface",
      "model_name": "google/gemma-3-270m-it"
    }
    ```
    
    **返回值：**
    - message: 操作結果消息
    - details: 下載過程的詳細信息
    """
    try:
        resolved_model_source = model_source or (request.model_source if request else None)
        resolved_model_name = model_name or (request.model_name if request else None)

        if resolved_model_source is None or resolved_model_name is None:
            raise HTTPException(status_code=422, detail="model_source 與 model_name 為必填欄位")

        result = model_manager.download_model(resolved_model_source.value, resolved_model_name)
        return {"message": "模型下載請求已處理", "details": result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/local", summary="列出本地模型")
def list_local_models_endpoint(model_source: Optional[ModelSource] = Query(None, description="指定模型來源；選擇 ollama 時僅列出 Ollama 模型，未指定時列出 HuggingFace 下載的本地模型")):
    """
    列出本地暫存區中已下載的模型
    
    **查詢參數說明：**
    - **model_source** (可選): 指定模型來源類型
      - "ollama": 僅列出 Ollama 模型
      - 不指定: 列出 HuggingFace 下載的模型
    
    **使用範例：**
    - GET /models/local
    - GET /models/local?model_source=ollama
    
    **返回值：**
    - 模型列表，包含模型名稱、路徑、大小等信息
    """
    return model_manager.list_local_models(model_source.value if model_source else None)

@router.post("/upload_to_lakefs", summary="將本地模型上傳至 lakeFS")
def upload_model_to_lakefs_endpoint(request: UploadRequest):
    """
    將本地暫存區的模型上傳到 lakeFS 儲存庫
    
    **輸入參數說明：**
    - **repo_name**: 目標 lakeFS 儲存庫名稱，格式限制：僅允許小寫字母、數字、連字號(-)
    - **local_model_name**: 本地暫存區中的模型資料夾名稱
    - **branch_name**: 目標 lakeFS 分支名稱，默認 "main"
    
    **使用範例：**
    ```json
    {
      "repo_name": "my-model-repo",
      "local_model_name": "gemma-3-270m-it",
      "branch_name": "main"
    }
    ```
    
    **返回值：**
    - message: 上傳結果消息
    - lakefs_uri: 上傳後的 lakeFS URI 位置
    """
    try:
        local_path = os.path.join(model_manager.models_tmp_root, request.local_model_name)
        logger.info(f"Uploading model from {local_path} to LakeFS repo {request.repo_name}")
        result = model_manager.upload_to_lakefs(request.repo_name, local_path, request.branch_name)
        if "Failed" in result:
            logger.error(f"LakeFS upload failed: {result}")
            raise HTTPException(status_code=500, detail=result)
        return {"message": "上傳成功", "lakefs_uri": result}
    except HTTPException:
        raise
    except Exception as e:
        error_detail = str(e)
        logger.error(f"Error uploading to LakeFS: {error_detail}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to ensure lakeFS repository: {request.repo_name}. Error: {error_detail}")

@router.get("/lakefs_repos", summary="列出 lakeFS 中的模型儲存庫")
def list_remote_models_endpoint():
    """
    列出 lakeFS 中所有的模型儲存庫
    
    **無需輸入參數**
    
    **使用範例：**
    - GET /models/lakefs_repos
    
    **返回值：**
    - lakeFS 儲存庫列表，包含儲存庫名稱、描述等信息
    """
    return model_manager.list_lakefs_repos()

@router.post("/register_from_lakefs", summary="將 lakeFS 模型註冊到 MLflow")
def register_lakefs_model_endpoint(request: RegisterLakeFSRequest):
    """
    將 lakeFS 中指定儲存庫的模型註冊到 MLflow Registry
    
    **輸入參數說明：**
    - **registered_name** (必填): 在 MLflow 中註冊的名稱
    - **task** (必填): 模型任務類型；舊名可沿用（如 text-generation-hf），新版建議使用 canonical task family
        （如 text-generation、vlm、asr、ocr）
    - **engine** (必填): 註冊時指定的推理 runtime（如 transformers、vllm、custom）；
        對 spec-driven HF 模型通常填 `transformers`
    - **model_params** (必填): 模型參數量（單位：B，十億）
    - **lakefs_repo** (必填): 來源 lakeFS 儲存庫名稱
    - **commit_id** (可選): 指定 commit ID，不填則使用最新提交
    - **lakefs_branch** (可選): 分支名稱，默認 "main"
    - **version_description** (可選): 版本描述
    - **stage** (可選): 註冊後的階段（production, staging, archived, none）
    - **quantization** (可選): 量化類型，可選值: "4bit", "8bit", None (無量化)
    - **model_class** (條件必填): `transformers` 模型 class 名稱，例如
        `AutoModelForCausalLM`、`Qwen2_5_VLForConditionalGeneration`；若未提供 `pipeline_task`，此欄必填
    - **processor_class** (可選): processor / tokenizer class，例如 `AutoProcessor`、`AutoTokenizer`；
        多模態、ASR、OCR 等模型通常建議一併提供
    - **pipeline_task** (條件必填): 若提供，推理端會走 `transformers.pipeline` 捷徑，例如
        `text-generation`、`automatic-speech-recognition`、`image-to-text`；提供後可省略 `model_class`
    - **trust_remote_code** (可選): 是否信任模型倉庫中的自訂程式碼；某些 HF 客製模型必須設為 `true`
    - **custom_handler** (可選): 自訂推理 handler 類別路徑（格式 `pkg.mod:Class`），用於非標準
        encode / generate / decode 流程

    **重要限制：**
    - 對 `engine="transformers"` 的 spec-driven 註冊，`model_class` 與 `pipeline_task` 至少要提供一個
    - 若是標準 HF pipeline 模型，通常填 `pipeline_task` 即可；若是 VLM 或特殊架構，通常需填 `model_class`
    
    **使用範例：**
    ```json
    {
      "registered_name": "gemma-270m-it",
      "task": "text-generation",
      "engine": "transformers",
      "model_params": 0.27,
      "lakefs_repo": "gemma-270m-it",
      "lakefs_branch": "main",
      "version_description": "Initial registration",
      "stage": "staging",
      "pipeline_task": "text-generation",
      "processor_class": "AutoTokenizer",
      "trust_remote_code": true
    }
    ```
    
    **返回值：**
    - message: 註冊結果消息
    - result: 註冊詳細信息
    """
    result = model_manager.register_lakefs_to_mlflow(
        registered_name=request.registered_name,
        lakefs_repo=request.lakefs_repo,
        task=request.task,
        engine=request.engine,
        model_params=request.model_params,
        commit_id=request.commit_id,
        lakefs_branch=request.lakefs_branch,
        version_description=request.version_description,
        stage=request.stage,
        quantization=request.quantization,
        model_class=request.model_class,
        processor_class=request.processor_class,
        pipeline_task=request.pipeline_task,
        trust_remote_code=request.trust_remote_code,
        torch_dtype=request.torch_dtype,
        custom_handler=request.custom_handler,
    )
    if isinstance(result, dict) and result.get("error"):
        raise HTTPException(status_code=400, detail=result["error"])
    return {"message": "模型已註冊", "result": result}

@router.post("/register_ollama", summary="將本地 Ollama 模型註冊到 MLflow")
def register_ollama_model_endpoint(request: RegisterOllamaRequest):
    """
    將本地已安裝的 Ollama 模型註冊到 MLflow Registry
    
    **輸入參數說明：**
    - **local_model_name**: 本地 Ollama 模型完整名稱（如 llama3:latest）
    - **task**: 模型任務類型，默認 "text-generation-ollama"
    - **model_params**: 模型參數量（可選，不填則自動估算）
    - **registered_name**: 在 MLflow 中的註冊名稱（可選）
    - **version_description**: 版本描述
    - **stage**: 註冊後的階段
    - **quantization** (可選): 量化類型，可選值: "4bit", "8bit", None (使用模型本身的量化級別)
    
    **使用範例：**
    ```json
    {
      "local_model_name": "llama3:latest",
      "task": "text-generation-ollama",
      "model_params": 8,
      "registered_name": "llama3-local",
      "version_description": "Register llama3 local model",
      "stage": "staging"
    }
    ```
    
    **返回值：**
    - message: 註冊成功消息
    - result: 註冊詳細信息
    """
    try:
        result = model_manager.register_ollama_to_mlflow(
            local_model_name=request.local_model_name,
            task=request.task,
            model_params=request.model_params,
            registered_name=request.registered_name,
            version_description=request.version_description,
            stage=request.stage,
            quantization=request.quantization
        )
        if isinstance(result, dict) and result.get("error"):
            # model_manager 將錯誤訊息包在 {'error': msg}
            raise HTTPException(status_code=400, detail=result["error"])
        return {"message": f"Ollama 模型 '{request.local_model_name}' 註冊成功", "result": result}
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_msg = f"Failed to register Ollama model: {str(e)}"
        logger.error(f"Error registering Ollama model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail={
            "error": error_msg,
            "error_type": type(e).__name__
        })


@router.get("/registered_in_mlflow", summary="列出在 MLflow 中已註冊的模型")
def list_mlflow_models_endpoint(show_all: bool = False):
    """
    列出在 MLflow Model Registry 中已註冊的所有模型
    
    **查詢參數說明：**
    - **show_all** (可選): 是否顯示所有版本，默認 false（僅最新版本）
    
    **使用範例：**
    - GET /models/registered_in_mlflow
    - GET /models/registered_in_mlflow?show_all=true
    
    **返回值：**
    - MLflow 中註冊的模型列表，包含名稱、版本、階段等信息
    """
    return model_manager.list_mlflow_models(show_all)

@router.get("/registered_in_mlflow/{model_name}", summary="獲取特定註冊模型的詳細資訊")
def get_model_details_endpoint(model_name: str, all_version: bool = False):
    """
    獲取指定 MLflow 模型的詳細信息
    
    **路徑參數說明：**
    - **model_name**: MLflow 中註冊的模型名稱
    
    **查詢參數說明：**
    - **all_version** (可選): 是否顯示所有版本，默認 false（僅最新版本）
    
    **使用範例：**
    - GET /models/registered_in_mlflow/my-model
    - GET /models/registered_in_mlflow/my-model?all_version=true
    
    **返回值：**
    - 模型的詳細信息，包含版本歷史、標籤、參數等
    """
    details = model_manager.get_model_details_from_mlflow(model_name, all_version)
    if "error" in details:
        raise HTTPException(status_code=404, detail=details["error"])
    return details

@router.post("/transition_stage", summary="轉換 MLflow 中模型版本的階段")
def transition_model_stage_endpoint(request: TransitionStageRequest):
    """
    轉換 MLflow 中指定模型版本的階段狀態
    
    **輸入參數說明：**
    - **model_name**: MLflow 中註冊的模型名稱
    - **version**: 要轉換的模型版本號
    - **stage**: 目標階段（production, staging, archived, none）
    - **archive_existing_versions**: 是否將目標階段中現有的其他版本封存
    
    **使用範例：**
    ```json
    {
      "model_name": "my-model",
      "version": "1",
      "stage": "production",
      "archive_existing_versions": false
    }
    ```
    
    **返回值：**
    - message: 轉換結果消息
    """
    try:
        result = model_manager.model_transition_stage_on_mlflow(
            model_name=request.model_name,
            version=request.version,
            stage=request.stage,
            archive_existing_versions=request.archive_existing_versions
        )
        return {"message": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ===== 新增的高級模型管理功能 =====

class BatchDownloadRequest(BaseModel):
    downloads: List[DownloadRequest] = Field(..., description="批量下載請求列表")

class ModelSearchRequest(BaseModel):
    query: str = Field(..., description="搜索關鍵詞")
    task_filter: Optional[str] = Field(None, description="按任務類型過濾")
    stage_filter: Optional[str] = Field(None, description="按階段過濾")

class ModelComparisonRequest(BaseModel):
    model_names: List[str] = Field(..., description="要比較的模型名稱列表")
    metrics: Optional[List[str]] = Field(None, description="要比較的指標")

class ModelMetadataUpdate(BaseModel):
    model_name: str = Field(..., description="模型名稱")
    version: str = Field(..., description="模型版本")
    metadata: Dict[str, Any] = Field(..., description="要更新的元數據")

@router.post("/batch_download", summary="批量下載模型")
def batch_download_models(request: BatchDownloadRequest):
    """
    批量下載多個模型到本地暫存區
    
    **輸入參數說明：**
    - **downloads**: 下載請求列表，每個請求包含：
      - model_source: 模型來源（"huggingface" 或 "ollama"）
      - model_name: 模型名稱
    
    **使用範例：**
    ```json
    {
      "downloads": [
        {
          "model_source": "huggingface",
          "model_name": "google/gemma-3-270m-it"
        },
        {
          "model_source": "ollama",
          "model_name": "llama3:latest"
        }
      ]
    }
    ```
    
    **返回值：**
    - message: 批量下載結果摘要
    - results: 每個模型的下載結果列表，包含狀態和詳細信息
    """
    try:
        results = []
        for download_req in request.downloads:
            try:
                result = model_manager.download_model(
                    download_req.model_source.value,
                    download_req.model_name
                )
                results.append({
                    "model_name": download_req.model_name,
                    "status": "success",
                    "details": result
                })
            except Exception as e:
                results.append({
                    "model_name": download_req.model_name,
                    "status": "failed",
                    "error": str(e)
                })
        
        success_count = sum(1 for r in results if r["status"] == "success")
        return {
            "message": f"批量下載完成，成功: {success_count}/{len(results)}",
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量下載失敗: {str(e)}")

@router.get("/stats", summary="獲取模型統計信息")
def get_model_statistics(show_all: bool = False):
    """
    獲取 MLflow 中註冊模型的統計概況信息
    
    **查詢參數說明：**
    - **show_all** (可選): 是否統計所有版本，默認 false（僅最新版本）
    
    **統計內容說明：**
    - **stage_distribution**: 各階段（production, staging, archived, none）的模型數量
    - **task_distribution**: 各任務類型（text-generation-hf, vlm 等）的模型數量
    - **local_models**: 本地暫存區的模型數量
    - **lakefs_repos**: lakeFS 儲存庫的數量
    
    **使用範例：**
    - GET /models/stats
    - GET /models/stats?show_all=true
    
    **返回值：**
    - success: 是否成功
    - total_models: 總模型數量
    - stage_distribution: 階段分布統計
    - task_distribution: 任務分布統計
    - local_models: 本地模型數量
    - lakefs_repos: lakeFS 儲存庫數量
    """
    try:
        all_models = model_manager.list_mlflow_models(show_all)  # 取得所有模型
        stage_counts: Dict[str, int] = {}
        task_counts: Dict[str, int] = {}

        for m in all_models:
            # 最新版本資訊只有一個在 versions[0] (list_mlflow_models 中對最新版本情境)
            versions = m.get("versions", [])
            if not versions:  # 版本列表為空，跳過
                continue
            if show_all:
                # 若顯示所有版本，則統計所有版本
                for mv in versions:
                    stage = mv.get("stage", "none").lower()
                    stage_counts[stage] = stage_counts.get(stage, 0) + 1
                    tags = mv.get("tags", {}) or {}
                    task = tags.get("task_family") or tags.get("inference_task") or tags.get("model_type") or "unknown"
                    task_counts[task] = task_counts.get(task, 0) + 1
                continue
            latest = max(versions, key=lambda v: int(v.get("version", 0)))               # 取最大版本號
            stage = latest.get("stage", "none").lower()                                  # 預設為 none
            stage_counts[stage] = stage_counts.get(stage, 0) + 1                         # 統計 stage
            tags = latest.get("tags", {}) or {}                                          # 預設為空 dict
            task = tags.get("inference_task") or tags.get("model_type") or "unknown"     # 優先取 inference_task，沒有再取 model_type，兩者都沒有則標為 unknown
            task_counts[task] = task_counts.get(task, 0) + 1                             # 統計 task

        return {
            "success": True,
            "total_models": len(all_models),
            "stage_distribution": stage_counts,
            "task_distribution": task_counts,
            "local_models": len(model_manager.list_local_models()),
            "lakefs_repos": len(model_manager.list_lakefs_repos())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"獲取統計信息失敗: {str(e)}")

@router.get("/health", summary="檢查模型管理服務健康狀態")
def check_model_service_health():
    """
    檢查模型管理相關服務的連接和運行狀態
    
    **檢查的服務包括：**
    - **mlflow**: MLflow Model Registry 連接狀態
    - **lakefs**: lakeFS 儲存庫連接狀態
    - **local_storage**: 本地模型儲存目錄存在性
    - **huggingface**: HuggingFace 訪問令牌配置狀態
    - **ollama**: Ollama 服務連接狀態
    
    **無需輸入參數**
    
    **使用範例：**
    - GET /models/health
    
    **返回值：**
    - success: 是否成功執行檢查
    - overall_healthy: 整體健康狀態（所有服務都正常時為 true）
    - services: 各服務的詳細健康狀態
    """
    try:
        health_status = {
            "mlflow": False,
            "lakefs": False,
            "local_storage": False,
            "huggingface": False,
            "ollama": False
        }
        
        # 檢查 MLflow 連接
        try:
            model_manager.list_mlflow_models(show_all=False)
            health_status["mlflow"] = True
        except:
            pass
        
        # 檢查 lakeFS 連接
        try:
            model_manager.list_lakefs_repos()
            health_status["lakefs"] = True
        except:
            pass
        
        # 檢查本地存儲
        try:
            import os
            health_status["local_storage"] = os.path.exists(model_manager.models_tmp_root)
        except:
            pass
        
        # 檢查 HuggingFace 令牌
        health_status["huggingface"] = bool(model_manager.hf_token)
        
        # 檢查 Ollama（簡單檢查）
        try:
            local_ollama_models = model_manager.list_local_models("ollama")
            health_status["ollama"] = True
        except:
            pass
        
        overall_health = all(health_status.values())
        
        return {
            "success": True,
            "overall_healthy": overall_health,
            "services": health_status
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"健康檢查失敗: {str(e)}")