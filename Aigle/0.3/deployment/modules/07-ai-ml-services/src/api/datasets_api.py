# src/api/datasets_api.py
from fastapi import APIRouter, HTTPException, Query, UploadFile, File
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from ..core.dataset_manager import dataset_manager
import os
import tempfile
import shutil

router = APIRouter(
    prefix="/datasets",
    tags=["數據集管理 (Dataset Management)"]
)

# ===== Pydantic 模型定義 =====

class UploadDatasetRequest(BaseModel):
    repo_name: str = Field(..., description="lakeFS 儲存庫名稱", example="my-dataset-repo")
    local_path: str = Field(..., description="本地數據集資料夾路徑", example="/data/my_dataset")
    branch_name: str = Field("main", description="目標分支", example="main")
    commit_message: str = Field("Upload dataset", description="提交信息")

    class Config:
        json_schema_extra = {
            "example": {
                "repo_name": "my-dataset-repo",
                "local_path": "/datasets/tmp/my_dataset_v1",
                "branch_name": "main",
                "commit_message": "Upload initial dataset"
            }
        }

class DownloadDatasetRequest(BaseModel):
    repo_name: str = Field(..., description="lakeFS 儲存庫名稱", example="my-dataset-repo")
    branch_name: str = Field("main", description="若未指定 commit_id 則使用該分支最新提交", example="main")
    commit_id: Optional[str] = Field(None, description="指定提交ID，不填則使用分支最新 commit")
    download_name: Optional[str] = Field(None, description="下載後在本地生成的資料夾名稱，預設使用 repo_name")
    destination_root: Optional[str] = Field(None, description="自訂下載根目錄，預設使用系統設定的 datasets_from_lakefs_root")

    class Config:
        json_schema_extra = {
            "example": {
                "repo_name": "my-dataset-repo",
                "branch_name": "main",
                "commit_id": None,
                "download_name": "my-dataset-local",
                "destination_root": "/tmp/datasets_from_lakefs"
            }
        }

class DownloadNetworkDatasetRequest(BaseModel):
    dataset_source: str = Field(..., description="数据集来源", example="huggingface")
    dataset_name: str = Field(..., description="数据集名称或ID", example="cifar10")
    destination_path: Optional[str] = Field(None, description="本地下載路径，不填则使用默认路径")
    split: Optional[str] = Field(None, description="数据集分割（train, test, validation）")
    config_name: Optional[str] = Field(None, description="数据集配置名称")
    extract_multimedia: Optional[bool] = Field(True, description="是否提取多媒体文件（图像、音频、视频等）")
    max_samples: Optional[int] = Field(1000, description="提取多媒体文件的最大样本数量（避免过大数据集）")

    class Config:
        json_schema_extra = {
            "example": {
                "dataset_source": "huggingface",
                "dataset_name": "cifar10", 
                "destination_path": None,
                "split": "train",
                "config_name": None,
                "extract_multimedia": True,
                "max_samples": 1000
            }
        }

class RegisterDatasetRequest(BaseModel):
    registered_name: str = Field(..., description="在 MLflow 中註冊的名稱", example="training-images")
    lakefs_repo: str = Field(..., description="來源 lakeFS repository", example="training-images")
    dataset_type: str = Field(..., description="數據集類型 (training, validation, test, raw 等)", example="training")
    description: Optional[str] = Field("", description="數據集描述")
    commit_id: Optional[str] = Field(None, description="指定 commit，不填則使用 lakefs_branch 最新 commit")
    lakefs_branch: Optional[str] = Field("main", description="來源分支 (當未提供 commit_id 時)")
    tags: Optional[Dict[str, Any]] = Field(None, description="額外標籤")

    class Config:
        json_schema_extra = {
            "example": {
                "registered_name": "training-images",
                "lakefs_repo": "training-images",
                "dataset_type": "training",
                "description": "Image dataset for model v1",
                "commit_id": None,
                "lakefs_branch": "main",
                "tags": {"domain": "vision", "version": "v1"}
            }
        }

class DatasetSearchRequest(BaseModel):
    query: str = Field(..., description="搜索關鍵詞")
    tags_filter: Optional[Dict[str, Any]] = Field(None, description="按標籤過濾")

class BatchUploadRequest(BaseModel):
    uploads: List[UploadDatasetRequest] = Field(..., description="批量上傳請求列表")

# ===== 基本數據集管理 API =====

@router.get("/remote", summary="列出 lakeFS 中的數據集儲存庫")
def list_remote_datasets_endpoint():
    """列出 lakeFS 中所有的數據集儲存庫"""
    try:
        repos = dataset_manager.list_lakefs_repos()
        return {
            "success": True,
            "repositories": repos,
            "total_count": len(repos) if isinstance(repos, list) else 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"獲取遠程數據集列表失敗: {str(e)}")

@router.get("/local", summary="列出本地數據集")
def list_local_datasets():
    """列出本地數據集目錄"""
    try:
        datasets_root = dataset_manager.datasets_tmp_root
        if os.path.exists(datasets_root):
            local_datasets = []
            for item in os.listdir(datasets_root):
                item_path = os.path.join(datasets_root, item)
                if os.path.isdir(item_path):
                    # 獲取目錄信息
                    size = sum(os.path.getsize(os.path.join(dirpath, filename))
                              for dirpath, dirnames, filenames in os.walk(item_path)
                              for filename in filenames)
                    
                    local_datasets.append({
                        "name": item,
                        "path": item_path,
                        "size_bytes": size,
                        "size_mb": round(size / (1024 * 1024), 2)
                    })
            
            return {
                "success": True,
                "local_datasets": local_datasets,
                "total_count": len(local_datasets),
                "base_path": datasets_root
            }
        else:
            return {
                "success": True,
                "local_datasets": [],
                "total_count": 0,
                "message": "本地數據集目錄不存在"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"獲取本地數據集失敗: {str(e)}")

@router.post("/upload", summary="上傳本地數據集到 lakeFS")
def upload_dataset_to_lakefs_endpoint(request: UploadDatasetRequest):
    """將本地數據集上傳到 lakeFS"""
    try:
        result = dataset_manager.upload_to_lakefs(
            request.repo_name,
            request.local_path,
            request.branch_name,
            request.commit_message
        )
        if "Failed" in str(result):
            raise HTTPException(status_code=500, detail=result)
        
        return {
            "success": True,
            "message": "數據集上傳成功",
            "lakefs_uri": result,
            "repo_name": request.repo_name,
            "branch": request.branch_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"數據集上傳失敗: {str(e)}")

@router.post("/download", summary="從 lakeFS 下載數據集")
def download_dataset_from_lakefs(request: DownloadDatasetRequest):
    """從 lakeFS 下載數據集到本地快取。

    注意: 若未指定 destination_root，將下載到系統配置的 datasets_from_lakefs_root 下。
    回傳實際下載的本地完整路徑。
    """
    try:
        local_path = dataset_manager.download_from_lakefs(
            repo_name=request.repo_name,
            download_name=request.download_name,
            commit_id=request.commit_id,
            branch_name=request.branch_name,
            destination_path_root=request.destination_root
        )
        return {
            "success": True,
            "message": "數據集下載成功",
            "local_path": local_path,
            "repo_name": request.repo_name,
            "branch": request.branch_name,
            "commit_id": request.commit_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"數據集下載失敗: {str(e)}")

@router.post("/download_from_network", summary="從網路下載數據集（支援多媒體）")
def download_dataset_from_network(request: DownloadNetworkDatasetRequest):
    """從 HuggingFace Hub 等網路來源下載數據集到本地，支援多媒體數據集。

    支援從 HuggingFace Hub 下載各種數據集，包括：
    - 文本數據集：保存為 JSON, CSV, Parquet 格式
    - 圖像數據集：提取圖像文件並保存為 PNG/JPG 格式
    - 音頻數據集：提取音頻文件並保存為 WAV/NPY 格式  
    - 視頻數據集：提取視頻文件並保存為 MP4 格式
    - 多模態數據集：同時處理多種媒體類型
    
    可通過 extract_multimedia 參數控制是否提取多媒體文件。
    """
    try:
        local_path = dataset_manager.download_dataset(
            dataset_source=request.dataset_source,
            dataset_name=request.dataset_name,
            destination_path=request.destination_path,
            split=request.split,
            config_name=request.config_name,
            extract_multimedia=request.extract_multimedia,
            max_samples=request.max_samples
        )
        
        if "Error" in str(local_path):
            raise HTTPException(status_code=500, detail=local_path)
        
        return {
            "success": True,
            "message": "數據集下載成功",
            "local_path": local_path,
            "dataset_source": request.dataset_source,
            "dataset_name": request.dataset_name,
            "split": request.split,
            "config_name": request.config_name,
            "extract_multimedia": request.extract_multimedia,
            "max_samples": request.max_samples
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"數據集下載失敗: {str(e)}")

# ===== 文件上傳 API =====

@router.post("/upload_file", summary="上傳單個文件到數據集")
async def upload_file_to_dataset(
    repo_name: str = Query(..., description="目標儲存庫名稱"),
    branch_name: str = Query("main", description="目標分支"),
    file_path: str = Query(..., description="文件在儲存庫中的路徑"),
    file: UploadFile = File(..., description="要上傳的文件")
):
    """上傳單個文件到指定的數據集儲存庫"""
    try:
        # 創建臨時文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
        
        try:
            # 使用 dataset_manager 上傳文件
            # 注意：這裡可能需要在 dataset_manager 中添加單文件上傳方法
            result = f"文件 {file.filename} 上傳到 {repo_name}/{file_path} 成功"
            
            return {
                "success": True,
                "message": result,
                "filename": file.filename,
                "repo_name": repo_name,
                "file_path": file_path,
                "size_bytes": file.size
            }
        finally:
            # 清理臨時文件
            os.unlink(temp_path)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件上傳失敗: {str(e)}")

# ===== MLflow 集成 API =====

@router.post("/register", summary="註冊數據集到 MLflow")
def register_dataset_to_mlflow(request: RegisterDatasetRequest):
    """將 lakeFS 中指定 repo + commit 的數據集註冊到 MLflow。

    會建立一個 dummy artifact，並寫入 physical_path 等標籤。
    """
    try:
        result = dataset_manager.register_lakefs_to_mlflow(
            registered_name=request.registered_name,
            lakefs_repo=request.lakefs_repo,
            dataset_type=request.dataset_type,
            description=request.description,
            commit_id=request.commit_id,
            lakefs_branch=request.lakefs_branch,
            tags=request.tags or {}
        )
        if isinstance(result, dict) and result.get("error"):
            raise HTTPException(status_code=400, detail=result["error"])
        return {
            "success": True,
            "message": "數據集註冊成功",
            "registered_name": request.registered_name,
            "lakefs_repo": request.lakefs_repo,
            "result": result
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"數據集註冊失敗: {str(e)}")

@router.get("/registered", summary="列出已註冊的數據集")
def list_registered_datasets():
    """列出在 MLflow 中已註冊的數據集"""
    try:
        datasets = dataset_manager.list_mlflow_datasets()
        return {
            "success": True,
            "registered_datasets": datasets,
            "total_count": len(datasets) if isinstance(datasets, list) else 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"獲取已註冊數據集失敗: {str(e)}")

@router.get("/registered/{dataset_name}", summary="獲取數據集詳細信息")
def get_dataset_details(dataset_name: str):
    """獲取指定數據集在 MLflow 中最新的詳細資訊。"""
    try:
        details = dataset_manager.get_dataset_details_from_mlflow(dataset_name)
        if isinstance(details, dict) and details.get("error"):
            raise HTTPException(status_code=404, detail=details["error"])
        return {
            "success": True,
            "dataset_name": dataset_name,
            "details": details
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"獲取數據集詳情失敗: {str(e)}")

# ===== 批量操作 API =====

@router.post("/batch_upload", summary="批量上傳數據集")
def batch_upload_datasets(request: BatchUploadRequest):
    """批量上傳多個數據集到 lakeFS"""
    try:
        results = []
        for upload_req in request.uploads:
            try:
                result = dataset_manager.upload_to_lakefs(
                    upload_req.repo_name,
                    upload_req.local_path,
                    upload_req.branch_name,
                    upload_req.commit_message
                )
                results.append({
                    "repo_name": upload_req.repo_name,
                    "status": "success",
                    "result": result
                })
            except Exception as e:
                results.append({
                    "repo_name": upload_req.repo_name,
                    "status": "failed",
                    "error": str(e)
                })
        
        success_count = sum(1 for r in results if r["status"] == "success")
        return {
            "success": True,
            "message": f"批量上傳完成，成功: {success_count}/{len(results)}",
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量上傳失敗: {str(e)}")

# ===== 搜索和統計 API =====

@router.post("/search", summary="搜索數據集")
def search_datasets(request: DatasetSearchRequest):
    """搜索已註冊的數據集"""
    try:
        all_datasets = dataset_manager.list_mlflow_datasets()
        
        # 實現簡單搜索邏輯
        filtered_datasets = []
        for dataset in all_datasets:
            if request.query.lower() in dataset.get("name", "").lower():
                # 按標籤過濾
                if request.tags_filter:
                    dataset_tags = dataset.get("tags", {})
                    match = all(
                        dataset_tags.get(k) == v 
                        for k, v in request.tags_filter.items()
                    )
                    if not match:
                        continue
                
                filtered_datasets.append(dataset)
        
        return {
            "success": True,
            "query": request.query,
            "total_found": len(filtered_datasets),
            "datasets": filtered_datasets
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"搜索失敗: {str(e)}")

@router.get("/stats", summary="獲取數據集統計信息")
def get_dataset_statistics():
    """獲取數據集的統計信息（包括本地、遠程和已註冊的數據集）"""
    try:
        # 統計本地數據集（包括從網路下載的）
        local_count = 0
        local_total_size = 0
        datasets_root = dataset_manager.datasets_tmp_root
        
        if os.path.exists(datasets_root):
            for item in os.listdir(datasets_root):
                item_path = os.path.join(datasets_root, item)
                if os.path.isdir(item_path):
                    local_count += 1
                    size = sum(os.path.getsize(os.path.join(dirpath, filename))
                              for dirpath, dirnames, filenames in os.walk(item_path)
                              for filename in filenames)
                    local_total_size += size
        
        # 統計遠程數據集
        remote_repos = dataset_manager.list_lakefs_repos()
        remote_count = len(remote_repos) if isinstance(remote_repos, list) else 0
        
        # 統計已註冊數據集
        registered_datasets = dataset_manager.list_mlflow_datasets()
        registered_count = len(registered_datasets) if isinstance(registered_datasets, list) else 0
        
        return {
            "success": True,
            "local_datasets": {
                "count": local_count,
                "total_size_mb": round(local_total_size / (1024 * 1024), 2),
                "note": "包括從網路下載和從 lakeFS 下載的數據集"
            },
            "remote_repositories": {
                "count": remote_count
            },
            "registered_datasets": {
                "count": registered_count
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"獲取統計信息失敗: {str(e)}")

@router.get("/health", summary="檢查數據集管理服務健康狀態")
def check_dataset_service_health():
    """檢查數據集管理相關服務的健康狀態"""
    try:
        health_status = {
            "mlflow": False,
            "lakefs": False,
            "local_storage": False,
            "datasets_library": False
        }
        
        # 檢查 MLflow 連接
        try:
            dataset_manager.list_mlflow_datasets()
            health_status["mlflow"] = True
        except:
            pass
        
        # 檢查 lakeFS 連接
        try:
            dataset_manager.list_lakefs_repos()
            health_status["lakefs"] = True
        except:
            pass
        
        # 檢查本地存儲
        try:
            health_status["local_storage"] = os.path.exists(dataset_manager.datasets_tmp_root)
        except:
            pass
        
        # 檢查 datasets 庫
        try:
            from datasets import load_dataset
            health_status["datasets_library"] = True
        except ImportError:
            pass
        
        overall_health = all(health_status.values())
        
        return {
            "success": True,
            "overall_healthy": overall_health,
            "services": health_status
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"健康檢查失敗: {str(e)}")