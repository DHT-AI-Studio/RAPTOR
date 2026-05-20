# src/api/config_api.py
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from ..core.config import config
import yaml
import json
import os

router = APIRouter(
    prefix="/config",
    tags=["配置管理 (Configuration Management)"]
)

# ===== Pydantic 模型定義 =====

class ConfigUpdateRequest(BaseModel):
    config_path: List[str] = Field(..., description="配置路径，如 ['inference', 'memory_manager', 'safe_margin_mb']")
    value: Any = Field(..., description="新的配置值")

class ConfigReloadRequest(BaseModel):
    config_type: str = Field(..., description="配置類型 (base, inference)")
    force_reload: bool = Field(False, description="是否強制重新加載")

# ===== 配置查詢 API =====

@router.get("/", summary="獲取完整配置")
def get_full_config():
    """獲取系統的完整配置信息"""
    try:
        # 獲取整個配置樹
        full_config = config._config
        
        return {
            "success": True,
            "config": full_config,
            "config_keys": list(full_config.keys()) if isinstance(full_config, dict) else []
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"獲取配置失敗: {str(e)}"
        )

@router.get("/sections", summary="獲取配置區段列表")
def get_config_sections():
    """獲取所有可用的配置區段"""
    try:
        full_config = config._config
        sections = list(full_config.keys()) if isinstance(full_config, dict) else []
        
        section_info = []
        for section in sections:
            section_data = config.get_config(section)
            section_info.append({
                "name": section,
                "type": type(section_data).__name__,
                "has_subsections": isinstance(section_data, dict),
                "subsections": list(section_data.keys()) if isinstance(section_data, dict) else []
            })
        
        return {
            "success": True,
            "sections": section_info,
            "total_count": len(sections)
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"獲取配置區段失敗: {str(e)}"
        )

@router.get("/get/{section}", summary="獲取特定配置區段")
def get_config_section(section: str, subsection: Optional[str] = None):
    """獲取指定的配置區段或子區段"""
    try:
        if subsection:
            config_data = config.get_config(section, subsection)
        else:
            config_data = config.get_config(section)
        
        if config_data is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"配置區段不存在: {section}" + (f".{subsection}" if subsection else "")
            )
        
        return {
            "success": True,
            "section": section,
            "subsection": subsection,
            "config_data": config_data
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"獲取配置區段失敗: {str(e)}"
        )

@router.get("/search", summary="搜索配置項")
def search_config(query: str, case_sensitive: bool = False):
    """在配置中搜索特定的鍵或值"""
    try:
        def search_in_dict(data, path="", results=None):
            if results is None:
                results = []
            
            if isinstance(data, dict):
                for key, value in data.items():
                    current_path = f"{path}.{key}" if path else key
                    
                    # 搜索鍵名
                    key_match = (query.lower() in key.lower()) if not case_sensitive else (query in key)
                    if key_match:
                        results.append({
                            "path": current_path,
                            "key": key,
                            "value": value,
                            "match_type": "key"
                        })
                    
                    # 搜索值（如果是字符串）
                    if isinstance(value, str):
                        value_match = (query.lower() in value.lower()) if not case_sensitive else (query in value)
                        if value_match:
                            results.append({
                                "path": current_path,
                                "key": key,
                                "value": value,
                                "match_type": "value"
                            })
                    
                    # 遞歸搜索
                    search_in_dict(value, current_path, results)
            
            return results
        
        search_results = search_in_dict(config._config)
        
        return {
            "success": True,
            "query": query,
            "case_sensitive": case_sensitive,
            "results": search_results,
            "total_matches": len(search_results)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"搜索配置失敗: {str(e)}"
        )

# ===== 推理配置專用 API =====

@router.get("/inference", summary="獲取推理配置")
def get_inference_config():
    """獲取推理相關的完整配置"""
    try:
        inference_config = config.get_config("inference")
        
        if inference_config is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="推理配置不存在"
            )
        
        return {
            "success": True,
            "inference_config": inference_config
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"獲取推理配置失敗: {str(e)}"
        )

@router.get("/inference/tasks", summary="獲取支持的推理任務配置")
def get_inference_tasks_config():
    """獲取所有支持的推理任務及其配置"""
    try:
        task_configs = config.get_config("inference", "task_to_models")
        
        if task_configs is None:
            return {
                "success": True,
                "tasks": {},
                "total_count": 0,
                "message": "未找到任務配置"
            }
        
        # 格式化任務信息
        formatted_tasks = []
        for task_name, task_config in task_configs.items():
            formatted_tasks.append({
                "task_name": task_name,
                "strategy": task_config.get("strategy", "priority"),
                "discovery": task_config.get("discovery", {}),
                "config": task_config
            })
        
        return {
            "success": True,
            "tasks": formatted_tasks,
            "total_count": len(formatted_tasks),
            "raw_config": task_configs
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"獲取推理任務配置失敗: {str(e)}"
        )

@router.get("/inference/engines", summary="獲取推理引擎配置")
def get_inference_engines_config():
    """獲取推理引擎的配置信息"""
    try:
        engines_config = config.get_config("inference", "engines")
        
        return {
            "success": True,
            "engines_config": engines_config or {},
            "available_engines": list(engines_config.keys()) if engines_config else []
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"獲取推理引擎配置失敗: {str(e)}"
        )

@router.get("/inference/memory", summary="獲取記憶體管理配置")
def get_memory_manager_config():
    """獲取推理記憶體管理的配置"""
    try:
        memory_config = config.get_config("inference", "memory_manager")
        
        return {
            "success": True,
            "memory_manager_config": memory_config or {}
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"獲取記憶體管理配置失敗: {str(e)}"
        )

# ===== 存儲和服務配置 API =====

@router.get("/storage", summary="獲取存儲配置")
def get_storage_config():
    """獲取存儲路徑和相關配置"""
    try:
        storage_config = config.get_config("storage")
        
        # 檢查路徑是否存在
        if storage_config:
            for key, path in storage_config.items():
                if isinstance(path, (str, os.PathLike)) and "path" in key.lower():
                    storage_config[f"{key}_exists"] = os.path.exists(path)
                    if os.path.exists(path):
                        try:
                            storage_config[f"{key}_permissions"] = oct(os.stat(path).st_mode)[-3:]
                        except:
                            storage_config[f"{key}_permissions"] = "unknown"
        
        return {
            "success": True,
            "storage_config": storage_config or {}
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"獲取存儲配置失敗: {str(e)}"
        )

@router.get("/services", summary="獲取外部服務配置")
def get_services_config():
    """獲取 MLflow, lakeFS 等外部服務的配置（隱藏敏感信息）"""
    try:
        services_config = {}
        
        # MLflow 配置
        mlflow_config = config.get_config("mlflow")
        if mlflow_config:
            services_config["mlflow"] = {
                "tracking_uri": mlflow_config.get("tracking_uri"),
                "configured": bool(mlflow_config.get("tracking_uri"))
            }
        
        # lakeFS 配置
        lakefs_config = config.get_config("lakefs")
        if lakefs_config:
            services_config["lakefs"] = {
                "endpoint": lakefs_config.get("endpoint"),
                "model_storage_namespace": lakefs_config.get("model_storage_namespace"),
                "dataset_storage_namespace": lakefs_config.get("dataset_storage_namespace"),
                "has_credentials": bool(lakefs_config.get("access_key") and lakefs_config.get("secret_key"))
            }
        
        # HuggingFace 配置
        hf_config = config.get_config("huggingface")
        if hf_config:
            services_config["huggingface"] = {
                "has_token": bool(hf_config.get("hf_token"))
            }
        
        return {
            "success": True,
            "services_config": services_config
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"獲取服務配置失敗: {str(e)}"
        )

# ===== 配置驗證和健康檢查 API =====

@router.get("/validate", summary="驗證配置完整性")
def validate_config():
    """驗證當前配置的完整性和正確性"""
    try:
        validation_results = {
            "overall_valid": True,
            "issues": [],
            "warnings": [],
            "sections_checked": []
        }
        
        # 檢查必需的配置區段
        required_sections = ["storage", "mlflow", "lakefs", "inference"]
        for section in required_sections:
            section_config = config.get_config(section)
            validation_results["sections_checked"].append(section)
            
            if section_config is None:
                validation_results["issues"].append(f"缺少必需的配置區段: {section}")
                validation_results["overall_valid"] = False
            elif not section_config:
                validation_results["warnings"].append(f"配置區段為空: {section}")
        
        # 檢查存儲路徑
        storage_config = config.get_config("storage")
        if storage_config:
            for key, path in storage_config.items():
                if isinstance(path, (str, os.PathLike)) and "path" in key.lower():
                    if not os.path.exists(path):
                        validation_results["warnings"].append(f"存儲路徑不存在: {key} -> {path}")
        
        # 檢查推理配置
        inference_config = config.get_config("inference")
        if inference_config:
            if not inference_config.get("task_to_models"):
                validation_results["issues"].append("推理配置中缺少任務模型映射")
                validation_results["overall_valid"] = False
        
        return {
            "success": True,
            "validation_results": validation_results
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"配置驗證失敗: {str(e)}"
        )

@router.get("/health", summary="配置服務健康檢查")
def config_health_check():
    """檢查配置管理服務的健康狀態"""
    try:
        health_status = {
            "config_loaded": False,
            "config_files_accessible": {},
            "environment_variables": {},
            "overall_healthy": True
        }
        
        # 檢查配置是否成功加載
        try:
            full_config = config._config
            health_status["config_loaded"] = bool(full_config)
        except:
            health_status["overall_healthy"] = False
        
        # 檢查配置文件可訪問性
        from ..core.config import PROJECT_ROOT
        config_files = [
            PROJECT_ROOT / "src" / "core" / "configs" / "base.yaml",
            PROJECT_ROOT / "src" / "core" / "configs" / "inference.yaml",
            PROJECT_ROOT / ".env"
        ]
        
        for config_file in config_files:
            health_status["config_files_accessible"][str(config_file)] = config_file.exists()
            if not config_file.exists() and config_file.name != ".env":
                health_status["overall_healthy"] = False
        
        # 檢查重要的環境變量
        import os
        important_env_vars = ["MLFLOW_TRACKING_URI", "LAKEFS_ENDPOINT", "HF_TOKEN"]
        for env_var in important_env_vars:
            health_status["environment_variables"][env_var] = bool(os.getenv(env_var))
        
        return {
            "success": True,
            "health_status": health_status
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"配置健康檢查失敗: {str(e)}"
        )

# ===== 配置統計 API =====

@router.get("/stats", summary="獲取配置統計信息")
def get_config_stats():
    """獲取配置的統計信息"""
    try:
        def count_config_items(data, depth=0):
            """遞歸計算配置項目數量"""
            if isinstance(data, dict):
                count = len(data)
                for value in data.values():
                    count += count_config_items(value, depth + 1)
                return count
            elif isinstance(data, list):
                count = len(data)
                for item in data:
                    count += count_config_items(item, depth + 1)
                return count
            else:
                return 0
        
        full_config = config._config
        stats = {
            "total_sections": len(full_config) if isinstance(full_config, dict) else 0,
            "total_config_items": count_config_items(full_config),
            "section_breakdown": {},
            "config_size_estimate": len(str(full_config))
        }
        
        if isinstance(full_config, dict):
            for section_name, section_data in full_config.items():
                stats["section_breakdown"][section_name] = {
                    "items_count": count_config_items(section_data),
                    "type": type(section_data).__name__
                }
        
        return {
            "success": True,
            "config_stats": stats
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"獲取配置統計失敗: {str(e)}"
        )