# src/inference/exceptions.py
"""
推理系統的統一異常定義

提供分層的異常類型，便於 API 層根據異常類型返回適當的 HTTP 狀態碼。

異常層次結構：
    InferenceError (基類)
    ├── ValidationError (4xx)
    │   └── UnsupportedTaskError
    ├── ResourceNotFoundError (4xx)
    │   └── ModelNotFoundError
    ├── ModelLoadError (5xx)
    ├── InferenceExecutionError (5xx)
    ├── EngineError (5xx)
    └── ResourceExhaustedError (503)
"""


class InferenceError(Exception):
    """
    推理相關異常的基類
    
    所有推理系統的自定義異常都應該繼承此類。
    """
    pass


# ===== 4xx 客戶端錯誤 =====

class ValidationError(InferenceError):
    """
    輸入驗證錯誤
    
    當請求參數不符合要求時拋出，對應 HTTP 400 Bad Request。
    
    Examples:
        - 必需參數缺失
        - 參數類型錯誤
        - 參數格式不正確
        - 數據字段缺失
    """
    pass


class UnsupportedTaskError(ValidationError):
    """
    不支持的任務類型或引擎組合
    
    當請求的任務類型、引擎類型或其組合不被支持時拋出，
    對應 HTTP 400 Bad Request。
    
    Examples:
        - 不存在的任務類型
        - 不支持的引擎類型
        - 引擎與任務類型不兼容（如 Ollama 引擎用於 VLM 任務）
    """
    pass


class ResourceNotFoundError(InferenceError):
    """
    資源未找到錯誤（通用）
    
    當請求的資源不存在時拋出，對應 HTTP 404 Not Found。
    
    Examples:
        - 模型未找到
        - 文件未找到
        - 配置未找到
    """
    pass


class ModelNotFoundError(ResourceNotFoundError):
    """
    模型未找到錯誤
    
    當請求的模型在 MLflow 或本地不存在時拋出，
    對應 HTTP 404 Not Found。
    
    Examples:
        - MLflow 中未註冊該模型
        - 模型文件不存在
        - 模型版本不存在
    """
    pass


# ===== 5xx 服務器錯誤 =====

class ModelLoadError(InferenceError):
    """
    模型加載失敗
    
    當模型加載過程中發生錯誤時拋出，對應 HTTP 500 Internal Server Error。
    
    Examples:
        - 模型文件損壞
        - 模型格式不兼容
        - 內存不足以加載模型
        - 模型依賴缺失
    """
    pass


class InferenceExecutionError(InferenceError):
    """
    推理執行失敗
    
    當推理過程中發生錯誤時拋出，對應 HTTP 500 Internal Server Error。
    
    Examples:
        - 推理過程拋出異常
        - 數據預處理失敗
        - 結果後處理失敗
        - 引擎內部錯誤
    """
    pass


class EngineError(InferenceError):
    """
    引擎相關錯誤
    
    當推理引擎（Ollama、Transformers 等）發生錯誤時拋出，
    對應 HTTP 500 Internal Server Error。
    
    Examples:
        - Ollama 服務連接失敗
        - Transformers 模型推理失敗
        - 引擎配置錯誤
        - 引擎初始化失敗
    """
    pass


# ===== 503 服務不可用 =====

class ResourceExhaustedError(InferenceError):
    """
    資源耗盡錯誤
    
    當系統資源（GPU、內存等）不足時拋出，
    對應 HTTP 503 Service Unavailable。
    
    Examples:
        - GPU 內存不足
        - 系統內存不足
        - 所有 GPU 都在使用中
        - 並發請求數超過限制
    
    客戶端應該在收到此錯誤後等待一段時間再重試。
    """
    pass


# ===== 異常工具函數 =====

def get_http_status_code(exception: Exception) -> int:
    """
    根據異常類型返回建議的 HTTP 狀態碼
    
    Args:
        exception: 異常實例
        
    Returns:
        int: HTTP 狀態碼
    """
    if isinstance(exception, (ValidationError, UnsupportedTaskError)):
        return 400  # Bad Request
    elif isinstance(exception, (ResourceNotFoundError, ModelNotFoundError)):
        return 404  # Not Found
    elif isinstance(exception, ResourceExhaustedError):
        return 503  # Service Unavailable
    elif isinstance(exception, InferenceError):
        return 500  # Internal Server Error
    else:
        return 500  # Internal Server Error (默認)


def format_error_response(exception: Exception, include_traceback: bool = False) -> dict:
    """
    格式化異常為統一的錯誤響應格式
    
    Args:
        exception: 異常實例
        include_traceback: 是否包含堆棧追蹤（僅用於調試）
        
    Returns:
        dict: 格式化的錯誤響應
    """
    response = {
        "error_type": type(exception).__name__,
        "message": str(exception),
        "status_code": get_http_status_code(exception)
    }
    
    if include_traceback:
        import traceback
        response["traceback"] = traceback.format_exc()
    
    return response
