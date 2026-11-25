# src/inference/exceptions.py
"""
推理系統的統一異常定義

提供分層的異常類型，便於 API 層根據異常類型返回適當的 HTTP 狀態碼。
每個類別帶有穩定的 `code`（給呼叫端程式判斷用，不隨訊息文字變動）與
建議的 `http_status`；個別實例可用 `details` 帶出額外欄位（例如 model_limit）。

異常層次結構（括號內為 HTTP status / code）：
    InferenceError (500 / inference_error)
    ├── ValidationError (400 / invalid_request)
    │   ├── UnsupportedTaskError (400 / unsupported_task)
    │   ├── InvalidParameterError (400 / invalid_parameter)
    │   ├── ConflictingParametersError (400 / conflicting_parameters)
    │   ├── UnsupportedParameterError (400 / unsupported_parameter)
    │   └── UnsupportedCapabilityError (400 / unsupported_capability)
    ├── PolicyViolationError (400 / policy_violation)
    ├── ResourceNotFoundError (404 / resource_not_found)
    │   └── ModelNotFoundError (404 / model_not_found)
    │       ├── ModelVersionNotFoundError (404 / model_version_not_found)
    │       └── ModelAliasNotFoundError (404 / model_alias_not_found)
    ├── MaxTokensExceededError (422 / max_tokens_exceeded)
    ├── StructuredOutputError (502 / structured_output_error)
    ├── ModelLoadError (500 / model_load_failed)
    ├── InferenceExecutionError (500 / inference_failed)
    ├── EngineError (500 / engine_error)
    │   ├── InferenceTimeoutError (504 / inference_timeout)
    │   └── RuntimeUnavailableError (503 / runtime_unavailable)
    ├── RegistryUnavailableError (503 / registry_unavailable)
    └── ResourceExhaustedError (503 / resource_exhausted)
"""

from typing import Any


class InferenceError(Exception):
    """
    推理相關異常的基類

    所有推理系統的自定義異常都應該繼承此類。
    """

    code = "inference_error"
    http_status = 500

    def __init__(self, message: str = "", *, code: str | None = None, **details: Any):
        super().__init__(message)
        if code:
            self.code = code
        self.details: dict[str, Any] = details


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

    code = "invalid_request"
    http_status = 400


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

    code = "unsupported_task"


class InvalidParameterError(ValidationError):
    """共用生成參數的值不合法（如 temperature 超出 0～2、max_tokens 非正整數）。"""

    code = "invalid_parameter"


class ConflictingParametersError(ValidationError):
    """同義或互斥的參數同時提供且值互相矛盾（如 max_tokens 與 max_new_tokens 不同）。"""

    code = "conflicting_parameters"


class UnsupportedParameterError(ValidationError):
    """已宣告的共用參數在此模型／runtime 路徑無法套用。"""

    code = "unsupported_parameter"


class UnsupportedCapabilityError(ValidationError):
    """請求要的能力（如結構化輸出）此模型／runtime 做不到；拒絕而不是假裝做到。"""

    code = "unsupported_capability"


class PolicyViolationError(InferenceError):
    """
    護欄政策阻擋（GB-6）

    當 Guardrail Service（module 23）對輸入或輸出回傳 action="block" 時拋出，
    對應 HTTP 400 Bad Request。

    Attributes:
        category:  違規類別（regex 標籤如 "jailbreak"，或 Llama Guard 碼如 "S1"）
        direction: "input"（送入 LLM 前）或 "output"（LLM 回應後）
    """

    code = "policy_violation"
    http_status = 400

    def __init__(self, category: str | None = None, direction: str = "input"):
        self.category = category
        self.direction = direction
        super().__init__(f"policy_violation ({direction}, category={category})")


class ResourceNotFoundError(InferenceError):
    """
    資源未找到錯誤（通用）

    當請求的資源不存在時拋出，對應 HTTP 404 Not Found。

    Examples:
        - 模型未找到
        - 文件未找到
        - 配置未找到
    """

    code = "resource_not_found"
    http_status = 404


class ModelNotFoundError(ResourceNotFoundError):
    """
    模型未找到錯誤

    當請求的模型在 MLflow 或本地不存在時拋出，
    對應 HTTP 404 Not Found。

    Examples:
        - MLflow 中未註冊該模型
        - 模型文件不存在
    """

    code = "model_not_found"


class ModelVersionNotFoundError(ModelNotFoundError):
    """模型存在，但指定的 model_version 不存在（不回退到其他版本）。"""

    code = "model_version_not_found"


class ModelAliasNotFoundError(ModelNotFoundError):
    """模型存在，但指定的 model_alias 未設定（不回退到其他版本）。"""

    code = "model_alias_not_found"


class MaxTokensExceededError(InferenceError):
    """
    要求的 max_tokens 超過該模型版本已知的輸出上限，對應 HTTP 422。

    只在上限已配置（非 null）時拋出；details 帶 model_limit / requested。
    """

    code = "max_tokens_exceeded"
    http_status = 422

    def __init__(self, requested: int, model_limit: int, model_name: str = "", model_version: str = ""):
        super().__init__(
            f"max_tokens={requested} exceeds the output-token limit {model_limit} "
            f"of model '{model_name}' v{model_version}",
            requested=requested,
            model_limit=model_limit,
        )
        self.requested = requested
        self.model_limit = model_limit


class StructuredOutputError(InferenceError):
    """
    模型輸出不符合請求的 response_format / format（不是 JSON，或違反 schema），對應 HTTP 502。

    錯在上游模型的輸出，不在請求；details 帶 violations（違規位置與原因）。
    不合格的輸出絕不會當成合格的結果傳回。
    """

    code = "structured_output_error"
    http_status = 502


# ===== 5xx 服務器錯誤 =====

class ModelLoadError(InferenceError):
    """
    模型加載失敗

    當模型加載過程中發生錯誤時拋出，對應 HTTP 500 Internal Server Error。

    Examples:
        - 模型文件損壞
        - 模型格式不兼容
        - 模型依賴缺失
    """

    code = "model_load_failed"


class InferenceExecutionError(InferenceError):
    """
    推理執行失敗

    當推理過程中發生錯誤時拋出，對應 HTTP 500 Internal Server Error。

    Examples:
        - 推理過程拋出異常
        - 數據預處理失敗
        - 結果後處理失敗
    """

    code = "inference_failed"


class EngineError(InferenceError):
    """
    引擎相關錯誤

    當推理引擎（Ollama、Transformers 等）發生錯誤時拋出，
    對應 HTTP 500 Internal Server Error。

    Examples:
        - Transformers 模型推理失敗
        - Ollama 回傳非 200 且無法歸類的錯誤
        - 引擎配置錯誤
    """

    code = "engine_error"


class InferenceTimeoutError(EngineError):
    """上游 runtime（如 Ollama daemon）在逾時內沒有回應，對應 HTTP 504。"""

    code = "inference_timeout"
    http_status = 504


class RuntimeUnavailableError(EngineError):
    """runtime（如 Ollama daemon）無法連線，對應 HTTP 503。"""

    code = "runtime_unavailable"
    http_status = 503


class RegistryUnavailableError(InferenceError):
    """模型 registry（MLflow）無法連線或回傳非預期錯誤，對應 HTTP 503。

    與 ModelNotFoundError 區分：registry 故障時不能斷定模型不存在。
    """

    code = "registry_unavailable"
    http_status = 503


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

    code = "resource_exhausted"
    http_status = 503


# ===== 異常工具函數 =====

def get_http_status_code(exception: Exception) -> int:
    """
    根據異常類型返回建議的 HTTP 狀態碼

    Args:
        exception: 異常實例

    Returns:
        int: HTTP 狀態碼
    """
    return getattr(exception, "http_status", 500) if isinstance(exception, InferenceError) else 500


def get_error_code(exception: Exception) -> str:
    """回傳穩定錯誤碼；非推理系統異常一律為 internal_error。"""
    return getattr(exception, "code", "inference_error") if isinstance(exception, InferenceError) else "internal_error"


def error_detail(exception: Exception) -> tuple[int, dict[str, Any]]:
    """(HTTP status, detail dict)：error_type / message / code，加上異常實例的 details 欄位。"""
    detail: dict[str, Any] = {
        "error_type": type(exception).__name__,
        "message": str(exception),
        "code": get_error_code(exception),
    }
    detail.update(getattr(exception, "details", None) or {})
    if isinstance(exception, ResourceExhaustedError):
        detail.setdefault("retry_after", 60)
    return get_http_status_code(exception), detail


def format_error_response(exception: Exception, include_traceback: bool = False) -> dict:
    """
    格式化異常為統一的錯誤響應格式

    Args:
        exception: 異常實例
        include_traceback: 是否包含堆棧追蹤（僅用於調試）

    Returns:
        dict: 格式化的錯誤響應
    """
    status_code, response = error_detail(exception)
    response["status_code"] = status_code

    if include_traceback:
        import traceback
        response["traceback"] = traceback.format_exc()

    return response
