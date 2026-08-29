from fastapi import HTTPException

class AppBaseException(HTTPException):
    """所有自定義異常的基礎類別"""
    def __init__(self, detail: str, status_code: int = 500):
        super().__init__(status_code=status_code, detail=detail)

# === Embedding 相關錯誤 ===
class EmbeddingError(AppBaseException):
    def __init__(self, detail: str):
        super().__init__(detail=detail, status_code=503)

# === 搜尋錯誤 ===
class SearchError(AppBaseException):
    def __init__(self, detail: str):
        super().__init__(detail=detail, status_code=503)

# === 批次寫入錯誤 ===
class IngestError(AppBaseException):
    def __init__(self, detail: str):
        super().__init__(detail=detail, status_code=500)

# === 請求參數錯誤 ===
class ValidationError(AppBaseException):
    def __init__(self, detail: str):
        super().__init__(detail=detail, status_code=400)

# === 外部服務錯誤 ===
class ExternalServiceError(AppBaseException):
    def __init__(self, detail: str):
        super().__init__(detail=detail, status_code=502)