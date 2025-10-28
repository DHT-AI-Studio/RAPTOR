from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny
from sentence_transformers import SentenceTransformer
import uvicorn
import time

# ====== 新增：CacheManager 匯入 ======
from cache_manager import CacheManager

# ========== 資料模型 ==========
class SearchRequest(BaseModel):
    """搜索請求模型"""
    query_text: str = Field(..., description="搜索關鍵字", min_length=1)
    embedding_type: str = Field(..., description="搜索類型: summary 或 text")
    type: Optional[str] = Field(None, description="集合類型: audio/video/document")
    filename: Optional[List[str]] = Field(None, description="文件名列表")
    source: Optional[str] = Field(None, description="檔案類型: csv/pdf/docx/xlsx 等")
    limit: int = Field(5, description="返回結果數量", ge=1, le=100)

class SearchResult(BaseModel):
    """單個搜索結果"""
    score: float
    id: str
    payload: dict

class SearchResponse(BaseModel):
    """搜索響應模型"""
    success: bool
    total: int
    results: List[SearchResult]

class IndexResponse(BaseModel):
    """索引操作響應"""
    success: bool
    message: str
    indexes: Optional[dict] = None


# ========== 全局配置 ==========
app = FastAPI(
    title="文檔相似度搜索 API",
    description="基於 Qdrant 的文檔內容相似度搜索服務",
    version="1.0.0"
)

# 全局變數
client = None
model = None
collection_name = "documents"

# ====== 新增：CacheManager 初始化 ======
cm = CacheManager(
    host='192.168.157.123',
    port=7000,
    password="dht888888",
    max_connections=1000,
    ttl=3600,
    ttl_multiplier=1e-2,
    is_cluster=True
)


# ========== 啟動事件 ==========
@app.on_event("startup")
async def startup_event():
    """應用啟動時初始化"""
    global client, model
    
    print("🔌 正在連接 Qdrant...")
    client = QdrantClient(host="localhost", port=6333)
    
    # 檢查 collection
    try:
        client.get_collection(collection_name)
        print(f"✅ 成功連接到 collection: {collection_name}")
    except Exception as e:
        print(f"❌ Collection '{collection_name}' 不存在: {e}")
        raise
    
    # 載入模型
    print("🤖 正在載入向量模型 (BAAI/bge-m3)...")
    model = SentenceTransformer("BAAI/bge-m3")
    print("✅ 模型載入完成")


# ========== 建立篩選條件 ==========
def build_filter(
    embedding_type: str,
    type_value: Optional[str] = None,
    filenames: Optional[List[str]] = None,
    source: Optional[str] = None
) -> Optional[Filter]:
    """建立篩選條件"""
    must_conditions = []

    # status (必要)
    must_conditions.append(FieldCondition(key="status", match=MatchValue(value="active")))
    
    # embedding_type (必要)
    must_conditions.append(FieldCondition(key="embedding_type", match=MatchValue(value=embedding_type)))
    
    # type
    if type_value:
        must_conditions.append(FieldCondition(key="type", match=MatchValue(value=type_value)))
    
    # filename
    if filenames:
        if len(filenames) == 1:
            must_conditions.append(FieldCondition(key="filename", match=MatchValue(value=filenames[0])))
        else:
            must_conditions.append(FieldCondition(key="filename", match=MatchAny(any=filenames)))
    
    # source
    if source:
        must_conditions.append(FieldCondition(key="source", match=MatchValue(value=source)))
    
    return Filter(must=must_conditions) if must_conditions else None


# ====== 新增：快取版搜尋函數 ======
@cm.cache
def cached_search(collection_name, query_vector, query_filter, limit):
    """具快取的 Qdrant 搜尋"""
    results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
        with_vectors=False
    )
    return results


# ========== API 端點 ==========
@app.get("/", tags=["系統"])
async def root():
    """API 根路徑"""
    return {
        "message": "文檔相似度搜索 API",
        "version": "1.0.0",
        "docs": "/docs",
        "collection": collection_name
    }


@app.get("/health", tags=["系統"])
async def health_check():
    """健康檢查"""
    try:
        collection_info = client.get_collection(collection_name)
        return {
            "status": "healthy",
            "collection": collection_name,
            "vectors_count": collection_info.vectors_count,
            "points_count": collection_info.points_count
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"服務不可用: {str(e)}")


# ========== 文件搜尋 (加入快取) ==========
@app.post("/document_search", response_model=SearchResponse, tags=["搜索"])
async def search_documents(request: SearchRequest):
    """
    執行文檔相似度搜索 (支援快取)
    """
    try:
        start = time.perf_counter()

        # 驗證 embedding_type
        if request.embedding_type not in ["summary", "text"]:
            raise HTTPException(status_code=400, detail="embedding_type 必須是 'summary' 或 'text'")
        
        # 建立篩選條件
        query_filter = build_filter(
            embedding_type=request.embedding_type,
            type_value=request.type,
            filenames=request.filename,
            source=request.source
        )
        
        # 生成查詢向量
        query_vector = model.encode(request.query_text).tolist()
        
        # ✅ 使用快取搜尋
        results = cached_search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=request.limit
        )
        
        formatted_results = [
            SearchResult(score=res.score, id=str(res.id), payload=res.payload)
            for res in results
        ]
        
        end = time.perf_counter()
        print(f"[TIMED] /document_search took {end - start:.3f}s")

        return SearchResponse(success=True, total=len(formatted_results), results=formatted_results)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"搜索失敗: {str(e)}")


# 其餘 API (indexes、collection info、stats...) 保持不變
# =====================================================
# ... [略，原本內容完全不變]
# =====================================================

if __name__ == "__main__":
    uvicorn.run(
        "api_document_search_with_cache:app",
        host="192.168.157.124",
        port=8813,
        reload=True
    )