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
    type: Optional[str] = Field(None, description="集合類型: audio/video/document/image")
    filename: Optional[List[str]] = Field(None, description="文件名列表")
    source: Optional[str] = Field(None, description="圖像格式: jpg/png/jpeg/gif/bmp 等")
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
    title="圖像相似度搜索 API",
    description="基於 Qdrant 的圖像內容相似度搜索服務",
    version="1.0.0"
)

# 全局變數
client = None
model = None
collection_name = "images"

# ====== 新增：CacheManager 初始化 ======
cm = CacheManager(
    host='192.168.157.123',
    port=7000,
    password="dht888888",
    max_connections=1000,
    ttl=3600,           # 快取一小時
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
    
    try:
        client.get_collection(collection_name)
        print(f"✅ 成功連接到 collection: {collection_name}")
    except Exception as e:
        print(f"❌ Collection '{collection_name}' 不存在: {e}")
        raise
    
    print("🤖 正在載入向量模型 (BAAI/bge-m3)...")
    model = SentenceTransformer("BAAI/bge-m3")
    print("✅ 模型載入完成")


# ========== 篩選條件 ==========
def build_filter(
    embedding_type: str,
    type_value: Optional[str] = None,
    filenames: Optional[List[str]] = None,
    source: Optional[str] = None
) -> Optional[Filter]:
    """建立篩選條件"""
    must_conditions = []

    must_conditions.append(FieldCondition(key="status", match=MatchValue(value="active")))
    must_conditions.append(FieldCondition(key="embedding_type", match=MatchValue(value=embedding_type)))
    
    if type_value:
        must_conditions.append(FieldCondition(key="type", match=MatchValue(value=type_value)))
    
    if filenames:
        if len(filenames) == 1:
            must_conditions.append(FieldCondition(key="filename", match=MatchValue(value=filenames[0])))
        else:
            must_conditions.append(FieldCondition(key="filename", match=MatchAny(any=filenames)))
    
    if source:
        must_conditions.append(FieldCondition(key="source", match=MatchValue(value=source)))
    
    return Filter(must=must_conditions) if must_conditions else None


# ====== 新增：快取版搜尋函數 ======
@cm.cache
def cached_search(collection_name, query_vector, query_filter, limit):
    """具快取的 Qdrant 搜尋"""
    print("[CACHE] miss → querying Qdrant")
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
        "message": "圖像相似度搜索 API",
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


# ========== 搜尋端點 (整合 Cache) ==========
@app.post("/image_search", response_model=SearchResponse, tags=["搜索"])
async def search_images(request: SearchRequest):
    """執行圖像相似度搜索 (支援快取)"""
    try:
        start = time.perf_counter()

        if request.embedding_type not in ["summary", "text"]:
            raise HTTPException(status_code=400, detail="embedding_type 必須是 'summary' 或 'text'")
        
        query_filter = build_filter(
            embedding_type=request.embedding_type,
            type_value=request.type,
            filenames=request.filename,
            source=request.source
        )
        
        query_vector = model.encode(request.query_text).tolist()
        
        # ✅ 使用快取查詢
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
        print(f"[TIMED] /image_search took {end - start:.3f}s")

        return SearchResponse(success=True, total=len(formatted_results), results=formatted_results)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"搜索失敗: {str(e)}")


# ========== 其他 API 保持不變 ==========
@app.post("/indexes/create", response_model=IndexResponse, tags=["索引管理"])
async def create_indexes():
    """建立所有必要的索引"""
    try:
        index_fields = [
            ("embedding_type", "keyword"),
            ("type", "keyword"),
            ("filename", "keyword"),
            ("source", "keyword"),
        ]
        created, existing, errors = [], [], []
        for field_name, field_type in index_fields:
            try:
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=field_type
                )
                created.append(field_name)
            except Exception as e:
                if "already exists" in str(e).lower():
                    existing.append(field_name)
                else:
                    errors.append(f"{field_name}: {str(e)}")
        return IndexResponse(
            success=len(errors) == 0,
            message=f"建立 {len(created)} 個新索引，{len(existing)} 個已存在",
            indexes={"created": created, "existing": existing, "errors": errors}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"建立索引失敗: {str(e)}")


@app.get("/indexes", response_model=IndexResponse, tags=["索引管理"])
async def get_indexes():
    """查看現有索引"""
    try:
        collection_info = client.get_collection(collection_name)
        indexes = {}
        if collection_info.payload_schema:
            indexes = {field: str(schema) for field, schema in collection_info.payload_schema.items()}
        return IndexResponse(success=True, message=f"找到 {len(indexes)} 個索引", indexes=indexes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"獲取索引失敗: {str(e)}")


@app.get("/collection/info", tags=["集合管理"])
async def get_collection_info():
    """獲取 collection 詳細資訊"""
    try:
        info = client.get_collection(collection_name)
        return {
            "name": collection_name,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": info.status,
            "config": {
                "vector_size": info.config.params.vectors.size if hasattr(info.config.params, 'vectors') else None,
                "distance": str(info.config.params.vectors.distance) if hasattr(info.config.params, 'vectors') else None
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"獲取資訊失敗: {str(e)}")


# ========== 啟動服務 ==========
if __name__ == "__main__":
    uvicorn.run(
        "api_image_search_with_cache:app",
        host="0.0.0.0",
        port=8814,
        reload=True
    )