from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny
from sentence_transformers import SentenceTransformer
import uvicorn
import time

from cache_manager import CacheManager

class SearchRequest(BaseModel):
    query_text: str = Field(..., description="搜索關鍵字", min_length=1)
    embedding_type: str = Field(..., description="搜索類型: summary 或 text")
    type: Optional[str] = Field(None, description="集合類型: audio/video/document/image")
    filename: Optional[List[str]] = Field(None, description="文件名列表")
    speaker: Optional[List[str]] = Field(None, description="說話者列表 (僅 text 模式)")
    limit: int = Field(5, description="返回結果數量", ge=1, le=100)

class SearchResult(BaseModel):
    score: float
    id: str
    payload: dict

class SearchResponse(BaseModel):
    success: bool
    total: int
    results: List[SearchResult]

class IndexResponse(BaseModel):
    success: bool
    message: str
    indexes: Optional[dict] = None

app = FastAPI(
    title="影片相似度搜索 API",
    description="基於 Qdrant 的影片內容相似度搜索服務",
    version="1.0.0"
)

client = None
model = None
collection_name = "videos"

cm = CacheManager(
    host='192.168.157.123',
    port=7000,
    password="dht888888",
    max_connections=1000,
    ttl=3600,
    ttl_multiplier=1e-2,
    is_cluster=True
)

@app.on_event("startup")
async def startup_event():
    global client, model
    
    client = QdrantClient(host="localhost", port=6333)
    
    try:
        client.get_collection(collection_name)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Collection '{collection_name}' 不存在: {e}")
    
    model = SentenceTransformer("BAAI/bge-m3")

def build_filter(
    embedding_type: str,
    type_value: Optional[str] = None,
    filenames: Optional[List[str]] = None,
    speakers: Optional[List[str]] = None
) -> Optional[Filter]:
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
    
    if speakers and embedding_type == "text":
        if len(speakers) == 1:
            must_conditions.append(FieldCondition(key="speaker", match=MatchValue(value=speakers[0])))
        else:
            must_conditions.append(FieldCondition(key="speaker", match=MatchAny(any=speakers)))
    
    return Filter(must=must_conditions) if must_conditions else None

@cm.cache
def cached_search(collection_name, query_vector, query_filter, limit):
    results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
        with_vectors=False
    )
    return results

@app.get("/", tags=["系統"])
async def root():
    return {
        "message": "影片相似度搜索 API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", tags=["系統"])
async def health_check():
    try:
        client.get_collection(collection_name)
        return {"status": "healthy", "collection": collection_name}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"服務不可用: {str(e)}")

@app.post("/video_search", response_model=SearchResponse, tags=["搜索"])
async def search_videos(request: SearchRequest):
    try:
        if request.embedding_type not in ["summary", "text"]:
            raise HTTPException(status_code=400, detail="embedding_type 必須是 'summary' 或 'text'")
        
        query_filter = build_filter(
            embedding_type=request.embedding_type,
            type_value=request.type,
            filenames=request.filename,
            speakers=request.speaker
        )
        
        query_vector = model.encode(request.query_text).tolist()
        
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

        return SearchResponse(success=True, total=len(formatted_results), results=formatted_results)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"搜索失敗: {str(e)}")

@app.post("/indexes/create", response_model=IndexResponse, tags=["索引管理"])
async def create_indexes():
    try:
        index_fields = [
            ("embedding_type", "keyword"),
            ("type", "keyword"),
            ("filename", "keyword"),
            ("speaker", "keyword"),
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
    try:
        info = client.get_collection(collection_name)
        return {
            "name": collection_name,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": info.status
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"獲取資訊失敗: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "api_video_search_with_cache:app",
        host="0.0.0.0",
        port=8811,
        reload=True
    )