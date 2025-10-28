import os
import json
import uuid
from typing import Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

app = FastAPI(title="Qdrant Data Inserter API")

client = None
model = None

@app.on_event("startup")
async def startup_event():
    """應用啟動時初始化"""
    global client, model
    
    qdrant_host = os.getenv("QDRANT_HOST", "qdrant")
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
    
    print(f"🔌 正在連接 Qdrant ({qdrant_host}:{qdrant_port})...")
    try:
        client = QdrantClient(host=qdrant_host, port=qdrant_port, timeout=10)
        collections = client.get_collections()
        print(f"✅ 成功連接到 Qdrant")
    except Exception as e:
        print(f"❌ 無法連接到 Qdrant: {e}")
        raise
    
    print("🤖 正在載入向量模型 (BAAI/bge-m3)...")
    try:
        model = SentenceTransformer("BAAI/bge-m3")
        print("✅ 模型載入完成")
    except Exception as e:
        print(f"❌ 模型載入失敗: {e}")
        raise
    
    print("🚀 Insert API 已就緒！")


def ensure_collection_exists(collection_name: str) -> None:
    """確保 collection 存在"""
    try:
        client.get_collection(collection_name)
        print(f"✅ Collection '{collection_name}' 已存在")
    except:
        print(f"⚠️  Collection '{collection_name}' 不存在，正在創建...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
        )
        print(f"✅ 成功創建 collection: {collection_name}")


def extract_embedding_content(payload: Dict[str, Any]) -> str:
    """提取用於生成向量的內容"""
    embedding_type = payload.get("embedding_type", "")
    if embedding_type == "summary" and payload.get("summary"):
        return payload["summary"]
    elif embedding_type == "text" and payload.get("text"):
        return payload["text"]
    return payload.get("summary") or payload.get("text") or ""


@app.get("/")
async def root():
    """根路徑"""
    return {
        "message": "Qdrant Data Inserter API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health():
    """健康檢查"""
    try:
        client.get_collections()
        return {"status": "healthy"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"服務不可用: {str(e)}")


@app.post("/insert_json")
async def insert_json(file: UploadFile = File(...)):
    """插入 JSON 數據"""
    try:
        raw_data = await file.read()
        data = json.loads(raw_data.decode("utf-8"))
        
        if isinstance(data, dict):
            data = [data]
        
        print(f"📊 收到 {len(data)} 筆數據")
        
        grouped_data = {}
        for item in data:
            payload = item.get("payload", {})
            collection_name = payload.get("type", "")
            if not collection_name:
                continue
            grouped_data.setdefault(collection_name, []).append(item)
        
        results = {}
        for collection_name, items in grouped_data.items():
            ensure_collection_exists(collection_name)
            
            points = []
            for item in items:
                payload = item.get("payload", {})
                content = extract_embedding_content(payload)
                
                if not content:
                    continue
                
                vector = model.encode(content).tolist()
                point_id = item.get("id", str(uuid.uuid4()))
                
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload
                    )
                )
            
            if points:
                client.upsert(collection_name=collection_name, points=points)
                results[collection_name] = len(points)
                print(f"✅ 插入 {len(points)} 筆數據到 {collection_name}")
        
        return {
            "status": "success",
            "message": "成功插入數據",
            "results": results
        }
        
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"JSON 格式錯誤: {str(e)}")
    except Exception as e:
        print(f"❌ 插入失敗: {str(e)}")
        raise HTTPException(status_code=500, detail=f"插入失敗: {str(e)}")
