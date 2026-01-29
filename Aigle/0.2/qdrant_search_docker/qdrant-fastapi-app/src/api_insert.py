class QdrantDataInserter:
    def __init__(self, host: str = "localhost", port: int = 6333, model_name: str = "BAAI/bge-m3"):
        self.client = QdrantClient(host=host, port=port)
        self.model = SentenceTransformer(model_name)
        self.vector_size = 1024

    def ensure_collection_exists(self, collection_name: str) -> None:
        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
            )

    def extract_embedding_content(self, payload: Dict[str, Any]) -> str:
        embedding_type = payload.get("embedding_type", "")
        if embedding_type == "summary" and payload.get("summary"):
            return payload["summary"]
        elif embedding_type == "text" and payload.get("text"):
            return payload["text"]
        return payload.get("summary") or payload.get("text") or ""

    def insert_json_data(self, data: Any) -> Dict[str, int]:
        if isinstance(data, dict):
            data = [data]

        grouped_data = {}
        for item in data:
            payload = item.get("payload", {})
            collection_name = payload.get("type", "")
            if not collection_name:
                continue
            grouped_data.setdefault(collection_name, []).append(item)

        results = {}
        for collection_name, items in grouped_data.items():
            self.ensure_collection_exists(collection_name)
            points = []
            for item in items:
                payload = item.get("payload", {})
                content = self.extract_embedding_content(payload)
                if not content:
                    continue
                vector = self.model.encode(content).tolist()
                point_id = item.get("id", str(uuid.uuid4()))
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload
                    )
                )
            if points:
                self.client.upsert(collection_name=collection_name, points=points)
                results[collection_name] = len(points)
        return results


# 建立單例
inserter = QdrantDataInserter()

@app.post("/insert_json")
async def insert_json(file: UploadFile = File(...)):
    try:
        raw_data = await file.read()
        data = json.loads(raw_data.decode("utf-8"))
        results = inserter.insert_json_data(data)
        return {"status": "success", "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))