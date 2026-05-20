from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from qdrant_client.models import (
    Filter, 
    FieldCondition, 
    MatchValue, 
    MatchAny
)
from qdrant_client.http.exceptions import ResponseHandlingException

import asyncio
import logging

from app.core.config import settings
from app.schemas import DocumentItem, SearchRequest, SearchResult, PayloadSchema

class QdrantService:
    def __init__(self):
        self.client = AsyncQdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
        )

    async def initialize(self):
        await self._ensure_collection()
        logging.info("QdrantService initialization complete")
        
    async def _ensure_collection(self):
        exists = await self.client.collection_exists(
            collection_name=settings.QDRANT_COLLECTION
        )
        if not exists:
            await self.client.create_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config=models.VectorParams(
                    size=settings.VECTOR_DIM,
                    distance=models.Distance.COSINE,
                ),
            )

    async def batch_insert(
        self,
        items: list[DocumentItem],
        batch_size: int = 128,
        max_concurrency: int = 2,
        max_retries: int = 3,
    ):
        points = [
            models.PointStruct(
                id=item.id,
                vector=item.vector,
                payload=item.payload,
            )
            for item in items
        ]

        semaphore = asyncio.Semaphore(max_concurrency)

        async def upsert_batch(batch, batch_idx: int):
            async with semaphore:
                for attempt in range(1, max_retries + 1):
                    try:
                        logging.info(
                            f"[Qdrant] Upserting batch {batch_idx} "
                            f"({len(batch)} points), attempt {attempt}"
                        )
                        await self.client.upsert(
                            collection_name=settings.QDRANT_COLLECTION,
                            points=batch,
                            wait=True,
                        )
                        return
                    except ResponseHandlingException as e:
                        logging.warning(
                            f"[Qdrant] Batch {batch_idx} failed (attempt {attempt}): {e}"
                        )
                        if attempt == max_retries:
                            raise
                        await asyncio.sleep(1.5 * attempt)

        tasks = []
        for idx, i in enumerate(range(0, len(points), batch_size), start=1):
            batch = points[i:i + batch_size]
            tasks.append(upsert_batch(batch, idx))

        await asyncio.gather(*tasks)

    @staticmethod
    def build_filters(req: SearchRequest, schema: PayloadSchema) -> Filter:
        conditions = [FieldCondition(key="payload_schema", match=MatchValue(value=schema.name))]

        if not schema.skip_status_filter:
            conditions.insert(0, FieldCondition(key="status", match=MatchValue(value="active")))

        if req.branch_id:
            conditions.append(FieldCondition(key="branch_id", match=MatchValue(value=req.branch_id)))

        if not schema.skip_req_filters:
            if req.embedding_type:
                conditions.append(FieldCondition(key="embedding_type", match=MatchValue(value=req.embedding_type)))
            if req.type:
                if isinstance(req.type, list):
                    conditions.append(FieldCondition(key="type", match=MatchAny(any=req.type)))
                else:
                    conditions.append(FieldCondition(key="type", match=MatchValue(value=req.type)))
            if req.filename:
                conditions.append(FieldCondition(key="filename", match=MatchAny(any=req.filename)))
            if req.speaker:
                conditions.append(FieldCondition(key="speaker", match=MatchAny(any=req.speaker)))
            if req.source:
                conditions.append(FieldCondition(key="source", match=MatchValue(value=req.source)))

        return Filter(must=conditions)

    async def execute_search(self, query_body: dict) -> list[SearchResult]:
        res = await self.client.search(
            collection_name=settings.QDRANT_COLLECTION,
            query_vector=query_body.get("vector"),
            query_filter=query_body.get("filter"),
            limit=query_body.get("limit"),
            with_payload=True,
            with_vectors=False
        )
        return [
            SearchResult(
                id=r.id,
                score=r.score,
                payload=r.payload
            ) for r in res
        ]

    async def liveness(self) -> bool:
        try:
            await asyncio.wait_for(
                self.client.http.service_api.healthz(),
                timeout=2,
            )
            return True
        except Exception:
            return False

    async def readiness(self) -> bool:
        try:
            await asyncio.wait_for(
                self.client.get_collection(settings.QDRANT_COLLECTION),
                timeout=3,
            )
            return True
        except Exception:
            return False