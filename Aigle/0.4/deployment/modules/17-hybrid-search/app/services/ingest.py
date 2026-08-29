import ijson
import asyncio
import logging
import uuid
from typing import List, IO, Optional

from app.schemas import DocumentItem
from app.services.qdrant_service import QdrantService
from app.services.opensearch_service import OpenSearchService
from app.core.embedding import EmbeddingManager
from app.core.task_manager import TaskManager
from app.core.payload import PayloadSchemaManager


class IngestService:
    def __init__(self, qdrant: QdrantService, opensearch: OpenSearchService, payload_schema_manager: PayloadSchemaManager):
        self.qdrant = qdrant
        self.opensearch = opensearch
        self.schema_manager = payload_schema_manager

    async def ingest_docs(
        self,
        docs: List[DocumentItem],
        embedder: EmbeddingManager,
        batch_size: int = 16,
        payload_schema: Optional[str] = None
    ) -> int:
        texts = []
        valid_docs = []

        logging.info(f"Getting extractor for payload schema: {payload_schema}")
        extractor = self.schema_manager.get_extractor(payload_schema)

        for doc in docs:
            payload = doc.payload
            content = extractor(payload)
            doc.payload["payload_schema"] = payload_schema

            if content:
                texts.append(content)
                valid_docs.append(doc)
        
        logging.info(f"Valid docs: {len(valid_docs)}")

        if texts:
            embeddings = await embedder.encode_async(texts, batch_size=batch_size)
            for doc, vector in zip(valid_docs, embeddings):
                doc.vector = vector

        for doc in valid_docs:
            if doc.id is None:
                doc.id = str(uuid.uuid4())

        await asyncio.gather(
            self.qdrant.batch_insert(valid_docs),
            self.opensearch.batch_insert(valid_docs)
        )
        return len(valid_docs)

    async def ingest_file(
        self,
        file_obj: IO,
        embedder: EmbeddingManager,
        batch_size: int = 128,
        payload_schema: Optional[str] = None
    ) -> int:
        """同步 ingest JSON file"""
        batch_docs: List[DocumentItem] = []
        total_count = 0

        logging.info("Ingesting file...")
        parser = ijson.items(file_obj, "item")
        for obj in parser:
            batch_docs.append(DocumentItem(**obj))
            if len(batch_docs) >= batch_size:
                count = await self.ingest_docs(batch_docs, embedder, payload_schema=payload_schema)
                total_count += count
                batch_docs = []

        if batch_docs:
            count = await self.ingest_docs(batch_docs, embedder, payload_schema=payload_schema)
            total_count += count

        return total_count

    async def ingest_file_task(
        self,
        file_obj: IO,
        embedder: EmbeddingManager,
        task_manager: TaskManager,
        task_id: str,
        batch_size: int = 128,
        payload_schema: Optional[str] = None
    ):
        """在 service 裡面做背景 ingest 任務"""
        batch_docs: List[DocumentItem] = []
        total_count = 0

        try:
            parser = ijson.items(file_obj, "item")
            for obj in parser:
                batch_docs.append(DocumentItem(**obj))
                if len(batch_docs) >= batch_size:
                    count = await self.ingest_docs(batch_docs, embedder, payload_schema=payload_schema)
                    total_count += count
                    task_manager.update_progress(task_id, done=total_count)
                    batch_docs = []

            if batch_docs:
                count = await self.ingest_docs(batch_docs, embedder, payload_schema=payload_schema)
                total_count += count
                task_manager.update_progress(task_id, done=total_count)

            task_manager.mark_done(task_id, total=total_count)
        except Exception as e:
            task_manager.mark_failed(task_id, str(e))
            logging.error(f"Background ingest failed: {e}")