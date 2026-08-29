import logging
import threading
from typing import Any

from app.core.embedding import EmbeddingManager
from app.services.ingest import IngestService
from app.services.search import SearchService
from app.services.opensearch_service import OpenSearchService
from app.services.qdrant_service import QdrantService
from app.core.task_manager import TaskManager
from app.core.rerank import RerankerManager
from app.core.payload import PayloadSchemaManager

class ServiceContainer:
    """統一管理全域服務實例"""
    def __init__(self):
        self._instances: dict[str, Any] = {}
        self._lock = threading.RLock()

    def _get_instance(self, key: str, factory):
        if key not in self._instances:
            with self._lock:
                if key not in self._instances:
                    try:
                        self._instances[key] = factory()
                        logging.info(f"Initialized service: {key}")
                    except Exception as e:
                        logging.error(f"Failed to initialize {key}: {e}")
                        raise
        return self._instances[key]

    def get_embedding(self) -> EmbeddingManager:
        return self._get_instance("embedding", EmbeddingManager)

    def get_opensearch(self) -> OpenSearchService:
        return self._get_instance("opensearch", OpenSearchService)

    def get_qdrant(self) -> QdrantService:
        return self._get_instance("qdrant", QdrantService)

    def get_ingest(self) -> IngestService:
        return self._get_instance(
            "ingest",
            lambda: IngestService(
                opensearch=self.get_opensearch(),
                qdrant=self.get_qdrant(),
                payload_schema_manager=self.get_payload_schema_manager()
            )
        )

    def get_search(self) -> SearchService:
        return self._get_instance(
            "search",
            lambda: SearchService(
                opensearch=self.get_opensearch(),
                qdrant=self.get_qdrant(),
                reranker=self.get_reranker(),
                payload_schema_manager=self.get_payload_schema_manager()
            )
        )

    def get_task_manager(self) -> TaskManager:
        return self._get_instance("task_manager", TaskManager)

    def get_reranker(self) -> RerankerManager:
        return self._get_instance("reranker", RerankerManager)

    def get_payload_schema_manager(self) -> PayloadSchemaManager:
        return self._get_instance("payload_schema_manager", PayloadSchemaManager)

_container = ServiceContainer()

def get_embedding_service() -> EmbeddingManager:
    return _container.get_embedding()

def get_opensearch_service() -> OpenSearchService:
    return _container.get_opensearch()

def get_qdrant_service() -> QdrantService:
    return _container.get_qdrant()

def get_ingest_service() -> IngestService:
    return _container.get_ingest()

def get_search_service() -> SearchService:
    return _container.get_search()

def get_task_manager() -> TaskManager:
    return _container.get_task_manager()

def get_reranker_service() -> RerankerManager:
    return _container.get_reranker()

def get_payload_schema_manager() -> PayloadSchemaManager:
    return _container.get_payload_schema_manager()
