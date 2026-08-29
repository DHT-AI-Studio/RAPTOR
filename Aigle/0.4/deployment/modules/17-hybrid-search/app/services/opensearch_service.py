import json
import logging
import httpx
from typing import List
from opensearchpy import AsyncOpenSearch
from opensearchpy.helpers import async_bulk

from app.core.config import settings
from app.schemas import DocumentItem, SearchRequest, SearchResult, PayloadSchema

class OpenSearchService:
    def __init__(self):
        self.client = AsyncOpenSearch(
            hosts=[{"host": settings.OPENSEARCH_HOST, "port": settings.OPENSEARCH_PORT}],
            http_auth=(settings.OPENSEARCH_USER, settings.OPENSEARCH_PASSWORD),
            use_ssl=True,
            verify_certs=settings.OPENSEARCH_VERIFY_CERTS,
            ssl_show_warn=False,
        )

    async def initialize(self):
        await self._ensure_pipeline()
        await self._ensure_index()
        await self._create_dashboards_pattern()
        await self._warmup()
        logging.info("OpenSearchService initialization complete")

    async def _warmup(self):
        try:
            await self.client.search(
                index=settings.OPENSEARCH_INDEX,
                body={"size": 1, "query": {"match_all": {}}},
            )
            logging.info("OpenSearch warm-up query done")
        except Exception:
            pass  # index 不存在或無資料時不影響啟動

    async def _ensure_pipeline(self):
        pipeline_id = "rrf-hybrid-pipeline"

        pipeline_body = {
            "description": "RRF pipeline for hybrid search",
            "phase_results_processors": [
                {
                    "score-ranker-processor": {
                        "combination": {
                            "technique": "rrf",
                            "rank_constant": 60,
                        }
                    }
                }
            ],
        }

        await self.client.transport.perform_request(
            method="PUT",
            url=f"/_search/pipeline/{pipeline_id}",
            body=pipeline_body,
        )

    async def _ensure_index(self):
        exists = await self.client.indices.exists(
            index=settings.OPENSEARCH_INDEX
        )

        if not exists:
            mapping = {
                "settings": {
                    "index": {
                        "number_of_shards": 1,
                        "knn": True,
                    }
                },
                "mappings": {
                    "properties": {
                        "my_vector": {
                            "type": "knn_vector",
                            "dimension": settings.VECTOR_DIM,
                            "method": {
                                "name": "hnsw",
                                "space_type": "cosinesimil",
                                "engine": "lucene",
                            },
                        },
                        "payload": {
                            "properties": {
                                "summary": {"type": "text"},
                                "text": {"type": "text"},
                                "status": {"type": "keyword"},
                                "embedding_type": {"type": "keyword"},
                                "type": {"type": "keyword"},
                                "filename": {"type": "keyword"},
                                "speaker": {"type": "keyword"},
                                "source": {"type": "keyword"},
                                "video_id": { "type": "keyword" },
                                "payload_schema": { "type": "keyword" },
                                "asset_path": { "type": "keyword" },
                                "version_id": { "type": "keyword" },
                                "branch_id": { "type": "keyword" },
                                # temporal reasoning fields
                                "duration": { "type": "float" },
                                "events": {
                                    "type": "nested",
                                    "properties": {
                                        "start_timestamp": { "type": "float" },
                                        "end_timestamp": { "type": "float" },
                                        "description": {
                                            "type": "text"
                                        },
                                        "objects": {
                                            "type": "keyword"
                                        }
                                    }
                                }
                            }
                        },
                    }
                },
            }

            await self.client.indices.create(
                index=settings.OPENSEARCH_INDEX,
                body=mapping,
            )

    async def _create_dashboards_pattern(self):
        dashboards_url = (
            f"http://{settings.OPENSEARCH_DASHBOARDS_HOST}:5601"
            f"/api/saved_objects/index-pattern/hybrid_index_pattern"
        )

        headers = {
            "osd-xsrf": "true",
            "Content-Type": "application/json",
        }

        payload = {
            "attributes": {
                "title": settings.OPENSEARCH_INDEX,
            }
        }

        async with httpx.AsyncClient(verify=False) as client:
            try:
                await client.post(
                    dashboards_url,
                    json=payload,
                    headers=headers,
                    auth=(
                        settings.OPENSEARCH_USER,
                        settings.OPENSEARCH_PASSWORD,
                    ),
                )
            except Exception as e:
                logging.warning(
                    f"Dashboards pattern creation failed: {e}"
                )

    async def batch_insert(self, items: List[DocumentItem]):
        logging.info(f"Total items to index: {len(items)}")

        async def actions():
            for item in items:
                yield {
                    "_op_type": "index",
                    "_index": settings.OPENSEARCH_INDEX,
                    "_id": item.id,
                    "_source": {
                        "my_vector": item.vector,
                        "payload": item.payload,
                    },
                }

        success, failed = await async_bulk(
            self.client,
            actions(),
            raise_on_error=False,
            request_timeout=30,
        )

        if failed:
            logging.error(f"[OpenSearch] {len(failed)} document(s) failed to index: {failed[:3]}")
            raise RuntimeError(f"OpenSearch batch_insert: {len(failed)} doc(s) failed, first error: {failed[0]}")

        return {
            "success_count": success,
            "failed_count": 0,
        }

    @staticmethod
    def build_filters(req: SearchRequest, schema: PayloadSchema) -> list:
        filters = [{"term": {"payload.payload_schema": schema.name}}]

        if not schema.skip_status_filter:
            filters.insert(0, {"term": {"payload.status": "active"}})

        if req.branch_id:
            filters.append({"term": {"payload.branch_id": req.branch_id}})

        if not schema.skip_req_filters:
            if req.embedding_type:
                filters.append({"term": {"payload.embedding_type": req.embedding_type}})
            if req.type:
                if isinstance(req.type, list):
                    filters.append({"terms": {"payload.type": req.type}})
                else:
                    filters.append({"term": {"payload.type": req.type}})
            if req.filename:
                filters.append({"terms": {"payload.filename": req.filename}})
            if req.speaker:
                filters.append({"terms": {"payload.speaker": req.speaker}})
            if req.source:
                filters.append({"term": {"payload.source": req.source}})

        return filters

    async def execute_search(self, query_body: dict, use_pipeline: bool = False) -> list[SearchResult]:
        params = {"search_pipeline": "rrf-hybrid-pipeline"} if use_pipeline else {}
        response = await self.client.search(
            index=settings.OPENSEARCH_INDEX, 
            body=query_body, 
            params=params
        )

        return [
            SearchResult(
                id=hit["_id"],
                score=hit["_score"],
                payload=hit["_source"]["payload"]
            ) for hit in response['hits']['hits']
        ]