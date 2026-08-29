import time
from typing import List, Dict
from collections import defaultdict
import asyncio

from app.services.qdrant_service import QdrantService
from app.services.opensearch_service import OpenSearchService
from app.core.rerank import RerankerManager
from app.core.config import settings
from app.schemas import SearchRequest, SearchResult, SearchResponse
from app.utils import measure_time
from app.core.payload import PayloadSchemaManager


def rrf_fusion(
    result_lists: List[List[SearchResult]],
    k: int = 60
) -> List[SearchResult]:
    scores = defaultdict(float)
    payload_map: Dict[str, Dict] = {}

    for results in result_lists:
        for rank, res in enumerate(results, start=1):
            scores[res.id] += 1.0 / (k + rank)
            payload_map.setdefault(res.id, res.payload)

    fused = [
        SearchResult(
            id=doc_id,
            score=score,
            payload=payload_map[doc_id]
        )
        for doc_id, score in scores.items()
    ]

    fused.sort(key=lambda x: x.score, reverse=True)
    return fused


class SearchService:
    def __init__(
        self,
        qdrant: QdrantService,
        opensearch: OpenSearchService,
        reranker: RerankerManager,
        payload_schema_manager: PayloadSchemaManager
    ):
        self.qdrant = qdrant
        self.opensearch = opensearch
        self.reranker = reranker
        self.schema_manager = payload_schema_manager

    def _build_bm25_clauses(self, query: str, fields: list[str], nested_paths: list[str], nested_operator: str = "should"):
        """
        Build BM25 clauses:
        - Nested fields -> nested + match, multi nested fields -> bool.should / bool.must
        - Non-nested fields -> single match or multi_match
        """
        nested_map = {}  # nested_path -> list of fields
        non_nested_fields = []

        for field in fields:
            nested_path = next((path for path in nested_paths if field.startswith(path + ".")), None)
            if nested_path:
                nested_map.setdefault(nested_path, []).append(field)
            else:
                non_nested_fields.append(field)

        clauses = []

        # 非 nested
        if len(non_nested_fields) == 1:
            clauses.append({
                "match": {non_nested_fields[0]: {"query": query, "operator": "or"}}
            })
        elif len(non_nested_fields) > 1:
            clauses.append({
                "multi_match": {"query": query, "fields": non_nested_fields, "operator": "or"}
            })

        # nested
        for path, nested_fields in nested_map.items():
            if len(nested_fields) == 1:
                clauses.append({
                    "nested": {
                        "path": path,
                        "query": {
                            "match": {nested_fields[0]: {"query": query, "operator": "or"}}
                        }
                    }
                })
            else:
                clauses.append({
                    "nested": {
                        "path": path,
                        "query": {
                            "bool": {
                                nested_operator: [
                                    {"match": {f: {"query": query, "operator": "or"}}}
                                    for f in nested_fields
                                ]
                            }
                        }
                    }
                })

        return clauses

    async def _apply_rerank(self, query: str, results: List[SearchResult], payload_schema: str, top_k: int) -> List[SearchResult]:
        if not results:
            return []
        
        content_extractor = self.schema_manager.get_extractor(payload_schema)

        return await self.reranker.rerank_async(
            query=query,
            documents=results,
            content_extractor=content_extractor,
            top_k=top_k
        )

    async def bm25_search_by_opensearch(self, req: SearchRequest, limit: int = None) -> SearchResponse:
        async with measure_time() as elapsed:
            search_limit = limit or req.top_k
            schema = self.schema_manager.get_schema(req.payload_schema)
            filters = self.opensearch.build_filters(req, schema)
            bm25_fields, nested_paths = schema.bm25_fields, schema.nested_paths

            must_clauses = self._build_bm25_clauses(
                query=req.query,
                fields=bm25_fields,
                nested_paths=nested_paths,
                nested_operator="should"  # or "must" 視需求
            )

            query_body = {
                "size": search_limit,
                "_source": ["payload"],
                "query": {
                    "bool": {
                        "must": must_clauses,
                        "filter": filters
                    }
                }
            }

            res = await self.opensearch.execute_search(query_body)

        return SearchResponse(results=res, timing={"total_sec": elapsed()})

    async def vector_search_by_opensearch(self, req: SearchRequest, vector: list, limit: int = None) -> SearchResponse:
        async with measure_time() as elapsed:
            search_limit = limit or req.top_k
            schema = self.schema_manager.get_schema(req.payload_schema)
            filters = self.opensearch.build_filters(req, schema)
            query_body = {
                "size": search_limit,
                "_source": ["payload"],
                "query": {
                    "knn": {
                        "my_vector": {
                            "vector": vector,
                            "k": req.top_k,
                            "filter": {"bool": {"must": filters}}
                        }
                    }
                }
            }
            res = await self.opensearch.execute_search(query_body)

        return SearchResponse(results=res, timing={"total_sec": elapsed()})

    async def hybrid_search_by_opensearch(self, req: SearchRequest, vector: list, limit: int = None) -> SearchResponse:
        async with measure_time() as elapsed:
            search_limit = limit or req.top_k
            schema = self.schema_manager.get_schema(req.payload_schema)
            filters = self.opensearch.build_filters(req, schema)
            bm25_clauses = self._build_bm25_clauses(
                query=req.query,
                fields=schema.bm25_fields,
                nested_paths=schema.nested_paths,
                nested_operator="should"
            )
            query_body = {
                "size": search_limit,
                "_source": ["payload"],
                "query": {
                    "hybrid": {
                        "queries": [
                            {"bool": {"must": bm25_clauses, "filter": filters}},
                            {"knn": {"my_vector": {"vector": vector, "k": req.top_k, "filter": {"bool": {"must": filters}}}}}
                        ]
                    }
                }
            }
            res = await self.opensearch.execute_search(query_body, use_pipeline=True)

        return SearchResponse(results=res, timing={"total_sec": elapsed()})

    async def vector_search_by_qdrant(self, req: SearchRequest, vector: list, limit: int = None) -> SearchResponse:
        async with measure_time() as elapsed:
            search_limit = limit or req.top_k
            schema = self.schema_manager.get_schema(req.payload_schema)
            filters = self.qdrant.build_filters(req, schema)
            query_body = {
                "vector": vector,
                "filter": filters,
                "limit": search_limit
            }
            res = await self.qdrant.execute_search(query_body)
        return SearchResponse(results=res, timing={"total_sec": elapsed()})

    async def hybrid_search(self, req: SearchRequest, vector: list) -> SearchResponse:
        total_start = time.perf_counter()

        search_depth = max(req.top_k * 3, settings.RERANK_DEPTH)

        bm25_exec, vector_exec = await asyncio.gather(
            self.bm25_search_by_opensearch(req, limit=search_depth),
            self.vector_search_by_qdrant(req, vector, limit=search_depth)
        )

        fusion_start = time.perf_counter()
        fused_results = rrf_fusion(
            [bm25_exec.results, vector_exec.results],
            k=settings.RRF_K_FACTOR
        )
        rerank_depth = max(req.top_k, min(len(fused_results), settings.RERANK_DEPTH))
        fused_results = fused_results[:rerank_depth]
        fusion_time = time.perf_counter() - fusion_start

        rerank_start = time.perf_counter()
        rerank_results = await self._apply_rerank(req.query, fused_results, req.payload_schema, req.top_k)
        rerank_time = time.perf_counter() - rerank_start

        total_time = time.perf_counter() - total_start

        return SearchResponse(
            results=rerank_results,
            timing={
                "total_sec": total_time,
                "bm25_sec": bm25_exec.timing["total_sec"] if bm25_exec else 0,
                "vector_sec": vector_exec.timing["total_sec"] if vector_exec else 0,
                "fusion_sec": fusion_time,
                "rerank_sec": rerank_time
            }
        )