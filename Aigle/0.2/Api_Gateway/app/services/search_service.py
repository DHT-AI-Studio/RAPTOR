"""
Search service for unified cross-collection search functionality.
Handles parallel searches across multiple Qdrant collections and result aggregation.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from typing import Any, Dict, List, Literal, Optional

import httpx

_logger = logging.getLogger(__name__)


class SearchService:
    """Service for handling unified searches across multiple collections."""

    # Collection configuration mapping
    COLLECTION_CONFIG = {
        "video": {"port": 8821, "endpoint": "/video_search"},
        "audio": {"port": 8822, "endpoint": "/audio_search"},
        "document": {"port": 8823, "endpoint": "/document_search"},
        "image": {"port": 8824, "endpoint": "/image_search"},
    }

    def __init__(
        self,
        qdrant_host: str,
        http_client: httpx.AsyncClient,
    ):
        """
        Initialize the search service.

        Args:
            qdrant_host: Qdrant host address
            http_client: Async HTTP client for making requests
        """
        self.qdrant_host = qdrant_host
        self.http_client = http_client

    async def _search_collection(
        self,
        collection_type: str,
        query_text: str,
        embedding_type: str,
        filters: Dict[str, Any],
        limit: int,
        timeout: float = 30.0,
    ) -> Dict[str, Any]:
        """
        Search a single collection.

        Args:
            collection_type: Type of collection (video, audio, document, image)
            query_text: Search query text
            embedding_type: Embedding type (text or summary)
            filters: Collection-specific filters
            limit: Number of results to return
            timeout: Request timeout in seconds

        Returns:
            Search results from the collection
        """
        if collection_type not in self.COLLECTION_CONFIG:
            raise ValueError(f"Unknown collection type: {collection_type}")

        config = self.COLLECTION_CONFIG[collection_type]
        url = f"http://{self.qdrant_host}:{config['port']}{config['endpoint']}"

        # Build request payload - merge filters properly
        payload = {
            "query_text": query_text,
            "embedding_type": embedding_type,
            "limit": limit,
        }
        
        # Add filters without overwriting existing keys
        if filters:
            payload.update(filters)

        _logger.debug(
            f"Searching {collection_type} collection",
            extra={
                "url": url,
                "payload": payload,
            }
        )

        try:
            response = await self.http_client.post(
                url,
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            response_data = response.json()
            
            _logger.debug(
                f"Search result for {collection_type}",
                extra={
                    "response_keys": list(response_data.keys()) if isinstance(response_data, dict) else "not_dict",
                }
            )
            
            return {
                "collection": collection_type,
                "success": True,
                "data": response_data,
                "error": None,
            }
        except httpx.HTTPStatusError as e:
            _logger.error(
                f"Search failed for {collection_type} with status {e.response.status_code}: {e.response.text}"
            )
            return {
                "collection": collection_type,
                "success": False,
                "data": None,
                "error": f"HTTP {e.response.status_code}: {e.response.text[:200]}",
            }
        except Exception as e:
            _logger.error(f"Search error for {collection_type}: {str(e)}")
            return {
                "collection": collection_type,
                "success": False,
                "data": None,
                "error": str(e),
            }

    async def unified_search(
        self,
        query_text: str,
        embedding_type: str = "text",
        filters: Optional[Dict[str, Dict[str, Any]]] = None,
        limit_per_collection: int = 5,
        global_limit: Optional[int] = None,
        score_threshold: Optional[float] = None,
        search_timeout: float = 30.0,
    ) -> Dict[str, Any]:
        """
        Perform unified search across multiple collections.

        Args:
            query_text: Search query text
            embedding_type: Type of embedding ('text' or 'summary')
            filters: Per-collection filter dictionaries
            limit_per_collection: Number of results per collection
            global_limit: Maximum total results after aggregation
            score_threshold: Minimum score threshold for results

        Returns:
            Unified search results with metadata
        """
        start_time = time.time()

        # Search all collections by default
        collections = list(self.COLLECTION_CONFIG.keys())

        # Default filters
        if filters is None:
            filters = {}

        # Create search tasks for parallel execution
        search_tasks = []
        for collection in collections:
            collection_filters = filters.get(collection, {})
            task = self._search_collection(
                collection_type=collection,
                query_text=query_text,
                embedding_type=embedding_type,
                filters=collection_filters,
                limit=limit_per_collection,
                timeout=search_timeout,
            )
            search_tasks.append(task)

        # Execute searches in parallel
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        # Process results
        successful_results = []
        failed_collections = []
        collection_stats = {}

        for result in search_results:
            if isinstance(result, Exception):
                _logger.error(f"Search task failed with exception: {result}")
                failed_collections.append({"collection": "unknown", "error": str(result)})
                continue

            collection = result["collection"]

            if result["success"]:
                data = result["data"]
                # Extract results - handle different response formats
                if isinstance(data, dict):
                    items = data.get("results", data.get("data", []))
                elif isinstance(data, list):
                    items = data
                else:
                    items = []

                _logger.debug(
                    f"Successfully retrieved {len(items)} items from {collection}",
                    extra={"collection": collection, "item_count": len(items)}
                )

                successful_results.append({
                    "collection": collection,
                    "items": items,
                })

                # Collect stats
                scores = [item.get("score", 0) for item in items if isinstance(item, dict)]
                collection_stats[collection] = {
                    "count": len(items),
                    "avg_score": sum(scores) / len(scores) if scores else 0,
                }
            else:
                _logger.warning(
                    f"Search failed for {collection}: {result['error']}",
                    extra={"collection": collection, "error": result["error"]}
                )
                failed_collections.append({
                    "collection": collection,
                    "error": result["error"],
                })

        # Aggregate results (merge strategy)
        aggregated_results = self._aggregate_merge(
            successful_results,
            score_threshold,
            global_limit,
        )

        execution_time = (time.time() - start_time) * 1000  # Convert to ms

        # Build response
        response = {
            "query": query_text,
            "embedding_type": embedding_type,
            "total_results": len(aggregated_results),
            "collections_searched": collections,
            "results": aggregated_results,
            "collection_stats": collection_stats,
            "execution_time_ms": round(execution_time, 2),
        }

        # Add warnings if any collections failed
        if failed_collections:
            response["warnings"] = {
                "failed_collections": failed_collections,
                "message": f"{len(failed_collections)} collection(s) failed to search",
            }

        return response

    def _aggregate_merge(
        self,
        successful_results: List[Dict[str, Any]],
        score_threshold: Optional[float],
        global_limit: Optional[int],
    ) -> List[Dict[str, Any]]:
        """Merge all results and sort by score."""
        all_results = []

        for result in successful_results:
            collection = result["collection"]
            items = result["items"]

            for item in items:
                if not isinstance(item, dict):
                    continue

                # Add collection type to each result
                enriched_item = {
                    "type": collection,
                    "collection": collection,
                    **item,
                }

                # Apply score threshold
                if score_threshold is not None:
                    score = item.get("score", 0)
                    if score < score_threshold:
                        continue

                all_results.append(enriched_item)

        # Sort by score descending
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)

        # Apply global limit
        if global_limit is not None:
            all_results = all_results[:global_limit]

        return all_results

    @staticmethod
    def generate_cache_key(
        query_text: str,
        embedding_type: str,
        collections: List[str],
        filters: Dict[str, Dict[str, Any]],
    ) -> str:
        """
        Generate a cache key for the search query.

        Args:
            query_text: Search query text
            embedding_type: Embedding type
            collections: List of collections
            filters: Filter dictionary

        Returns:
            Cache key string
        """
        # Create a stable representation of the query parameters
        key_parts = [
            query_text,
            embedding_type,
            "|".join(sorted(collections)),
            str(sorted(filters.items())),
        ]
        key_string = "::".join(key_parts)

        # Hash for consistent key length
        key_hash = hashlib.md5(key_string.encode()).hexdigest()
        return f"unified_search:{key_hash}"
