from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue
from opensearchpy import AsyncOpenSearch
from opensearchpy.helpers import async_bulk
import httpx
import uuid
import asyncio
import logging

from .config import settings

logger = logging.getLogger(__name__)


class SearchSync:
    def __init__(self):
        self.qdrant = AsyncQdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
        self.opensearch = AsyncOpenSearch(
            hosts=[{"host": settings.opensearch_host, "port": settings.opensearch_port}],
            http_auth=(settings.opensearch_user, settings.opensearch_password),
            use_ssl=True,
            verify_certs=settings.opensearch_verify_certs,
            ssl_show_warn=False,
        )
        self._initialized = False

    async def initialize(self):
        if self._initialized:
            return
        try:
            await self.qdrant.get_collection(settings.qdrant_collection)
            logger.info(f"[SearchSync] Qdrant collection '{settings.qdrant_collection}' confirmed.")
        except Exception as e:
            logger.warning(f"[SearchSync] Qdrant collection '{settings.qdrant_collection}' not reachable: {e}")

        try:
            exists = await self.opensearch.indices.exists(index=settings.opensearch_index)
            if exists:
                logger.info(f"[SearchSync] OpenSearch index '{settings.opensearch_index}' confirmed.")
            else:
                logger.warning(f"[SearchSync] OpenSearch index '{settings.opensearch_index}' does not exist.")
        except Exception as e:
            logger.warning(f"[SearchSync] OpenSearch index check failed: {e}")

        self._initialized = True

    async def close(self):
        await self.qdrant.close()
        await self.opensearch.close()
        self._initialized = False
        logger.info("[SearchSync] Connections closed.")

    # ---------- shared filter builders ----------

    def _qdrant_filter(self, asset_path: str, version_id: str, branch_id: str = None) -> Filter:
        conditions = [
            FieldCondition(key="asset_path", match=MatchValue(value=asset_path)),
            FieldCondition(key="version_id", match=MatchValue(value=version_id)),
        ]
        if branch_id:
            conditions.append(FieldCondition(key="branch_id", match=MatchValue(value=branch_id)))
        return Filter(must=conditions)

    def _opensearch_query(self, asset_path: str, version_id: str, branch_id: str = None) -> dict:
        must = [
            {"term": {"payload.asset_path": asset_path}},
            {"term": {"payload.version_id": version_id}},
        ]
        if branch_id:
            must.append({"term": {"payload.branch_id": branch_id}})
        return {"bool": {"must": must}}

    # ---------- public API ----------

    async def check_indexed(self, asset_path: str, version_id: str, branch: str) -> bool:
        """Return True if this asset version has been indexed in Qdrant (i.e. analysis completed)."""
        try:
            result = await self.qdrant.count(
                collection_name=settings.qdrant_collection,
                count_filter=self._qdrant_filter(asset_path, version_id, branch),
                exact=False,
            )
            return result.count > 0
        except Exception as e:
            logger.warning(f"[Qdrant] check_indexed failed for {asset_path}/{version_id}: {e}")
            return True  # assume indexed on error to avoid spurious re-processing

    async def archive_metadata(self, asset_path: str, version_id: str, branch: str):
        await asyncio.gather(
            self._qdrant_set_status(asset_path, version_id, "archived", branch_id=branch),
            self._opensearch_set_status(asset_path, version_id, "archived", branch_id=branch),
            self._graph_set_status(asset_path, version_id, "archived", branch_id=branch),
        )

    async def reactivate_metadata(self, asset_path: str, version_id: str, branch: str):
        await asyncio.gather(
            self._qdrant_set_status(asset_path, version_id, "active", branch_id=branch),
            self._opensearch_set_status(asset_path, version_id, "active", branch_id=branch),
            self._graph_set_status(asset_path, version_id, "active", branch_id=branch),
        )

    async def delete_metadata(self, asset_path: str, version_id: str, branch: str):
        await asyncio.gather(
            self._qdrant_delete(asset_path, version_id, branch_id=branch),
            self._opensearch_delete(asset_path, version_id, branch_id=branch),
            self._graph_delete(asset_path, version_id, branch_id=branch),
        )

    async def clone_point(
        self,
        source_path: str,
        source_version: str,
        target_path: str,
        target_version: str,
        branch: str,
    ):
        await asyncio.gather(
            self._qdrant_clone(source_path, source_version, target_path, target_version, branch_id=branch),
            self._opensearch_clone(source_path, source_version, target_path, target_version, branch_id=branch),
        )

    # ---------- Graph Service internals ----------

    async def _graph_set_status(self, asset_path: str, version_id: str, status: str, branch_id: str = None):
        url = settings.graph_service_url
        if not url:
            return
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{url}/source/set_status",
                    json={
                        "asset_path": asset_path,
                        "version_id": version_id,
                        "branch_id":  branch_id or "",
                        "status":     status,
                    },
                )
                resp.raise_for_status()
            logger.info(f"[Graph] status={status} for {asset_path}/{version_id}")
        except Exception as e:
            logger.error(f"[Graph] Failed to set status for {asset_path}/{version_id}: {e}")

    async def _graph_delete(self, asset_path: str, version_id: str, branch_id: str = None):
        url = settings.graph_service_url
        if not url:
            return
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{url}/source/delete",
                    json={
                        "asset_path": asset_path,
                        "version_id": version_id,
                        "branch_id":  branch_id or "",
                    },
                )
                resp.raise_for_status()
            logger.info(f"[Graph] Deleted Neo4j nodes for {asset_path}/{version_id}")
        except Exception as e:
            logger.error(f"[Graph] Failed to delete Neo4j nodes for {asset_path}/{version_id}: {e}")

    # ---------- Qdrant internals ----------

    async def _qdrant_set_status(self, asset_path: str, version_id: str, status: str, branch_id: str = None):
        try:
            await self.qdrant.set_payload(
                collection_name=settings.qdrant_collection,
                payload={"status": status},
                points=self._qdrant_filter(asset_path, version_id, branch_id),
            )
            logger.info(f"[Qdrant] status={status} for {asset_path}/{version_id}")
        except Exception as e:
            logger.error(f"[Qdrant] Failed to set status for {asset_path}/{version_id}: {e}")

    async def _qdrant_delete(self, asset_path: str, version_id: str, branch_id: str = None):
        try:
            await self.qdrant.delete(
                collection_name=settings.qdrant_collection,
                points_selector=self._qdrant_filter(asset_path, version_id, branch_id),
            )
            logger.info(f"[Qdrant] Deleted points for {asset_path}/{version_id}")
        except Exception as e:
            logger.error(f"[Qdrant] Failed to delete points for {asset_path}/{version_id}: {e}")

    async def _qdrant_clone(
        self,
        source_path: str,
        source_version: str,
        target_path: str,
        target_version: str,
        branch_id: str = None,
    ):
        offset = None
        cloned_count = 0
        try:
            while True:
                points, next_offset = await self.qdrant.scroll(
                    collection_name=settings.qdrant_collection,
                    scroll_filter=self._qdrant_filter(source_path, source_version),
                    limit=100,
                    offset=offset,
                    with_vectors=True,
                )
                if not points:
                    break

                new_points = [
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=p.vector,
                        payload={
                            **p.payload,
                            "asset_path": target_path,
                            "version_id": target_version,
                            "status": "active",
                            # overwrite branch_id to target user; never inherit from source
                            "branch_id": branch_id,
                        },
                    )
                    for p in points
                ]
                await self.qdrant.upsert(
                    collection_name=settings.qdrant_collection,
                    points=new_points,
                )
                cloned_count += len(new_points)

                offset = next_offset
                if offset is None:
                    break

            logger.info(f"[Qdrant] Cloned {cloned_count} points: {source_path} → {target_path}")
        except Exception as e:
            logger.error(f"[Qdrant] Clone failed {source_path} → {target_path}: {e}")

    # ---------- OpenSearch internals ----------

    async def _opensearch_set_status(self, asset_path: str, version_id: str, status: str, branch_id: str = None):
        try:
            await self.opensearch.update_by_query(
                index=settings.opensearch_index,
                body={
                    "query": self._opensearch_query(asset_path, version_id, branch_id),
                    "script": {
                        "source": "ctx._source.payload.status = params.status",
                        "params": {"status": status},
                    },
                },
            )
            logger.info(f"[OpenSearch] status={status} for {asset_path}/{version_id}")
        except Exception as e:
            logger.error(f"[OpenSearch] Failed to set status for {asset_path}/{version_id}: {e}")

    async def _opensearch_delete(self, asset_path: str, version_id: str, branch_id: str = None):
        try:
            await self.opensearch.delete_by_query(
                index=settings.opensearch_index,
                body={"query": self._opensearch_query(asset_path, version_id, branch_id)},
            )
            logger.info(f"[OpenSearch] Deleted docs for {asset_path}/{version_id}")
        except Exception as e:
            logger.error(f"[OpenSearch] Failed to delete docs for {asset_path}/{version_id}: {e}")

    async def _opensearch_clone(
        self,
        source_path: str,
        source_version: str,
        target_path: str,
        target_version: str,
        branch_id: str = None,
        batch_size: int = 500,
    ):
        cloned_count = 0
        search_after = None
        try:
            while True:
                body = {
                    "size": batch_size,
                    "sort": [{"_id": "asc"}],
                    "query": self._opensearch_query(source_path, source_version),
                }
                if search_after:
                    body["search_after"] = search_after

                response = await self.opensearch.search(
                    index=settings.opensearch_index,
                    body=body,
                )
                hits = response["hits"]["hits"]
                if not hits:
                    break

                async def make_actions(hits_batch):
                    for hit in hits_batch:
                        src = hit["_source"]
                        yield {
                            "_op_type": "index",
                            "_index": settings.opensearch_index,
                            "_id": str(uuid.uuid4()),
                            "_source": {
                                "my_vector": src.get("my_vector"),
                                "payload": {
                                    **src.get("payload", {}),
                                    "asset_path": target_path,
                                    "version_id": target_version,
                                    "status": "active",
                                    # overwrite branch_id to target user; never inherit from source
                                    "branch_id": branch_id,
                                },
                            },
                        }

                success, failed = await async_bulk(
                    self.opensearch,
                    make_actions(hits),
                    raise_on_error=False,
                )
                cloned_count += success
                if failed:
                    logger.warning(f"[OpenSearch] {len(failed)} docs failed during clone")

                if len(hits) < batch_size:
                    break
                search_after = hits[-1].get("sort")
                if not search_after:
                    break

            logger.info(f"[OpenSearch] Cloned {cloned_count} docs: {source_path} → {target_path}")
        except Exception as e:
            logger.error(f"[OpenSearch] Clone failed {source_path} → {target_path}: {e}")
