from redis import Redis, ConnectionPool as SyncConnectionPool
from redis.asyncio import Redis as AsyncRedis, ConnectionPool as AsyncConnectionPool, RedisCluster as AsyncRedisCluster
from redis.cluster import RedisCluster
import pickle
from typing import Optional, Any, Dict, List

class BaseCache:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        max_connections: int = 100,
        ttl: int = 3600,
        is_cluster: bool = False,
        **kwargs
    ):
        self.ttl = ttl
        self.is_cluster = is_cluster
        self._closed = False
        
        pool_kwargs = {
            "max_connections": max_connections,
        }

        if is_cluster:
            self.sync_client = RedisCluster.from_url(
                f"redis://{host}:{port}", 
                password=password, 
                **pool_kwargs
            )
            self.async_client = AsyncRedisCluster.from_url(
                f"redis://{host}:{port}", 
                password=password, 
                **pool_kwargs
            )
        else:
            self.sync_client = Redis(
                connection_pool=SyncConnectionPool.from_url(
                    f"redis://{host}:{port}/{db}", 
                    password=password, 
                    **pool_kwargs
                )
            )
            self.async_client = AsyncRedis(
                connection_pool=AsyncConnectionPool.from_url(
                    f"redis://{host}:{port}/{db}", 
                    password=password, 
                    **pool_kwargs
                )
            )

    def close(self):
        """Close sync connections and schedule async cleanup"""
        if self._closed:
            return
        self._closed = True

        try:
            if hasattr(self.sync_client, 'close'):
                self.sync_client.close()
        except Exception:
            pass

    async def aclose(self):
        if self._closed:
            return
        self._closed = True
        try:
            if hasattr(self.async_client, 'aclose'):
                await self.async_client.aclose()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.aclose()

    def get(self, key: str) -> Optional[Any]:
        value = self.sync_client.get(key)
        return pickle.loads(value) if value else None

    def set(self, key: str, value: Any, ttl: Optional[int] = None, **kwargs) -> None:
        if ttl is None:
            ttl = self.ttl
        self.sync_client.setex(key, ttl, pickle.dumps(value))

    def delete(self, key: str) -> None:
        self.sync_client.delete(key)

    async def aget(self, key: str) -> Optional[Any]:
        value = await self.async_client.get(key)
        return pickle.loads(value) if value else None

    async def aset(self, key: str, value: Any, ttl: Optional[int] = None, **kwargs) -> None:
        if ttl is None:
            ttl = self.ttl
        await self.async_client.setex(key, ttl, pickle.dumps(value))

    async def adelete(self, key: str) -> None:
        await self.async_client.delete(key)

    def mget(self, keys: List[str]) -> List[Optional[Any]]:
        values = self.sync_client.mget(keys)
        return [pickle.loads(v) if v else None for v in values]

    def mset(self, items: Dict[str, Any], ttl: Optional[int] = None) -> None:
        if ttl is None:
            ttl = self.ttl
        pipeline = self.sync_client.pipeline()
        for key, value in items.items():
            pipeline.setex(key, ttl, pickle.dumps(value))
        pipeline.execute()

    async def amget(self, keys: List[str]) -> List[Optional[Any]]:
        values = await self.async_client.mget(keys)
        return [pickle.loads(v) if v else None for v in values]

    async def amset(self, items: Dict[str, Any], ttl: Optional[int] = None) -> None:
        if ttl is None:
            ttl = self.ttl
        async with self.async_client.pipeline() as pipeline:
            for key, value in items.items():
                pipeline.setex(key, ttl, pickle.dumps(value))
            await pipeline.execute()

    def clear_by_pattern(self, pattern: str):
        """Delete keys matching the given pattern (sync)"""
        pipeline = self.sync_client.pipeline()
        count = 0
        for key in self.sync_client.scan_iter(match=pattern):
            pipeline.delete(key)
            count += 1
            if count % 100 == 0:
                pipeline.execute()
        if count % 100 != 0:
            pipeline.execute()

    async def aclear_by_pattern(self, pattern: str):
        """Delete keys matching the given pattern (async)"""
        count = 0
        pipeline = self.async_client.pipeline()
        async for key in self.async_client.scan_iter(match=pattern):
            pipeline.delete(key)
            count += 1
            if count % 100 == 0:
                await pipeline.execute()
        if count % 100 != 0:
            await pipeline.execute()

    def exists(self, key: str) -> bool:
        """Check if a key exists in Redis (sync)"""
        return self.sync_client.exists(key)

    async def aexists(self, key: str) -> bool:
        """Check if a key exists in Redis (async)"""
        return await self.async_client.exists(key)