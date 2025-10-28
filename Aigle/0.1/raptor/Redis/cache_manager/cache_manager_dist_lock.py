import functools
import inspect
import logging
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
import weakref
from typing import (
    Callable,
    Dict,
    Optional,
    Union,
    Any,
    TypeVar,
    Coroutine,
)
from .base_cache import BaseCache
from .semantic_redis_cache import SemanticRedisCache
from .utils import hash_query
from .distributed_lock import RedisLock, AsyncRedisLock
import asyncio
import threading


logger = logging.getLogger(__name__)
T = TypeVar('T')
R = TypeVar('R')


class CacheManager:
    def __init__(
        self,
        host: str = 'localhost',
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        max_connections: int = 100,
        max_workers: Optional[int] = 4,
        ttl: int = 3600,
        ttl_multiplier: Optional[float] = None,
        is_cluster: bool = False,
        cleanup_interval: int = 3600,
        semantic: bool = False,
        embedding_model: str = "BAAI/bge-m3",
        ollama_url: Optional[str] = None,
        similarity_threshold: float = 0.8
    ):
        self.caches: Dict[str, Dict] = {}
        self._cache_pools: Dict[tuple, BaseCache] = {}
        self.default_config = {
            'host': host,
            'port': port,
            'db': db,
            'password': password,
            'max_connections': max_connections,
            'max_workers': max_workers,
            'ttl': ttl,
            'is_cluster': is_cluster,
            'semantic': semantic,
            'embedding_model': embedding_model,
            'ollama_url': ollama_url,
            'similarity_threshold': similarity_threshold
        }
        self.ttl_multiplier = ttl_multiplier
        self._cleanup_interval = cleanup_interval
        self._scheduled_cleanup = False
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown = False
        self._schedule_cleanup()

    def __del__(self):
        if not self._shutdown:
            self.close()

    def close(self):
        """Clean up resources"""
        if self._shutdown:
            return
        self._shutdown = True
        # Cancel cleanup task
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
        # Shutdown thread pool executors
        for cache_entry in self.caches.values():
            if isinstance(cache_entry['concurrent_executor'], ThreadPoolExecutor):
                cache_entry['concurrent_executor'].shutdown(wait=True)
        # Close all BaseCache instances
        for cache in self._cache_pools.values():
            cache.close()
        
        self.caches.clear()
        self._cache_pools.clear()

    async def aclose(self):
        """Async close all BaseCache instances"""
        if self._shutdown:
            return
        self._shutdown = True
        # Cancel cleanup task
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
        # Close all BaseCache instances
        for cache in self._cache_pools.values():
            await cache.aclose()

    def _schedule_cleanup(self):
        """Schedule periodic cleanup of expired locks and counters"""
        if self._cleanup_interval <= 0 or self._scheduled_cleanup:
            return

        async def _cleanup():
            logger.debug("Starting cleanup task...")
            try:
                while not self._shutdown:
                    await asyncio.sleep(self._cleanup_interval)
                    if self._shutdown:
                        break
                    
                    expired_keys = []
                    caches_to_remove = []

                    for cache_name, cache_entry in list(self.caches.items()):
                        cache = cache_entry['cache']
                        keys_to_check = list(cache_entry['hit_counter'].keys())
                        exists_checks = await asyncio.gather(*[
                            cache.aexists(key) for key in keys_to_check
                        ], return_exceptions=True)

                        for key, exists in zip(keys_to_check, exists_checks):
                            if isinstance(exists, Exception) or not exists:
                                cache_entry['hit_counter'].pop(key, None)
                                expired_keys.append(key)
                        
                        if await self._is_cache_empty(cache_name, cache):
                            caches_to_remove.append(cache_name)
                            
                    for cache_name in caches_to_remove:
                        logger.debug(f"Removing cache {cache_name}")
                        self.caches.pop(cache_name, None)
                        
                    logger.debug(f"Cleaned up {len(expired_keys)} keys")
                    logger.debug(f"Expired keys: {expired_keys}")
                    
            except asyncio.CancelledError:
                logger.debug("Cleanup task cancelled")
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")

        self._scheduled_cleanup = True
        try:
            loop = asyncio.get_running_loop()
            self._cleanup_task = loop.create_task(_cleanup())
        except RuntimeError:
            def run_loop():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(_cleanup())
                except asyncio.CancelledError:
                    pass
                finally:
                    loop.close()

            thread = threading.Thread(target=run_loop, daemon=True)
            thread.start()

    def register_cache(self, name: str, func_type: str, **kwargs):
        if func_type not in ("sync", "async"):
            raise ValueError(f"Invalid func_type: {func_type}")

        cache_key = tuple(sorted(kwargs.items()))
        if cache_key not in self._cache_pools:
            if kwargs.get("semantic", False):
                self._cache_pools[cache_key] = SemanticRedisCache(
                    base_cache=BaseCache(**kwargs), 
                    model_name=kwargs.get("embedding_model", "BAAI/bge-m3"),
                    ollama_url=kwargs.get("ollama_url", None),
                    similarity_threshold=kwargs.get("similarity_threshold", 0.8)
                )
            else:
                self._cache_pools[cache_key] = BaseCache(**kwargs)

        if name not in self.caches:
            self.caches[name] = {
                "cache": self._cache_pools[cache_key],
                "hit_counter": defaultdict(int),
                "ttl": kwargs.get("ttl", self.default_config['ttl']),
                "in_progress_tasks": weakref.WeakValueDictionary(),
                "concurrent_executor": (
                    ThreadPoolExecutor(max_workers=kwargs.get("max_workers", self.default_config['max_workers']))
                    if func_type == "sync"
                    else asyncio.Semaphore(kwargs.get("max_workers", self.default_config['max_workers']))
                ),
                "func_type": func_type,
                "semantic": kwargs.get("semantic", False)
            }

    def _make_key(self, name: str, *args, **kwargs) -> str:
        safe_kwargs = {
            k: v for k, v in kwargs.items()
            if not callable(v) and not hasattr(v, '__dict__')
        }        
        key_data = {"name": name, "args": args, "kwargs": sorted(safe_kwargs.items())}
        return f"{name}:{hash_query(key_data)}"

    def _process_cache_hit(self, cache_name: str, key: str, value: Any, query: Optional[str] = None) -> Any:
        entry = self.caches.get(cache_name)
        if entry is None:
            raise KeyError(f"Cache {cache_name} not found")
        
        entry['hit_counter'][key] += 1

        if self.ttl_multiplier is not None:
            dynamic_ttl = int(entry['ttl'] * (entry['hit_counter'][key] * self.ttl_multiplier + 1))
            cache = entry['cache']
            try:
                if entry['func_type'] == 'sync':
                    cache.set(key, value, ttl=dynamic_ttl, query=query)
                else:
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(cache.aset(key, value, ttl=dynamic_ttl, query=query))
                    except RuntimeError:
                        pass
            except Exception as e:
                logger.warning(f"Failed to update TTL for {key}: {e}")

        return value

    def _make_decorator(self, cache_name: str, func_type: str, **default_options):
        if cache_name not in self.caches:
            self.register_cache(cache_name, func_type, **default_options)

        entry = self.caches[cache_name]
        cache = entry['cache']
        ttl = entry['ttl']
        semantic = entry['semantic']
        base_cache = cache.base_cache if isinstance(cache, SemanticRedisCache) else cache

        def decorator(func: Callable[..., Union[T, Coroutine[Any, Any, T]]]) -> Callable[..., Union[T, Coroutine[Any, Any, T]]]:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs) -> T:
                key = self._make_key(cache_name, *args, **kwargs)
                query = str(kwargs.get("query", args[0] if args else ""))
                lock_key = f"lock:{key}"

                with entry['concurrent_executor']:
                    try:
                        if semantic:
                            value = cache.get(key, query)
                        else:
                            value = cache.get(key)

                        if value is not None:
                            logger.debug(f"[CACHE HIT] cache_name: {cache_name} key: {key}")
                            return self._process_cache_hit(cache_name, key, value, query)
                    except Exception as e:
                        logger.error(f"Cache get error for {key}: {e}")

                logger.debug(f"[CACHE MISS] cache_name: {cache_name} key: {key}")
                
                if key in entry['in_progress_tasks']:
                    logger.debug(f"[CACHE IN PROGRESS] cache_name: {cache_name} key: {key}")
                    try:
                        return entry['in_progress_tasks'][key].result(timeout=60)
                    except Exception as e:
                        logger.warning(f"In-progress task error for {key}: {e}")
                
                lock = RedisLock(base_cache, key=lock_key)
                with lock.context(blocking=True, blocking_timeout=None, auto_extend=True):
                    try:
                        if semantic:
                            value = cache.get(key, query)
                        else:
                            value = cache.get(key)

                        if value is not None:
                            logger.debug(f"[CACHE HIT] cache_name: {cache_name} key: {key}")
                            return self._process_cache_hit(cache_name, key, value, query)
                    except Exception as e:
                        logger.error(f"Cache get error post-lock for {key}: {e}")

                    if key in entry['in_progress_tasks']:
                        try:
                            return entry['in_progress_tasks'][key].result()
                        except Exception as e:
                            logger.warning(f"In-progress task error post-lock for {key}: {e}")

                    future = Future()
                    entry['in_progress_tasks'][key] = future

                    try:
                        result = func(*args, **kwargs)
                        future.set_result(result)
                        if semantic:
                            cache.set(key, result, ttl=ttl, query=query)
                        else:
                            cache.set(key, result, ttl=ttl)
                        return result
                    except Exception as e:
                        future.set_exception(e)
                        raise
                    finally:
                        entry['in_progress_tasks'].pop(key, None)

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs) -> T:
                key = self._make_key(cache_name, *args, **kwargs)
                query = str(kwargs.get("query", args[0] if args else ""))
                lock_key = f"lock:{key}"

                async with entry['concurrent_executor']:
                    try:
                        if semantic:
                            value = await cache.aget(key, query)
                        else:
                            value = await cache.aget(key)
                        if value is not None:
                            logger.debug(f"[CACHE HIT] cache_name: {cache_name} key: {key}")
                            return self._process_cache_hit(cache_name, key, value, query)
                    except Exception as e:
                        logger.error(f"Cache get error for {key}: {e}")

                logger.debug(f"[CACHE MISS] cache_name: {cache_name} key: {key}")

                if key in entry['in_progress_tasks']:
                    logger.debug(f"[CACHE IN PROGRESS] cache_name: {cache_name} key: {key}")
                    try:
                        return await entry['in_progress_tasks'][key]
                    except Exception as e:
                        logger.warning(f"In-progress task error for {key}: {e}")

                lock = AsyncRedisLock(base_cache, key=lock_key)
                async with lock.context(blocking=True, blocking_timeout=None, auto_extend=True):
                    try:
                        if semantic:
                            value = await cache.aget(key, query)
                        else:
                            value = await cache.aget(key)
                        if value is not None:
                            logger.debug(f"[CACHE HIT] cache_name: {cache_name} key: {key}")
                            return self._process_cache_hit(cache_name, key, value, query)
                    except Exception as e:
                        logger.error(f"Cache get error post-lock for {key}: {e}")

                    if key in entry['in_progress_tasks']:
                        try:
                            return await entry['in_progress_tasks'][key]
                        except Exception as e:
                            logger.warning(f"In-progress task error post-lock for {key}: {e}")

                    loop = asyncio.get_event_loop()
                    future = loop.create_future()
                    entry['in_progress_tasks'][key] = future

                    try:
                        result = await asyncio.wait_for(func(*args, **kwargs), timeout=60)
                        if semantic:
                            await cache.aset(key, result, ttl=ttl, query=query)
                        else:
                            await cache.aset(key, result, ttl=ttl)
                        future.set_result(result)
                        return result
                    except Exception as e:
                        future.set_exception(e)
                        raise
                    finally:
                        entry['in_progress_tasks'].pop(key, None)

            return async_wrapper if func_type == 'async' else sync_wrapper

        return decorator

    def cache(self, name=None, **kwargs):
        def _wrapper(func):
            nonlocal name
            if name is None:
                name = f"{func.__module__}.{func.__qualname__}"

            func_type = 'async' if inspect.iscoroutinefunction(func) else 'sync'
            merged_kwargs = self.default_config.copy()
            merged_kwargs.update(kwargs)

            return self._make_decorator(name, func_type, **merged_kwargs)(func)

        if callable(name):
            func = name
            name = None
            return _wrapper(func)
        return _wrapper

    def clear_cache(self, cache_name_or_func: Union[str, Callable]) -> bool:
        """Clear only keys related to the specified cache name or function."""
        if callable(cache_name_or_func):
            func = cache_name_or_func
            cache_name = f"{func.__module__}.{func.__qualname__}"
        else:
            cache_name = cache_name_or_func

        if cache_name not in self.caches:
            logger.warning(f"Cache {cache_name} not found")
            return False

        entry = self.caches[cache_name]
        cache = entry['cache']
        try:
            pattern = f"{cache_name}:*"

            if entry['func_type'] == 'sync':
                cache.clear_by_pattern(pattern)
            else:
                async def _clear():
                    try:
                        await cache.aclear_by_pattern(pattern)
                    except Exception as e:
                        logger.error(f"Error clearing cache {cache_name}: {e}")

                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(_clear())
                except RuntimeError:
                    cache.clear_by_pattern(pattern)

            entry['hit_counter'].clear()
            entry['in_progress_tasks'].clear()
            return True
        
        except Exception as e:
            logger.error(f"Error clearing cache {cache_name}: {e}")
            return False

    async def async_has_redis_keys(self, cache: BaseCache, pattern: str) -> bool:
        try:
            async for key in cache.async_client.scan_iter(match=pattern):
                return True
            return False
        except Exception as e:
            logger.warning(f"Error checking Redis keys for {pattern}: {e}")
            return True

    def has_redis_keys(self, cache: BaseCache, pattern: str) -> bool:
        try:
            for key in cache.sync_client.scan_iter(match=pattern):
                return True
            return False
        except Exception as e:
            logger.warning(f"Error checking Redis keys for {pattern}: {e}")
            return True

    async def _is_cache_empty(self, cache_name: str, cache: Union[BaseCache, SemanticRedisCache]) -> bool:
        if isinstance(cache, SemanticRedisCache):
            return not await self.async_has_redis_keys(cache.base_cache, f"sem_cache:{cache_name}:*")
        else:
            return not await self.async_has_redis_keys(cache, f"{cache_name}:*")

