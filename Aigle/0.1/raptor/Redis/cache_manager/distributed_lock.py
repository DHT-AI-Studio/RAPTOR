import logging
import asyncio
import uuid
import random
import threading
import time
from contextlib import contextmanager, asynccontextmanager
from typing import Optional, Any, Union

from .base_cache import BaseCache


logger = logging.getLogger(__name__)


class BaseRedisLock:
    """
    Base class providing shared logic for Redis-based distributed locks.
    Supports both sync and async implementations.
    """

    RELEASE_SCRIPT = """
    if redis.call("get", KEYS[1]) == ARGV[1] then
        return redis.call("del", KEYS[1])
    else
        return 0
    end
    """

    EXTEND_SCRIPT = """
    if redis.call("get", KEYS[1]) == ARGV[1] then
        return redis.call("pexpire", KEYS[1], ARGV[2])
    else
        return 0
    end
    """

    def __init__(self, key: str, timeout_ms: int = 5000, max_ttl_ms: Optional[int] = None):
        self.key = key
        self.timeout_ms = timeout_ms
        self.max_ttl_ms = max_ttl_ms if max_ttl_ms is not None else 30000
        self.token = str(uuid.uuid4())
        self.acquired = False
        self._start_time: Optional[float] = None

        # Watchdog control
        self._stop_event: Optional[Union[threading.Event, asyncio.Event]] = None
        self._extend_thread: Optional[threading.Thread] = None
        self._extend_task: Optional[asyncio.Task] = None

    def _gen_jitter(self) -> float:
        """Randomized sleep interval to avoid thundering herd (in seconds)."""
        return random.uniform(0.05, 0.1)

    @property
    def is_locked(self) -> bool:
        return self.acquired

    def get_owner_token(self) -> str:
        return self.token


# =========================================================
# Sync Version (RedisLock)
# =========================================================
class RedisLock(BaseRedisLock):
    """Synchronous Redis distributed lock implementation."""

    def __init__(self, base_cache: BaseCache, key: str, timeout_ms: int = 5000, max_ttl_ms: Optional[int] = None):
        super().__init__(key, timeout_ms, max_ttl_ms)
        self._sync_client = base_cache.sync_client

    def acquire(self, blocking: bool = True, blocking_timeout: Optional[float] = None) -> bool:
        self._start_time = time.monotonic()

        while True:
            ok = self._sync_client.set(self.key, self.token, nx=True, px=self.timeout_ms)
            if ok:
                self.acquired = True
                logger.debug(f"[SyncLock] Acquired lock for {self.key}")
                return True

            if not blocking:
                return False

            if blocking_timeout and (time.monotonic() - self._start_time) >= blocking_timeout:
                logger.warning(f"[SyncLock] Acquire timeout for {self.key}")
                return False

            time.sleep(self._gen_jitter())

    def release(self) -> bool:
        if not self.acquired:
            return False

        if self._extend_thread and self._stop_event:
            self._stop_event.set()
            self._extend_thread.join(timeout=1.0)
            self._extend_thread = None
            self._stop_event = None

        released = False
        try:
            res = self._sync_client.eval(self.RELEASE_SCRIPT, 1, self.key, self.token)
            if res == 1:
                logger.debug(f"[SyncLock] Released lock for {self.key}")
                released = True
            else:
                logger.warning(f"[SyncLock] Lock {self.key} already expired or owned by others.")
        except Exception as e:
            logger.error(f"[SyncLock] Error releasing lock {self.key}: {e}")
        finally:
            self.acquired = False

        return released

    def _auto_extend(self):
        if self._extend_thread:
            return  # Avoid multiple starts

        self._stop_event = threading.Event()
        start_time = self._start_time
        interval = self.timeout_ms / 3000  # Extend every ~ TTL/3

        def _extend_loop():
            try:
                while not self._stop_event.is_set():
                    if self.max_ttl_ms and (time.monotonic() - start_time) * 1000 >= self.max_ttl_ms:
                        logger.warning(f"[SyncLock] Max TTL reached for {self.key}")
                        self.acquired = False
                        break

                    try:
                        res = self._sync_client.eval(self.EXTEND_SCRIPT, 1, self.key, self.token, self.timeout_ms)
                        if res != 1:
                            logger.warning(f"[SyncLock] Failed to extend lock {self.key}, lock lost.")
                            self.acquired = False
                            break
                    except Exception as e:
                        logger.error(f"[SyncLock] Error extending lock {self.key}: {e}")
                        self.acquired = False
                        break

                    if self._stop_event.wait(interval):
                        break

            finally:
                logger.debug(f"[SyncLock] Auto-extend thread for {self.key} exited.")

        self._extend_thread = threading.Thread(target=_extend_loop, daemon=True)
        self._extend_thread.start()

    @contextmanager
    def context(self, blocking: bool = True, blocking_timeout: Optional[float] = None, auto_extend: bool = False):
        if self.acquire(blocking, blocking_timeout):
            try:
                if auto_extend:
                    self._auto_extend()
                yield self
            finally:
                self.release()
        else:
            raise TimeoutError(f"Failed to acquire lock for {self.key}")


# =========================================================
# Async Version (AsyncRedisLock)
# =========================================================
class AsyncRedisLock(BaseRedisLock):
    """Asynchronous Redis distributed lock implementation."""

    def __init__(self, base_cache: BaseCache, key: str, timeout_ms: int = 5000, max_ttl_ms: Optional[int] = None):
        super().__init__(key, timeout_ms, max_ttl_ms)
        self._async_client = base_cache.async_client

    async def acquire(self, blocking: bool = True, blocking_timeout: Optional[float] = None) -> bool:
        loop = asyncio.get_event_loop()
        self._start_time = loop.time()

        while True:
            ok = await self._async_client.set(self.key, self.token, nx=True, px=self.timeout_ms)
            if ok:
                self.acquired = True
                logger.debug(f"[AsyncLock] Acquired lock for {self.key}")
                return True

            if not blocking:
                return False

            if blocking_timeout and (loop.time() - self._start_time) >= blocking_timeout:
                logger.warning(f"[AsyncLock] Acquire timeout for {self.key}")
                return False

            await asyncio.sleep(self._gen_jitter())

    async def release(self) -> bool:
        if not self.acquired:
            return False

        # Stop auto extend
        if self._extend_task and not self._extend_task.done():
            self._stop_event.set()
            # ensure the task is cancelled
            self._extend_task.cancel()
            try:
                await self._extend_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"[AsyncLock] Error waiting for extend task: {e}")
            finally:
                self._extend_task = None
                self._stop_event = None

        released = False
        try:
            res = await self._async_client.eval(self.RELEASE_SCRIPT, 1, self.key, self.token)
            if res == 1:
                logger.debug(f"[AsyncLock] Released lock for {self.key}")
                released = True
            elif res == 0:
                logger.warning(f"[AsyncLock] Lock {self.key} already expired or owned by others.")
        except Exception as e:
            logger.error(f"[AsyncLock] Error releasing lock {self.key}: {e}")
        finally:
            # Ensure the lock is released
            self.acquired = False

        return released

    async def _auto_extend(self):
        if self._extend_task:
            return  # Avoid multiple starts

        self._stop_event = asyncio.Event()
        start_time = self._start_time
        interval = self.timeout_ms / 3000  # Extend ~ TTL/3

        async def _extend_loop():
            while not self._stop_event.is_set():
                if self.max_ttl_ms and (asyncio.get_event_loop().time() - start_time) * 1000 >= self.max_ttl_ms:
                    logger.warning(f"[AsyncLock] Max TTL reached for {self.key}")
                    self.acquired = False
                    return

                await asyncio.sleep(interval)
                if self._stop_event.is_set():
                    return

                try:
                    res = await self._async_client.eval(self.EXTEND_SCRIPT, 1, self.key, self.token, self.timeout_ms)
                    if res != 1:
                        logger.warning(f"[AsyncLock] Failed to extend lock {self.key}, lock lost.")
                        self.acquired = False
                        return
                except Exception as e:
                    logger.error(f"[AsyncLock] Error extending lock {self.key}: {e}")
                    self.acquired = False
                    return

        self._extend_task = asyncio.create_task(_extend_loop())

    @asynccontextmanager
    async def context(self, blocking: bool = True, blocking_timeout: Optional[float] = None, auto_extend: bool = False):
        if await self.acquire(blocking, blocking_timeout):
            try:
                if auto_extend:
                    await self._auto_extend()
                yield self
            finally:
                await self.release()
        else:
            raise asyncio.TimeoutError(f"Failed to acquire async lock for {self.key}")