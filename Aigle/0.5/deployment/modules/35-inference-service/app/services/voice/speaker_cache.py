"""Redis conditioning cache with atomic LRU admission, TTL and user isolation."""
from __future__ import annotations

import hashlib
import time

from app.errors import ServiceError

_PUT = """
local members = redis.call('ZRANGE', KEYS[1], 0, -1)
for _, id in ipairs(members) do
  if redis.call('EXISTS', ARGV[1] .. id) == 0 then redis.call('ZREM', KEYS[1], id) end
end
redis.call('SET', KEYS[2], ARGV[3], 'EX', ARGV[4])
redis.call('ZADD', KEYS[1], ARGV[5], ARGV[2])
local extra = redis.call('ZCARD', KEYS[1]) - tonumber(ARGV[6])
if extra > 0 then
  local victims = redis.call('ZRANGE', KEYS[1], 0, extra - 1)
  for _, id in ipairs(victims) do
    redis.call('DEL', ARGV[1] .. id)
    redis.call('ZREM', KEYS[1], id)
  end
end
redis.call('EXPIRE', KEYS[1], ARGV[4])
return 1
"""
_GET = """
local value = redis.call('GET', KEYS[2])
if value then redis.call('ZADD', KEYS[1], ARGV[2], ARGV[1])
else redis.call('ZREM', KEYS[1], ARGV[1]) end
return value
"""
_DELETE = """
redis.call('DEL', KEYS[2])
redis.call('ZREM', KEYS[1], ARGV[1])
return 1
"""


class SpeakerCache:
    """Each verified user has a separate cache namespace and capacity."""

    def __init__(self, redis, user_id: str, ttl: int = 86400, maximum: int = 100) -> None:
        if ttl < 1 or maximum < 1:
            raise ValueError("speaker cache TTL and maximum must be positive")
        tenant = hashlib.sha256(user_id.encode()).hexdigest()
        # Redis hash tags also keep Lua keys in one slot if cluster is enabled later.
        self.prefix = f"inf:speaker:{{{tenant}}}:"
        self.index = self.prefix + "_lru"
        self.redis, self.ttl, self.maximum = redis, ttl, maximum

    def key(self, speaker_id: str) -> str:
        """Validate IDs so management routes cannot address the index key.

        192, not 128: Cloning.clone() prefixes a backend name + "." when 2+
        clone backends are configured (e.g. "chatterbox-multilingual-v3."),
        on top of a speaker_name/hash that's itself allowed up to 128 chars.
        """
        if (not speaker_id or len(speaker_id) > 192 or speaker_id.startswith("_")
                or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-." for c in speaker_id)):
            raise ServiceError(400, "invalid_speaker_id")
        return self.prefix + speaker_id

    async def get(self, speaker_id: str) -> bytes | None:
        """Touch recency on hit without extending the embedding's TTL."""
        return await self.redis.eval(_GET, 2, self.index, self.key(speaker_id),
                                     speaker_id, time.time_ns())

    async def put(self, speaker_id: str, embedding: bytes) -> None:
        """Atomically insert, remove expired members and evict the least recently used."""
        await self.redis.eval(_PUT, 2, self.index, self.key(speaker_id), self.prefix,
                              speaker_id, embedding, self.ttl, time.time_ns(), self.maximum)

    async def list(self) -> list[dict]:
        """List only live embeddings and their remaining TTL in seconds."""
        ids = await self.redis.zrange(self.index, 0, -1)
        speakers = []
        for value in ids:
            speaker_id = value.decode() if isinstance(value, bytes) else value
            ttl = await self.redis.ttl(self.key(speaker_id))
            if ttl > 0:
                speakers.append({"id": speaker_id, "ttl_remaining_s": ttl})
            else:
                await self.redis.zrem(self.index, speaker_id)
        return speakers

    async def delete(self, speaker_id: str) -> None:
        """Evict one speaker; deletion is idempotent."""
        await self.redis.eval(_DELETE, 2, self.index, self.key(speaker_id), speaker_id)

    async def _namespace_keys(self) -> list:
        """Every key in this user's namespace, embeddings and LRU index alike.
        Scans the prefix rather than reading the index, so entries the index
        lost track of are found too."""
        return [key async for key in self.redis.scan_iter(match=self.prefix + "*")]

    def _embeddings_among(self, keys: list) -> int:
        index = self.index.encode()
        return sum(1 for key in keys if (key if isinstance(key, bytes) else key.encode()) != index)

    async def purge(self) -> int:
        """Delete every key in this user's namespace -- all embeddings and the
        LRU index -- for GDPR erasure (GD-1). Returns how many speaker
        embeddings were deleted."""
        keys = await self._namespace_keys()
        if not keys:
            return 0
        await self.redis.delete(*keys)
        return self._embeddings_among(keys)

    async def count(self) -> int:
        """How many speaker embeddings this user has cached -- for the GDPR
        export manifest (GD-2), which notes their existence without exporting
        them. Read-only: unlike list(), never prunes the index."""
        return self._embeddings_among(await self._namespace_keys())
