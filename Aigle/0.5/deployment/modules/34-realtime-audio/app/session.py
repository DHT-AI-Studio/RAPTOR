"""WebSocket tickets, Redis session records and session limits."""
from __future__ import annotations

import json
import secrets
import time
import uuid
from typing import Any

from redis.asyncio import Redis

TICKET_PREFIX = "rt:ws_ticket:"
SESSION_PREFIX = "rt:session:"
ACTIVE_KEY = "rt:sessions:active"
USER_ACTIVE_PREFIX = "rt:sessions:user:"

# Session state machine.  LISTENING: nothing to do but listen; PROCESSING: a final transcript
# or an answer is being prepared; RESPONDING: answer audio is playing.  PROCESSING -> LISTENING
# when a transcript starts no turn; RESPONDING -> PROCESSING when another turn is already being
# prepared as an answer finishes; RESPONDING -> LISTENING also on barge-in (RTA-X5).
# LISTENING -> RESPONDING when queued answer audio starts playing after its turn has finished
# (e.g. once a dropped client is back).
# ENDED: END, error or lifetime reached; EXPIRED: dropped and not resumed in time (RTA-X3).
# Whether a client is connected is the separate ``connection`` field.
INIT, LISTENING, PROCESSING, RESPONDING = "INIT", "LISTENING", "PROCESSING", "RESPONDING"
ENDED, EXPIRED = "ENDED", "EXPIRED"
TERMINAL = {ENDED, EXPIRED}
TRANSITIONS = {
    INIT: {LISTENING, ENDED, EXPIRED},
    LISTENING: {PROCESSING, RESPONDING, ENDED, EXPIRED},
    PROCESSING: {RESPONDING, LISTENING, ENDED, EXPIRED},
    RESPONDING: {LISTENING, PROCESSING, ENDED, EXPIRED},
    ENDED: set(),
    EXPIRED: set(),
}

# Atomic "reclaim stale, check capacity (global and per-user), add": KEYS[1]=global active
# set, KEYS[2]=this user's active set. ARGV = stale_before, max_sessions, now, session_id,
# max_sessions_per_user. Returns 1=ok, 0=global cap hit, -1=per-user cap hit.
_ACQUIRE_LUA = """
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', ARGV[1])
redis.call('ZREMRANGEBYSCORE', KEYS[2], '-inf', ARGV[1])
if redis.call('ZCARD', KEYS[1]) >= tonumber(ARGV[2]) then
  return 0
end
if redis.call('ZCARD', KEYS[2]) >= tonumber(ARGV[5]) then
  return -1
end
redis.call('ZADD', KEYS[1], ARGV[3], ARGV[4])
redis.call('ZADD', KEYS[2], ARGV[3], ARGV[4])
return 1
"""

#: acquire_slot()'s denial reasons, also used as the WS/gRPC error code.
TOO_MANY_SESSIONS = "too_many_sessions"
TOO_MANY_SESSIONS_FOR_USER = "too_many_sessions_for_user"


class InvalidTransition(Exception):
    pass


class TicketStore:
    def __init__(self, redis: Redis, ttl_s: int) -> None:
        self._redis = redis
        self.ttl_s = ttl_s

    async def issue(self, user_id: str, roles: list[str] | tuple[str, ...] = ()) -> str:
        token = secrets.token_urlsafe(32)
        payload = json.dumps({"user_id": user_id, "roles": sorted(roles), "issued_at": time.time()})
        await self._redis.set(TICKET_PREFIX + token, payload, ex=self.ttl_s, nx=True)
        return token

    async def exists(self, token: str | None) -> bool:
        return bool(token) and bool(await self._redis.exists(TICKET_PREFIX + token))

    async def consume(self, token: str | None) -> dict[str, Any] | None:
        """Atomically read and delete; a ticket can be consumed exactly once."""
        if not token:
            return None
        raw = await self._redis.getdel(TICKET_PREFIX + token)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except ValueError:
            return None

    async def peek(self, token: str | None) -> dict[str, Any] | None:
        """Read without consuming -- used to learn the ticket's user_id before acquiring a
        session slot, so a capacity denial doesn't burn the caller's one-time ticket."""
        if not token:
            return None
        raw = await self._redis.get(TICKET_PREFIX + token)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except ValueError:
            return None


class SessionStore:
    def __init__(self, redis: Redis, ttl_s: int, max_sessions: int, heartbeat_s: float,
                 max_sessions_per_user: int = 3) -> None:
        self._redis = redis
        self.ttl_s = ttl_s
        self.max_sessions = max_sessions
        self.max_sessions_per_user = max_sessions_per_user
        self.heartbeat_s = heartbeat_s
        self._acquire = redis.register_script(_ACQUIRE_LUA)

    @staticmethod
    def key(session_id: str) -> str:
        return SESSION_PREFIX + session_id

    @staticmethod
    def user_key(user_id: str) -> str:
        return USER_ACTIVE_PREFIX + user_id

    async def acquire_slot(self, user_id: str) -> tuple[str | None, str]:
        """Reserve one of RT_MAX_SESSIONS slots, within this user's own
        RT_MAX_SESSIONS_PER_USER. Returns (new session id, "ok") or
        (None, TOO_MANY_SESSIONS | TOO_MANY_SESSIONS_FOR_USER)."""
        session_id = uuid.uuid4().hex
        now = time.time()
        result = await self._acquire(
            keys=[ACTIVE_KEY, self.user_key(user_id)],
            args=[now - 3 * self.heartbeat_s, self.max_sessions, now, session_id, self.max_sessions_per_user],
        )
        if result == 1:
            return session_id, "ok"
        return None, TOO_MANY_SESSIONS if result == 0 else TOO_MANY_SESSIONS_FOR_USER

    async def release_slot(self, session_id: str, user_id: str) -> None:
        await self._redis.zrem(ACTIVE_KEY, session_id)
        await self._redis.zrem(self.user_key(user_id), session_id)

    async def heartbeat(self, session_id: str, user_id: str) -> None:
        await self._redis.zadd(ACTIVE_KEY, {session_id: time.time()}, xx=True)
        await self._redis.zadd(self.user_key(user_id), {session_id: time.time()}, xx=True)

    async def active_count(self) -> int:
        await self._redis.zremrangebyscore(ACTIVE_KEY, "-inf", time.time() - 3 * self.heartbeat_s)
        return await self._redis.zcard(ACTIVE_KEY)

    async def register(self, session_id: str, user_id: str) -> None:
        key = self.key(session_id)
        now = time.time()
        await self._redis.hset(key, mapping={
            "user_id": user_id, "state": INIT, "started_at": now, "updated_at": now,
            "codec": "", "sample_rate": 0, "language": "", "utterances": 0, "connection": "connected",
        })
        await self._redis.expire(key, self.ttl_s)

    async def update(self, session_id: str, **fields: Any) -> None:
        # HSET on an existing key keeps its TTL, so the absolute session lifetime is unchanged.
        fields["updated_at"] = time.time()
        await self._redis.hset(self.key(session_id), mapping={k: _enc(v) for k, v in fields.items()})

    async def delete(self, session_id: str) -> None:
        await self._redis.delete(self.key(session_id))

    async def get(self, session_id: str) -> dict[str, str]:
        raw = await self._redis.hgetall(self.key(session_id))
        return {_dec(k): _dec(v) for k, v in raw.items()}


def _enc(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return "" if value is None else value


def _dec(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


class SessionState:
    """In-memory state machine for one connection, mirrored to Redis on every transition."""

    def __init__(self, store: SessionStore, session_id: str) -> None:
        self._store = store
        self.session_id = session_id
        self.state = INIT

    async def to(self, new_state: str, **fields: Any) -> None:
        if new_state not in TRANSITIONS[self.state]:
            raise InvalidTransition(f"{self.state} -> {new_state}")
        self.state = new_state
        await self._store.update(self.session_id, state=new_state, **fields)
