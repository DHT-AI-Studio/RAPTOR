# agent_protocol/discovery.py

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

_host = os.getenv("REDIS_HOST", "raptor-redis-standalone")
_port = os.getenv("REDIS_PORT", "6379")
_db   = os.getenv("REDIS_DB", "0")
_pw   = os.getenv("REDIS_PASSWORD", "")
_auth = f":{_pw}@" if _pw else "@"
REDIS_URL = f"redis://{_auth}{_host}:{_port}/{_db}"
AGENT_TTL  = int(os.getenv("AGENT_TTL", "300"))   # Redis key TTL (seconds)
_HEARTBEAT = int(AGENT_TTL * 0.8)                  # re-register at 80% of TTL


class DiscoveryService:
    """Peer-agent discovery backed by Redis (a2a:agents:{agent_id})."""

    AGENT_TTL = AGENT_TTL

    def __init__(self):
        self._redis:          Optional[aioredis.Redis]  = None
        self._agent_card:     Optional[Dict]            = None
        self._heartbeat_task: Optional[asyncio.Task]    = None

    async def start(self):
        self._redis = await aioredis.from_url(REDIS_URL, decode_responses=True)

    async def stop(self):
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        if self._redis:
            await self._redis.aclose()

    # ── Registration ────────────────────────────────────────────────────────

    async def register_self(self, agent_card: Dict) -> None:
        self._agent_card = agent_card
        agent_id = agent_card.get("id", "unknown")
        key = f"a2a:agents:{agent_id}"
        await self._redis.setex(key, AGENT_TTL, json.dumps(agent_card))
        logger.info(f"Registered self as agent: {agent_id}")
        if self._heartbeat_task is None or self._heartbeat_task.done():
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def _heartbeat_loop(self) -> None:
        while True:
            await asyncio.sleep(_HEARTBEAT)
            if self._agent_card:
                try:
                    await self.register_self(self._agent_card)
                    logger.debug("Heartbeat: re-registered self in Redis")
                except Exception as exc:
                    logger.warning(f"Heartbeat: re-registration failed: {exc}")

    # ── External agent registration ──────────────────────────────────────────

    async def register_peer(self, agent_card: Dict) -> None:
        agent_id = agent_card.get("id")
        if not agent_id:
            raise ValueError("agent_card must have an 'id' field")
        key = f"a2a:agents:{agent_id}"
        await self._redis.setex(key, AGENT_TTL, json.dumps(agent_card))
        logger.info(f"Registered peer agent: {agent_id}")

    async def deregister_peer(self, agent_id: str) -> bool:
        key = f"a2a:agents:{agent_id}"
        deleted = await self._redis.delete(key)
        logger.info(f"Deregistered peer agent: {agent_id} (found={bool(deleted)})")
        return bool(deleted)

    async def heartbeat_peer(self, agent_id: str) -> bool:
        key = f"a2a:agents:{agent_id}"
        refreshed = await self._redis.expire(key, AGENT_TTL)
        logger.debug(f"Heartbeat peer agent: {agent_id} (found={bool(refreshed)})")
        return bool(refreshed)

    # ── Listing ─────────────────────────────────────────────────────────────

    async def list_agents(self) -> list:
        keys = await self._redis.keys("a2a:agents:*")
        agents = []
        for key in keys:
            raw = await self._redis.get(key)
            if raw:
                agents.append(json.loads(raw))
        return agents

    # ── Delegation ───────────────────────────────────────────────────────────

    async def delegate(self, task: str, payload: Dict, target_agent_id: Optional[str]) -> Dict:
        return {
            "status":          "accepted",
            "task":            task,
            "target_agent_id": target_agent_id,
            "timestamp":       datetime.now(timezone.utc).isoformat(),
        }

    # ── Query — runs the full RAG pipeline ──────────────────────────────────

    async def query(self, query_str: str, params: Dict) -> Dict:
        from pipeline import run_rag_pipeline
        top_k      = int(params.get("top_k", 5))
        request_id = params.get("request_id")
        result = await run_rag_pipeline(query_str, top_k=top_k, request_id=request_id)
        return result.model_dump()
