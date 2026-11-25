"""INF-3: per-task default models and per-request model resolution.

Resolution order for every inference call: ``request.model`` > Redis default
(``inf:model:default:{task}``, Module 02) > factory default from ``DEFAULT_<TASK>_MODEL``.
A per-request ``model`` never writes to Redis.
"""
from __future__ import annotations

import logging

from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.config import DEFAULT_TASKS, TASK_MODEL_TYPE
from app.errors import ServiceError
from app.services.model_registry import ModelEntry, ModelRegistry

logger = logging.getLogger(__name__)

KEY_PREFIX = "inf:model:default:"


def default_key(task: str) -> str:
    return f"{KEY_PREFIX}{task}"


class ModelDefaults:
    def __init__(self, redis: Redis, factory_defaults: dict[str, str | None]) -> None:
        self._redis = redis
        self._factory = {task: factory_defaults.get(task) for task in DEFAULT_TASKS}

    async def get_all(self) -> dict[str, str | None]:
        try:
            stored = await self._redis.mget([default_key(t) for t in DEFAULT_TASKS])
        except RedisError as exc:
            logger.warning("Redis unavailable, using factory defaults: %s", exc)
            stored = [None] * len(DEFAULT_TASKS)
        return {
            task: (_decode(value) or self._factory[task])
            for task, value in zip(DEFAULT_TASKS, stored)
        }

    async def get(self, task: str) -> str | None:
        try:
            value = _decode(await self._redis.get(default_key(task)))
        except RedisError as exc:
            logger.warning("Redis unavailable, using factory default for %s: %s", task, exc)
            value = None
        return value or self._factory[task]

    async def set(self, task: str, model: str) -> None:
        try:
            await self._redis.set(default_key(task), model)
        except RedisError as exc:
            raise ServiceError(503, "redis_unavailable", f"cannot persist default model: {exc}") from exc


def _decode(value) -> str | None:
    if value is None:
        return None
    return value.decode() if isinstance(value, bytes) else str(value)


async def model_not_found(registry: ModelRegistry, task: str, model: str, message: str = "") -> ServiceError:
    wanted = TASK_MODEL_TYPE[task]
    available: list[str] = [m.name for m in await registry.list_models() if m.type == wanted]
    return ServiceError(422, "model_not_found", message, model=model, available=available)


class ModelResolver:
    def __init__(self, registry: ModelRegistry, defaults: ModelDefaults) -> None:
        self.registry = registry
        self.defaults = defaults

    async def validate(self, task: str, model: str) -> ModelEntry:
        """The model must be listed by the registry with the type the task needs."""
        entry = await self.registry.get(model)
        if entry is None or entry.type != TASK_MODEL_TYPE[task]:
            raise await model_not_found(self.registry, task, model)
        return entry

    async def resolve(self, task: str, requested: str | None) -> ModelEntry:
        name = requested or await self.defaults.get(task)
        if not name:
            raise ServiceError(
                400, "model_required",
                f"no default model configured for task '{task}'; pass 'model' or set one via POST /models/default",
            )
        return await self.validate(task, name)
