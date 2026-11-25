"""Guardrail enable/disable switches — Redis-backed, immediate effect, no TTL.

`guardrail:enabled` is the same key app/services/checker.py's GB-4 checker
already read ("0" = off, anything else/missing = on, fail-open). This module
is the single place that reads and writes it so the admin API
(app/routers/system.py) and every check family agree on the same switch.

`guardrail:policy_enabled` is a second, narrower switch gating only
/policy/check/llm/* and /debug/policy/check/llm/* (app/routers/check_policy_llm.py,
app/routers/debug_policy_check_llm.py) via is_policy_check_enabled() below — it
has no effect on /guard/check/* or the GB-4 proxy checker, which keep reading
is_enabled() directly. Same fail-open semantics as ENABLED_KEY, so it needs no
.env-driven cold-start seed: a missing key already means "on".
"""
from __future__ import annotations

from typing import Literal

import yaml
from fastapi import HTTPException
from pydantic import ValidationError

from app.core.config import get_settings
from app.db import redis_conn
from app.engine import policy_store
from app.models.policy import GuardrailPolicy, parse_standard_policies
from app.models.system import SystemStatus

ENABLED_KEY = "guardrail:enabled"
POLICY_ENABLED_KEY = "guardrail:policy_enabled"


async def is_enabled() -> bool:
    redis = await redis_conn.get_redis()
    return await redis.get(ENABLED_KEY) != "0"


async def set_enabled(enabled: bool) -> None:
    redis = await redis_conn.get_redis()
    await redis.set(ENABLED_KEY, "1" if enabled else "0")


async def is_policy_enabled() -> bool:
    redis = await redis_conn.get_redis()
    return await redis.get(POLICY_ENABLED_KEY) != "0"


async def set_policy_enabled(enabled: bool) -> None:
    redis = await redis_conn.get_redis()
    await redis.set(POLICY_ENABLED_KEY, "1" if enabled else "0")


async def is_policy_check_enabled() -> bool:
    """True only when both the global switch and the policy-check switch are
    on — the combined gate used by /policy/check/llm/* and
    /debug/policy/check/llm/*."""
    return await is_enabled() and await is_policy_enabled()


async def init_default_state() -> None:
    """Seed the switch from GR_DEFAULT_ENABLED on a cold Redis (key not yet set)."""
    redis = await redis_conn.get_redis()
    if await redis.get(ENABLED_KEY) is None:
        await set_enabled(get_settings().gr_default_enabled)


def _detect_policy_format(raw_content: str) -> Literal["standard", "checker_rules"] | None:
    if parse_standard_policies(raw_content):
        return "standard"
    try:
        GuardrailPolicy.from_raw(raw_content)
    except (yaml.YAMLError, ValueError, ValidationError):
        return None
    return "checker_rules"


def _proxy_mounted() -> bool:
    from app.routers import proxy_openai  # lazy: the router imports this module
    return proxy_openai.MOUNTED


async def get_status() -> SystemStatus:
    enabled = await is_enabled()
    policy_enabled = await is_policy_enabled()
    try:
        policy = await policy_store.get_active_policy()
        name, version = policy.name, policy.version
        policy_format = _detect_policy_format(policy.raw_content)
    except HTTPException:
        name, version, policy_format = None, None, None
    return SystemStatus(enabled=enabled, policy_enabled=policy_enabled,
                         active_policy_name=name, active_policy_version=version,
                         active_policy_format=policy_format,
                         conflict_strategy=get_settings().guard_conflict_strategy,
                         proxy_mounted=_proxy_mounted())
