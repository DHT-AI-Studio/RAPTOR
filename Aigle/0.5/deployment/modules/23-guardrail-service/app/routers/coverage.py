"""Guardrail coverage matrix — GB-22.

Endpoints
---------
GET  /guardrail/coverage                     — legacy static enforcement-point list (backward compat)
GET  /guardrail/coverage/integration         — dynamic per-entry-point platform matrix (GB-22)
GET  /guardrail/coverage/integration/summary — aggregate counts + coverage_pct
POST /guardrail/integration/register         — called by integrating modules at startup to set
                                               their screened flag in Redis (used by 07 and 18
                                               which have no direct Redis dependency)

Redis key schema:  guardrail:integration:{entry_point}  →  "1"  (no TTL: survives restarts)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.db.redis_conn import get_redis

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Guardrail coverage"])


# ── Legacy static enforcement-point list (kept for backward compat) ──────────

class CoverageEntry(BaseModel):
    endpoint: str
    method: str
    policy_aware: bool
    deprecated: bool = False
    successor: str | None = None
    story: str
    description: str


ENFORCEMENT_POINTS: list[CoverageEntry] = [
    CoverageEntry(
        endpoint="/guardrail/policy/decide", method="POST", policy_aware=True, story="GRD-X1",
        description="Canonical decision API: allow | redact | warn | block for one input or output",
    ),
    CoverageEntry(
        endpoint="/api/v0.2/check/input", method="POST", policy_aware=True, story="GRD-X1",
        description="User input, CheckResponse shape",
    ),
    CoverageEntry(
        endpoint="/api/v0.2/check/output", method="POST", policy_aware=True, story="GRD-X1",
        description="AI response, CheckResponse shape",
    ),
    CoverageEntry(
        endpoint="/api/v0.2/check/conversation", method="POST", policy_aware=True, story="GRD-X1",
        description="Last turn of a conversation, CheckResponse shape",
    ),
    CoverageEntry(
        endpoint="/v1/chat/completions", method="POST", policy_aware=True, story="GB-7",
        description="OpenAI-compatible proxy: last user message in, choices[0] content out",
    ),
    CoverageEntry(
        endpoint="/guardrail/check/input", method="POST", policy_aware=True, story="GB-4",
        description="Rule-engine check of an input",
    ),
    CoverageEntry(
        endpoint="/guardrail/check/output", method="POST", policy_aware=True, story="GB-4",
        description="Rule-engine check of an output",
    ),
    CoverageEntry(
        endpoint="/guard/check/input", method="POST", policy_aware=False, deprecated=True,
        successor="/guardrail/policy/decide", story="GB-6",
        description="Fixed guard prompts only, no policy",
    ),
    CoverageEntry(
        endpoint="/guard/check/output", method="POST", policy_aware=False, deprecated=True,
        successor="/guardrail/policy/decide", story="GB-6",
        description="Fixed guard prompts only, no policy",
    ),
    CoverageEntry(
        endpoint="/guard/check/conversation", method="POST", policy_aware=False, deprecated=True,
        successor="/api/v0.2/check/conversation", story="GB-6",
        description="Fixed guard prompts only, no policy",
    ),
    CoverageEntry(
        endpoint="/guard/check/raw", method="POST", policy_aware=False, story="GB-6",
        description="Debug view: each guard model's raw output, no verdict",
    ),
]


@router.get(
    "/coverage",
    response_model=list[CoverageEntry],
    summary="Module 23 enforcement points — static list (GB-22 legacy)",
)
async def coverage() -> list[CoverageEntry]:
    """Static list of module 23's own content-screening endpoints.
    Preserved for backward compatibility; see /coverage/integration
    for the full platform-integration matrix (GB-22)."""
    return ENFORCEMENT_POINTS


# ── GB-22 platform-integration matrix ────────────────────────────────────────

@dataclass(frozen=True)
class _IntegrationSpec:
    entry_point: str
    module: str
    tier: str
    method: str
    env_var: str
    story_reference: str


_ENTRY_POINTS: list[_IntegrationSpec] = [
    _IntegrationSpec(
        entry_point="module_07_hook",
        module="07-ai-ml-services",
        tier="inference",
        method="guardrail_hook.check() before/after LLM inference (sync requests)",
        env_var="GUARDRAIL_URL",
        story_reference="GB-6",
    ),
    _IntegrationSpec(
        entry_point="module_15_proxy",
        module="15-chat-service",
        tier="chat",
        method="LLM_BASE_URL proxied through module 23 /v1/chat/completions",
        env_var="GUARDRAIL_URL",
        story_reference="GB-12",
    ),
    _IntegrationSpec(
        entry_point="mcp_tool_calls",
        module="18-query-orchestrator",
        tier="orchestration",
        method="MCP tool-call pre/post check via guardrail middleware",
        env_var="GUARDRAIL_URL",
        story_reference="GB-18",
    ),
    _IntegrationSpec(
        entry_point="gateway_middleware",
        module="13-api-services",
        tier="gateway",
        method="GuardrailMiddleware wraps POST /chat/completions, /a2a/query, /chat/chat",
        env_var="GUARDRAIL_URL",
        story_reference="GB-19",
    ),
    _IntegrationSpec(
        entry_point="grpc_interceptor",
        module="36-grpc-gateway",
        tier="grpc",
        method="gRPC interceptor pre/post check on every RPC with text payload",
        env_var="GUARDRAIL_URL",
        story_reference="GB-20",
    ),
    _IntegrationSpec(
        entry_point="a2a_host_orchestrator",
        module="21-agent-protocol",
        tier="agent",
        method="guardrail_precheck() on incoming A2A task content",
        env_var="GUARDRAIL_URL",
        story_reference="GB-21",
    ),
]

_REDIS_KEY_PREFIX = "guardrail:integration:"


class IntegrationCoverageEntry(BaseModel):
    entry_point: str = Field(..., description="Unique entry-point identifier")
    module: str = Field(..., description="Deployment module directory name")
    tier: str = Field(..., description="Architectural tier")
    screened: bool = Field(..., description="True = module registered at startup with GUARDRAIL_URL set")
    method: str = Field(..., description="How the module integrates with the guardrail service")
    env_var: str = Field(..., description="Environment variable that enables the integration")
    story_reference: str = Field(..., description="Implementing Jira story")


class CoverageSummary(BaseModel):
    total: int
    screened: int
    unscreened: int
    coverage_pct: float = Field(..., description="screened / total * 100, rounded to 2 dp")


# ── Registration endpoint (used by modules without direct Redis access) ───────

class RegistrationRequest(BaseModel):
    entry_point: str = Field(..., description="Must match one of the known _ENTRY_POINTS entry_point values")
    module: str = Field(..., description="Caller's module identifier, for audit logging")


_KNOWN_ENTRY_POINTS: frozenset[str] = frozenset(ep.entry_point for ep in _ENTRY_POINTS)


@router.post(
    "/integration/register",
    status_code=204,
    summary="Register a module's guardrail integration at startup (GB-22)",
)
async def register_integration(body: RegistrationRequest) -> None:
    """Called by integrating modules at startup when GUARDRAIL_URL is set.

    Sets guardrail:integration:{entry_point} = "1" in Redis (no TTL).
    Modules that have a direct Redis connection (13, 15, 21, 36) write the
    key themselves; modules without one (07, 18) POST here instead.

    Returns 204 on success.  Returns 422 for an unknown entry_point so
    misconfigured callers fail visibly rather than silently polluting the
    coverage matrix with phantom keys.
    """
    if body.entry_point not in _KNOWN_ENTRY_POINTS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown entry_point {body.entry_point!r}. "
                   f"Known: {sorted(_KNOWN_ENTRY_POINTS)}",
        )
    redis = await get_redis()
    key = f"{_REDIS_KEY_PREFIX}{body.entry_point}"
    await redis.set(key, "1")
    logger.info("[GB-22] guardrail integration registered: entry_point=%s module=%s",
                body.entry_point, body.module)


# ── Read endpoints ────────────────────────────────────────────────────────────

async def _get_screened_flags() -> dict[str, bool]:
    """Single MGET for all entry-point keys."""
    redis = await get_redis()
    keys = [f"{_REDIS_KEY_PREFIX}{ep.entry_point}" for ep in _ENTRY_POINTS]
    values = await redis.mget(*keys)
    return {ep.entry_point: (val == "1") for ep, val in zip(_ENTRY_POINTS, values)}


@router.get(
    "/coverage/integration",
    response_model=list[IntegrationCoverageEntry],
    summary="Platform-wide guardrail integration coverage matrix (GB-22)",
)
async def integration_coverage() -> list[IntegrationCoverageEntry]:
    """One row per platform entry point.  screened=true means the module
    called register_integration (or wrote the Redis key directly) at its
    last startup with GUARDRAIL_URL configured."""
    flags = await _get_screened_flags()
    return [
        IntegrationCoverageEntry(
            entry_point=ep.entry_point,
            module=ep.module,
            tier=ep.tier,
            screened=flags[ep.entry_point],
            method=ep.method,
            env_var=ep.env_var,
            story_reference=ep.story_reference,
        )
        for ep in _ENTRY_POINTS
    ]


@router.get(
    "/coverage/integration/summary",
    response_model=CoverageSummary,
    summary="Guardrail integration coverage summary (GB-22)",
)
async def integration_coverage_summary() -> CoverageSummary:
    """Aggregate counts: {total, screened, unscreened, coverage_pct}."""
    flags = await _get_screened_flags()
    screened_count = sum(1 for v in flags.values() if v)
    total = len(_ENTRY_POINTS)
    return CoverageSummary(
        total=total,
        screened=screened_count,
        unscreened=total - screened_count,
        coverage_pct=round(screened_count / total * 100, 2),
    )
