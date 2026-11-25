"""
Proxy router that forwards guardrail policy-management and system-switch
requests to remote service: raptor-guardrail-service:8026 (module 23)

Distinct from app/middlewares/guardrail.py, which auto-wraps three hardcoded
routes (chat/completions, a2a/query, chat/chat) with calls to module 23's
POST /guardrail/policy/decide and has no CRUD and no system switch. This
router is the gateway proxy for exactly two of module 23's /guardrail/*
route groups:

  * "Guardrail Policies" (module 23 app/routers/guardrail_policies.py) —
    versioned policy CRUD, one active at a time
  * "Guardrail (system)" (module 23 app/routers/system.py) — the global
    enable/disable switches and status

Parameters, request bodies and response models mirror module 23's so the
gateway's Swagger matches the backend's. Mirrors app/routers/benchmark.py's
forward_request(request, path) pattern.

Every endpoint here requires the `guardrail-admin` Keycloak role, checked
before ever reaching module 23 -- same pattern as moderation.py's
`_is_moderation_admin`/`_require_moderation_admin`. Without this, any logged-in
user (even the lowest `user_basic` tier) could globally disable guardrail
checking or rewrite the live policy content, confirmed live.
"""

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

import httpx
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Request,
    status,
)
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.api.dependencies import get_current_user
from app.core.config import get_settings

router = APIRouter(tags=["Guardrail Proxy"])


def _is_guardrail_admin(payload: dict[str, Any]) -> bool:
    roles = payload.get("realm_access", {}).get("roles", [])
    return "guardrail-admin" in roles


def _require_guardrail_admin(current_user: dict[str, Any]) -> None:
    if not _is_guardrail_admin(current_user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Requires the guardrail-admin role")

# Mirrors module 23's app/engine/policy_targets.py
PolicyTarget = Literal["original", "llama-guard", "granite-guardian", "gpt-oss-safeguard"]


# --- Models (mirror module 23's app/engine/models.py + app/models/system.py, for Swagger typing only) ---

class PolicyCreateResponse(BaseModel):
    id: UUID
    name: str
    version: str
    created_at: datetime


class PolicySummary(BaseModel):
    """Lightweight overview of a stored policy (no content body)."""
    id: UUID
    name: str
    version: str
    is_active: bool
    created_at: datetime


class PolicyDetail(BaseModel):
    """Full stored policy — identity fields plus content per inference target.

    `raw_content` is the shared/"original" content, used by every check
    family unchanged, and as the /policy/check/llm/* fallback for any guard
    model without its own override below."""
    id: UUID
    name: str
    version: str
    raw_content: str
    content_llama_guard: str | None = None
    content_granite_guardian: str | None = None
    content_gpt_oss_safeguard: str | None = None
    is_active: bool
    created_at: datetime


class PolicyTemplate(BaseModel):
    """Suggested YAML/JSON format and example content for a policy."""
    description: str
    example_json: str
    example_yaml: str


class PolicyContentUpdate(BaseModel):
    """Any field left out (or explicitly omitted from the JSON body) is left
    unchanged — only fields actually present in the request are written."""
    raw_content: str | None = Field(None, description="Update the shared/default ('original') content")
    content_llama_guard: str | None = Field(None, description="Update the llama-guard-specific prompt")
    content_granite_guardian: str | None = Field(None, description="Update the granite-guardian-specific prompt")
    content_gpt_oss_safeguard: str | None = Field(None, description="Update the gpt-oss-safeguard-specific prompt")


class SystemEnabledResponse(BaseModel):
    enabled: bool = Field(..., description="Guardrail checking master switch state")


class PolicyEnabledResponse(BaseModel):
    policy_enabled: bool = Field(
        ..., description="Policy-based LLM check switch state (/policy/check/llm/*, /debug/policy/check/llm/*)"
    )


class SystemStatus(BaseModel):
    enabled: bool = Field(..., description="Guardrail checking master switch state")
    policy_enabled: bool = Field(
        ...,
        description=(
            "Policy-based LLM check switch state; when False, /policy/check/llm/* and "
            "/debug/policy/check/llm/* short-circuit to safe=true (independent of `enabled`, "
            "has no effect on /guard/check/*)"
        ),
    )
    active_policy_name: str | None = Field(None, description="Name of the active policy, or null if none")
    active_policy_version: str | None = Field(None, description="Version of the active policy, or null if none")
    active_policy_format: Literal["standard", "checker_rules"] | None = Field(
        None, description="Format of the active policy's content, or null if none is active or it parses as neither.",
    )
    conflict_strategy: Literal["strict", "majority", "permissive"] = Field(
        ..., description="How disagreeing multi-guard-model verdicts are resolved (GUARD_CONFLICT_STRATEGY).",
    )
    proxy_mounted: bool = Field(
        False, description="Whether the OpenAI-compatible proxy (/v1/chat/completions, /v1/models) is registered",
    )


# --- Core Proxy Function ---

async def forward_request(request: Request, path: str, body: bytes | None = None):
    settings = get_settings()
    url = f"{settings.guardrail_url.rstrip('/')}/guardrail{path}"

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.request(
                method=request.method,
                url=url,
                headers={
                    key: value for key, value in request.headers.items()
                    if key.lower() != "host"
                },
                params=request.query_params,
                content=body if body is not None else await request.body(),
            )

        return JSONResponse(
            status_code=response.status_code,
            content=response.json() if response.content else None,
        )

    except httpx.RequestError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to connect to guardrail service: {e!s}",
        )


# --- Proxy Endpoints: Guardrail Policies (module 23 app/routers/guardrail_policies.py) ---

# create_policy_proxy reads the raw body (any text, any Content-Type), so
# FastAPI has nothing to introspect for docs — same openapi_extra body schema
# module 23 attaches to its own POST /guardrail/policies.
_POLICY_BODY_SCHEMA = {
    "type": "string",
    "description": "Arbitrary policy content, stored and used exactly as written. No structure is required.",
}
_POLICY_BODY_EXAMPLE = "description: Block weapon instructions; escalate anything that looks malicious.\n"


@router.post(
    "/policies",
    response_model=PolicyCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new policy (name/version as query params, body is free-form content)",
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "text/plain":       {"schema": _POLICY_BODY_SCHEMA, "example": _POLICY_BODY_EXAMPLE},
                "application/json": {"schema": _POLICY_BODY_SCHEMA, "example": _POLICY_BODY_EXAMPLE},
                "application/yaml": {"schema": _POLICY_BODY_SCHEMA, "example": _POLICY_BODY_EXAMPLE},
            },
        },
    },
)
async def create_policy_proxy(
    request: Request,
    name: str = Query(..., min_length=1, description="Identifies this policy in listings"),
    version: str = Query(..., min_length=1, description="Identifies this policy version"),
    target: PolicyTarget = Query(
        "original",
        description="Which inference target this content applies to. `original` is the shared content; "
                    "the other three are per-guard-model prompts for /policy/check/llm/*, and a model "
                    "without its own prompt falls back to `original`.",
    ),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """
    Upload content for one policy version + target. `name`/`version` are
    query parameters; the request body is stored and used exactly as
    written — any text, YAML, JSON, whatever — no structure is required.

    `target` selects which content this call sets. Uploading again with the
    same `name`+`version` but a different `target` adds/replaces just that
    target's content on the same policy — it does not create a new version.

    See `GET .../guardrail/policies/template` for a suggested example, or
    `POST .../guardrail/policies/upload` to upload a `.yaml`/`.json` file instead.
    """
    _require_guardrail_admin(current_user)
    return await forward_request(request, "/policies")


@router.post(
    "/policies/upload",
    response_model=PolicyCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a policy file (name/version as form fields)",
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "file": {"type": "string", "format": "binary",
                                      "description": "A policy file — any text content, no structure required"},
                            "name": {"type": "string", "minLength": 1,
                                     "description": "Identifies this policy in listings"},
                            "version": {"type": "string", "minLength": 1,
                                        "description": "Identifies this policy version"},
                            "target": {"type": "string", "enum": list(PolicyTarget.__args__), "default": "original",
                                       "description": "Which inference target this content applies to"},
                        },
                        "required": ["file", "name", "version"],
                    },
                },
            },
        },
    },
)
async def upload_policy_proxy(request: Request, current_user: dict[str, Any] = Depends(get_current_user)):
    """Same as `POST .../guardrail/policies`, but for an actual uploaded file —
    `name`/`version`/`target` are form fields.

    Declared with a bare `Request` rather than `UploadFile`/`Form(...)` params —
    FastAPI would parse those eagerly and drain the multipart body stream before
    this handler runs, leaving `forward_request`'s `request.body()` nothing to
    read (`RuntimeError: Stream consumed`). Forwarding the raw, still-unread
    body + original `Content-Type` (multipart boundary and all) is what actually
    works."""
    _require_guardrail_admin(current_user)
    return await forward_request(request, "/policies/upload")


@router.put("/{policy_id}", response_model=PolicyDetail, summary="Modify an existing policy's content directly by id")
async def update_policy_proxy(
    policy_id: UUID, body: PolicyContentUpdate, request: Request,
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """Modify an existing policy's content fields (`name`/`version`/`is_active`
    are unaffected). Fields not provided are left unchanged; each provided field
    is replaced in full. 400 if no content field is provided. If this policy is
    the active one, module 23's Redis cache is refreshed so /policy/check/*
    applies the new content immediately."""
    _require_guardrail_admin(current_user)
    return await forward_request(request, f"/{policy_id}")


@router.get(
    "/policies/template",
    response_model=PolicyTemplate,
    summary="View the suggested format and example content for a policy",
)
async def get_policy_template_proxy(
    request: Request,
    format: Literal["standard", "checker_rules", "moderation"] = Query(
        "checker_rules",
        description="'checker_rules' (default) is the existing YAML/JSON detector-rule format; "
                    "'standard' is the Standard Policy Format (a JSON array of category objects); "
                    "'moderation' is the same S1-S13 policy plus the CM-1 moderation taxonomy "
                    "(object form: {\"categories\": [...], \"moderation\": {...}}).",
    ),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """Returns example policy content — copy, edit, and upload via
    `POST .../guardrail/policies` (with `?name=...&version=...`) or `.../upload`
    (with `name`/`version` form fields)."""
    _require_guardrail_admin(current_user)
    return await forward_request(request, "/policies/template")


@router.get("/policies", response_model=list[PolicySummary], summary="List all policies")
async def list_policies_proxy(request: Request, current_user: dict[str, Any] = Depends(get_current_user)):
    _require_guardrail_admin(current_user)
    return await forward_request(request, "/policies")


@router.get("/policies/active", response_model=PolicyDetail, summary="Get the currently active policy")
async def get_active_policy_proxy(request: Request, current_user: dict[str, Any] = Depends(get_current_user)):
    """Returns the currently active policy's full content. 404 if none is active."""
    _require_guardrail_admin(current_user)
    return await forward_request(request, "/policies/active")


@router.get("/policies/{policy_id}", response_model=PolicyDetail, summary="Get full policy content")
async def get_policy_proxy(
    policy_id: UUID, request: Request, current_user: dict[str, Any] = Depends(get_current_user),
):
    _require_guardrail_admin(current_user)
    return await forward_request(request, f"/policies/{policy_id}")


@router.put("/policies/{policy_id}/activate", response_model=PolicyDetail, summary="Activate a policy")
async def activate_policy_proxy(
    policy_id: UUID, request: Request, current_user: dict[str, Any] = Depends(get_current_user),
):
    """Deactivate whatever policy is currently active and activate this one instead."""
    _require_guardrail_admin(current_user)
    return await forward_request(request, f"/policies/{policy_id}/activate")


@router.put("/policies/{policy_id}/deactivate", response_model=PolicyDetail, summary="Deactivate a policy")
async def deactivate_policy_proxy(
    policy_id: UUID, request: Request, current_user: dict[str, Any] = Depends(get_current_user),
):
    """
    Deactivate this policy. Unless this policy IS the built-in
    `default-standard-policy`, deactivating it immediately falls back to
    activating that default instead (creating it first if it's never
    existed) — there's always an active policy. This response still reports
    the policy you asked to deactivate, not the fallback.

    404 if the policy doesn't exist, 409 if it exists but isn't currently active.
    """
    _require_guardrail_admin(current_user)
    return await forward_request(request, f"/policies/{policy_id}/deactivate")


@router.delete(
    "/policies/{policy_id}",
    response_model=None,
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a policy",
)
async def delete_policy_proxy(
    policy_id: UUID, request: Request, current_user: dict[str, Any] = Depends(get_current_user),
):
    """Delete a stored policy. 409 if it's the currently active policy, or if
    it's the built-in `default-standard-policy` (any version) — that one can
    never be deleted, since it's the fallback every deactivation falls back to."""
    _require_guardrail_admin(current_user)
    return await forward_request(request, f"/policies/{policy_id}")


# --- Proxy Endpoints: Guardrail (system) (module 23 app/routers/system.py) ---

@router.post("/system/enable", response_model=SystemEnabledResponse, summary="Enable guardrail checking globally")
async def enable_proxy(request: Request, current_user: dict[str, Any] = Depends(get_current_user)):
    _require_guardrail_admin(current_user)
    return await forward_request(request, "/system/enable")


@router.post("/system/disable", response_model=SystemEnabledResponse, summary="Disable guardrail checking globally")
async def disable_proxy(request: Request, current_user: dict[str, Any] = Depends(get_current_user)):
    _require_guardrail_admin(current_user)
    return await forward_request(request, "/system/disable")


@router.post("/system/policy/enable", response_model=PolicyEnabledResponse,
             summary="Enable policy-based LLM checking (/policy/check/llm/*, /debug/policy/check/llm/*)")
async def enable_policy_proxy(request: Request, current_user: dict[str, Any] = Depends(get_current_user)):
    _require_guardrail_admin(current_user)
    return await forward_request(request, "/system/policy/enable")


@router.post("/system/policy/disable", response_model=PolicyEnabledResponse,
             summary="Disable policy-based LLM checking (/policy/check/llm/*, /debug/policy/check/llm/*)")
async def disable_policy_proxy(request: Request, current_user: dict[str, Any] = Depends(get_current_user)):
    _require_guardrail_admin(current_user)
    return await forward_request(request, "/system/policy/disable")


@router.get("/system/status", response_model=SystemStatus, summary="Current guardrail switch state + active policy")
async def status_proxy(request: Request, current_user: dict[str, Any] = Depends(get_current_user)):
    _require_guardrail_admin(current_user)
    return await forward_request(request, "/system/status")
