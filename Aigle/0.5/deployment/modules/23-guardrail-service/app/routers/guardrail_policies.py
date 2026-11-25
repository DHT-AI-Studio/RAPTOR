"""Guardrail Policy endpoints — versioned policy content, one active at a time.

  POST   /guardrail/policies                       → upload a new policy version (raw body + name/version query params)
  POST   /guardrail/policies/upload                 → upload a new policy version (file + name/version form fields)
  PUT    /guardrail/{policy_id}                     → modify an existing policy's content directly by id
  GET    /guardrail/policies/template               → suggested format + example content
  GET    /guardrail/policies/active                 → the currently active policy (404 if none)
  GET    /guardrail/policies                        → list all stored policies
  GET    /guardrail/policies/{policy_id}            → full policy content
  PUT    /guardrail/policies/{policy_id}/activate   → make this the active policy
  PUT    /guardrail/policies/{policy_id}/deactivate → deactivate this policy (must be the active one)
  DELETE /guardrail/policies/{policy_id}            → delete (rejected if it's the active one)

`name`/`version` are supplied as request parameters, not read from the
uploaded content — the content itself is opaque: any UTF-8 text, no
structure required, stored and used completely unprocessed. See
app/engine/models.py's comment on PolicyDetail for how the check endpoints
consume it.

`target` selects which content POST /guardrail/policies[/upload] sets:
`original` (the shared/default content, used unchanged by /guardrail/check/*
and /policy/check/*) or one of `llama-guard`/`granite-guardian`/`gpt-oss-safeguard`
— a per-guard-model prompt override that /policy/check/llm/* uses for that
model specifically, falling back to `original` when unset. Uploading again
with the same name+version but a different target adds/replaces just that
one target's content on the same policy — it does not create a new version.
The dropdown-name <-> column <-> guard-adapter-family mapping is the single
table in app/engine/policy_targets.py.

Writes go straight to the guardrail_policies table in Postgres; activating a
policy (or editing an active one via PUT /guardrail/{policy_id}) also
refreshes the Redis `guardrail:active_policy` cache that policy_engine.py /
check_policy_llm.py read for /policy/check/*.
"""
from __future__ import annotations

import json
import logging
from typing import Literal
from uuid import UUID

import jsonschema
import yaml
from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from app.engine import policy_store
from app.engine.models import (
    PolicyCreateResponse,
    PolicyDetail,
    PolicySummary,
    PolicyTemplate,
)
from app.engine.policy_targets import PolicyTarget
from app.models.moderation import ModerationPolicy
from app.models.policy import STANDARD_OBJECT_KEYS, StandardPolicyCategory
from app.services.default_moderation import DEFAULT_MODERATION_STANDARD_POLICY
from app.services.default_policy import DEFAULT_STANDARD_POLICY

_STANDARD_POLICY_SCHEMA = TypeAdapter(list[StandardPolicyCategory]).json_schema()


def _validate_if_standard_format(raw_content: str, target: str) -> None:
    """If `raw_content` looks like a JSON array (the Standard Policy Format
    shape) AND it's going into the shared "original" content, validate it
    strictly against the format's JSON Schema — malformed standard-format
    JSON is rejected here rather than silently falling back to being used as
    opaque free text. Anything that isn't a JSON array (existing YAML
    checker_rules policies, free text) is left completely alone, matching
    today's lenient behavior. Per-model override targets are always free-text
    verbatim and never validated against this schema.

    CM-1: the object form `{"categories": [...], "moderation": {...}}` is
    validated the same way -- the array, plus the moderation block against
    app/models/moderation.py. Any other JSON object (e.g. checker_rules) is
    left alone as before."""
    if target != "original":
        return
    try:
        data = json.loads(raw_content)
    except json.JSONDecodeError:
        return
    moderation = None
    if isinstance(data, dict) and "categories" in data:
        extra = sorted(set(data) - STANDARD_OBJECT_KEYS)
        if extra:
            raise HTTPException(status_code=422, detail=f"Invalid Standard Policy Format: unexpected key(s) {extra}; "
                                                        f"the object form takes only 'categories' and 'moderation'")
        moderation, data = data.get("moderation"), data["categories"]
    if not isinstance(data, list):
        return
    try:
        jsonschema.validate(instance=data, schema=_STANDARD_POLICY_SCHEMA)
    except jsonschema.ValidationError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid Standard Policy Format: {exc.message}") from exc
    if moderation is not None:
        try:
            ModerationPolicy.model_validate(moderation)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=f"Invalid moderation block: {exc}") from exc

_logger = logging.getLogger(__name__)
router = APIRouter(tags=["Guardrail Policies"])

# create_policy takes a raw Request (to accept any text body regardless of
# Content-Type), so FastAPI has nothing to introspect for docs — attach a
# minimal schema manually via openapi_extra instead. name/version are real
# Query parameters (FastAPI documents those automatically).
_POLICY_BODY_SCHEMA = {
    "type": "string",
    "description": "Arbitrary policy content, stored and used exactly as written. No structure is required.",
}
_POLICY_BODY_EXAMPLE = "description: Block weapon instructions; escalate anything that looks malicious.\n"

_EXAMPLE_YAML = """\
description: Standard content safety policy for general use.

# This content is stored and used exactly as written — it becomes the
# verbatim system prompt for /policy/check/llm/*. Including a 'rules' array
# (as shown here) also lets /policy/check/input|output|conversation use it
# for deterministic matching; if you omit 'rules', those three endpoints
# return an error telling you to use /policy/check/llm/* instead.
default_action: allow
default_reason: No policy rules matched — content appears safe
rules:
  - id: GEN-001
    priority: 100
    description: Block weapon / drug instructions
    conditions:
      risk_flags_any: [weapon_instructions, drug_synthesis]
      confidence: [high, medium]
    action: block
    reason: Dangerous instructions that could enable real-world harm
  - id: GEN-002
    priority: 90
    description: Escalate suspected malicious intent
    conditions:
      intent: [malicious]
    action: escalate
    reason: Content exhibits potentially malicious intent
"""
_EXAMPLE_JSON = json.dumps(yaml.safe_load(_EXAMPLE_YAML), indent=2)


def _decode(raw: bytes) -> str:
    """No structure is required or validated — just readable text."""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Content must be UTF-8 text: {exc}")


class PolicyContentUpdate(BaseModel):
    """Any field left out (or explicitly omitted from the JSON body) is left
    unchanged — only fields actually present in the request are written."""
    raw_content:                str | None = Field(None, description="更新「original」共用內容")
    content_llama_guard:        str | None = Field(None, description="更新 llama-guard 專用 prompt")
    content_granite_guardian:   str | None = Field(None, description="更新 granite-guardian 專用 prompt")
    content_gpt_oss_safeguard:  str | None = Field(None, description="更新 gpt-oss-safeguard 專用 prompt")


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
async def create_policy(
    request: Request,
    name: str = Query(..., min_length=1, description="Identifies this policy in listings"),
    version: str = Query(..., min_length=1, description="Identifies this policy version"),
    target: PolicyTarget = Query(
        "original",
        description="這份內容套用到哪個推理目標。original 是共用內容；其餘三個是 /policy/check/llm/* "
                     "各別 guard model 專用的 prompt，未設定時該 model 會退回使用 original。",
    ),
) -> PolicyCreateResponse:
    """
    Upload content for one policy version + target. `name`/`version` are
    query parameters; the request body is stored and used exactly as
    written — any text, YAML, JSON, whatever — no structure is required.

    `target` selects which content this call sets. Uploading again with the
    same `name`+`version` but a different `target` adds/replaces just that
    target's content on the same policy — it does not create a new version,
    so a single policy can accumulate all four targets across separate calls.

    See `GET /guardrail/policies/template` for a suggested example, or
    `POST /guardrail/policies/upload` to upload a `.yaml`/`.json` file instead.
    """
    raw_content = _decode(await request.body())
    _validate_if_standard_format(raw_content, target)
    _logger.info("guardrail policy: create name=%s version=%s target=%s", name, version, target)
    return await policy_store.create_policy(name, version, raw_content, target)


@router.post(
    "/policies/upload",
    response_model=PolicyCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a policy file (name/version as form fields)",
)
async def upload_policy(
    file: UploadFile = File(..., description="A policy file — any text content, no structure required"),
    name: str = Form(..., min_length=1, description="Identifies this policy in listings"),
    version: str = Form(..., min_length=1, description="Identifies this policy version"),
    target: PolicyTarget = Form("original", description="這份內容套用到哪個推理目標"),
) -> PolicyCreateResponse:
    """Same as `POST /guardrail/policies`, but for an actual uploaded file — `name`/`version`/`target` are form fields."""
    raw_content = _decode(await file.read())
    _validate_if_standard_format(raw_content, target)
    _logger.info("guardrail policy: upload filename=%s name=%s version=%s target=%s", file.filename, name, version, target)
    return await policy_store.create_policy(name, version, raw_content, target)


@router.put("/{policy_id}", response_model=PolicyDetail, summary="Modify an existing policy's content directly by id")
async def update_policy(policy_id: UUID, body: PolicyContentUpdate) -> PolicyDetail:
    """
    直接修改一份已存在 policy 的內容欄位（`name`/`version`/`is_active` 不受影響）。
    body 中沒有提供的欄位維持不變；提供的欄位（`raw_content` 或任一
    model-specific override）會整段取代。如果這份 policy 目前是啟用中的
    active policy，Redis 快取會一併更新，/policy/check/* 立即套用新內容。
    """
    fields = body.model_dump(exclude_unset=True, exclude_none=True)
    if not fields:
        raise HTTPException(status_code=400, detail="No content fields provided to update")
    _logger.info("guardrail policy: update id=%s fields=%s", policy_id, list(fields))
    return await policy_store.update_policy_content(policy_id, fields)


_STANDARD_POLICY_EXAMPLE_JSON = json.dumps(DEFAULT_STANDARD_POLICY, indent=2, ensure_ascii=False)
_MODERATION_POLICY_EXAMPLE_JSON = json.dumps(DEFAULT_MODERATION_STANDARD_POLICY, indent=2, ensure_ascii=False)


@router.get(
    "/policies/template",
    response_model=PolicyTemplate,
    summary="View the suggested format and example content for a policy",
)
async def get_policy_template(
    format: Literal["standard", "checker_rules", "moderation"] = Query(
        "checker_rules",
        description="'checker_rules' (default) is the existing YAML/JSON detector-rule format; "
                     "'standard' is the Standard Policy Format (a JSON array of category objects); "
                     "'moderation' is the same S1-S13 policy plus the CM-1 moderation taxonomy "
                     "(object form: {\"categories\": [...], \"moderation\": {...}}).",
    ),
) -> PolicyTemplate:
    """Returns example policy content — copy, edit, and upload via
    `POST /guardrail/policies` (with `?name=...&version=...`) or `.../upload`
    (with `name`/`version` form fields)."""
    if format == "moderation":
        return PolicyTemplate(
            description=(
                "Standard Policy Format, object form (CM-1): `categories` is the 13-category S1-S13 "
                "default (text decisions unchanged), `moderation` is the v0.5 moderation taxonomy -- "
                "nsfw, violence, hate, self_harm -- each mapped to Llama Guard 3 S-codes and "
                "ShieldGemma 2 policies, with a decision per modality (text, transcript, image, "
                "video_frame) and a 0-1 score threshold for image and video_frame. The image guards "
                "(/infer/*) apply the active policy's block when called with `modality`; without "
                "one they use this same built-in default. Upload with `target=original`, then "
                "activate. See README.md, 'Moderation Taxonomy (CM-1)'."
            ),
            example_yaml="",
            example_json=_MODERATION_POLICY_EXAMPLE_JSON,
        )
    if format == "standard":
        return PolicyTemplate(
            description=(
                "Standard Policy Format: a JSON array of category objects, each mapping a "
                "category (id/name/description) to a severity/decision plus block/allow "
                "criteria — no per-model prompt authoring needed, every configured guard model "
                "(Llama Guard 3 / Granite Guardian / GPT-OSS-Safeguard) gets its own correctly "
                "formatted prompt generated from this same source of truth at activation time. "
                "Below is the 13-category default (S1-S13, Llama Guard 3's official taxonomy — "
                "S14 'Code Interpreter Abuse' is a later addition, not part of the original 13, "
                "so it's excluded), pre-filled with descriptions/criteria you can edit. Upload "
                "with `target=original` (the default); JSON only, not YAML — "
                "see policy_example/standard_policy_format.md for the full field reference."
            ),
            example_yaml="",
            example_json=_STANDARD_POLICY_EXAMPLE_JSON,
        )
    return PolicyTemplate(
        description=(
            "`name`/`version` are supplied as request parameters (query params for "
            "POST /guardrail/policies, form fields for .../upload) — not part of this content. "
            "Everything below is stored and used exactly as written — it becomes the verbatim "
            "system prompt for /policy/check/llm/*. Including a 'rules' array (as shown) also "
            "lets /policy/check/input|output|conversation use it for deterministic matching; "
            "if you omit 'rules', those three endpoints will return an error telling you to "
            "use /policy/check/llm/* instead."
        ),
        example_yaml=_EXAMPLE_YAML,
        example_json=_EXAMPLE_JSON,
    )


@router.get("/policies", response_model=list[PolicySummary], summary="List all policies")
async def list_policies() -> list[PolicySummary]:
    return await policy_store.list_policies()


@router.get(
    "/policies/active",
    response_model=PolicyDetail,
    summary="Get the currently active policy",
)
async def get_active_policy() -> PolicyDetail:
    """Returns the currently active policy's full content. 404 if none is active."""
    return await policy_store.get_active_policy()


@router.get("/policies/{policy_id}", response_model=PolicyDetail, summary="Get full policy content")
async def get_policy(policy_id: UUID) -> PolicyDetail:
    return await policy_store.get_policy(policy_id)


@router.put("/policies/{policy_id}/activate", response_model=PolicyDetail, summary="Activate a policy")
async def activate_policy(policy_id: UUID) -> PolicyDetail:
    """Deactivate whatever policy is currently active and activate this one instead."""
    _logger.info("guardrail policy: activate id=%s", policy_id)
    return await policy_store.activate_policy(policy_id)


@router.put("/policies/{policy_id}/deactivate", response_model=PolicyDetail, summary="Deactivate a policy")
async def deactivate_policy(policy_id: UUID) -> PolicyDetail:
    """
    Deactivate this policy. Unless this policy IS the built-in
    `default-standard-policy`, deactivating it immediately falls back to
    activating that default instead (creating it first if it's never
    existed) — there's always an active policy, so `GET /guardrail/policies/active`
    never 404s just because a user-created policy was deactivated. This
    response still reports the policy you asked to deactivate, not the
    fallback.

    404 if the policy doesn't exist, 409 if it exists but isn't currently active.
    """
    _logger.info("guardrail policy: deactivate id=%s", policy_id)
    return await policy_store.deactivate_policy(policy_id)


@router.delete(
    "/policies/{policy_id}",
    response_model=None,
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a policy",
)
async def delete_policy(policy_id: UUID) -> None:
    """Delete a stored policy. Returns 409 if it's the currently active
    policy, or if it's the built-in `default-standard-policy` (any version)
    — that one can never be deleted, since it's the fallback every
    deactivation and startup falls back to."""
    _logger.info("guardrail policy: delete id=%s", policy_id)
    await policy_store.delete_policy(policy_id)
