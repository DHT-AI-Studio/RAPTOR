"""GD-1 / VIE01-246: GDPR/CCPA erasure — public gateway surface.

Module 13 is the only externally-reachable service in this deployment
(same pattern as every other router here); Module 33 (the orchestrator)
and modules 04/25/26's own GDPR endpoints are internal-only. This router
does the JWT + gdpr-admin role check, then proxies to Module 33 presenting
X-Internal-Token -- Module 33 has no session of its own to authenticate
the caller with, so it trusts that shared secret instead.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.api.dependencies import get_current_user, get_http_client
from app.core.config import Settings, get_settings

_logger = logging.getLogger(__name__)
router = APIRouter(tags=["GDPR"])

# Unlike the two POST endpoints below (explicitly "(authenticated as self)"
# and "(requires gdpr-admin Keycloak role)" per the ticket), GET .../jobs/
# {job_id} states no auth requirement at all -- deliberately: self-erasure
# disables the submitter's own account and revokes their session as step 2
# of the erasure sequence, before the job even finishes, so requiring a
# still-valid JWT here would make it impossible for the very user who
# requested their own erasure to ever confirm it completed. The job_id
# itself (a securely random UUID, returned once, only to the submitter or
# the DPO who acted on their behalf) is the access control instead -- the
# same pattern as a package-tracking number or a payment-status link.
public_router = APIRouter(tags=["GDPR"])


def _is_gdpr_admin(payload: dict[str, Any]) -> bool:
    roles = payload.get("realm_access", {}).get("roles", [])
    return "gdpr-admin" in roles


def _headers(settings: Settings, user_id: str, requested_by: str | None = None) -> dict[str, str]:
    headers = {"X-Internal-Token": settings.gdpr_internal_token, "X-User-ID": user_id}
    if requested_by:
        headers["X-Requested-By"] = requested_by
    return headers


async def _forward(client: httpx.AsyncClient, settings: Settings, method: str, path: str, **kwargs) -> dict[str, Any]:
    return (await _forward_response(client, settings, method, path, **kwargs)).json()


async def _forward_response(client: httpx.AsyncClient, settings: Settings, method: str, path: str, **kwargs) -> httpx.Response:
    """Like _forward, but keeps Module 33's own status code -- for routes
    whose success status itself carries meaning (DSAR's 200 vs 202)."""
    try:
        resp = await client.request(method, f"{settings.privacy_compliance_url}{path}", timeout=settings.request_timeout, **kwargs)
        resp.raise_for_status()
        return resp
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=f"Privacy service error: {exc.response.text}") from exc
    except httpx.RequestError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Could not connect to privacy compliance service.") from exc


@router.post(
    "/erasure",
    summary="Submit a GDPR/CCPA erasure request for yourself",
    description="Deletes all personal data associated with your account across every RAPTOR data store, "
                "within 30 days (GDPR Article 17 / CCPA § 1798.105). Submitting for an already-erased "
                "account returns the previous job instead of starting a new one. Under a legal hold the "
                "request is deferred until the hold ends (`status: DEFERRED`, `reason`, `hold_until`). "
                "`exception_fields` lists the data kept by law, with its retention and legal basis.",
)
async def submit_self_erasure(
    current_user: dict[str, Any] = Depends(get_current_user),
    client: httpx.AsyncClient = Depends(get_http_client),
    settings: Settings = Depends(get_settings),
):
    user_id = current_user["sub"]
    return await _forward(client, settings, "POST", "/erasure", headers=_headers(settings, user_id))


@router.post(
    "/erasure/{user_id}",
    summary="Submit a GDPR/CCPA erasure request for another user (gdpr-admin only)",
    description="Same as POST /gdpr/erasure, but on behalf of any user_id. Requires the `gdpr-admin` Keycloak realm role.",
)
async def submit_admin_erasure(
    user_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),
    client: httpx.AsyncClient = Depends(get_http_client),
    settings: Settings = Depends(get_settings),
):
    if not _is_gdpr_admin(current_user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Requires the gdpr-admin role")
    return await _forward(client, settings, "POST", "/erasure", headers=_headers(settings, user_id, requested_by=current_user["sub"]))


@public_router.get(
    "/erasure/jobs/{job_id}",
    summary="Get an erasure job's status",
    description="No login required -- the job_id itself (a securely random UUID, "
                "returned once at submission time) is the access control. This is "
                "deliberate: self-erasure disables the submitter's own account and "
                "session as step 2 of the process, before the job finishes, so "
                "requiring a still-valid login here would make it impossible for a "
                "user to ever confirm their own erasure completed.",
)
async def get_erasure_job(
    job_id: str,
    client: httpx.AsyncClient = Depends(get_http_client),
    settings: Settings = Depends(get_settings),
):
    return await _forward(client, settings, "GET", f"/erasure/jobs/{job_id}", headers={"X-Internal-Token": settings.gdpr_internal_token})


@router.post(
    "/erasure/jobs/{job_id}/retry",
    summary="Retry the stores still failed on a PARTIAL_FAILURE job (gdpr-admin only)",
    description="409 while the user is under a legal hold: the retry would delete data the hold keeps.",
)
async def retry_erasure_job(
    job_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),
    client: httpx.AsyncClient = Depends(get_http_client),
    settings: Settings = Depends(get_settings),
):
    if not _is_gdpr_admin(current_user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Requires the gdpr-admin role")
    return await _forward(client, settings, "POST", f"/erasure/jobs/{job_id}/retry", headers={"X-Internal-Token": settings.gdpr_internal_token})


# ── GD-2 / VIE01-247: data portability export ───────────────────────────────
#
# Unlike erasure's GET .../jobs/{job_id}, export's poll endpoint stays
# JWT-gated: submitting an export doesn't disable the submitter's account or
# revoke their session the way erasure's step 2 does, so there's no
# self-lockout problem to design around here. Ownership is checked
# explicitly (job.user_id must match the caller, unless gdpr-admin) rather
# than trusting the job_id alone as the credential.

# GET is the ticket's method (GD-2 / VIE01-247); POST, the method this route
# shipped with, stays so existing callers keep working. Both start a job.
@router.get(
    "/export",
    summary="Submit a GDPR/CCPA data export request for yourself",
    description="Same as POST /gdpr/export -- GET is the method the GD-2 ticket specifies. "
                "Starts a fresh export job each call; returns {job_id, status: \"pending\"}.",
)
@router.post(
    "/export",
    summary="Submit a GDPR/CCPA data export request for yourself",
    description="Assembles a downloadable ZIP of your personal data across every RAPTOR data store "
                "(GDPR Article 20 / CCPA § 1798.100). Always starts a fresh export job.",
)
async def submit_self_export(
    current_user: dict[str, Any] = Depends(get_current_user),
    client: httpx.AsyncClient = Depends(get_http_client),
    settings: Settings = Depends(get_settings),
):
    user_id = current_user["sub"]
    return await _forward(client, settings, "POST", "/export", headers=_headers(settings, user_id))


@router.post(
    "/export/{user_id}",
    summary="Submit a GDPR/CCPA data export request for another user (gdpr-admin only)",
    description="Same as POST /gdpr/export, but on behalf of any user_id. Requires the `gdpr-admin` Keycloak realm role.",
)
async def submit_admin_export(
    user_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),
    client: httpx.AsyncClient = Depends(get_http_client),
    settings: Settings = Depends(get_settings),
):
    if not _is_gdpr_admin(current_user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Requires the gdpr-admin role")
    return await _forward(client, settings, "POST", "/export", headers=_headers(settings, user_id, requested_by=current_user["sub"]))


@router.get(
    "/export/jobs/{job_id}",
    summary="Get an export job's status and, once completed, its download URL(s)",
)
async def get_export_job(
    job_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),
    client: httpx.AsyncClient = Depends(get_http_client),
    settings: Settings = Depends(get_settings),
):
    job = await _forward(client, settings, "GET", f"/export/jobs/{job_id}", headers={"X-Internal-Token": settings.gdpr_internal_token})
    if job.get("user_id") != current_user["sub"] and not _is_gdpr_admin(current_user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="This export job belongs to a different user")
    return job


# ── GD-3 / VIE01-248: data retention policy engine ──────────────────────────
#
# All three routes require the gdpr-admin role -- unlike erasure/export,
# there's no "self" variant here (a retention policy isn't a per-user
# resource). The ticket's own text says "GET /api/0.5/gdpr/retention/..."
# but every other route in this codebase (including this file's own
# erasure/export routes) is mounted at /api/{RAPTOR_API_VERSION}/gdpr/... ,
# which resolves to 0.4 in this deployment -- kept consistent with that
# rather than hardcoding the ticket's literal (and inconsistent with the
# rest of the ticket set) "0.5".

@router.get(
    "/retention/policy",
    summary="Get the current data retention policy table (gdpr-admin only)",
)
async def get_retention_policy(
    current_user: dict[str, Any] = Depends(get_current_user),
    client: httpx.AsyncClient = Depends(get_http_client),
    settings: Settings = Depends(get_settings),
):
    if not _is_gdpr_admin(current_user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Requires the gdpr-admin role")
    return await _forward(client, settings, "GET", "/retention/policy", headers={"X-Internal-Token": settings.gdpr_internal_token})


@router.put(
    "/retention/policy",
    summary="Update one or more retention policy rows (gdpr-admin only)",
)
async def update_retention_policy(
    updates: list[dict[str, Any]],
    current_user: dict[str, Any] = Depends(get_current_user),
    client: httpx.AsyncClient = Depends(get_http_client),
    settings: Settings = Depends(get_settings),
):
    if not _is_gdpr_admin(current_user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Requires the gdpr-admin role")
    return await _forward(
        client, settings, "PUT", "/retention/policy", json=updates,
        headers={"X-Internal-Token": settings.gdpr_internal_token, "X-Requested-By": current_user["sub"]},
    )


@router.get(
    "/retention/policy/dry-run",
    summary="Count records that WOULD be deleted if enforcement ran now (gdpr-admin only, read-only)",
)
async def dry_run_retention_policy(
    current_user: dict[str, Any] = Depends(get_current_user),
    client: httpx.AsyncClient = Depends(get_http_client),
    settings: Settings = Depends(get_settings),
):
    if not _is_gdpr_admin(current_user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Requires the gdpr-admin role")
    return await _forward(client, settings, "GET", "/retention/policy/dry-run", headers={"X-Internal-Token": settings.gdpr_internal_token})


# ── GD-4 / VIE01-249: data subject access request (right of access) ─────────
#
# A normal account's DSAR is answered synchronously (200, the report
# itself); one with more than GD_DSAR_ASYNC_THRESHOLD indexed items -- or
# whose assembly overruns Module 33's sync budget -- gets 202
# {job_id, status: "pending"} instead, polled via GET /dsar/jobs/{job_id}.
# Module 33's status code is passed through unchanged so a client can tell
# the two apart without inspecting the body. The poll endpoint stays
# JWT-gated with an ownership check, same as export's and for the same
# reason: a DSAR doesn't revoke the requester's session.

async def _dsar(client: httpx.AsyncClient, settings: Settings, headers: dict[str, str]) -> JSONResponse:
    resp = await _forward_response(client, settings, "GET", "/dsar", headers=headers)
    return JSONResponse(status_code=resp.status_code, content=resp.json())


@router.get(
    "/dsar",
    summary="Get a summary of all personal data RAPTOR holds about you",
    description="Structured summary (counts and categories, not content) of your personal data across every "
                "RAPTOR data store (GDPR Article 15 / CCPA § 1798.100). 200 with the report for a normal "
                "account; 202 with {job_id, status: \"pending\"} for an account with more than 10k indexed "
                "items -- poll GET /gdpr/dsar/jobs/{job_id}.",
)
async def get_self_dsar(
    current_user: dict[str, Any] = Depends(get_current_user),
    client: httpx.AsyncClient = Depends(get_http_client),
    settings: Settings = Depends(get_settings),
):
    return await _dsar(client, settings, _headers(settings, current_user["sub"]))


@router.get(
    "/dsar/jobs/{job_id}",
    summary="Get an async DSAR job's status and, once completed, its report",
)
async def get_dsar_job(
    job_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),
    client: httpx.AsyncClient = Depends(get_http_client),
    settings: Settings = Depends(get_settings),
):
    job = await _forward(client, settings, "GET", f"/dsar/jobs/{job_id}", headers={"X-Internal-Token": settings.gdpr_internal_token})
    if job.get("user_id") != current_user["sub"] and not _is_gdpr_admin(current_user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="This DSAR job belongs to a different user")
    return job


@router.get(
    "/dsar/{user_id}",
    summary="Get the personal-data summary for another user (gdpr-admin only)",
    description="Same as GET /gdpr/dsar, but for any user_id. Requires the `gdpr-admin` Keycloak realm role.",
)
async def get_admin_dsar(
    user_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),
    client: httpx.AsyncClient = Depends(get_http_client),
    settings: Settings = Depends(get_settings),
):
    if not _is_gdpr_admin(current_user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Requires the gdpr-admin role")
    return await _dsar(client, settings, _headers(settings, user_id, requested_by=current_user["sub"]))


# ── GD-5 / VIE01-250: consent management ────────────────────────────────────
#
# Self only -- consent is the data subject's own choice, so there's no
# gdpr-admin variant. The caller's X-Forwarded-For (or, with no proxy in
# front, their socket address) goes to Module 33, which stores only its
# SHA-256 in gdpr_consent_history.

class ConsentUpdate(BaseModel):
    granted: bool


def _consent_headers(settings: Settings, request: Request, user_id: str) -> dict[str, str]:
    headers = _headers(settings, user_id)
    forwarded_for = request.headers.get("x-forwarded-for") or (request.client.host if request.client else None)
    if forwarded_for:
        headers["X-Forwarded-For"] = forwarded_for
    return headers


@router.get(
    "/consent",
    summary="Get your consent for each data processing purpose",
    description="memory, analytics, benchmark, voice_recording, face_detection and voice_clone -- each with granted, "
                "granted_at and a description (GDPR Article 7).",
)
async def get_consent(
    current_user: dict[str, Any] = Depends(get_current_user),
    client: httpx.AsyncClient = Depends(get_http_client),
    settings: Settings = Depends(get_settings),
):
    return await _forward(client, settings, "GET", "/consent", headers=_headers(settings, current_user["sub"]))


@router.put(
    "/consent/{purpose}",
    summary="Grant or withdraw your consent for one data processing purpose",
    description="Withdrawing memory, benchmark, voice_recording, face_detection or voice_clone consent also deletes "
                "that purpose's data (a partial erasure -- your account stays); the response carries the erasure "
                "job's id. face_detection: your faces, face groups and named people are deleted, and face detection, "
                "seed photos and People Search stop. voice_clone: your cloned-voice speaker embeddings are deleted, "
                "and voice cloning stops.",
)
async def put_consent(
    purpose: str,
    body: ConsentUpdate,
    request: Request,
    current_user: dict[str, Any] = Depends(get_current_user),
    client: httpx.AsyncClient = Depends(get_http_client),
    settings: Settings = Depends(get_settings),
):
    return await _forward(
        client, settings, "PUT", f"/consent/{purpose}", json=body.model_dump(),
        headers=_consent_headers(settings, request, current_user["sub"]),
    )


# ── GD-7 / VIE01-252: data access audit log ─────────────────────────────────
#
# Records are written by AuditMiddleware (app/middlewares/audit_middleware.py)
# for every request to a personal-data route; Module 33 stores them. The
# admin listing is gdpr-admin only; /audit-log/my is always scoped to the
# caller's own JWT sub, never a caller-chosen id.

@router.get(
    "/audit-log",
    summary="Data access audit log (gdpr-admin only)",
    description="Every recorded access to personal data, newest first. Filters: `user_id` (whose data), "
                "`from`/`to` (ISO 8601). Paginated with `page` / `page_size` (max 1000).",
)
async def get_audit_log(
    request: Request,
    current_user: dict[str, Any] = Depends(get_current_user),
    client: httpx.AsyncClient = Depends(get_http_client),
    settings: Settings = Depends(get_settings),
):
    if not _is_gdpr_admin(current_user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Requires the gdpr-admin role")
    return await _forward(
        client, settings, "GET", "/audit-log", params=dict(request.query_params),
        headers={"X-Internal-Token": settings.gdpr_internal_token},
    )


@router.get(
    "/audit-log/my",
    summary="Who accessed your personal data",
    description="Accesses to your own data, newest first: accessor (shown as a SHA-256 hash when it wasn't you), "
                "endpoint and time. Paginated with `page` / `page_size` (max 1000).",
)
async def get_my_audit_log(
    request: Request,
    current_user: dict[str, Any] = Depends(get_current_user),
    client: httpx.AsyncClient = Depends(get_http_client),
    settings: Settings = Depends(get_settings),
):
    params = {k: v for k, v in request.query_params.items() if k in ("page", "page_size")}
    return await _forward(
        client, settings, "GET", "/audit-log/my", params=params,
        headers=_headers(settings, current_user["sub"]),
    )


# ── V05-10 / VIE01-364: operator view of every GDPR job ─────────────────────
#
# Erasure, export, DSAR and consent-withdrawal jobs in one list -- to find
# partial failures (with the stores still failing) and stuck jobs, instead
# of opening one job id at a time. gdpr-admin only.

@router.get(
    "/jobs",
    summary="List GDPR jobs for operators (gdpr-admin only)",
    description="Erasure, export, DSAR and consent-withdrawal jobs, newest first, each with `failed_stores`. "
                "Filters: `type`, `status` (e.g. `partial_failure`), `user_id`, `from`/`to` (ISO 8601, on "
                "requested_at), `stuck_hours` (only jobs still pending/in progress after that many hours). "
                "Paginated with `page` / `page_size` (max 1000). Retry a failed erasure with "
                "POST /gdpr/erasure/jobs/{job_id}/retry.",
)
async def list_jobs(
    request: Request,
    current_user: dict[str, Any] = Depends(get_current_user),
    client: httpx.AsyncClient = Depends(get_http_client),
    settings: Settings = Depends(get_settings),
):
    if not _is_gdpr_admin(current_user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Requires the gdpr-admin role")
    return await _forward(
        client, settings, "GET", "/jobs", params=dict(request.query_params),
        headers={"X-Internal-Token": settings.gdpr_internal_token},
    )


@router.get(
    "/jobs/summary",
    summary="GDPR job counts by type and status (gdpr-admin only)",
    description="How many jobs of each type are in each status, the total in `partial_failure`, and how many "
                "are stuck (still pending/in progress after `stuck_hours`, default 24).",
)
async def get_jobs_summary(
    request: Request,
    current_user: dict[str, Any] = Depends(get_current_user),
    client: httpx.AsyncClient = Depends(get_http_client),
    settings: Settings = Depends(get_settings),
):
    if not _is_gdpr_admin(current_user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Requires the gdpr-admin role")
    params = {k: v for k, v in request.query_params.items() if k == "stuck_hours"}
    return await _forward(
        client, settings, "GET", "/jobs/summary", params=params,
        headers={"X-Internal-Token": settings.gdpr_internal_token},
    )


# ── PRIV-X3 / VIE01-357: rectification, restriction, objection ─────────────
#
# Rectification (Art. 16): your own Keycloak profile -- email, firstName,
# lastName, one field per request. There is no separate ArcadeDB user
# profile to propagate to: a user's name and email are stored in Keycloak
# only. Restriction (Art. 18): self-service on, gdpr-admin only to lift;
# while on, every write request outside /gdpr/* is refused with 451 (see
# app/middlewares/processing_restriction.py). Objection (Art. 21): stops one
# purpose's future processing (its consent is withdrawn) without deleting
# past data -- unlike PUT /consent/{purpose}=false, which also erases it.

class RectifyBody(BaseModel):
    field: str
    value: str


class RestrictBody(BaseModel):
    reason: str | None = None


class LiftRestrictionBody(BaseModel):
    user_id: str


class ObjectBody(BaseModel):
    processing_purpose: str


@router.post(
    "/rectify",
    summary="Correct your own profile data (GDPR Article 16)",
    description="One field per request: `email`, `firstName` or `lastName` (your Keycloak profile -- the only "
                "place RAPTOR stores them). A changed email is marked unverified. Response: `fields_updated`, "
                "`propagation_status`. An email change is refused (409) while the server has no mail server to "
                "send the verification link; ask an administrator instead.",
)
async def rectify(
    body: RectifyBody,
    current_user: dict[str, Any] = Depends(get_current_user),
    client: httpx.AsyncClient = Depends(get_http_client),
    settings: Settings = Depends(get_settings),
):
    return await _forward(client, settings, "POST", "/rectify", json=body.model_dump(),
                          headers=_headers(settings, current_user["sub"]))


@router.post(
    "/restrict",
    summary="Restrict processing of your data (GDPR Article 18)",
    description="While restricted, every write request you make (uploads, chat, processing, ...) is refused with "
                "451 `{\"reason\": \"processing_restricted\"}`; reads and your GDPR rights (export, erasure, "
                "consent, objection) stay available. Only a gdpr-admin can lift it. Restricting again while "
                "restricted returns the restriction in force.",
)
async def restrict(
    body: RestrictBody | None = None,
    current_user: dict[str, Any] = Depends(get_current_user),
    client: httpx.AsyncClient = Depends(get_http_client),
    settings: Settings = Depends(get_settings),
):
    return await _forward(client, settings, "POST", "/restrict", json=(body or RestrictBody()).model_dump(),
                          headers=_headers(settings, current_user["sub"]))


@router.post(
    "/restrict/lift",
    summary="Lift a user's processing restriction (gdpr-admin only)",
    description="Body: `user_id` (whose restriction). 404 if that user isn't restricted. "
                "Requires the `gdpr-admin` Keycloak realm role.",
)
async def lift_restriction(
    body: LiftRestrictionBody,
    current_user: dict[str, Any] = Depends(get_current_user),
    client: httpx.AsyncClient = Depends(get_http_client),
    settings: Settings = Depends(get_settings),
):
    if not _is_gdpr_admin(current_user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Requires the gdpr-admin role")
    return await _forward(client, settings, "POST", "/restrict/lift", json=body.model_dump(),
                          headers={"X-Internal-Token": settings.gdpr_internal_token,
                                   "X-Requested-By": current_user["sub"]})


@router.post(
    "/object",
    summary="Object to one processing purpose (GDPR Article 21)",
    description="`processing_purpose`: memory, analytics, benchmark, voice_recording, face_detection or voice_clone. "
                "Stops future processing "
                "for that purpose (its consent is withdrawn) but does not delete past data -- request an "
                "erasure for that. Response: `purposes_stopped`, `data_deleted: false`.",
)
async def object_to_processing(
    body: ObjectBody,
    request: Request,
    current_user: dict[str, Any] = Depends(get_current_user),
    client: httpx.AsyncClient = Depends(get_http_client),
    settings: Settings = Depends(get_settings),
):
    return await _forward(client, settings, "POST", "/object", json=body.model_dump(),
                          headers=_consent_headers(settings, request, current_user["sub"]))


# ── PRIV-X4 / VIE01-358: legal holds and erasure exceptions ────────────────
#
# A gdpr-admin places a legal hold (GDPR Art. 17(3)) on a user. While it is
# active, POST /erasure answers {"status": "DEFERRED", "reason": "legal_hold",
# "hold_until": ...} and records the request; Module 33's daily expiry job
# runs it once the hold ends. Every erasure response lists in
# exception_fields the data kept by law, with its legal basis.

class LegalHoldBody(BaseModel):
    user_id: str
    reason: str
    hold_until: date
    authority: str


@router.post(
    "/admin/legal-hold",
    summary="Place a legal hold on a user's erasure (gdpr-admin only)",
    description="Body: `user_id`, `reason` (e.g. `regulatory_audit`), `hold_until` (a future date; the hold "
                "ends at its start, UTC) and `authority` (e.g. `DPO`). Until then an erasure request for that "
                "user is deferred, not carried out. 409 if the user already has an open hold, is being erased "
                "or was erased. Requires the `gdpr-admin` Keycloak realm role.",
)
async def place_legal_hold(
    body: LegalHoldBody,
    current_user: dict[str, Any] = Depends(get_current_user),
    client: httpx.AsyncClient = Depends(get_http_client),
    settings: Settings = Depends(get_settings),
):
    if not _is_gdpr_admin(current_user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Requires the gdpr-admin role")
    return await _forward(client, settings, "POST", "/admin/legal-hold", json=body.model_dump(mode="json"),
                          headers={"X-Internal-Token": settings.gdpr_internal_token,
                                   "X-Requested-By": current_user["sub"]})


@router.get(
    "/admin/legal-holds",
    summary="List active legal holds (gdpr-admin only)",
    description="Every legal hold in force now, the soonest to end first. Requires the `gdpr-admin` Keycloak "
                "realm role.",
)
async def list_legal_holds(
    current_user: dict[str, Any] = Depends(get_current_user),
    client: httpx.AsyncClient = Depends(get_http_client),
    settings: Settings = Depends(get_settings),
):
    if not _is_gdpr_admin(current_user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Requires the gdpr-admin role")
    return await _forward(client, settings, "GET", "/admin/legal-holds",
                          headers={"X-Internal-Token": settings.gdpr_internal_token})

# ── GD-9 / VIE01-xxx: Article 30 Record of Processing Activities (RoPA) ─────
#
# Read-only registry of all processing activities -- gdpr-admin only,
# no "self" variant (a RoPA is a controller-level record, not per-user).
# CCPA equivalent is a separate endpoint with sells_data: false hardcoded.

@router.get(
    "/processing-records",
    summary="Get the Article 30 Record of Processing Activities (gdpr-admin only)",
    description="Returns the controller's full RoPA as required by GDPR Article 30. "
                "Requires the `gdpr-admin` Keycloak realm role.",
)
async def get_processing_records(
    current_user: dict[str, Any] = Depends(get_current_user),
    client: httpx.AsyncClient = Depends(get_http_client),
    settings: Settings = Depends(get_settings),
):
    if not _is_gdpr_admin(current_user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Requires the gdpr-admin role")
    return await _forward(client, settings, "GET", "/processing-records",
                          headers={"X-Internal-Token": settings.gdpr_internal_token})


@router.get(
    "/processing-records/ccpa",
    summary="Get the CCPA equivalent of the processing records (gdpr-admin only)",
    description="Same as GET /gdpr/processing-records but shaped for CCPA: sells_data is always false. "
                "Requires the `gdpr-admin` Keycloak realm role.",
)
async def get_processing_records_ccpa(
    current_user: dict[str, Any] = Depends(get_current_user),
    client: httpx.AsyncClient = Depends(get_http_client),
    settings: Settings = Depends(get_settings),
):
    if not _is_gdpr_admin(current_user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Requires the gdpr-admin role")
    return await _forward(client, settings, "GET", "/processing-records/ccpa",
                          headers={"X-Internal-Token": settings.gdpr_internal_token})

