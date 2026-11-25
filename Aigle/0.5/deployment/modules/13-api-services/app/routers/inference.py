"""Authenticated proxy for Module 35 (inference-service). Mounted at /api/0.5/inference
(app/main.py), so every route path below is relative to that prefix.

One declared route per Module 35 endpoint (17; `/internal/gdpr/*` is Module 33's own direct,
internal-token-gated path to Module 35 and was never reachable through this public, JWT-gated
proxy either way), mirroring every other module's own router in this package (guardrail.py,
search.py, training.py, ...) -- this one used to be the lone exception, a six-verb generic
reverse proxy (`/{path:path}`) whose own OpenAPI showed nothing but "Proxy Inference" six times.
The tradeoff of individually declaring routes, same as every sibling router already accepts: a
route Module 35 adds has to be added here too, by hand -- there is no way around that and still
have this gateway's own Swagger describe the real request/response shapes.

A route's *external* path here is not always its *upstream* Module 35 path: Module 35's text
and voice routers are themselves mounted under its own `/inference/...` prefix, its models/health
routes are not (`forward()`'s `upstream_path` argument carries the real one in each case).

Multipart routes (diarize/stt/tts/clone) are declared with a bare ``Request``, not typed
``UploadFile``/``Form(...)`` params -- same reasoning as guardrail.py's ``upload_policy_proxy``:
FastAPI would otherwise drain the multipart body stream before the handler runs, leaving nothing
for ``request.body()`` to forward (``RuntimeError: Stream consumed``). Their request shape is
still fully documented via ``openapi_extra``.

JSON-body text routes forward the raw, already-cached ``request.body()`` rather than
re-serializing a locally re-validated Pydantic model -- declaring (and therefore owning) a second
copy of Module 35's own request validation here would drift the moment either side's rules
change without the other noticing; ``openapi_extra`` documents the real shape without enforcing
it a second time. Module 35 remains the single source of truth for what is actually valid.

A catch-all fallback stays at the bottom, methods=* on both `""` and `/{path:path}` (mirroring
this router's own old shape) -- not a second, parallel API surface (`include_in_schema=False`),
only so a path not yet mirrored above (a brand-new Module 35 endpoint, or a genuine typo) still
gets a verdict *after* authentication rather than a bare 404 before auth ever runs.
"""
from __future__ import annotations

from typing import Annotated
from urllib.parse import quote

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.api.dependencies import get_current_user
from app.core.config import get_settings

router = APIRouter(tags=["Inference"], dependencies=[Depends(get_current_user)])
_HOP_HEADERS = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "transfer-encoding", "upgrade", "host", "content-length",
}


def forwarding_headers(headers) -> dict[str, str]:
    """Remove hop headers, including custom names in the Connection header."""
    drop = _HOP_HEADERS | {h.strip().lower() for h in headers.get("connection", "").split(",")}
    return {k: v for k, v in headers.items() if k.lower() not in drop}


async def forward(request: Request, current_user: dict, upstream_path: str) -> StreamingResponse:
    """Preserve bytes/query/status/stream; derive user identity from the verified JWT.
    ``upstream_path`` is this call's fixed, already-quoted Module 35 path."""
    sub = current_user["sub"]
    if request.headers.get("x-user-id", sub) != sub:
        raise HTTPException(403, "Cross-user access denied")
    settings = get_settings()
    headers = forwarding_headers(request.headers)
    headers["x-user-id"] = sub
    client = httpx.AsyncClient(timeout=settings.upload_timeout)
    try:
        outbound = client.build_request(
            request.method, settings.inference_service_url.rstrip("/") + upstream_path,
            headers=headers, params=request.query_params.multi_items(), content=await request.body(),
        )
        response = await client.send(outbound, stream=True)
    except httpx.RequestError as exc:
        await client.aclose()
        raise HTTPException(502, "Inference service unavailable") from exc
    except BaseException:
        await client.aclose()
        raise

    async def chunks():
        try:
            async for chunk in response.aiter_raw():
                yield chunk
        finally:
            await response.aclose()
            await client.aclose()

    return StreamingResponse(
        chunks(), status_code=response.status_code, headers=forwarding_headers(response.headers)
    )


# --- Text inference (INF-5..8) -----------------------------------------------------------

_TEXT_BODY_EXAMPLES = {
    "translate": {"input": "早安", "options": {"target_language": "en", "source_language": "zh-TW"}},
    "summarize": {"input": "一段很長的文字……", "options": {"style": "bullet", "max_sentences": 3}},
    "complete": {"input": "寫一首關於秋天的短詩", "options": {"max_tokens": 256, "temperature": 0.7}},
    "transform": {"input": "原始文字", "options": {"instruction": "改寫成正式語氣"}},
}


class TextTaskResponse(BaseModel):
    """Mirrors Module 35's actual response shape -- for Swagger typing only, see module docstring."""
    task: str
    model: str
    model_version: str | None = None
    output: str
    usage: dict | None = None
    latency_ms: int


def _text_route(task: str, route_path: str, summary: str):
    example = _TEXT_BODY_EXAMPLES[task]
    upstream_path = f"/inference{route_path}"

    @router.post(
        route_path, summary=summary, response_model=TextTaskResponse, name=f"text_{task}",
        openapi_extra={"requestBody": {"required": True, "content": {
            "application/json": {"example": {"model": None, **example}},
        }}},
    )
    async def _proxy(request: Request, current_user: Annotated[dict, Depends(get_current_user)]):
        return await forward(request, current_user, upstream_path)

    return _proxy


translate_proxy = _text_route("translate", "/text/translate", "Translate text (INF-5)")
summarize_proxy = _text_route("summarize", "/text/summarize", "Summarize text (INF-6)")
complete_proxy = _text_route("complete", "/text/complete", "Text completion / generation (INF-7)")
transform_proxy = _text_route("transform", "/text/transform", "Rewrite / transform text (INF-8)")


@router.post(
    "/text/complete/stream",
    summary="SSE streaming text completion (INF-18)",
    responses={200: {"content": {"text/event-stream": {}}, "description": "SSE token stream"}},
    openapi_extra={"requestBody": {"required": True, "content": {
        "application/json": {"example": {"model": None, **_TEXT_BODY_EXAMPLES["complete"]}},
    }}},
)
async def complete_stream_proxy(request: Request, current_user: Annotated[dict, Depends(get_current_user)]):
    """``data: {"token": "...", "finish_reason": null}`` per token, final frame carries
    ``finish_reason``, sentinel ``data: [DONE]`` -- see Module 35's own API.md for the exact
    frame shapes; this gateway does not re-parse the stream, only forwards it."""
    return await forward(request, current_user, "/inference/text/complete/stream")


@router.post(
    "/text",
    summary="Generic text dispatcher: routes on the `task` field",
    response_model=TextTaskResponse,
    openapi_extra={"requestBody": {"required": True, "content": {
        "application/json": {"example": {"task": "translate", **_TEXT_BODY_EXAMPLES["translate"]}},
    }}},
)
async def text_dispatch_proxy(request: Request, current_user: Annotated[dict, Depends(get_current_user)]):
    """Body = ``{"task": "translate|summarize|complete|transform", ...}`` plus that task's own
    fields, exactly as the matching typed route above."""
    return await forward(request, current_user, "/inference/text")


# --- Voice inference (INF-9..11, 13) -----------------------------------------------------

class DiarizeSegment(BaseModel):
    speaker: str
    text: str
    start: float
    end: float


class DiarizeResponse(BaseModel):
    speakers: list[str]
    segments: list[DiarizeSegment]
    latency_ms: float


_AUDIO_FILE_SCHEMA = {"type": "string", "format": "binary", "description": "WAV / MP3 / FLAC"}


def _multipart_extra(required: list[str], properties: dict) -> dict:
    """``openapi_extra`` for a multipart/form-data body -- built, not hand-nested, so the
    bracket depth can't be miscounted."""
    schema = {"type": "object", "required": required, "properties": properties}
    return {"requestBody": {"required": True, "content": {"multipart/form-data": {"schema": schema}}}}


@router.post(
    "/voice/diarize",
    summary="Speaker diarization with WhisperX transcription (INF-9)",
    response_model=DiarizeResponse,
    openapi_extra=_multipart_extra(["audio"], {
        "audio": _AUDIO_FILE_SCHEMA,
        "model": {"type": "string", "default": "pyannote-3.1"},
        "num_speakers": {"type": "integer", "minimum": 1, "maximum": 64},
        "language": {"type": "string", "description": "BCP-47; omit to auto-detect"},
    }),
)
async def diarize_proxy(request: Request, current_user: Annotated[dict, Depends(get_current_user)]):
    return await forward(request, current_user, "/inference/voice/diarize")


class SttResponse(BaseModel):
    text: str
    language: str | None = None
    language_probability: float | None = None
    confidence: float | None = None


@router.post(
    "/voice/stt",
    summary="Speech-to-text (INF-10) -- forwards to Module 34's loaded Whisper model",
    response_model=SttResponse,
    openapi_extra=_multipart_extra(["audio"], {
        "audio": _AUDIO_FILE_SCHEMA,
        "language": {"type": "string", "description": "BCP-47; omit to auto-detect"},
    }),
)
async def stt_proxy(request: Request, current_user: Annotated[dict, Depends(get_current_user)]):
    return await forward(request, current_user, "/inference/voice/stt")


@router.post(
    "/voice/tts",
    summary="Text-to-speech (INF-11) -- forwards to Module 34's loaded TTS model",
    responses={200: {"content": {"audio/wav": {}}, "description": "Synthesized WAV audio"}},
    openapi_extra=_multipart_extra(["text"], {
        "text": {"type": "string"}, "language": {"type": "string", "description": "BCP-47"},
    }),
)
async def tts_proxy(request: Request, current_user: Annotated[dict, Depends(get_current_user)]):
    return await forward(request, current_user, "/inference/voice/tts")


@router.post(
    "/voice/clone",
    summary="Voice clone synthesis — XTTS-v2 / Chatterbox Multilingual V3 (INF-13)",
    responses={200: {"content": {"audio/wav": {}},
                     "description": "Synthesized WAV; X-Speaker-Id / X-Watermarked response headers"}},
    openapi_extra=_multipart_extra(["reference_audio", "text", "language"], {
        "reference_audio": {"type": "string", "format": "binary", "description": "WAV / MP3 / FLAC; 1-60 s"},
        "text": {"type": "string", "minLength": 1, "maxLength": 3000},
        "language": {"type": "string", "description": "BCP-47"},
        "speaker_name": {"type": "string", "description": "1-128 chars; reuse to skip re-encoding"},
        "model": {"type": "string", "description": "xtts-v2 or chatterbox-multilingual-v3; omit for the deployment default"},
    }),
)
async def clone_proxy(request: Request, current_user: Annotated[dict, Depends(get_current_user)]):
    """GD-11: requires the caller's `voice_clone` GDPR consent (checked by Module 35 against
    Module 33) -- `403 consent_required` without it."""
    return await forward(request, current_user, "/inference/voice/clone")


class SpeakerEntry(BaseModel):
    id: str
    ttl_remaining_s: int


class SpeakersResponse(BaseModel):
    speakers: list[SpeakerEntry]


@router.get("/voice/clone/speakers", summary="List caller's cached speaker embeddings",
            response_model=SpeakersResponse)
async def list_speakers_proxy(request: Request, current_user: Annotated[dict, Depends(get_current_user)]):
    return await forward(request, current_user, "/inference/voice/clone/speakers")


@router.delete("/voice/clone/speakers/{speaker_id}", status_code=204,
               summary="Evict a caller-owned speaker embedding")
async def delete_speaker_proxy(
    speaker_id: str, request: Request, current_user: Annotated[dict, Depends(get_current_user)],
):
    return await forward(request, current_user, f"/inference/voice/clone/speakers/{quote(speaker_id, safe='')}")


# --- Models (INF-2/3) ----------------------------------------------------------------------
# Module 35's models.py router is mounted at its own root (/models), not under its /inference
# prefix -- upstream paths below match that, not the /inference/... shape text/voice routes use.

@router.get("/models", summary="List models from Module 07 MLflow + Ollama (cached)")
async def list_models_proxy(request: Request, current_user: Annotated[dict, Depends(get_current_user)]):
    return await forward(request, current_user, "/models")


@router.get("/models/default", summary="Current default model per task")
async def get_defaults_proxy(request: Request, current_user: Annotated[dict, Depends(get_current_user)]):
    return await forward(request, current_user, "/models/default")


@router.post("/models/default", status_code=204, summary="Set the default model for a task (persisted in Redis)",
             openapi_extra={"requestBody": {"required": True, "content": {
                 "application/json": {"example": {"task": "text", "model": "gemma-prod"}},
             }}})
async def set_default_proxy(request: Request, current_user: Annotated[dict, Depends(get_current_user)]):
    return await forward(request, current_user, "/models/default")


@router.get("/models/{name:path}", summary="One model's registry entry")
async def get_model_proxy(name: str, request: Request, current_user: Annotated[dict, Depends(get_current_user)]):
    # A decoded traversal must never escape the fixed upstream surface.
    if any(part in {".", ".."} for part in name.split("/")) or "\\" in name:
        raise HTTPException(400, "Invalid model name")
    return await forward(request, current_user, f"/models/{quote(name, safe='/')}")


# --- System ----------------------------------------------------------------------------------

@router.get("/health", summary="Module 35 health")
async def health_proxy(request: Request, current_user: Annotated[dict, Depends(get_current_user)]):
    return await forward(request, current_user, "/health")


# --- Fallback: anything not mirrored above yet -------------------------------------------

async def unmirrored_proxy(
    request: Request, current_user: Annotated[dict, Depends(get_current_user)], path: str = "",
):
    """Not a second, undocumented way to reach Module 35 -- the 17 routes above are the real
    interface. This exists only so a path this router hasn't been taught yet (a brand-new
    Module 35 endpoint not mirrored here, or a genuine typo) still gets a verdict *after*
    authentication, matching every declared route above, rather than FastAPI 404ing on an
    unmatched path before auth ever runs -- which is the property
    test_all_paths_deny_without_bearer_before_http pins for every path, including unknown
    ones. Never shown in Swagger (`include_in_schema=False` on both registrations below):
    this must never read as "a second API surface" -- Module 35's new endpoints belong
    mirrored above, not left to fall through here."""
    if any(part in {".", ".."} for part in path.split("/")) or "\\" in path:
        raise HTTPException(400, "Invalid inference path")
    return await forward(request, current_user, f"/inference/{quote(path, safe='/')}")


for _path in ("", "/{path:path}"):
    router.add_api_route(
        _path, unmirrored_proxy,
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
        include_in_schema=False,
    )
