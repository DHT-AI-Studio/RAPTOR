"""POST /inference/text/{translate,summarize,complete,transform} and the /inference/text dispatcher."""
from __future__ import annotations

import time
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import StreamingResponse as _StreamingResponse
from pydantic import ValidationError

from app.errors import ServiceError
from app.metrics import inf_request_duration_seconds, inf_request_errors_total
from app.schemas import (
    CompleteRequest,
    SummarizeRequest,
    TransformRequest,
    TranslateRequest,
)
from app.services.text_inference import TextInference

router = APIRouter(prefix="/inference/text", tags=["text inference"])

_TASK_MODELS = {
    "translate": TranslateRequest,
    "summarize": SummarizeRequest,
    "complete": CompleteRequest,
    "transform": TransformRequest,
}


def get_text_inference(request: Request) -> TextInference:
    return request.app.state.text_inference

async def _run_text_task(task: str, model: str | None, coro) -> Any:
    """Execute a text inference coroutine and record duration / error metrics.

    Parameters
    ----------
    task:  one of translate / summarize / complete / transform
    model: resolved model name (may be None before resolution; stored as-is)
    coro:  awaitable returned by the TextInference method
    """
    label_model = model or "default"
    t0 = time.perf_counter()
    try:
        result = await coro
        inf_request_duration_seconds.labels(task=task, model=label_model).observe(
            time.perf_counter() - t0
        )
        return result
    except ServiceError as exc:
        inf_request_duration_seconds.labels(task=task, model=label_model).observe(
            time.perf_counter() - t0
        )
        inf_request_errors_total.labels(task=task, error_type=exc.error).inc()
        raise
    except Exception:
        inf_request_duration_seconds.labels(task=task, model=label_model).observe(
            time.perf_counter() - t0
        )
        inf_request_errors_total.labels(task=task, error_type="unknown").inc()
        raise

@router.post("/translate", summary="Translate text (INF-5)")
async def translate(body: TranslateRequest, svc: Annotated[TextInference, Depends(get_text_inference)]):
    return await _run_text_task("translate", body.model, svc.translate(body.model, body.input, body.options))


@router.post("/summarize", summary="Summarize text (INF-6)")
async def summarize(body: SummarizeRequest, svc: Annotated[TextInference, Depends(get_text_inference)]):
    return await _run_text_task("summarize", body.model, svc.summarize(body.model, body.input, body.options))


@router.post("/complete", summary="Text completion / generation (INF-7)")
async def complete(body: CompleteRequest, svc: Annotated[TextInference, Depends(get_text_inference)]):
    return await _run_text_task("complete", body.model, svc.complete(body.model, body.input, body.options))


@router.post(
    "/complete/stream",
    summary="SSE streaming text completion (INF-18)",
    response_class=_StreamingResponse,
    responses={200: {"content": {"text/event-stream": {}}, "description": "SSE token stream"}},
)
async def complete_stream(
    body: CompleteRequest,
    svc: Annotated[TextInference, Depends(get_text_inference)],
):
    """Stream text completion tokens via Server-Sent Events.

    Response: text/event-stream
    Each token:  data: {"token": "<tok>", "finish_reason": null}
    Final frame: data: {"token": "", "finish_reason": "stop|length"}
    Sentinel:    data: [DONE]
    Error frame: data: {"error": "upstream_error", "message": "..."}
    """
    # All validation + guardrail runs inside stream_complete() *before* the
    # generator is returned, so ServiceError still propagates as a normal
    # HTTP error response (not inside the SSE stream).
    generator = await svc.stream_complete(body.model, body.input, body.options)
    return _StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

@router.post("/transform", summary="Rewrite / transform text (INF-8)")
async def transform(body: TransformRequest, svc: Annotated[TextInference, Depends(get_text_inference)]):
    return await _run_text_task("transform", body.model, svc.transform(body.model, body.input, body.options))


@router.post("", summary="Generic text dispatcher: routes on the `task` field")
async def dispatch(
    payload: Annotated[
        dict[str, Any],
        Body(examples=[{"task": "translate", "input": "早安", "options": {"target_language": "en"}}]),
    ],
    svc: Annotated[TextInference, Depends(get_text_inference)],
):
    """Body = ``{"task": "translate|summarize|complete|transform", ...}`` plus that task's own
    fields; validated exactly like the matching typed route."""
    task = payload.get("task")
    if task not in _TASK_MODELS:
        raise ServiceError(
            400, "unsupported_task", f"task must be one of {list(_TASK_MODELS)}",
            task=task, supported=list(_TASK_MODELS),
        )
    fields = {k: v for k, v in payload.items() if k != "task"}
    try:
        body = _TASK_MODELS[task].model_validate(fields)
    except ValidationError as exc:
        raise RequestValidationError(
            [{**e, "loc": ("body", *e["loc"])} for e in exc.errors()]
        ) from exc
    return await _run_text_task(task, body.model, getattr(svc, task)(body.model, body.input, body.options))
