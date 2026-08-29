"""Pipeline executor (BM-4).

Calls any supported Raptor pipeline with a test-case input and returns a
normalized ``{output, latency_ms, raw_response}`` result. Downstream services
are called directly (no auth overhead). HTTP/timeout errors never raise — they
are returned as ``{output: "", latency_ms, error}`` so a single bad case does
not abort a whole run (consumer-safe).
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from app.core.config import get_settings

logger = logging.getLogger(__name__)


def _pipeline_registry() -> Dict[str, Dict[str, str]]:
    s = get_settings()
    return {
        "chat": {"base": s.chat_url, "path": "/api/v1/chat"},
        "search": {"base": s.search_url, "path": "/personal/search/hybrid"},
        "rag": {"base": s.rag_url, "path": "/query"},
        "classify": {"base": s.classify_url, "path": "/classify"},
        # Serve a (fine-tuned) local checkpoint via Module 16. model_path comes
        # from the run's config_override, so the same schema can score different
        # models — the core train → serve → score loop.
        "local_infer": {"base": s.local_infer_url, "path": "/api/v1/inference/infer"},
        # Benchmark a model registered in Module 07 (AI Lifecycle API). model_name
        # / engine come from the run's config_override, so the same schema can
        # score different registered models against the same rubric.
        "lifecycle_infer": {"base": s.lifecycle_infer_url, "path": "/inference/infer"},
    }


def _prompt_text(data: Dict[str, Any]) -> str:
    """Best-effort extraction of the prompt string from a test-case input."""
    return (data.get("inputs") or data.get("message") or data.get("question")
            or data.get("query") or "")


def _build_body(pipeline: str, data: Dict[str, Any],
                config_override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Map a test-case input dict to the pipeline's request body."""
    if pipeline == "chat":
        body = {"user_id": data.get("user_id", "benchmark-user"), "message": data.get("message", "")}
        if data.get("session_id"):
            body["session_id"] = data["session_id"]
        return body
    if pipeline == "search":
        return {"query": data.get("query", ""), "top_k": data.get("top_k", 5)}
    if pipeline == "rag":
        return {"question": data.get("question", data.get("message", "")), "mode": "direct"}
    if pipeline == "classify":
        return {"query": data.get("query", data.get("message", ""))}
    if pipeline == "local_infer":
        cfg = config_override or {}
        body: Dict[str, Any] = {
            "model_path": cfg.get("model_path") or data.get("model_path"),
            "inputs": _prompt_text(data),
        }
        for key in ("max_new_tokens", "temperature"):
            if key in cfg:
                body[key] = cfg[key]
        return body
    if pipeline == "lifecycle_infer":
        s = get_settings()
        cfg = config_override or {}
        return {
            "task": "text-generation",
            "engine": cfg.get("engine", s.lifecycle_infer_engine),
            "model_name": cfg.get("model_name") or data.get("model_name"),
            "data": {"inputs": _prompt_text(data)},
            "options": {
                "temperature": cfg.get("temperature", s.lifecycle_infer_temperature),
                # accept either max_length or max_new_tokens from the override
                "max_length": cfg.get("max_length", cfg.get("max_new_tokens", s.lifecycle_infer_max_length)),
                "think": cfg.get("think", s.lifecycle_infer_think),
            },
        }
    raise ValueError(f"unsupported pipeline: {pipeline}")


def _build_headers(pipeline: str, data: Dict[str, Any],
                   run_branch_id: Optional[str] = None) -> Optional[Dict[str, str]]:
    """Extra headers search/rag need beyond the JSON body -- both ultimately hit
    Module 25's per-branch ArcadeDB, so both need a branch_id. Precedence: a
    test case's own input.branch_id/input.user_id wins, else run_branch_id
    (the run submitter's own sub, injected by Module 13 -- see
    routers/benchmark.py). Errors rather than silently searching empty/wrong
    data: rag in particular degrades silently (Module 21's /query treats a
    missing X-Branch-ID as "no results", not an error) and answers as if
    nothing was found -- an ungrounded answer that still looks like a normal
    completed run.
    """
    if pipeline not in ("search", "rag"):
        return None
    branch_id = data.get("branch_id") or data.get("user_id") or run_branch_id
    if not branch_id:
        raise ValueError(
            f"target_pipeline={pipeline} has no branch_id to scope the search to -- "
            "either the test case sets input.branch_id/input.user_id, or the "
            "run must be submitted by an authenticated caller (Module 13 "
            "injects the submitter's branch_id automatically)"
        )
    if pipeline == "rag":
        return {"X-User-ID": branch_id, "X-Branch-ID": branch_id}
    return {"X-Branch-ID": branch_id}


def _extract_output(pipeline: str, payload: Any) -> str:
    """Best-effort extraction of a comparable text output from a response."""
    if isinstance(payload, str):
        return payload
    if not isinstance(payload, dict):
        return str(payload)

    if pipeline == "lifecycle_infer":
        # Module 07 shape: {"result": {"response": "...", "metadata": {...}}, ...}.
        # Must run before the generic key loop, which would otherwise stringify
        # the whole `result` dict.
        result = payload.get("result")
        if isinstance(result, dict):
            for key in ("response", "text", "generated_text", "output"):
                if result.get(key):
                    return str(result[key])
        if isinstance(result, str):
            return result

    if pipeline == "search":
        results = payload.get("results") or payload.get("hits") or payload.get("documents")
        if isinstance(results, list):
            parts = []
            for r in results:
                if isinstance(r, dict):
                    parts.append(str(r.get("text") or r.get("content") or r.get("title") or r))
                else:
                    parts.append(str(r))
            return "\n".join(parts)

    for key in ("output", "answer", "response", "message", "content", "result",
                "intent", "pipeline", "label", "text"):
        if key in payload and payload[key] is not None:
            value = payload[key]
            return value if isinstance(value, str) else str(value)

    import json

    return json.dumps(payload, ensure_ascii=False)


async def _timed_post(url: str, body: Dict[str, Any], timeout: float,
                      headers: Optional[Dict[str, str]] = None
                      ) -> Tuple[Optional[Any], float, Optional[str]]:
    """POST json → (parsed_body_or_None, latency_ms, error_str_or_None).

    Never raises: a transport/HTTP error comes back as the error string so callers
    stay consumer-safe (a single bad case never aborts a whole run).
    """
    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=body, headers=headers)
        latency_ms = (time.perf_counter() - start) * 1000.0
        resp.raise_for_status()
        return (resp.json() if resp.content else {}), latency_ms, None
    except Exception as exc:  # noqa: BLE001 — consumer-safe by design
        return None, (time.perf_counter() - start) * 1000.0, str(exc)


async def call_pipeline(
    pipeline: str,
    data: Dict[str, Any],
    target_url: Optional[str] = None,
    config_override: Optional[Dict[str, Any]] = None,
    run_branch_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Call one pipeline and return {output, latency_ms, raw_response[, error]}."""
    registry = _pipeline_registry()
    if pipeline not in registry:
        return {"output": "", "latency_ms": 0.0, "raw_response": {},
                "error": f"unsupported pipeline: {pipeline}"}

    settings = get_settings()
    base = target_url or registry[pipeline]["base"]
    url = base.rstrip("/") + registry[pipeline]["path"]
    body = _build_body(pipeline, data, config_override)
    # Model-serving pipelines can be slow on the first request (model load onto GPU).
    if pipeline == "local_infer":
        timeout = settings.local_infer_timeout
    elif pipeline == "lifecycle_infer":
        timeout = settings.lifecycle_infer_timeout
    else:
        timeout = settings.pipeline_timeout

    try:
        headers = _build_headers(pipeline, data, run_branch_id)
    except ValueError as exc:
        return {"output": "", "latency_ms": 0.0, "raw_response": {}, "error": str(exc)}

    raw, latency_ms, error = await _timed_post(url, body, timeout, headers)
    if error is not None:
        logger.warning("Pipeline %s call failed: %s", pipeline, error)
        return {"output": "", "latency_ms": latency_ms, "raw_response": {}, "error": error}
    return {"output": _extract_output(pipeline, raw), "latency_ms": latency_ms, "raw_response": raw}


async def infer_batch_local(
    inputs: List[Dict[str, Any]],
    target_url: Optional[str] = None,
    config_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Batch local_infer: one request generates outputs for all test-case inputs.

    Returns {outputs: List[str], latency_ms[, error]}. On any failure returns an
    empty ``outputs`` list so the caller can fall back to per-case inference.
    """
    settings = get_settings()
    cfg = config_override or {}
    base = target_url or settings.local_infer_url
    url = base.rstrip("/") + "/api/v1/inference/infer_batch"
    body: Dict[str, Any] = {
        "model_path": cfg.get("model_path"),
        "inputs": [_prompt_text(d) for d in inputs],
    }
    for key in ("max_new_tokens", "temperature", "batch_size"):
        if key in cfg:
            body[key] = cfg[key]

    raw, latency_ms, error = await _timed_post(url, body, settings.local_infer_timeout)
    if error is not None:
        logger.warning("Batch local_infer failed (%s) — caller will fall back", error)
        return {"outputs": [], "latency_ms": latency_ms, "error": error}
    return {"outputs": [str(o) for o in (raw.get("outputs") or [])], "latency_ms": latency_ms}


async def unload_local_infer(target_url: Optional[str] = None,
                             model_path: Optional[str] = None) -> None:
    """Best-effort: ask Module 16 to unload the resident model and free VRAM.

    Called after a local_infer run finishes so the checkpoint does not sit on the
    GPU between benchmark/optimization iterations. Never raises — freeing memory
    is opportunistic, and a failure must not fail the run.
    """
    settings = get_settings()
    base = target_url or settings.local_infer_url
    url = base.rstrip("/") + "/api/v1/inference/unload"
    try:
        async with httpx.AsyncClient(timeout=settings.pipeline_timeout) as client:
            await client.post(url, json={"model_path": model_path})
        logger.info("Requested local_infer model unload (%s)", model_path)
    except Exception as exc:  # noqa: BLE001 — best-effort
        logger.warning("local_infer unload failed: %s", exc)
