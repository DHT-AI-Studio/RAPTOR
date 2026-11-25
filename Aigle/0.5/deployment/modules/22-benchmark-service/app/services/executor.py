"""Pipeline executor (BM-4).

Calls any supported Raptor pipeline with a test-case input and returns a
normalized ``{output, latency_ms, raw_response}`` result. Downstream services
are called directly (no auth overhead). HTTP/timeout errors never raise — they
are returned as ``{output: "", latency_ms, error}`` so a single bad case does
not abort a whole run (consumer-safe).
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

import httpx

from app.core.config import get_settings
from app.services import moderation

logger = logging.getLogger(__name__)


def _pipeline_registry() -> dict[str, dict[str, str]]:
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
        # Moderation model evaluation (CM-6) against Module 23. The text path
        # asks the guard models alone (no administrator policy), so the run
        # measures the models, not a policy. guardrail_image's path depends on
        # config_override.model (see _guardrail).
        "guardrail": {"base": s.guardrail_url, "path": "/guard/check/input"},
        "guardrail_image": {"base": s.guardrail_url, "path": ""},
    }


def _prompt_text(data: dict[str, Any]) -> str:
    """Best-effort extraction of the prompt string from a test-case input."""
    return (data.get("inputs") or data.get("message") or data.get("question")
            or data.get("query") or "")


def _build_body(pipeline: str, data: dict[str, Any],
                config_override: dict[str, Any] | None = None) -> dict[str, Any]:
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
        body: dict[str, Any] = {
            "model_path": cfg.get("model_path") or data.get("model_path"),
            "inputs": _prompt_text(data),
        }
        for key in ("max_new_tokens", "temperature"):
            if key in cfg:
                body[key] = cfg[key]
        return body
    if pipeline == "guardrail":
        return {"content": data.get("text", data.get("message", "")), "module": "benchmark"}
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


def _build_headers(pipeline: str, data: dict[str, Any],
                   run_branch_id: str | None = None) -> dict[str, str] | None:
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


async def _timed_post(url: str, body: dict[str, Any] | None, timeout: float,
                      headers: dict[str, str] | None = None,
                      files: dict[str, Any] | None = None,
                      ) -> tuple[Any | None, float, str | None]:
    """POST json (or multipart ``files``) → (parsed_body_or_None, latency_ms, error_str_or_None).

    Never raises: a transport/HTTP error comes back as the error string so callers
    stay consumer-safe (a single bad case never aborts a whole run).
    """
    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            if files is not None:
                resp = await client.post(url, files=files, data=body, headers=headers)
            else:
                resp = await client.post(url, json=body, headers=headers)
        latency_ms = (time.perf_counter() - start) * 1000.0
        resp.raise_for_status()
        return (resp.json() if resp.content else {}), latency_ms, None
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        return None, (time.perf_counter() - start) * 1000.0, str(exc)


async def call_pipeline(
    pipeline: str,
    data: dict[str, Any],
    target_url: str | None = None,
    config_override: dict[str, Any] | None = None,
    run_branch_id: str | None = None,
) -> dict[str, Any]:
    """Call one pipeline and return {output, latency_ms, raw_response[, error]}."""
    registry = _pipeline_registry()
    if pipeline not in registry:
        return {"output": "", "latency_ms": 0.0, "raw_response": {},
                "error": f"unsupported pipeline: {pipeline}"}
    if pipeline in ("guardrail", "guardrail_image"):
        return await _guardrail(pipeline, data, target_url or registry[pipeline]["base"],
                                config_override or {})

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


async def _guardrail(pipeline: str, data: dict[str, Any], base: str,
                     cfg: dict[str, Any]) -> dict[str, Any]:
    """Ask one Module 23 guard model about one case; the output is its verdict
    as JSON (see app/services/moderation.py).

    config_override.model names the model: for guardrail, which of Module 23's
    active guard models to score (omit to score the combined verdict); for
    guardrail_image, shieldgemma-2 or llama-guard-3-vision. Image cases give
    input.image_path, a file the benchmark service can read; ShieldGemma 2's
    policy threshold is config_override.threshold (default 0.5).
    """
    settings = get_settings()
    model = cfg.get("model")
    files = body = None
    try:
        if pipeline == "guardrail":
            url = base.rstrip("/") + "/guard/check/input"
            body = _build_body(pipeline, data)
        else:
            if model not in moderation.IMAGE_ENDPOINTS:
                raise ValueError(f"guardrail_image needs config_override.model, one of "
                                 f"{sorted(moderation.IMAGE_ENDPOINTS)}")
            url = base.rstrip("/") + moderation.IMAGE_ENDPOINTS[model]
            path = data.get("image_path")
            if not path:
                raise ValueError("guardrail_image case has no input.image_path")
            with open(path, "rb") as fh:  # noqa: ASYNC230 -- one local image per case, small
                files = {"image": (path.rsplit("/", 1)[-1], fh.read())}
    except (ValueError, OSError) as exc:
        return {"output": "", "latency_ms": 0.0, "raw_response": {}, "error": str(exc)}

    raw, latency_ms, error = await _timed_post(url, body if files is None else None,
                                               settings.guardrail_timeout, files=files)
    if error is not None:
        logger.warning("Pipeline %s call failed: %s", pipeline, error)
        return {"output": "", "latency_ms": latency_ms, "raw_response": {}, "error": error}
    try:
        if pipeline == "guardrail":
            verdict = moderation.text_verdict(raw, model)
        else:
            verdict = moderation.image_verdict(
                raw, model, float(cfg.get("threshold", moderation.DEFAULT_THRESHOLD)))
    except (KeyError, TypeError, ValueError) as exc:
        return {"output": "", "latency_ms": latency_ms, "raw_response": raw,
                "error": f"unexpected Module 23 answer: {exc!r}"}
    return {"output": json.dumps(verdict), "latency_ms": latency_ms, "raw_response": raw}


async def infer_batch_local(
    inputs: list[dict[str, Any]],
    target_url: str | None = None,
    config_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Batch local_infer: one request generates outputs for all test-case inputs.

    Returns {outputs: List[str], latency_ms[, error]}. On any failure returns an
    empty ``outputs`` list so the caller can fall back to per-case inference.
    """
    settings = get_settings()
    cfg = config_override or {}
    base = target_url or settings.local_infer_url
    url = base.rstrip("/") + "/api/v1/inference/infer_batch"
    body: dict[str, Any] = {
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


async def unload_local_infer(target_url: str | None = None,
                             model_path: str | None = None) -> None:
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
    except httpx.HTTPError as exc:
        logger.warning("local_infer unload failed: %s", exc)
