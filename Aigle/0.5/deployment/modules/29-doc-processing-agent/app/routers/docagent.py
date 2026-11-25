"""DA-09a — DocAgent API router.

Endpoints
---------
POST /api/v1/docagent/process   — upload a document + pick a task
GET  /api/v1/docagent/formats   — supported format matrix
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse

from app.adapters.doc12_adapter import (
    FORMAT_MATRIX,
    Doc12Adapter,
    _detect_format_category,
)
from app.agent.orchestrator import ALL_TASKS, DocAgentOrchestrator, TaskType

logger = logging.getLogger(__name__)

_TASK_DESC = f"One of: {', '.join(ALL_TASKS)}"

router = APIRouter(prefix="/api/v1/docagent", tags=["DocAgent"])


def _get_orchestrator(request: Request) -> DocAgentOrchestrator:
    orch = getattr(request.app.state, "orchestrator", None)
    if orch is None:
        raise RuntimeError("Orchestrator not initialised — check lifespan")
    return orch


def _get_adapter(request: Request) -> Doc12Adapter:
    adapter = getattr(request.app.state, "doc12_adapter", None)
    if adapter is None:
        raise RuntimeError("Doc12Adapter not initialised — check lifespan")
    return adapter


@router.post("/process", summary="Process a document with a DocAgent task")
async def process_document(
    request: Request,
    file: UploadFile = File(..., description="Document to process"),
    task: TaskType = Form(..., description=_TASK_DESC),
    options_json: str | None = Form(
        default=None,
        alias="options",
        description="JSON string of task-specific options",
    ),
) -> JSONResponse:
    """
    Upload a document and run one of the seven DocAgent tasks.

    **Tasks:**
    - `classify` — detect document type and category
    - `extract` — extract structured fields (pass `options.fields` list)
    - `summarize` — generate a summary (pass `options.max_length`)
    - `compare` — compare with a second document (pass `options.compare_text`)
    - `qa` — answer a question (pass `options.question`)
    - `parse_table` — extract all tables as structured JSON
    - `search` — semantic search (pass `options.query`, optional `options.top_k`)
    """
    # Parse options
    options: dict[str, Any] = {}
    if options_json:
        import json
        try:
            options = json.loads(options_json)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="options must be a valid JSON string",
            )

    # Read upload
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded file is empty",
        )

    filename = file.filename or "upload.bin"
    adapter = _get_adapter(request)
    orchestrator = _get_orchestrator(request)

    # Step 0 — reject unsupported formats before calling Module 12
    if _detect_format_category(filename) == "unknown":
        suffix = Path(filename).suffix.lower() or "none"
        logger.warning("Unsupported file type %r for %r", suffix, filename)
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {suffix}",
        )

    # Step 1 — Module 12: parse document → full_text + chunks
    try:
        doc = await adapter.process(file_bytes, filename)
    except httpx.HTTPStatusError as exc:
        upstream = exc.response.status_code
        if 400 <= upstream < 500:
            logger.warning("Module 12 rejected %r with %s", filename, upstream)
            raise HTTPException(
                status_code=upstream,
                detail="Document rejected by analysis service",
            )
        logger.error("Module 12 error for %r: %s", filename, upstream)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Upstream error: {upstream}",
        )
    except httpx.TimeoutException:
        logger.warning("Module 12 timeout for %r", filename)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Document analysis service timed out",
        )

    except Exception as exc:  # noqa: BLE001
        logger.error("Doc12Adapter failed for %r: %s: %s",
                     filename, type(exc).__name__, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Document analysis service unavailable",
        )

    # Step 2 — Orchestrator: run task
    result = await orchestrator.run(task=task, doc=doc, options=options)

    # Propagate tool-level errors as HTTP errors
    if result.get("error"):
        code = result.pop("status_code", 422)
        raise HTTPException(status_code=code, detail=result["error"])

    return JSONResponse(content=result)


@router.get("/formats", summary="Supported document format matrix")
async def get_formats() -> dict[str, Any]:
    """
    Returns the supported format matrix and tool/dependency availability.

    Each entry describes a format category, its file extensions, whether VLM
    (InternVL via Module 12) is required, and which backend handles it.
    """
    return {
        "formats": FORMAT_MATRIX,
        "supported_tasks": ALL_TASKS,
        "backends": {
            "module12": "Document parsing (PDF/OCR/Office/Spreadsheet/VLM)",
            "module07": "LLM inference (classify/extract/summarize/compare/qa/parse_table)",
            "qdrant":   "Vector search (search task via DA-6 EmbeddingSearchTool)",
        },
    }
