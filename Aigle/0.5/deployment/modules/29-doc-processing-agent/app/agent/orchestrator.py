"""DA-09a — DocAgent task orchestrator.

Routes an incoming task to the correct tool chain:

    classify    → ClassifyTool(full_text)
    extract     → ExtractTool(full_text, fields?)
    summarize   → SummarizeTool(full_text, max_length?)
    compare     → CompareTool(full_text, compare_text)
    qa          → QATool(full_text, question)
    parse_table → ParseTableTool(full_text, chunks)
    search      → EmbeddingSearchTool.forward(query, document_text, source, top_k)

The orchestrator is stateless — all state lives in Qdrant (search) or is
derived from the uploaded file on each request.
"""
from __future__ import annotations

import logging
from typing import Any, Literal

import httpx

from app.adapters.doc12_adapter import ProcessedDoc
from app.core.config import DocAgentSettings, get_settings
from app.tools.base import LLMToolError
from app.tools.classify import ClassifyTool
from app.tools.compare import CompareTool
from app.tools.extract import ExtractTool
from app.tools.parse_table import ParseTableTool
from app.tools.qa import QATool
from app.tools.summarize import SummarizeTool

logger = logging.getLogger(__name__)

TaskType = Literal[
    "classify", "extract", "summarize", "compare",
    "qa", "parse_table", "search",
]

ALL_TASKS: list[TaskType] = [
    "classify", "extract", "summarize", "compare",
    "qa", "parse_table", "search",
]


class DocAgentOrchestrator:
    """Stateless orchestrator — create once at app startup, reuse across requests."""

    def __init__(
        self,
        settings: DocAgentSettings | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._cfg = settings or get_settings()
        self._http = http_client

        # LLM-backed tools (share the same async client)
        self._classify    = ClassifyTool(self._cfg, http_client)
        self._extract     = ExtractTool(self._cfg, http_client)
        self._summarize   = SummarizeTool(self._cfg, http_client)
        self._compare     = CompareTool(self._cfg, http_client)
        self._qa          = QATool(self._cfg, http_client)
        self._parse_table = ParseTableTool(self._cfg, http_client)

        # DA-6 EmbeddingSearchTool (uses sync httpx + qdrant-client internally)
        self._search_tool = self._build_search_tool()

    def _build_search_tool(self):
        """Lazy-import EmbeddingSearchTool to keep startup fast when Qdrant is down."""
        try:
            from app.tools.embedding_search import (
                EmbeddingSearchConfig,
                EmbeddingSearchTool,
            )
            cfg = EmbeddingSearchConfig(
                ai_lifecycle_url=self._cfg.module07_url,
                embed_model=self._cfg.embed_model,
                qdrant_url=self._cfg.qdrant_url,
                collection=self._cfg.qdrant_doc_collection,
                vector_size=self._cfg.qdrant_vector_size,
                embed_timeout=self._cfg.embed_timeout,
                chunk_size=self._cfg.chunk_size,
                chunk_overlap=self._cfg.chunk_overlap,
                embed_batch_size=self._cfg.embed_batch_size,
            )
            return EmbeddingSearchTool(config=cfg)
        except Exception as exc:  # noqa: BLE001
            logger.warning("EmbeddingSearchTool init failed (Qdrant may be down): %s", exc)
            return None

    async def run(
        self,
        task: TaskType,
        doc: ProcessedDoc,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Dispatch to the correct tool and return a structured result dict."""
        opts = options or {}

        try:
            if task == "classify":
                result = await self._classify.run(full_text=doc.full_text)

            elif task == "extract":
                result = await self._extract.run(
                    full_text=doc.full_text,
                    fields=opts.get("fields"),
                )

            elif task == "summarize":
                result = await self._summarize.run(
                    full_text=doc.full_text,
                    max_length=opts.get("max_length", 500),
                )

            elif task == "compare":
                result = await self._compare.run(
                    full_text=doc.full_text,
                    compare_text=opts.get("compare_text"),
                )

            elif task == "qa":
                result = await self._qa.run(
                    full_text=doc.full_text,
                    question=opts.get("question"),
                )

            elif task == "parse_table":
                result = await self._parse_table.run(
                    full_text=doc.full_text,
                    chunks=doc.chunks,
                )

            elif task == "search":
                result = self._run_search(doc, opts)

            else:
                return {"error": f"Unknown task: {task!r}"}

        except LLMToolError as exc:
            logger.warning("Tool error for task=%s: %s", task, exc)
            return {
                "error": str(exc),
                "task": task,
                "status_code": exc.status_code if exc.status_code is not None else 502,
            }
        except Exception as exc:
            logger.exception("Unexpected error for task=%s", task)
            return {"error": str(exc), "task": task, "status_code": 500}

        return {"task": task, "filename": doc.filename, **result}

    def _run_search(self, doc: ProcessedDoc, opts: dict[str, Any]) -> dict[str, Any]:
        """Synchronous search via DA-6 EmbeddingSearchTool."""
        if self._search_tool is None:
            return {"error": "EmbeddingSearchTool unavailable (Qdrant not reachable)", "status_code": 503}

        query = opts.get("query", "")
        if not query:
            return {"error": "search task requires options.query"}

        return self._search_tool.forward(
            query=query,
            document_text=doc.full_text or None,
            source=doc.filename,
            top_k=opts.get("top_k", 10),
        )
