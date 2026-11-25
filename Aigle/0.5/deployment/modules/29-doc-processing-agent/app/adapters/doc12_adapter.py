"""Module 12 document-processing adapter (DA-09a).

Calls Module 12's sync API (POST /analyze) over HTTP.  Module 12 runs the
heavy lifting — InternVL VLM, OCR, docling, table extraction — inside its own
GPU container.  Module 29 never loads those models directly.

Supported formats (delegated entirely to Module 12):
    PDF (text-layer)  PDF (scanned/VLM)  DOCX  PPTX  XLSX  CSV  TXT  HTML  PNG/JPG

The adapter returns a ``ProcessedDoc`` dataclass so the rest of the service
never touches raw HTTP responses.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import httpx

from app.core.config import DocAgentSettings, get_settings

logger = logging.getLogger(__name__)

# Maps file extension → MIME type for the multipart upload to Module 12.
_EXT_MIME: dict[str, str] = {
    ".pdf":  "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".csv":  "text/csv",
    ".txt":  "text/plain",
    ".html": "text/html",
    ".htm":  "text/html",
    ".png":  "image/png",
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
}

# Canonical format categories for GET /formats
FORMAT_MATRIX: dict[str, dict] = {
    "pdf_text":    {"extensions": [".pdf"],          "vlm_required": False, "via": "module12"},
    "pdf_scanned": {"extensions": [".pdf"],          "vlm_required": True,  "via": "module12"},
    "office":      {"extensions": [".docx", ".pptx"],"vlm_required": False, "via": "module12"},
    "spreadsheet": {"extensions": [".xlsx", ".csv"], "vlm_required": False, "via": "module12"},
    "plaintext":   {"extensions": [".txt"],          "vlm_required": False, "via": "module12"},
    "html":        {"extensions": [".html", ".htm"], "vlm_required": False, "via": "module12"},
    "image":       {"extensions": [".png", ".jpg", ".jpeg"], "vlm_required": True, "via": "module12"},
}


@dataclass
class ProcessedDoc:
    """Normalised output from Module 12's /analyze endpoint."""
    filename: str
    full_text: str
    chunks: list[dict] = field(default_factory=list)
    total_chunks: int = 0
    summary: str = ""
    format_category: str = "unknown"

    @classmethod
    def from_module12_response(cls, filename: str, data: dict) -> ProcessedDoc:
        chunks = data.get("chunks", [])
        full_text = data.get("full_text") or "\n\n".join(
            c.get("payload", {}).get("text", "")
            for c in chunks
            if c.get("payload", {}).get("text")
        )
        return cls(
            filename=filename,
            full_text=full_text,
            chunks=chunks,
            total_chunks=data.get("total_chunks", len(chunks)),
            summary=data.get("summary", ""),
            format_category=_detect_format_category(filename),
        )


def _detect_format_category(filename: str) -> str:
    """Infer format category from file extension."""
    import pathlib
    ext = pathlib.Path(filename).suffix.lower()
    for category, info in FORMAT_MATRIX.items():
        if ext in info["extensions"]:
            return category
    return "unknown"


class Doc12Adapter:
    """Async HTTP adapter for Module 12's document sync API."""

    def __init__(
        self,
        settings: DocAgentSettings | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._cfg = settings or get_settings()
        self._client = http_client  # injected for tests; else created per-call

    async def process(
        self,
        file_bytes: bytes,
        filename: str,
        processing_mode: str = "default",
    ) -> ProcessedDoc:
        """Upload ``file_bytes`` to Module 12 /analyze and return a ProcessedDoc.

        Raises:
            httpx.TimeoutException: if Module 12 does not respond in time.
            httpx.HTTPStatusError: if Module 12 returns a 4xx/5xx.
            RuntimeError: if Module 12 is unreachable (connection error).
        """
        import pathlib
        ext = pathlib.Path(filename).suffix.lower()
        mime = _EXT_MIME.get(ext, "application/octet-stream")
        url = self._cfg.module12_sync_url.rstrip("/") + "/analyze"

        files = {"file": (filename, file_bytes, mime)}
        data  = {"processing_mode": processing_mode}

        own_client = self._client is None
        client = self._client or httpx.AsyncClient(
            timeout=httpx.Timeout(self._cfg.doc12_timeout)
        )
        try:
            logger.info("Doc12Adapter: POST %s  filename=%r  size=%d bytes",
                        url, filename, len(file_bytes))
            resp = await client.post(url, files=files, data=data)
            resp.raise_for_status()
            payload = resp.json()
            doc = ProcessedDoc.from_module12_response(filename, payload)
            logger.info("Doc12Adapter: got %d chunks, full_text=%d chars",
                        doc.total_chunks, len(doc.full_text))
            return doc
        except httpx.TimeoutException:
            logger.error("Doc12Adapter: timeout after %.1fs calling %s",
                         self._cfg.doc12_timeout, url)
            raise
        except httpx.ConnectError as exc:
            raise RuntimeError(
                f"Module 12 sync API unreachable at {self._cfg.module12_sync_url}: {exc}"
            ) from exc
        finally:
            if own_client:
                await client.aclose()

    async def health(self) -> bool:
        """Return True if Module 12 /health responds 200."""
        url = self._cfg.module12_sync_url.rstrip("/") + "/health"
        try:
            async with httpx.AsyncClient(timeout=5.0) as c:
                r = await c.get(url)
                return r.status_code == 200
        except Exception:  # noqa: BLE001
            return False
