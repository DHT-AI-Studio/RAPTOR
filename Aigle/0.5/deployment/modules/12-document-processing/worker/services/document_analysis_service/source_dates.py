"""When a document was written, as its own file says (temporal plan P13).

Module 25 dates a document's deictic phrases ("two years ago", "last month", 去年) from this; it validates the
value (placeholders, impossible dates) before using it. Read here because this is the one worker that opens
the original file:

    PDF            info dictionary /CreationDate           -> capture_time_source "pdf_creation_date"
    DOCX / PPTX    docProps/core.xml <dcterms:created>     -> capture_time_source "docx_core_created"

Never raises: a document without a readable date is processed exactly as before, with no anchor.
"""
from __future__ import annotations

import logging
import xml.etree.ElementTree as ET  # nosec B405 -- parses the document's own core.xml, already accepted for analysis
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

_DCTERMS = "{http://purl.org/dc/terms/}created"


def read_source_date(path: Path, filename: str) -> tuple[str | None, str | None]:
    """(the date as the file states it, where it came from), or (None, None)."""
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    try:
        if ext == "pdf":
            import fitz  # PyMuPDF, already a dependency of this worker
            with fitz.open(str(path)) as doc:
                created = (doc.metadata or {}).get("creationDate") or ""
            return (created, "pdf_creation_date") if created.strip() else (None, None)
        if ext in ("docx", "pptx", "xlsx"):
            with zipfile.ZipFile(path) as z:
                root = ET.fromstring(z.read("docProps/core.xml"))  # nosec B314 -- see the import
            node = root.find(_DCTERMS)
            created = (node.text or "").strip() if node is not None else ""
            return (created, "docx_core_created") if created else (None, None)
    except Exception as exc:  # noqa: BLE001 -- a damaged or unusual file only loses its date, never its analysis; logged
        logger.warning("could not read the creation date of %s: %s", filename, exc)
    return None, None
