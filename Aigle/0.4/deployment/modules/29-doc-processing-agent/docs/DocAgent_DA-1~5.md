# Document Processing Infrastructure Scaffold - DA-1

## Description

As a platform engineer, I want the DocAgent module skeleton (container, config, health check) registered in the Raptor stack, so that the module starts cleanly as part of deploy.sh.

## Acceptance Criteria:

 - 29-doc-processing-agent/ directory created with Dockerfile, docker-compose.yml, requirements.txt

 - app/core/config.py uses pydantic_settings.BaseSettings with prefix DA_

 - All settings documented in table: DA_AI_LIFECYCLE_URL, DA_ASSET_MGMT_URL, DA_QDRANT_URL, DA_LLM_MODEL, DA_VLM_MODEL, DA_EMBED_MODEL, DA_PDF_DPI, DA_MAX_PAGES, DA_AGENT_MAX_STEPS, DA_LOG_LEVEL

 - GET /health returns {"status": "ok", "service": "doc-processing-agent"}

 - PORT_DOC_AGENT=8029 added to deployment/modules/.env

 - Module 29 registered in build.py with deps=["02","03","04","07","13"]

 - bash deploy.sh -m 29 starts the container without errors

## Subtasks:

 - Create directory layout following Module 28 pattern

 - Write main.py with FastAPI lifespan (Module 07 reachability log, Qdrant collection placeholder)

 - Register in build.py


# Format Detector & Reader Tools - DA-2

## Description

As a DocAgent, I want to detect any uploaded document's format and read its content using the appropriate reader, so that subsequent tools receive clean, normalized content regardless of input format.

## Acceptance Criteria:

 - FormatDetectorTool uses python-magic (MIME sniffing) with extension fallback; returns {format: str, mime_type: str, size_bytes: int}; supported formats: pdf, docx, doc, xlsx, xls, csv, txt, html, image

 - PlainTextTool handles TXT/MD (chardet encoding detection) and HTML (BeautifulSoup tag stripping); returns {text: str, encoding: str, char_count: int}

 - SpreadsheetParseTool handles XLSX/XLS via openpyxl and CSV via csv.DictReader; returns {headers: [str], rows: [[str]], sheet_count: int}; skips empty rows

 - OfficeConversionTool runs soffice --headless --convert-to pdf --outdir {tmp} via asyncio.create_subprocess_exec; timeout configurable (DA_LIBREOFFICE_TIMEOUT); returns converted PDF path; cleans up temp on failure

 - PDFRenderTool uses PyMuPDF (fitz): renders pages at DA_PDF_DPI (default 150), extracts raw text; returns list of {page_num, text, image_b64, width, height}; respects DA_MAX_PAGES; cleans temp dirs on context exit

 - All tools are smolagents.Tool subclasses with name, description, inputs, output_type, and forward() method

 - Unit tests in tests/test_tools_reader.py with fixture files (1-page PDF, DOCX, XLSX, CSV, PNG)

## Subtasks:

 - tools/format_detector.py

 - tools/plain_text.py

 - tools/spreadsheet_parse.py

 - tools/office_converter.py

 - tools/pdf_render.py

 - Unit tests with sample files


# RaptorLLMModel — Module 07 Wrapper - DA-3

## Description

As a smolagents CodeAgent, I want a model wrapper that calls Module 07's inference endpoint, so that the agent can use Raptor's existing AI infrastructure instead of an external LLM service.

## Acceptance Criteria:

 - agent/raptor_llm_model.py implements the smolagents Model interface (or HfApiModel pattern): __call__(messages, **kwargs) → ChatMessage

 - Translates smolagents message list to Module 07 format: {task: "text-generation-ollama", engine: "ollama", model_name: DA_LLM_MODEL, data: {inputs: prompt_string}}

 - Parses response from body["result"]["generated_text"]

 - Supports stop_sequences via prompt suffix (Module 07 doesn't have native stop_sequences — append hint to prompt)

 - timeout from DA_LLM_TIMEOUT setting (default 120s)

 - On Module 07 error: raises RuntimeError with original HTTP status and body logged at WARNING

 - Unit test with mocked httpx.AsyncClient verifying request shape and response parsing

## Subtasks:

 - agent/raptor_llm_model.py

 - Unit test

# Field Extraction Tool - DA-5

## Description

As a DocAgent, I want to extract specific named fields from any document format, so that downstream services (like Module 28) can get structured data without knowing the document format.

## Acceptance Criteria:

 - FieldExtractionTool accepts {pages: [{page_num, text, image_b64}], fields: [str], context: str} → batches fields (max 5 per VLM call) → returns {field_name: extracted_value} for all requested fields

 - For text-rich pages (text length > 200 chars): uses LLM text extraction first, only calls VLM if field is null after text pass

 - For image-heavy pages (text length ≤ 200 chars): goes directly to VLM

 - Multi-page merge: if field found on multiple pages, last non-null value wins; list-type fields (e.g. factory names) are concatenated

 - Prompt template in prompts/field_extraction.py supports bilingual field names (handles both a Chinese field name, e.g. 申請者名稱, and its English equivalent, "Applicant Name," in the same request)

 - Returns {field_name: null} (not missing key) for fields not found

 - Unit test: given a 3-page PDF fixture with known field values, extraction achieves correct values for at least 4/5 test fields

 - Integration test: DOCX → OfficeConversionTool → PDFRenderTool → FieldExtractionTool → correct JSON output

## Subtasks:

 - tools/field_extraction.py

 - prompts/field_extraction.py

 - Unit + integration tests