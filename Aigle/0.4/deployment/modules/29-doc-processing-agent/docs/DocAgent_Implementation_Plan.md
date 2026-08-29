# DocAgent Implementation Plan (Module 29 — doc-processing-agent)

> Corresponding tickets: DA-1 (scaffold), DA-2 (format detection/reader tools), DA-3 (Module 07 LLM wrapper), DA-5 (field extraction), DA-6 (embedding search). **There is no DA-4.**
> Source docs: [`DocAgent_DA-1~5.md`](DocAgent_DA-1~5.md), [`embedding_search_DA-6.md`](embedding_search_DA-6.md)
> Status: **DA-6 is implemented and tested; DA-1~5 belong to someone else (the rest of this document is a whole-module design reference)**

---

## 0. Scope of this implementation pass (2026-07-22)

Per instructions, **only DA-6 (EmbeddingSearchTool) is implemented in this pass** — DA-1~5 belong to someone else. DA-6 is decoupled from DA-1~5 by consuming "already-extracted text," so it can be developed and tested independently and slotted in cleanly later.

- **Delivered at**: [`29-doc-processing-agent/`](../) (DA-6-specific files only — DA-1~5's `main.py`/`config.py`/`process.py`/`Dockerfile`/readers were not created).
- **Integration notes**: [`DA-6_integration.md`](DA-6_integration.md) (lifespan and `/process` snippets to paste in, configuration, test commands).
- **Module overlap analysis**: [`Module_Overlap_Analysis.md`](Module_Overlap_Analysis.md) (overlap and integration recommendations between DA-6/29 and 12-document-processing, 17-hybrid-search).
- **Dev/test endpoint**: Module 07 **v314 dev** at `http://localhost:9998` (`aiml-v314-api`, MLflow on 5557, `bge-m3` already registered); Qdrant uses in-memory `:memory:`, doesn't touch the shared `raptor-qdrant`.
- **Test results**: 7 unit tests (offline, Module 07 mocked via `MockTransport`) + 1 integration test (real bge-m3), all passing; query `"scope 3 supplier emissions"` → top **score 0.745 > 0.7** ✅ (DA-6 acceptance criterion met). The tool is verified to be a genuine `smolagents.Tool` subclass.
- ⚠ **Handoff note**: DA-1's planned `PORT_DOC_AGENT=8029` conflicts with the existing `raptor-audio-diarization` (already on 8029) — DA-1 will need to pick a different port.

---

## 1. Goal and scope

Build a new **smolagents-based** document-processing microservice, **`29-doc-processing-agent`** (host port **8029**), providing:

- Upload a document in any format → auto-detect format → read and normalize to text/pages (DA-2)
- Call **Module 07**'s existing LLM / VLM / embedding infrastructure (DA-3)
- **Field extraction** (DA-5): pull specified named fields out of a document as structured JSON
- **Semantic search** (DA-6): embed an uploaded document on the spot → write to Qdrant → nearest-neighbor search against a query
- A single entry point, `POST /api/v1/docagent/process`, dispatching on `task` (`search` / `extract` / …)

**Out of scope**: Module 28 (a downstream consumer that doesn't exist in the repo yet, treated as a future integration target); an async queue/batch pipeline for DocAgent (this phase completes synchronously within the request).

---

## 2. Current-state analysis: reusable assets

| Asset | Location | Use for DocAgent |
|---|---|---|
| **Module 07 inference endpoint** | `raptor-ai-lifecycle-api:8010`, `POST /inference/infer` | DA-3 LLM, DA-5 VLM, and DA-6 embedding all go through here (spec-driven; `task` determines the runtime) |
| **Working call example** | [`22-benchmark-service/app/services/judge.py`](../../22-benchmark-service/app/services/judge.py) | **Authoritative reference** for the exact request/response shapes of `_infer` (text-generation) and `_infer_embeddings` (embedding) |
| **Qdrant service pattern** | [`17-hybrid-search/app/services/qdrant_service.py`](../../17-hybrid-search/app/services/qdrant_service.py) | Directly mirrored by DA-6's `AsyncQdrantClient`, idempotent collection init, batch upsert, vector search |
| **smolagents / FastAPI microservice template** | [`21-agent-protocol/`](../../21-agent-protocol/) (`smolagents_host.py`'s `Tool` subclasses, `main.py` lifespan) | Template for DA-2/DA-3's `smolagents.Tool` interface, CodeAgent assembly, service scaffold |
| **Most recent module convention** | [`22-benchmark-service/`](../../22-benchmark-service/) (Dockerfile / docker-compose / .env / tests layout) | Template for DA-1's directory layout, compose (container port 8000 + host port mapping), pytest layout |
| **Module registration mechanism** | [`build.py`](../../build.py)'s `MODULES` list, `Module` dataclass, `_compose_up` | For DA-1 to register module 29 in build.py |

### Relationship to the existing video search (clarified)

The video search you're thinking of lives in **Module 17 (hybrid-search)**. It and DA-6 are **two deliberately separate vector-retrieval stacks**:

| | Module 17 (video/hybrid search) | Module 29 DA-6 (doc search) |
|---|---|---|
| Embedding source | **Local** `SentenceTransformer(BAAI/bge-m3)` | **Module 07** `/inference/infer` `task=embedding` |
| Qdrant collection | `raptor` | `doc_agent_embeddings` (new, cosine, 1024-dim) |
| Retrieval style | OpenSearch + Qdrant hybrid + RRF + rerank | Pure vector nearest-neighbor (this phase) |
| Positioning | Persistent platform-level index | DocAgent's index-at-upload, ad-hoc RAG retrieval |

> The two share the same dimensionality (bge-m3, 1024-dim), so routing DocAgent through Module 17's hybrid retrieval is a viable future extension — but **this phase follows DA-6's explicit requirement to use Module 07 embedding + a separate collection**.

---

## 3. Architecture design

### 3.1 Module positioning

- Name: `29-doc-processing-agent`; container `raptor-doc-processing-agent`
- Port: host `${PORT_DOC_AGENT:-8029}` → container `8000` (following Module 22's convention)
- deps: `["02","03","04","07","13"]` (Redis / DB / object storage (asset-mgmt) / AI-ML / API gateway)
- Runtime: Python 3.11-slim + LibreOffice + libmagic + PyMuPDF + smolagents (**a heavy image — see risk §6**)

### 3.2 Directory layout (following the Module 21/22 pattern + DA subtasks)

```
29-doc-processing-agent/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env
├── README.md
├── app/
│   ├── main.py                    # FastAPI + lifespan (logs Module 07 reachability, inits the Qdrant collection)
│   ├── core/
│   │   ├── config.py              # pydantic BaseSettings, env_prefix="DA_" (DA-1)
│   │   ├── qdrant.py              # AsyncQdrantClient + doc_agent_embeddings init (DA-6)
│   │   └── logging.py
│   ├── api/
│   │   └── process.py             # POST /api/v1/docagent/process (task dispatch)
│   ├── clients/
│   │   ├── module07.py            # shared httpx client: /inference/infer (LLM/VLM/embedding)
│   │   └── asset_mgmt.py          # fetches uploaded file bytes from Module 04/13 (asset_path/version_id)
│   ├── agent/
│   │   ├── raptor_llm_model.py    # DA-3: smolagents Model wrapper → Module 07
│   │   └── doc_agent.py           # smolagents CodeAgent assembly (agentic mode, optional)
│   ├── tools/
│   │   ├── format_detector.py     # DA-2
│   │   ├── plain_text.py          # DA-2
│   │   ├── spreadsheet_parse.py   # DA-2
│   │   ├── office_converter.py    # DA-2
│   │   ├── pdf_render.py          # DA-2
│   │   ├── field_extraction.py    # DA-5
│   │   └── embedding_search.py    # DA-6
│   ├── prompts/
│   │   └── field_extraction.py    # DA-5 (bilingual prompt template)
│   └── chunking.py                # DA-6 text chunking (configurable window/overlap)
└── tests/
    ├── fixtures/                  # 1-page PDF, DOCX, XLSX, CSV, PNG, 3-page scope3 PDF
    ├── test_tools_reader.py       # DA-2
    ├── test_raptor_llm_model.py   # DA-3
    ├── test_field_extraction.py   # DA-5
    └── test_embedding_search.py   # DA-6
```

### 3.3 `/api/v1/docagent/process` contract

```
POST /api/v1/docagent/process   (multipart/form-data or JSON+asset ref)
  file            : uploaded file (multipart) — or
  asset_path+version_id : fetched from asset-mgmt (JSON)
  task            : "search" | "extract"
  # task=search
  query           : str
  top_k           : int = 10
  # task=extract
  fields          : [str]
  context         : str = ""
→ 200 { "task": ..., "result": <task-specific> }
```

**Internal pipeline (deterministic route)**:

```
file bytes
  └─ FormatDetectorTool ─► format
       ├─ pdf                        → PDFRenderTool          → pages[]
       ├─ docx/doc/xls(x)→pdf        → OfficeConversionTool   → PDFRenderTool → pages[]
       ├─ xlsx/xls/csv (table lookup) → SpreadsheetParseTool   → rows → text
       ├─ txt/md/html                → PlainTextTool          → text
       └─ image                      → (extract goes through VLM; search goes through caption/OCR, see open item §8)
  └─ per task:
       task=search  → chunking → EmbeddingSearchTool.forward(chunks, query)
       task=extract → FieldExtractionTool.forward(pages, fields, context)
```

### 3.4 Where smolagents fits (an important design decision)

The DA tickets require both "tools are `smolagents.Tool` subclasses" (so they can be orchestrated by an agent) and "`/process` routes directly by task" (predictable, testable). This uses a **dual-track** approach:

- **`/process` uses a deterministic route** (dispatches directly to the matching Tool's `forward()` based on `task`) — analogous to Module 21's `mode=direct`, stable and testable, the default for this phase.
- **`RaptorLLMModel` + `CodeAgent`** (`agent/doc_agent.py`) preserves agentic orchestration capability (analogous to Module 21's `mode=agent`) — DA-3 was explicitly built for "a smolagents CodeAgent," for future dynamic tasks.
- Every tool is built as a `smolagents.Tool` subclass regardless, so both tracks share the same set of tools.

---

## 4. Per-ticket implementation design

### DA-1 — Scaffold

- `app/core/config.py`: `BaseSettings`, `model_config = SettingsConfigDict(env_prefix="DA_", ...)`. Settings table:

  | Setting | Default | Purpose |
  |---|---|---|
  | `DA_AI_LIFECYCLE_URL` | `http://raptor-ai-lifecycle-api:8010` | Module 07 |
  | `DA_ASSET_MGMT_URL` | `http://raptor-asset-management:8000` | Fetching uploaded files |
  | `DA_QDRANT_URL` | `http://raptor-qdrant:6333` | Vector store |
  | `DA_LLM_MODEL` | `qwen2.5-7b-ollama` | DA-3 |
  | `DA_VLM_MODEL` | `qwen2.5-vl-7b` (or internvl) | DA-5 |
  | `DA_EMBED_MODEL` | `bge-m3` | DA-6 |
  | `DA_PDF_DPI` | `150` | DA-2 PDFRender |
  | `DA_MAX_PAGES` | `50` | DA-2 cap |
  | `DA_AGENT_MAX_STEPS` | `10` | CodeAgent |
  | `DA_LOG_LEVEL` | `INFO` | — |
  | `DA_QDRANT_DOC_COLLECTION` | `doc_agent_embeddings` | DA-6 |
  | `DA_QDRANT_VECTOR_SIZE` | `1024` | DA-6 |
  | `DA_LLM_TIMEOUT` | `120` | DA-3 |
  | `DA_LIBREOFFICE_TIMEOUT` | `120` | DA-2 |
  | `DA_EMBED_TIMEOUT` | `30` | DA-6 |

- `app/main.py`: FastAPI + `lifespan`: (1) logs Module 07 reachability; (2) idempotently initializes the Qdrant collection `doc_agent_embeddings`.
- `GET /health` → `{"status":"ok","service":"doc-processing-agent"}`.
- `build.py` registration:
  ```python
  Module(id="29", name="29-doc-processing-agent",
         description="Doc Processing Agent — smolagents document reading/field extraction/semantic retrieval",
         deps=["02","03","04","07","13"],
         health_containers=["raptor-doc-processing-agent"],
         steps=[_compose_up("29-doc-processing-agent/docker-compose.yml")])
  ```
- `deployment/modules/.env`: add `PORT_DOC_AGENT=8029` under the `HOST PORT SETTINGS` block; add a new `── 29 Doc Processing Agent ──` block of `DA_*` vars; add `DOC_AGENT_URL=http://raptor-doc-processing-agent:8000` under the downstream-URL block.
- Acceptance: `bash deploy.sh -m 29` starts cleanly, `/health` returns ok.

### DA-2 — Format detection and reader tools (`smolagents.Tool` subclasses, each with `name/description/inputs/output_type/forward()`)

| Tool | Technology | Returns |
|---|---|---|
| `FormatDetectorTool` | `python-magic` (MIME) + extension fallback | `{format, mime_type, size_bytes}` |
| `PlainTextTool` | TXT/MD: `chardet`; HTML: `BeautifulSoup` tag stripping | `{text, encoding, char_count}` |
| `SpreadsheetParseTool` | XLSX/XLS: `openpyxl`; CSV: `csv.DictReader` (skips empty rows) | `{headers, rows, sheet_count}` |
| `OfficeConversionTool` | `soffice --headless --convert-to pdf` (via `asyncio.create_subprocess_exec`, timeout `DA_LIBREOFFICE_TIMEOUT`, cleans up temp on failure) | Path to the converted PDF |
| `PDFRenderTool` | PyMuPDF (`fitz`), renders at `DA_PDF_DPI` + extracts native text, respects `DA_MAX_PAGES`, cleans temp on context exit | `[{page_num, text, image_b64, width, height}]` |

- Tests in `tests/test_tools_reader.py` + fixtures (1-page PDF, DOCX, XLSX, CSV, PNG).

### DA-3 — RaptorLLMModel (Module 07 wrapper)

- `agent/raptor_llm_model.py` implements the smolagents `Model` interface: `__call__(messages, **kwargs) → ChatMessage`.
- Flattens messages into a prompt → calls Module 07:
  ```python
  {"task": "text-generation", "engine": "ollama",
   "model_name": DA_LLM_MODEL, "data": {"inputs": prompt}, "options": {...}}
  ```
  > ⚠ The original DA-3 text says `task:"text-generation-ollama"`. Module 07 canonicalizes the old name, but **the working example (judge.py) uses `task:"text-generation" + engine:"ollama"`** — this plan follows the latter.
- Response parsing: **don't hard-code `result.generated_text`**. In practice the ollama field is `result.response`; use robust parsing like judge.py's `_extract_text` (try `response/generated_text/text/content` in order).
- `stop_sequences`: prefer passing `options.stop` (ollama supports it); DA-3's "prompt-suffix hint" is a fallback.
- Timeout: `DA_LLM_TIMEOUT` (120s); on a Module 07 error → `raise RuntimeError`, log the original HTTP status + body at WARNING.
- Test: mock `httpx.AsyncClient`, verify request shape and response parsing.

### DA-5 — FieldExtractionTool

- Interface: `{pages:[{page_num,text,image_b64}], fields:[str], context:str}` → `{field: value}` (max 5 fields per VLM call).
- Text-rich (`len(text)>200`): LLM text extraction first, VLM fills in only fields still null afterward; image-heavy (`≤200`): goes straight to VLM.
- VLM call: `{"task":"vlm","model_name":DA_VLM_MODEL,"data":{"image":<b64/path>,"prompt":...}}`.
- Multi-page merge: a later non-null value overwrites an earlier one; list-type fields (e.g. factory names) are concatenated.
- `prompts/field_extraction.py`: bilingual field names (handles both a Chinese field name, e.g. 申請者名稱, and its English equivalent, "Applicant Name").
- Fields not found return `{field: null}` (key is kept, not omitted).
- Tests: a 3-page PDF fixture (≥4/5 fields correct); an integration test DOCX→OfficeConversion→PDFRender→FieldExtraction→JSON.

### DA-6 — EmbeddingSearchTool (the original request)

- `tools/embedding_search.py`, a `smolagents.Tool` subclass.
- **Chunking** (`app/chunking.py`): window+overlap chunking over the extracted text (defaults are configurable, see open item §8).
- **Embed (Module 07)**: mirrors judge.py's `_infer_embeddings`:
  ```python
  {"task":"embedding","model_name":DA_EMBED_MODEL,"data":{"inputs":[chunks...]}}
  # → resp.json()["result"]["embeddings"]  (list of 1024-dim)
  ```
  > ⚠ The original DA-6 text says `task:"text-embedding"`; Module 07's canonical task is **`embedding`** — this plan uses `embedding`.
- **Upsert to Qdrant** `doc_agent_embeddings` (cosine, 1024): point id (see §6 for dedup), payload `{text, source, page_num, chunk_idx, uploaded_at}`. Collection is created idempotently in lifespan.
- **Search**: embed the query (single item) → `client.search(limit=top_k, with_payload=True)` → return ranked `[{id, score, text, source}]`.
- **Within the same request**, embed-upsert happens before search (searching across both prior content and the current document).
- Test: upload a 3-page scope3 PDF → query `"scope 3 supplier emissions"` → at least one result with `score>0.7` (real or mocked Module 07).

---

## 5. Key decisions and risks (worth your attention)

1. **Spec-vs-implementation gaps (flagged with ⚠ above, collected here)**
   - DA-6 specified `task:"text-embedding"` → actually uses **`embedding`**.
   - DA-3 specified `result.generated_text` → the actual ollama field is **`response`**; use robust parsing.
   - DA-3 specified `task:"text-generation-ollama"` → uses `task:"text-generation"+engine:"ollama"`.
2. **Module 07 needs the models registered first (prerequisite — otherwise DA-3/5/6 fail at runtime)**: `bge-m3` (embedding), an LLM (e.g. `qwen2.5-7b-ollama`), a VLM (`qwen2.5-vl` / `internvl`). Corresponds to `07-ai-ml-services/scripts/{06,01,03,04}_*.sh`. Integration tests depend on this.
3. **Module 28 doesn't exist**: neither DA-1's "follow the Module 28 pattern" nor DA-5's "downstream Module 28" exist in the repo → this plan follows the **Module 21/22** pattern instead; Module 28 is treated as a future consumer (not built in this phase).
4. **Heavy Docker image**: LibreOffice + PyMuPDF + libmagic + smolagents(litellm) — slow build, large image. The Dockerfile needs `apt-get install libreoffice libmagic1`.
5. **Qdrant point id / re-upload dedup**: recommend a content hash as the id (re-sending the same file doesn't bloat the index) or a uuid (keeps history) — pending your choice (§8).
6. **Chunking strategy undefined** (DA-6 only says "document text chunks"): defaults to a configurable window/overlap.

---

## 6. Implementation phases and order (dependency-ordered)

| Phase | Content | Verifiable output |
|---|---|---|
| **P0 prerequisite** | Confirm Module 07 has embed/LLM/VLM registered; Qdrant reachable | Manual `curl` to `/inference/infer` succeeds |
| **P1 (DA-1)** | Scaffold: directory, config, main+lifespan, health, build.py, .env, compose, Dockerfile | `deploy.sh -m 29` starts, `/health` ok |
| **P2 (DA-2)** | 5 reader tools + unit tests | `test_tools_reader.py` green |
| **P3 (DA-6)** | chunking + EmbeddingSearchTool + Qdrant init + `/process?task=search` + integration test | scope3 PDF retrieval `score>0.7` (**the original requirement met**) |
| **P4 (DA-3)** | RaptorLLMModel + unit tests | Mock tests green |
| **P5 (DA-5)** | FieldExtractionTool + prompts + `/process?task=extract` + unit/integration tests | 3-page PDF ≥4/5 fields correct |

> DA-6 only depends on DA-1's scaffold + DA-2's `PDFRenderTool`/`FormatDetectorTool` + Module 07 embedding, **not on DA-3/DA-5**, so it's scheduled at P3 to land the originally-requested feature first.

---

## 7. Open questions

1. **Are the P0 models already registered in Module 07?** What are the exact registered names for `bge-m3` / LLM / VLM (the `DA_*_MODEL` defaults need to match)?
2. **Qdrant point id strategy**: content-hash dedup vs. uuid history retention?
3. **Chunking parameters**: are the proposed default chunk size / overlap (e.g. 512 chars / 64 overlap) acceptable?
4. **Image-type documents under task=search**: go through VLM captioning, OCR, or skip them for now and only index text that can be extracted directly?
5. **`/process` input shape**: primarily direct multipart upload — should it also support fetching by asset_path/version_id (Module 04)?
6. **Where to start**: "do P1–P5 in one pass," or "do P1+P3 first (shortest path to deliver DA-6), the rest later"?
