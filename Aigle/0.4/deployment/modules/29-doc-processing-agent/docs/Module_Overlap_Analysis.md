# Module 29 (DocAgent) Overlap & Integration Analysis vs 12 / 17

> Date: 2026-07-23　Scope: assess the functional overlap and integration opportunities between `29-doc-processing-agent` (current DA-6 + planned DA-1~5) and [`12-document-processing`](../../12-document-processing/), [`17-hybrid-search`](../../17-hybrid-search/).
> Related: [`DocAgent_Implementation_Plan.md`](DocAgent_Implementation_Plan.md), [`DA-6_integration.md`](DA-6_integration.md)

## TL;DR

- **DA-6 (current, vector search) vs 17**: real overlap (two separate Qdrant setups, the same bge-m3 model), but the positioning is distinct enough to be acceptable (ad-hoc upload vs. persistent platform index).
- **DA-2 / DA-5 (planned: readers + VLM extraction) vs 12**: **the worst overlap**. Module 12's `document_analysis` already has a synchronous `/analyze` (:8020) + full-format readers + VLM captioning + contextual chunking — DA-2 would amount to almost entirely rebuilding it.
- **Highest-value integration**: have Module 29 **call Module 12's `/analyze` to get chunks** instead of rebuilding extraction in DA-2/DA-5; have DA-6's vector search **hand off to Module 17's rerank** afterward.

## Positioning of the three modules

| Module | Role | Trigger | Output | Embedding path |
|---|---|---|---|---|
| **12-document-processing** | Platform-level document **extraction** pipeline | Kafka **+ synchronous `/analyze` (:8020)** | Format detection → per-format reader → VLM captioning → **contextual chunking** → chunks | Does not embed itself (leaves that to 17) |
| **17-hybrid-search** | Platform-level **hybrid retrieval** | REST `/api/v1/ingest`, `/api/v1/search{,/vector}` | Qdrant vectors + OpenSearch BM25 + **rerank** + RRF | **Local in-process** SentenceTransformer bge-m3 |
| **29-doc-processing-agent (DA-6)** | Ad-hoc document **semantic search** | `/api/v1/docagent/process?task=search` | Module 07 embed → Qdrant `doc_agent_embeddings` → pure vector NN | **Module 07** `/inference/infer` bge-m3 |

## Overlap comparison table

| Capability | Module 29 plan | Already provided by an existing module | Overlap |
|---|---|---|---|
| Format detection + readers (PDF / PDF-OCR / DOCX(office) / XLSX / CSV / HTML / TXT) | DA-2 (planned) | **12's `document_analysis` already has all of it**: `OfficeDocumentProcessor`, `PDFOCRProcessor`, `CSVXLSXProcessor`, `HTMLProcessor`, `TxtProcessor` | 🔴 Almost a complete rebuild |
| Image/photo VLM extraction | DA-5 (planned) | 12's `VLMAnnotator` (InternVL/Qwen VLM captioning) | 🟠 Partial (DA-5's "named-field extraction" is a genuinely new angle) |
| Chunking | DA-6 sentence-aware (char window) | 12's **contextual chunking** (each chunk gets LLM-generated document context, matching 17's `contextual` payload schema) | 🟠 12 is more mature |
| Embedding | DA-6 → Module 07 bge-m3 (1024) | 17 → local bge-m3 (1024) | 🟠 Same model, different path (vector-compatible) |
| Vector store + search | DA-6 Qdrant `doc_agent_embeddings` (pure vector) | 17 Qdrant `raptor` + BM25 + rerank (hybrid) | 🔴 DA-6 is a functional subset of 17 |
| Module 07 LLM/VLM call wrapper | DA-3 (planned, smolagents Model) | 12 already calls `/inference/infer` directly (LLM summary, VLM captioning) | 🟡 Overlapping call pattern |

## Detailed assessment

### 29 (DA-6) ↔ 17-hybrid-search
- **Overlap**: both do "document text → bge-m3 vector → Qdrant → similarity search." DA-6 currently rebuilds a subset of 17's vector search.
- **Differences (defensible reasons)**:
  - 17's `raptor` collection is a **persistent platform index** (carries `status`/`branch_id`/`embedding_type` filters, fed by the media pipeline), while DA-6 is **ad-hoc upload, single-session RAG**.
  - DA-6's ticket rationale — "power RAG **without a separate indexing pipeline**" — specifically means "don't want ad-hoc uploads to go through 12's Kafka pipeline." So **a separate collection is intentional design**, not an oversight.
  - Different embedding paths: 17 computes locally in-process, DA-6 goes through the Module 07 service (can share GPU/model lifecycle with other services).
- **Can be integrated**: after DA-6's vector search, call **17's rerank endpoint** (`bge-reranker-v2-m3`) to improve ranking instead of building one from scratch.

### 29 (DA-2/DA-5) ↔ 12-document-processing (the biggest overlap)
- Module 12's `document_analysis_service` already provides:
  - A **synchronous `/analyze`** endpoint (container `raptor-document-analysis`, :8020) — not just Kafka.
  - Coverage of the **same set of formats** planned for DA-2 (PDF/OCR, office/docx, csv/xlsx, html, txt) + VLM captioning.
  - **Contextual chunking** (better than DA-6's sentence splitting).
- So **DA-2 ≈ rebuilding Module 12's `document_analysis`**; DA-5's VLM portion also overlaps with `VLMAnnotator`.
- **Highest-value integration**: have Module 29's `/process` first call **12's `/analyze`** to get chunks (with contextual context), then have DA-6 embed / DA-5 do named-field extraction on top of that. This skips an entire GPU-tuned extraction stack and gets better chunking for free.

## Integration options (highest to lowest value)

| Option | Description | Benefit | Cost / risk |
|---|---|---|---|
| **A. Reuse readers** | 29 skips its own DA-2/DA-5 extraction, calls 12's `/analyze` (:8020) for chunks instead | Saves the most duplicate work; gets contextual chunking; single source of truth for extraction | Depends on 12's service being available; the `/analyze` contract needs confirming |
| **B. Reuse rerank** | DA-6's vector search hands off to 17's rerank | Better ranking quality, low cost | One extra network call |
| **C. Align embeddings** | Recognize DA-6 (Module 07) and 17 (local) use the same bge-m3, vector-compatible | Future interop/merge possible | Needs version alignment guaranteed |
| **D. Shared collection** | DA-6 uses 17's `/ingest`+`/search` directly (no separate `doc_agent_embeddings`) | Single source of truth, no second stack to maintain | Ad-hoc uploads would pollute the platform's `raptor` collection; conflicts with the ticket's "don't go through the pipeline" intent. **Not recommended** |

**Recommendation**: go with **A + B** — keep DA-6's separate collection (preserving ad-hoc semantics), but reuse 12 for extraction and 17 for ranking.

## Architectural tension (needs owner sign-off)

The DA-1~5 tickets explicitly call for **building new** readers (DA-2) / VLM extraction (DA-5) / embedding (DA-6) inside Module 29, which directly overlaps with existing capabilities in 12/17. DA-6 (vector search) is positioned distinctly enough from 17 on its own; **but if DA-2/DA-5 are rebuilt as the tickets specify, that's where the most waste is**. Since DA-1~5 belong to someone else, it's recommended to raise "**have DA-2/DA-5 call Module 12's `/analyze` instead**" with the DA-1~5 owner / architecture lead for a decision.

## References (sources reviewed for this analysis)

- `12-document-processing/worker/main.py` (WORKER_TYPE dispatch: orchestrator/analysis/summary/indexer/graph)
- `12-document-processing/worker/services/document_analysis_service/{main.py, document_analysis.py, sync_api.py}` (`/analyze`:8020, processors, contextual chunking)
- `12-document-processing/docker-compose.yml` (`HYBRID_SEARCH_INGEST_URL`, `INFERENCE_URL`)
- `17-hybrid-search/app/services/qdrant_service.py`, `app/core/{embedding,config}.py`, `app/api/v1/{ingest,search}.py`
- `29-doc-processing-agent/app/tools/embedding_search.py` (current DA-6 implementation)
