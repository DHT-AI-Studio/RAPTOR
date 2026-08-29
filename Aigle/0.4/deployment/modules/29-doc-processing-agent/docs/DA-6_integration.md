# DA-6 — EmbeddingSearchTool Delivery & Integration Notes

> Scope: **DA-6 (embedding search) only**. DA-1~5 (scaffolding, reader tools, LLM wrapper, field extraction) belong to someone else — this document explains how DA-6 connects to them.
> Full design background: [`DocAgent_Implementation_Plan.md`](DocAgent_Implementation_Plan.md).

## Delivered files (all DA-6-specific; none of DA-1~5's files are touched)

```
29-doc-processing-agent/
├── app/
│   ├── chunking.py                    # sentence-aware chunking (no external dependencies)
│   ├── core/doc_collection.py         # ensure_doc_collection(): idempotent collection creation (for lifespan)
│   └── tools/embedding_search.py      # EmbeddingSearchTool (smolagents.Tool subclass) + EmbeddingSearchConfig
├── tests/
│   ├── conftest.py
│   ├── test_embedding_search.py       # 7 unit tests (offline) + 1 integration test (hits real v314 bge-m3)
│   └── fixtures/
│       ├── make_scope3_pdf.py         # generates the 3-page scope3 PDF fixture
│       └── scope3_emissions.pdf
├── pytest.ini                         # registers the integration marker
└── requirements-da6.txt               # DA-6 runtime dependencies (to be merged into the module's requirements.txt later)
```

> **Not created** (belongs to DA-1~5, left alone to avoid conflicts): `app/main.py`, `app/core/config.py`, `app/api/process.py`, `Dockerfile`, `docker-compose.yml`, `.env`, reader tools.

## DA-6's boundaries and assumptions

- **Input is "already-extracted text"**: `EmbeddingSearchTool` takes `document_text` (str); it does not parse PDF/DOCX itself — that's the responsibility of **DA-2's `PDFRenderTool` / reader chain**. In the integrated flow, `/process` calls the reader first to get text, then hands it to this tool.
- **Embedding goes through Module 07**: `POST {DA_AI_LIFECYCLE_URL}/inference/infer`, `task="embedding"`, `model_name=bge-m3`, returns a 1024-dim vector.
  > ⚠ The original ticket text says `task:"text-embedding"`; Module 07's canonical task name is **`embedding`** (see `22-benchmark-service/app/services/judge.py`) — this tool uses `embedding`.
- **Vector store**: Qdrant collection `doc_agent_embeddings` (cosine, 1024), deliberately kept separate from Module 17's `raptor` collection.
- **Point id = content hash (uuid5)**: re-sending the same content from the same source overwrites rather than duplicates (idempotent).

## Configuration (`DA_*` env vars — aligned with DA-1's `config.py`)

| Env var | Default | Purpose |
|---|---|---|
| `DA_AI_LIFECYCLE_URL` | `http://raptor-ai-lifecycle-api:8010` | Module 07 endpoint |
| `DA_EMBED_MODEL` | `bge-m3` | Embedding model (must already be registered in Module 07) |
| `DA_QDRANT_URL` | `http://raptor-qdrant:6333` | Qdrant |
| `DA_QDRANT_DOC_COLLECTION` | `doc_agent_embeddings` | Collection name |
| `DA_QDRANT_VECTOR_SIZE` | `1024` | Dimension |
| `DA_EMBED_TIMEOUT` | `30` | Embedding call timeout (seconds) |
| `DA_CHUNK_SIZE` / `DA_CHUNK_OVERLAP` | `512` / `64` | Chunking |
| `DA_EMBED_BATCH_SIZE` | `64` | Batch size per embedding call |

`EmbeddingSearchConfig.from_env()` reads the vars above; you can also construct `EmbeddingSearchConfig(...)` directly and inject it (this is what the tests do).

## Integration point 1 — `app/main.py` lifespan (DA-1's file; paste this in as-is)

```python
from qdrant_client import QdrantClient
from app.core.doc_collection import ensure_doc_collection
from app.tools.embedding_search import EmbeddingSearchConfig

@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = EmbeddingSearchConfig.from_env()
    client = QdrantClient(url=cfg.qdrant_url)
    ensure_doc_collection(client, cfg.collection, cfg.vector_size)  # idempotent (DA-6 acceptance criterion)
    app.state.embedding_search = EmbeddingSearchTool(config=cfg, qdrant_client=client)
    # ... (Module 07 reachability logging etc. — DA-1's existing content)
    yield
```

## Integration point 2 — `task=search` on `POST /api/v1/docagent/process` (DA-1's router)

```python
# after the DA-2 reader has produced text:
tool = request.app.state.embedding_search
# reader (DA-2): raw file → pages/text; represented here as `text`
result = await asyncio.to_thread(              # forward() is synchronous (smolagents convention), offload to threadpool
    tool.forward,
    query=body.query,
    document_text=extracted_text,   # produced by the DA-2 reader; None = search existing content only
    source=body.source,             # filename / asset id
    top_k=body.top_k or 10,
)
return {"task": "search", "result": result}
# result = {"query", "indexed_chunks", "results": [{"id","score","text","source"}]}
```

`tool.index_document(text, source)` and `tool.search(query, top_k)` can also be called separately.

## Development / testing (using the v314 dev environment)

- Module 07 dev inference API: **`http://localhost:9998`** (container `aiml-v314-api`, MLflow on 5557, `bge-m3` already registered).
- Qdrant: integration tests use an in-memory `:memory:` instance, so they don't pollute the shared `raptor-qdrant:6333`.

```bash
cd deployment/modules/29-doc-processing-agent
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-da6.txt pytest reportlab pymupdf
python tests/fixtures/make_scope3_pdf.py          # generate the fixture (first time only)

pytest -m "not integration"                        # 7 unit tests (offline, Module 07 mocked via MockTransport)
DA_TEST_INFERENCE_URL=http://localhost:9998 pytest -m integration   # integration: real bge-m3
```

**Acceptance result (2026-07-22, v314 bge-m3)**: query `"scope 3 supplier emissions"` → top score **0.745 > 0.7** ✅ (DA-6 acceptance criterion met). 7 unit tests passed, 1 integration test passed.

## Left for others / follow-up

- **Port conflict**: DA-1 planned `PORT_DOC_AGENT=8029`, but host port **8029 is already used by `raptor-audio-diarization`** → DA-1 needs to pick a different port (build.py's port checker will also catch this).
- DA-6's runtime dependencies (`requirements-da6.txt`) need to be merged into DA-1's module-level `requirements.txt`.
- Handling of `image`-type documents in `task=search` (VLM captioning / OCR) isn't covered yet — see the open item in the implementation plan doc §8.
