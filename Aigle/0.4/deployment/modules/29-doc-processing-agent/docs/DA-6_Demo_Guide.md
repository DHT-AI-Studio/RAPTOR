# DA-6 Demo Guide — EmbeddingSearchTool (document semantic search)

> Audience: anyone giving a live demo of what DA-6 currently implements.
> Related: [`DA-6_integration.md`](DA-6_integration.md), [`DocAgent_Implementation_Plan.md`](DocAgent_Implementation_Plan.md)
>
> **Runnable version**: [`../tests/demo_da6.py`](../tests/demo_da6.py) (`# %%`-delimited cells, can be opened directly
> as a notebook or run cell-by-cell with `python`) plus an already-executed version with output and charts at
> [`../tests/demo_da6.ipynb`](../tests/demo_da6.ipynb).
> All numbers below are from an actual run against v314 bge-m3 on 2026-07-23.

---

## 0. What this demo shows

DA-6 = **`EmbeddingSearchTool`**: upload document text → embed via **Module 07** (bge-m3, 1024-dim) →
write to a Qdrant collection → semantic nearest-neighbor search, returning `{id, score, text, source}`.

> ⚠ DA-6 is currently a **tool-layer** component only: the `/api/v1/docagent/process` service endpoint belongs to
> **DA-1** (owned by someone else, not yet built). So this demo drives the tool directly from Python; the tool's
> `forward()` is exactly the code path `/process?task=search` will use in future (index at upload time, and be
> searched together with every document indexed before it).

## 1. Environment & prerequisites

| Component | Location | Notes |
|---|---|---|
| Embedding (Module 07 dev) | `http://localhost:9998` | v314 `aiml-v314-api`, `bge-m3` already registered in MLflow on 5557 |
| Vector store | In-memory `:memory:` (default) | Clean, re-runnable; set `DA_DEMO_REAL_QDRANT=1` to write to the shared `raptor-qdrant:6333` instead |
| PDF fixture | `tests/fixtures/scope3_emissions.pdf` | A 3-page Scope 3 emissions report |

```bash
# run from the host
cd deployment/modules/29-doc-processing-agent
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-da6.txt pymupdf matplotlib      # smolagents/httpx/qdrant-client are in requirements-da6

# confirm v314 embedding is reachable
curl -s -X POST http://localhost:9998/inference/infer \
  -H 'Content-Type: application/json' \
  -d '{"task":"embedding","model_name":"bge-m3","data":{"inputs":["ping"]}}' | head -c 120
```

## 2. How to run (either one)

**A. Script / cell-by-cell**
```bash
python tests/demo_da6.py            # runs end-to-end, prints each step's result
```

**B. Notebook**
```bash
jupyter lab tests/demo_da6.ipynb    # already contains execution results and charts; re-run with a kernel that has the packages above
```

> To see the written points in the Qdrant dashboard (`http://localhost:6333/dashboard`):
> `DA_DEMO_REAL_QDRANT=1 python tests/demo_da6.py` (creates/clears the `doc_agent_demo` collection).

## 3. Demo flow & expected output

| Step | What it demonstrates | Expected output (actual run) |
|---|---|---|
| **1. Index two documents** | scope3 PDF (3 pages) + an unrelated cooling-system manual | `indexed 7 chunk(s) ... scope3` / `1 chunk ... cooling` |
| **2. Cross-document semantic search** | each of two queries hits the correct source | `"scope 3 supplier emissions"` → **0.745** [scope3]; `"...radiator and coolant..."` → **0.748** [cooling], with the scope3 passage scoring only 0.35 for the second query (clear separation) |
| **3. `forward()` (= `/process?task=search`)** | a single call that first indexes an appendix, then searches | `indexed_chunks: 1`, results include both the existing scope3 content and the new appendix |
| **4. Re-upload dedup** | re-sending the same content from the same source doesn't flood the index (id = content hash) | `points before: 9 | after: 9 → no duplicates ✅` |
| **5. Acceptance check** | top score > 0.7 | `top score = 0.745 (PASS ✅ threshold 0.7)` |
| **6. (optional) Bar chart** | top-k scores vs. the 0.7 threshold line | embedded PNG in the notebook |

### Talking points for the live demo
- "Same **bge-m3** model, but served through **Module 07** (not computed locally in-process the way Module 17 does)" —
  highlights the division of labor with platform-level retrieval (see [`Module_Overlap_Analysis.md`](Module_Overlap_Analysis.md)).
- "The query hits the correct document, and the unrelated document's score drops to 0.35" — proves this is semantic retrieval, not keyword matching.
- "`forward()` is exactly the future `/process?task=search`: **index at upload time, search in the same call**, no separate indexing pipeline needed."

## 4. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Can't reach `localhost:9998` | v314 `aiml-v314-api` isn't running; or set `DA_AI_LIFECYCLE_URL` if using a different host |
| `Module 07 returned 0 vectors` / 404 | `bge-m3` isn't registered in that MLflow instance (see `07-ai-ml-services/scripts/06_embedding_bge_m3.sh`) |
| Top score is slightly below expectations | bge-m3's cosine range is fairly narrow; chunk size affects this — the demo uses `chunk_size=400/overlap=60` (see the implementation plan doc §5) |
| Notebook kernel can't find the packages | Select a venv kernel that has `requirements-da6` + pymupdf/matplotlib installed |
