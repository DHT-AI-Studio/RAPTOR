import math

from fastapi import APIRouter, Depends

from app.schemas import SearchRequest, SearchResponse, RerankRequest, RerankResponse, RerankResult, SearchResult
from app.core.dependencies import get_search_service, get_embedding_service, get_reranker_service
from app.core.config import settings
from app.services.search import SearchService
from app.core.embedding import EmbeddingManager
from app.core.rerank import RerankerManager
from app.core.exceptions import EmbeddingError, SearchError


router = APIRouter()

@router.post("/hybrid", response_model=SearchResponse)
async def hybrid_search(
    req: SearchRequest,
    search_service: SearchService = Depends(get_search_service),
    embedder: EmbeddingManager = Depends(get_embedding_service),
):
    """
    Execute a Hybrid Search pipeline combining BM25 and Vector retrieval.

    This API implements a **Multi-stage Retrieval Architecture**:

    1.  **Recall Stage**:
        Simultaneously retrieves candidate documents from **OpenSearch (BM25)** and **Qdrant (Vector)**.
        To ensure a high recall rate, the system automatically expands the initial search depth to **top_k * 5**.
    2.  **Fusion Stage**:
        Merges multi-source results using the **RRF (Reciprocal Rank Fusion)** algorithm to balance keyword relevance and semantic similarity.
    3.  **Reranking Stage (Precision)**:
        The fused candidates are processed by the **Reranker**.
        This stage performs deep semantic interaction between the query and the documents to re-score and provide the final optimized ranking.

    Parameters:
    - query (str): The search query string. **Required.**
    - top_k (int, optional): Maximum number of results to return. Defaults to 10.
    - payload_schema (str, optional): Schema name for filter/rerank behaviour (e.g., 'contextual', 'temporal'). Defaults to 'contextual'.

    Filters:
    - embedding_type (str, optional): Field to use for embedding, either 'text' or 'summary'. Defaults to None.
    - type (str or List[str], optional): Collection type(s) to search. Defaults to None.
    - filename (List[str], optional): Filter results by specific filenames. Defaults to None.
    - speaker (List[str], optional): Filter by speaker names. Only applicable for 'audios' and 'videos'. Defaults to None.
    - source (str, optional): Filter by file type (e.g., 'csv', 'pdf', 'docx'). Defaults to None.
    """
    try:
        vector = await embedder.encode_async(req.query)
    except Exception as e:
        raise EmbeddingError(f"Failed to encode query: {str(e)}")
    try:
        return await search_service.hybrid_search(req, vector)
    except Exception as e:
        raise SearchError(f"Failed to execute hybrid search: {str(e)}")

@router.post("/vector", response_model=SearchResponse)
async def vector_search(
    req: SearchRequest,
    search_service: SearchService = Depends(get_search_service),
    embedder: EmbeddingManager = Depends(get_embedding_service),
):
    """
    Vector Search (only Qdrant)

    Parameters:
    - query (str): The search query string. **Required.**
    - top_k (int, optional): Maximum number of results to return. Defaults to 10.
    - payload_schema (str, optional): Schema name. Defaults to 'contextual'.

    Filters:
    - embedding_type, type, filename, speaker, source
    """
    try:
        vector = await embedder.encode_async(req.query)
    except Exception as e:
        raise EmbeddingError(f"Failed to encode query: {str(e)}")
    try:
        return await search_service.vector_search_by_qdrant(req, vector)
    except Exception as e:
        raise SearchError(f"Failed to execute vector search: {str(e)}")

@router.post("/bm25", response_model=SearchResponse)
async def bm25_search(
    req: SearchRequest,
    search_service: SearchService = Depends(get_search_service),
):
    """
    BM25 Search (only OpenSearch)

    Parameters:
    - query (str): The search query string. **Required.**
    - top_k (int, optional): Maximum number of results to return. Defaults to 10.
    - payload_schema (str, optional): Schema name. Defaults to 'contextual'.

    Filters:
    - embedding_type, type, filename, speaker, source
    """
    try:
        return await search_service.bm25_search_by_opensearch(req)
    except Exception as e:
        raise SearchError(f"Failed to execute BM25 search: {str(e)}")

@router.post("/rerank", response_model=RerankResponse)
async def rerank_documents(
    req: RerankRequest,
    reranker: RerankerManager = Depends(get_reranker_service),
):
    """Rerank pre-retrieved documents using the cross-encoder model.

    Accepts arbitrary (id, text) pairs — caller is responsible for extracting
    the text to score. Returns all documents sorted by reranker score.
    """
    top_k = req.top_k if req.top_k is not None else len(req.documents)
    search_results = [
        SearchResult(id=doc.id, score=0.0, payload={"_text": doc.text, **(doc.payload or {})})
        for doc in req.documents
    ]
    reranked = await reranker.rerank_async(
        query=req.query,
        documents=search_results,
        content_extractor=lambda p: p.get("_text", ""),
        top_k=top_k,
    )
    t = max(settings.RERANKER_TEMPERATURE, 1e-6)
    return RerankResponse(results=[
        RerankResult(id=r.id, score=1.0 / (1.0 + math.exp(-r.score / t)), payload=r.payload)
        for r in reranked
    ])
