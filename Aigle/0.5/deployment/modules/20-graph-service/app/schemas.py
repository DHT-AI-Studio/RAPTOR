# graph-services/raptor_api/schemas.py
#
# Pydantic models for the FastAPI endpoints. Drives Swagger documentation.

from typing import Any, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

class IngestOptions(BaseModel):
    enable_per_moment_extraction: bool = Field(
        True,
        description=("Per-moment LLM entity extraction (batched). "
                     "Boosts APPEARS_IN precision but costs ~N/8 LLM calls."),
    )
    enable_moment_temporal: bool = Field(
        True,
        description=("Run moment-level temporal-fact extraction on moments "
                     "whose contextual_text contains time keywords. Cap=10."),
    )


class IngestRequest(BaseModel):
    records: list[dict[str, Any]] = Field(
        ...,
        description=("Qdrant payload records — typically the JSON array from "
                     "Multimodal Contextual Process or Text Contextual RAG."),
    )
    options: IngestOptions = Field(default_factory=IngestOptions)
    branch_id: str | None = Field(
        None,
        description="Branch (tenant) identifier — stored on Source and Moment nodes for multi-tenant filtering.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "records": [
                    {
                        "id": "uuid-1",
                        "payload": {
                            "embedding_type": "summary",
                            "version_id": "abc123",
                            "filename": "demo.mp4",
                            "asset_path": "video/demo",
                            "summary": "影片講述了某個故事",
                        },
                    },
                    {
                        "id": "uuid-2",
                        "payload": {
                            "embedding_type": "text",
                            "video_id": "demo.mp4_moment_0",
                            "chunk_index": 0,
                            "start_time": "0.0",
                            "end_time": "10.0",
                            "speaker": "SPEAKER_00",
                            "contextual": ("ocr:{frame1:{Hello}} / asr:{我是 Doey} / "
                                          "lvlm:{影片片段講述了角色的對話。}"),
                        },
                    },
                ],
                "options": {
                    "enable_per_moment_extraction": True,
                    "enable_moment_temporal": True,
                },
            }
        }
    }


class IngestResponse(BaseModel):
    source_id: str
    moments_ingested: int
    entities_upserted: int
    relations_upserted: int
    appears_in_created: int
    co_occurs_created: int
    temporal_facts: int
    elapsed_sec: float


class AsyncIngestAccepted(BaseModel):
    job_id: str
    status: Literal["accepted"] = "accepted"


class JobStatus(BaseModel):
    job_id: str
    status: Literal["pending", "running", "done", "failed"]
    result: IngestResponse | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

Strategy = Literal["literal", "semantic", "hybrid"]


class GraphRagRequest(BaseModel):
    query: str = Field(..., description="Natural-language query, EN or zh-TW.")
    max_depth: int = Field(2, ge=1, le=4, description="APOC subgraph max depth.")
    limit: int = Field(50, ge=1, le=200, description="Subgraph node cap per entity.")
    score_threshold: float = Field(0.5, ge=0.0, le=10.0,
                                    description="Lucene score floor for entity fulltext.")
    strategy: Strategy = Field(
        "hybrid",
        description=("'literal' = keyword in asr/lvlm only; "
                     "'semantic' = keyword in contextual_text only; "
                     "'hybrid' = no field filter (default)."),
    )
    branch_id: str | None = Field(
        None,
        description="Tenant filter — only return entities/moments reachable from this branch.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "原型體",
                "max_depth": 2,
                "limit": 50,
                "score_threshold": 0.5,
                "strategy": "hybrid",
            }
        }
    }


class FulltextRequest(BaseModel):
    query: str
    threshold: float = Field(0.3, ge=0.0, le=10.0)
    limit: int = Field(10, ge=1, le=100)


class MomentFulltextRequest(FulltextRequest):
    strategy: Strategy = "hybrid"


class GraphRagResponse(BaseModel):
    query: str
    strategy: str
    matched_entities: list[dict[str, Any]]
    matched_moments: list[dict[str, Any]]
    nodes: list[dict[str, Any]]
    relationships: list[dict[str, Any]]
    moment_ids: list[dict[str, Any]]
    citations: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# TKG Query
# ---------------------------------------------------------------------------

class TkgQueryRequest(BaseModel):
    query: str = Field(..., description="Natural-language query text")
    time_start: str | None = Field(None, description="ISO 8601 時間區間起點（可選）")
    time_end:   str | None = Field(None, description="ISO 8601 時間區間終點（可選）")
    max_depth:  int = Field(2, ge=1, le=4)
    limit:      int = Field(50, ge=1, le=200)
    score_threshold: float = Field(0.5, ge=0.0, le=10.0)
    branch_id: str | None = Field(
        None,
        description="Tenant filter — only return entities/moments reachable from this branch.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "川習會",
                "time_start": None,
                "time_end": None,
                "max_depth": 2,
                "limit": 50,
                "score_threshold": 0.5,
            }
        }
    }


class TkgQueryResponse(BaseModel):
    query: str
    matched_entities: list[dict[str, Any]]
    subgraph_nodes:   list[dict[str, Any]]
    subgraph_edges:   list[dict[str, Any]]
    temporal_facts:   list[dict[str, Any]]
    moment_ids:       list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Admin
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    neo4j_connected: bool
    llm_base_url: str


class StatsResponse(BaseModel):
    sources: int
    moments: int
    entities: int
    temporal_facts: int
    relation_edges: int
    appears_in_edges: int
    co_occurs_edges: int
