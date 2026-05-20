# Raptor 0.3 — Graph Service
# Port 8843
#
# GraphRAG
#   寫入：POST /ingest/payload  — Qdrant records → Neo4j 建圖
#         POST /ingest/upload   — 檔案上傳版本
#   查詢：POST /query/graph_rag, /query/entity/fulltext, /query/moment/fulltext
#         GET  /query/entity/{id}/subgraph, /query/moment/{id}/subgraph
#         GET  /graph/path, /graph/co-topics, /graph/by-predicate
#
# TKG
#   寫入：POST /temporal/facts  — upsert TemporalFact
#         POST /temporal/extract — LLM 萃取後寫入
#   查詢：GET  /temporal/timeline/{entity_id}
#         GET  /temporal/query, /temporal/window, /temporal/before, /temporal/after

from neo4j_reader import Neo4jClient as QueryNeo4jClient

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.responses import RedirectResponse

from pipeline import run_ingest_pipeline
from schemas import (
    FulltextRequest,
    GraphRagRequest,
    GraphRagResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    MomentFulltextRequest,
    StatsResponse,
    TkgQueryRequest,
    TkgQueryResponse,
)
from tkg import (
    ExtractRequest,
    ExtractResponse,
    TemporalFactIn,
    TemporalFactOut,
    llm_extract_temporal_facts,
    upsert_fact,
    _serialize_tf,
    _to_json_safe,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("graph_service")

# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------

state: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Graph Service")
    query_client = QueryNeo4jClient()
    try:
        await query_client.connect()
        state["query_client"] = query_client
        logger.info("Neo4j connected")
    except Exception as e:
        logger.warning(f"Neo4j connect failed: {e}")
        state["query_client"] = None
    yield
    if state.get("query_client"):
        await state["query_client"].close()
    logger.info("Graph Service shutdown")


app = FastAPI(
    title="Raptor 0.3 — Graph Service",
    version="0.3.0",
    description="GraphRAG + TKG 寫入與查詢服務（合併）。",
    lifespan=lifespan,
)


def _qc() -> QueryNeo4jClient:
    qc = state.get("query_client")
    if not qc:
        raise HTTPException(status_code=503, detail="Neo4j not available")
    return qc


# ---------------------------------------------------------------------------
# Admin
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["admin"])
async def health():
    qc = state.get("query_client")
    return HealthResponse(
        status="ok",
        neo4j_connected=qc is not None,
        llm_base_url=os.getenv("LLM_BASE_URL", "(unset)"),
    )


@app.get("/stats", response_model=StatsResponse, tags=["admin"])
async def stats():
    qc = _qc()
    cypher = """
    OPTIONAL MATCH (s:Source)        WITH count(s) AS sources
    OPTIONAL MATCH (m:Moment)        WITH sources, count(m) AS moments
    OPTIONAL MATCH (e:Entity)        WITH sources, moments, count(e) AS entities
    OPTIONAL MATCH (t:TemporalFact)  WITH sources, moments, entities, count(t) AS tf
    OPTIONAL MATCH ()-[r1:RELATION]->() WITH sources, moments, entities, tf, count(r1) AS rels
    OPTIONAL MATCH ()-[r2:APPEARS_IN]->() WITH sources, moments, entities, tf, rels, count(r2) AS app
    OPTIONAL MATCH ()-[r3:CO_OCCURS_WITH]-() WITH sources, moments, entities, tf, rels, app, count(r3) AS co
    RETURN sources, moments, entities, tf, rels, app, co
    """
    async with qc._driver.session() as session:
        result = await session.run(cypher)
        row = await result.single()
    if not row:
        return StatsResponse(sources=0, moments=0, entities=0, temporal_facts=0,
                             relation_edges=0, appears_in_edges=0, co_occurs_edges=0)
    return StatsResponse(
        sources=row["sources"], moments=row["moments"], entities=row["entities"],
        temporal_facts=row["tf"], relation_edges=row["rels"],
        appears_in_edges=row["app"], co_occurs_edges=row["co"] // 2,
    )


# ---------------------------------------------------------------------------
# GraphRAG — Ingest (write)
# ---------------------------------------------------------------------------

@app.post("/ingest/payload", response_model=IngestResponse,
          status_code=status.HTTP_200_OK, tags=["graphrag:ingest"])
async def ingest_payload(req: IngestRequest):
    """Qdrant payload records → Neo4j 建圖（Entity / Relation / APPEARS_IN / TemporalFact）。"""
    try:
        result = await run_ingest_pipeline(
            req.records,
            enable_per_moment_extraction=req.options.enable_per_moment_extraction,
            enable_moment_temporal=req.options.enable_moment_temporal,
            branch_id=req.branch_id or "",
        )
        return IngestResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/ingest/upload", response_model=IngestResponse,
          status_code=status.HTTP_200_OK, tags=["graphrag:ingest"])
async def ingest_upload(
    file: UploadFile = File(..., description="JSON file: array of Qdrant payload records"),
    enable_per_moment_extraction: bool = True,
    enable_moment_temporal: bool = True,
):
    if not file.filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="Only .json files accepted")
    raw = await file.read()
    try:
        records = json.loads(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
    if not isinstance(records, list):
        raise HTTPException(status_code=400, detail="JSON root must be an array")
    try:
        result = await run_ingest_pipeline(
            records,
            enable_per_moment_extraction=enable_per_moment_extraction,
            enable_moment_temporal=enable_moment_temporal,
        )
        return IngestResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# GraphRAG — Query
# ---------------------------------------------------------------------------

@app.post("/query/graph_rag", response_model=GraphRagResponse, tags=["graphrag:query"])
async def query_graph_rag(req: GraphRagRequest):
    """自然語言 → entity search → 子圖展開 → citations。"""
    qc = _qc()
    result = await qc.query_graph_rag(
        query_text=req.query, max_depth=req.max_depth,
        limit=req.limit, score_threshold=req.score_threshold,
        strategy=req.strategy,
        branch_id=req.branch_id or "",
    )
    return GraphRagResponse(**result)


@app.post("/query/entity/fulltext", tags=["graphrag:query"])
async def query_entity_fulltext(req: FulltextRequest):
    qc = _qc()
    return await qc.fulltext_search_entities(
        query_text=req.query, threshold=req.threshold, limit=req.limit,
    )


@app.post("/query/moment/fulltext", tags=["graphrag:query"])
async def query_moment_fulltext(req: MomentFulltextRequest):
    qc = _qc()
    return await qc.fulltext_search_moments(
        query_text=req.query, threshold=req.threshold,
        limit=req.limit, strategy=req.strategy,
    )


@app.get("/query/entity/{entity_id}/subgraph", tags=["graphrag:query"])
async def query_entity_subgraph(entity_id: str, max_depth: int = 2, limit: int = 50):
    qc = _qc()
    return await qc.get_subgraph(entity_id, max_depth=max_depth, limit=limit)


@app.get("/query/moment/{moment_id}/subgraph", tags=["graphrag:query"])
async def query_moment_subgraph(moment_id: str, max_depth: int = 1):
    qc = _qc()
    return await qc.get_moment_subgraph(moment_id, max_depth=max_depth)


@app.post("/tkg/query", response_model=TkgQueryResponse, tags=["tkg:query"])
async def tkg_query(req: TkgQueryRequest):
    """自然語言 → entity search → 子圖 + TemporalFact（含時間 filter）。

    同時回傳 GraphRAG 子圖結構與 TKG 時序事實，
    subgraph_nodes / subgraph_edges 供關係查詢，
    temporal_facts 供時序查詢，兩者共用同一批 matched_entities。
    """
    qc = _qc()
    branch_id = req.branch_id or ""
    entities = await qc.fulltext_search_entities(req.query, threshold=req.score_threshold, branch_id=branch_id)

    all_nodes: dict = {}
    all_edges: dict = {}
    all_moments: dict = {}

    if entities:
        sg_results = await asyncio.gather(*[
            qc.get_subgraph(entity["id"], max_depth=req.max_depth, limit=req.limit, branch_id=branch_id)
            for entity in entities
        ], return_exceptions=True)
        for entity, sg in zip(entities, sg_results):
            if isinstance(sg, Exception):
                logger.warning(f"Subgraph for {entity['id']} failed: {sg}")
                continue
            for n in sg["nodes"]:
                all_nodes[n["id"]] = n
            for r in sg["relationships"]:
                all_edges[(r["start_id"], r["type"], r["end_id"])] = r
            for m in sg["moment_ids"]:
                all_moments[m["moment_id"]] = m

    # Fallback: entity subgraph found no moments → search moments directly by text
    if not all_moments:
        try:
            moment_hits = await qc.fulltext_search_moments(
                req.query, threshold=req.score_threshold * 0.6, branch_id=branch_id,
            )
            for m in moment_hits:
                if m["id"] not in all_moments:
                    all_moments[m["id"]] = {
                        "moment_id":    m["id"],
                        "version_id":   m.get("version_id"),
                        "asset_path":   m.get("asset_path"),
                        "branch_id":    m.get("branch_id"),
                        "moment_index": m.get("moment_index"),
                        "start_sec":    m.get("start_sec"),
                        "end_sec":      m.get("end_sec"),
                        "asr_text":          m.get("asr_text"),
                        "lvlm_description":  m.get("lvlm_description"),
                        "contextual_text":   m.get("contextual_text"),
                        "ocr_text":          m.get("ocr_text"),
                    }
        except Exception as exc:
            logger.warning(f"TKG moment fallback search failed: {exc}")

    entity_ids = [e["id"] for e in entities]
    facts = await qc.collect_temporal_facts(entity_ids, req.time_start, req.time_end) if entity_ids else []

    return TkgQueryResponse(
        query=req.query,
        matched_entities=entities,
        subgraph_nodes=list(all_nodes.values()),
        subgraph_edges=list(all_edges.values()),
        temporal_facts=facts,
        moment_ids=list(all_moments.values()),
    )


@app.get("/tkg/entity/{entity_id}", tags=["tkg:query"])
async def tkg_entity_profile(entity_id: str, max_depth: int = 2, limit: int = 50):
    """Entity 完整 profile：子圖 + timeline 一次拿。"""
    qc = _qc()
    sg = await qc.get_subgraph(entity_id, max_depth=max_depth, limit=limit)
    facts = await qc.collect_temporal_facts([entity_id])
    return {
        "entity_id":      entity_id,
        "subgraph_nodes": sg["nodes"],
        "subgraph_edges": sg["relationships"],
        "moment_ids":     sg["moment_ids"],
        "temporal_facts": facts,
    }


@app.get("/graph/path", tags=["graphrag:query"])
async def shortest_path(from_entity: str, to_entity: str, max_depth: int = 5):
    """A → B 之間最短關係路徑。"""
    qc = _qc()
    cypher = f"""
    MATCH (a:Entity {{id: $from_id}}), (b:Entity {{id: $to_id}}),
          path = shortestPath((a)-[*1..{max_depth}]-(b))
    RETURN
        [n IN nodes(path) | {{id: n.id, labels: labels(n), name: n.name, type: n.type}}] AS nodes,
        [r IN relationships(path) | {{type: type(r), props: properties(r)}}]             AS rels
    LIMIT 1
    """
    async with qc._driver.session() as session:
        result = await session.run(cypher, from_id=from_entity, to_id=to_entity)
        row = await result.single()
    if not row:
        return {"nodes": [], "rels": [], "found": False}
    return {"nodes": list(row["nodes"]), "rels": list(row["rels"]), "found": True}


@app.get("/graph/co-topics", tags=["graphrag:query"])
async def co_topics(entity_id: str, limit: int = 20):
    """與 entity 共同出現次數最多的其他 entity（CO_OCCURS_WITH）。"""
    qc = _qc()
    cypher = """
    MATCH (e:Entity {id: $entity_id})-[r:CO_OCCURS_WITH]-(other:Entity)
    RETURN other.id AS id, other.name AS name, other.type AS type,
           count(r) AS co_count
    ORDER BY co_count DESC LIMIT $limit
    """
    async with qc._driver.session() as session:
        result = await session.run(cypher, entity_id=entity_id, limit=limit)
        rows = await result.data()
    return rows


@app.get("/graph/by-predicate", tags=["graphrag:query"])
async def by_predicate(
    entity_id: str, predicate: str,
    direction: str = "outgoing", limit: int = 50,
):
    """依 predicate 列出 entity 的鄰居。direction: outgoing | incoming | both。"""
    qc = _qc()
    if direction == "outgoing":
        pattern = "(e:Entity {id: $entity_id})-[r:RELATION {predicate: $predicate}]->(other:Entity)"
    elif direction == "incoming":
        pattern = "(other:Entity)-[r:RELATION {predicate: $predicate}]->(e:Entity {id: $entity_id})"
    else:
        pattern = "(e:Entity {id: $entity_id})-[r:RELATION {predicate: $predicate}]-(other:Entity)"
    cypher = f"""
    MATCH {pattern}
    RETURN other.id AS id, other.name AS name, other.type AS type,
           r.confidence AS confidence
    ORDER BY r.confidence DESC LIMIT $limit
    """
    async with qc._driver.session() as session:
        result = await session.run(cypher, entity_id=entity_id,
                                   predicate=predicate, limit=limit)
        rows = await result.data()
    return rows


# ---------------------------------------------------------------------------
# TKG — Write
# ---------------------------------------------------------------------------

@app.post("/temporal/facts", response_model=TemporalFactOut, tags=["tkg:write"])
async def create_fact(fact: TemporalFactIn):
    """Upsert a single TemporalFact. Idempotent on (entity_id, relation, value, time_start)。"""
    try:
        return TemporalFactOut(**await upsert_fact(_qc()._driver, fact))
    except Exception as e:
        logger.exception("create_fact failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/temporal/extract", response_model=ExtractResponse, tags=["tkg:write"])
async def extract_and_write(req: ExtractRequest):
    """LLM 從文字萃取時序事實並寫入 Neo4j。"""
    driver = _qc()._driver
    facts_dicts = await llm_extract_temporal_facts(req.text, max_facts=req.max_facts)
    written: List[Dict[str, Any]] = []
    for fd in facts_dicts:
        try:
            fact = TemporalFactIn(
                entity=fd.get("entity") or "",
                relation=fd.get("relation") or "RELATED_TO",
                value=fd.get("value") or "",
                time_start=fd.get("time_start"),
                time_end=fd.get("time_end"),
                confidence=float(fd.get("confidence") or 0.7),
                source_document_id=req.source_document_id,
                entity_id=fd.get("entity_id") or req.entity_id,
                moment_id=req.moment_id,
            )
            written.append(await upsert_fact(driver, fact))
        except Exception as e:
            logger.warning(f"Skipping malformed fact {fd}: {e}")
    return ExtractResponse(
        extracted=len(facts_dicts),
        written=len(written),
        facts=[TemporalFactOut(**w) for w in written],
    )


# ---------------------------------------------------------------------------
# TKG — Query
# ---------------------------------------------------------------------------

@app.get("/temporal/timeline/{entity_id}", response_model=List[TemporalFactOut], tags=["tkg:query"])
async def timeline(entity_id: str, limit: int = Query(100, ge=1, le=1000)):
    """entity 的事件時序軸，按 time_start 排序。"""
    cypher = """
    MATCH (e:Entity {id: $entity_id})-[:HAS_TEMPORAL_FACT]->(tf:TemporalFact)
    RETURN tf ORDER BY tf.time_start ASC, tf.created_at ASC LIMIT $limit
    """
    async with _qc()._driver.session() as session:
        result = await session.run(cypher, entity_id=entity_id, limit=limit)
        rows = await result.data()
    return [TemporalFactOut(**_serialize_tf(r["tf"])) for r in rows]


@app.get("/temporal/query", response_model=List[TemporalFactOut], tags=["tkg:query"])
async def query_facts(
    entity:         Optional[str] = Query(None),
    entity_id:      Optional[str] = Query(None),
    relation:       Optional[str] = Query(None),
    value:          Optional[str] = Query(None),
    time_after:     Optional[str] = Query(None),
    time_before:    Optional[str] = Query(None),
    moment_id:      Optional[str] = Query(None),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
    limit:          int   = Query(50, ge=1, le=500),
):
    """多條件過濾查詢 TemporalFact。"""
    where: List[str] = ["tf.confidence >= $min_confidence"]
    params: Dict[str, Any] = {"min_confidence": min_confidence}
    if entity:
        where.append("toLower(tf.entity) CONTAINS toLower($entity)")
        params["entity"] = entity
    if entity_id:
        where.append("tf.entity_id = $entity_id")
        params["entity_id"] = entity_id
    if relation:
        where.append("tf.relation = $relation")
        params["relation"] = relation
    if value:
        where.append("toLower(tf.value) CONTAINS toLower($value)")
        params["value"] = value
    if moment_id:
        where.append("tf.moment_id = $moment_id")
        params["moment_id"] = moment_id
    if time_after:
        where.append("(tf.time_end IS NULL OR tf.time_end >= $time_after)")
        params["time_after"] = time_after
    if time_before:
        where.append("(tf.time_start IS NULL OR tf.time_start <= $time_before)")
        params["time_before"] = time_before
    cypher = f"""
    MATCH (tf:TemporalFact) WHERE {" AND ".join(where)}
    RETURN tf ORDER BY tf.time_start ASC, tf.id LIMIT $limit
    """
    params["limit"] = limit
    async with _qc()._driver.session() as session:
        result = await session.run(cypher, **params)
        rows = await result.data()
    return [TemporalFactOut(**_serialize_tf(r["tf"])) for r in rows]


@app.get("/temporal/window", tags=["tkg:query"])
async def time_window(
    time_start: str, time_end: str,
    entity_id: Optional[str] = None, limit: int = 100,
):
    """某時間區間內所有 fact，可額外限定 entity。"""
    if entity_id:
        cypher = """
        MATCH (e:Entity {id: $entity_id})-[:HAS_TEMPORAL_FACT]->(tf:TemporalFact)
        WHERE (tf.time_end IS NULL OR tf.time_end >= $time_start)
          AND (tf.time_start IS NULL OR tf.time_start <= $time_end)
        RETURN tf ORDER BY tf.time_start LIMIT $limit
        """
        params: Dict[str, Any] = {"entity_id": entity_id, "time_start": time_start,
                                   "time_end": time_end, "limit": limit}
    else:
        cypher = """
        MATCH (tf:TemporalFact)
        WHERE (tf.time_end IS NULL OR tf.time_end >= $time_start)
          AND (tf.time_start IS NULL OR tf.time_start <= $time_end)
        RETURN tf ORDER BY tf.time_start LIMIT $limit
        """
        params = {"time_start": time_start, "time_end": time_end, "limit": limit}
    async with _qc()._driver.session() as session:
        result = await session.run(cypher, **params)
        rows = await result.data()
    return [_to_json_safe(dict(r["tf"].items())) for r in rows]


@app.get("/temporal/before", tags=["tkg:query"])
async def before(
    reference_time: str, entity_id: Optional[str] = None, limit: int = 50,
):
    """在 reference_time 之前發生的 fact。"""
    if entity_id:
        cypher = """
        MATCH (e:Entity {id: $entity_id})-[:HAS_TEMPORAL_FACT]->(tf:TemporalFact)
        WHERE tf.time_start IS NOT NULL AND tf.time_start < $reference_time
        RETURN tf ORDER BY tf.time_start DESC LIMIT $limit
        """
        params: Dict[str, Any] = {"entity_id": entity_id,
                                   "reference_time": reference_time, "limit": limit}
    else:
        cypher = """
        MATCH (tf:TemporalFact)
        WHERE tf.time_start IS NOT NULL AND tf.time_start < $reference_time
        RETURN tf ORDER BY tf.time_start DESC LIMIT $limit
        """
        params = {"reference_time": reference_time, "limit": limit}
    async with _qc()._driver.session() as session:
        result = await session.run(cypher, **params)
        rows = await result.data()
    return [_to_json_safe(dict(r["tf"].items())) for r in rows]


@app.get("/temporal/after", tags=["tkg:query"])
async def after(
    reference_time: str, entity_id: Optional[str] = None, limit: int = 50,
):
    """在 reference_time 之後發生的 fact。"""
    if entity_id:
        cypher = """
        MATCH (e:Entity {id: $entity_id})-[:HAS_TEMPORAL_FACT]->(tf:TemporalFact)
        WHERE tf.time_start IS NOT NULL AND tf.time_start > $reference_time
        RETURN tf ORDER BY tf.time_start ASC LIMIT $limit
        """
        params: Dict[str, Any] = {"entity_id": entity_id,
                                   "reference_time": reference_time, "limit": limit}
    else:
        cypher = """
        MATCH (tf:TemporalFact)
        WHERE tf.time_start IS NOT NULL AND tf.time_start > $reference_time
        RETURN tf ORDER BY tf.time_start ASC LIMIT $limit
        """
        params = {"reference_time": reference_time, "limit": limit}
    async with _qc()._driver.session() as session:
        result = await session.run(cypher, **params)
        rows = await result.data()
    return [_to_json_safe(dict(r["tf"].items())) for r in rows]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0",
                port=int(os.getenv("GRAPH_SERVICE_PORT", "8843")), reload=False)
