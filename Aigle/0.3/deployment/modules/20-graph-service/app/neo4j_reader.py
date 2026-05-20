# graph-services/graph_query_service/neo4j_client.py

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional
from neo4j import AsyncGraphDatabase

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Async Neo4j driver wrapper for graph queries（graph_query_service 用）。"""

    def __init__(self):
        self._driver = None

    async def connect(self):
        uri = os.getenv("NEO4J_URI", "bolt://raptor-neo4j:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "raptor0.3")
        self._driver = AsyncGraphDatabase.driver(
            uri, auth=(user, password), max_connection_pool_size=100
        )
        async with self._driver.session() as session:
            await (await session.run("RETURN 1 AS ping")).consume()
        await self._ensure_indexes()
        logger.info(f"Neo4j connected: {uri}")

    async def _ensure_indexes(self):
        """Idempotent: create fulltext indexes used by GraphRAG queries."""
        stmts = [
            ("CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS "
             "FOR (n:Entity) ON EACH [n.name, n.description]"),
            ("CREATE FULLTEXT INDEX moment_fulltext IF NOT EXISTS "
             "FOR (n:Moment) ON EACH [n.asr_text, n.lvlm_description, n.contextual_text]"),
        ]
        async with self._driver.session() as session:
            for stmt in stmts:
                try:
                    await (await session.run(stmt)).consume()
                except Exception as exc:
                    logger.warning(f"Index ensure failed (may already exist): {exc}")

    async def close(self):
        if self._driver:
            await self._driver.close()

    # ------------------------------------------------------------------
    # Full-text search — Entity + Moment
    # ------------------------------------------------------------------

    async def fulltext_search_entities(
        self, query_text: str, threshold: float = 0.5, limit: int = 10,
        branch_id: str = "",
    ) -> List[Dict[str, Any]]:
        """搜尋 entity_fulltext index（name + description）。

        branch_id 非空時，只回傳有路徑連到該 branch 的 Source 或 Moment 的 Entity，
        避免不同 tenant 的實體污染查詢結果。
        """
        cypher = """
        CALL db.index.fulltext.queryNodes("entity_fulltext", $query_text)
        YIELD node, score
        WHERE score > $threshold
          AND ($branch_id = ''
            OR EXISTS { MATCH (node)-[:MENTIONED_IN]->(s:Source {branch_id: $branch_id}) }
            OR EXISTS { MATCH (node)-[:APPEARS_IN]->(m:Moment {branch_id: $branch_id}) }
          )
        RETURN node.id AS id, node.name AS name, node.type AS type,
               node.description AS description, 'entity' AS node_kind, score
        ORDER BY score DESC
        LIMIT $limit
        """
        async with self._driver.session() as session:
            result = await session.run(
                cypher, query_text=query_text, threshold=threshold,
                limit=limit, branch_id=branch_id,
            )
            records = await result.data()
        logger.info(f"Entity fulltext '{query_text}' [branch={branch_id or '*'}]: {len(records)} hits")
        return records

    async def fulltext_search_moments(
        self,
        query_text: str,
        threshold: float = 0.3,
        limit: int = 5,
        strategy: str = "hybrid",
        branch_id: str = "",
    ) -> List[Dict[str, Any]]:
        """
        搜尋 moment_fulltext index（asr_text + lvlm_description + contextual_text）。

        strategy:
          'literal'  — 只要在 asr_text 或 lvlm_description 命中關鍵字才保留
                       （給「找特定字幕 / 畫面文字」這類事實型 query）
          'semantic' — 只要在 contextual_text 命中才保留
                       （給「找有...場景」這類概念型 query）
          'hybrid'   — 不過濾，全部回傳（預設）

        每個 record 額外標註 matched_via: ["asr_text", "contextual_text", ...]
        並在 snippets 提供關鍵字附近上下文（給 LLM citation 用）。
        """
        cypher = """
        CALL db.index.fulltext.queryNodes("moment_fulltext", $query_text)
        YIELD node, score
        WHERE score > $threshold
          AND ($branch_id = '' OR node.branch_id = $branch_id)
        RETURN node.id AS id,
               node.version_id AS version_id,
               node.asset_path AS asset_path,
               node.branch_id AS branch_id,
               node.moment_index AS moment_index,
               node.start_sec AS start_sec, node.end_sec AS end_sec,
               node.asr_text AS asr_text,
               node.lvlm_description AS lvlm_description,
               node.contextual_text AS contextual_text,
               'moment' AS node_kind, score
        ORDER BY score DESC
        LIMIT $limit
        """
        async with self._driver.session() as session:
            result = await session.run(
                cypher,
                query_text=query_text,
                threshold=threshold,
                limit=limit * 3 if strategy != "hybrid" else limit,  # over-fetch for filter
                branch_id=branch_id,
            )
            records = await result.data()

        # Annotate each record with matched_via + snippets, then apply strategy filter
        annotated: List[Dict[str, Any]] = []
        for r in records:
            matched_via, snippets = _detect_matches(query_text, r)
            r["matched_via"] = matched_via
            r["snippets"]    = snippets
            annotated.append(r)

        if strategy == "literal":
            filtered = [
                r for r in annotated
                if "asr_text" in r["matched_via"] or "lvlm_description" in r["matched_via"]
            ]
        elif strategy == "semantic":
            filtered = [r for r in annotated if "contextual_text" in r["matched_via"]]
        else:
            filtered = annotated

        filtered = filtered[:limit]
        logger.info(
            f"Moment fulltext '{query_text}' [strategy={strategy}]: "
            f"{len(records)} hits → {len(filtered)} after filter"
        )
        return filtered

    # ------------------------------------------------------------------
    # Subgraph traversal
    # ------------------------------------------------------------------

    async def get_subgraph(
        self, entity_id: str, max_depth: int = 2, limit: int = 50,
        branch_id: str = "",
    ) -> Dict[str, Any]:
        """
        從 Entity 出發做子圖遍歷。
        回傳結構化的 nodes（含各 label 的特定欄位）+ relationships + moment_ids。
        moment_ids 讓 caller 可直接透過 Clip Proxy 播放對應片段。
        """
        cypher = """
        MATCH (start:Entity {id: $entity_id})
        CALL apoc.path.subgraphAll(start, {
            maxLevel: $max_depth,
            limit: $limit,
            relationshipFilter: "RELATION|CO_OCCURS_WITH|MENTIONED_IN|HAS_TEMPORAL_FACT|HAS_MOMENT|APPEARS_IN"
        })
        YIELD nodes, relationships

        // 額外取出所有 Moment node 的 id，方便 Clip Proxy 直接使用
        WITH nodes, relationships,
             [n IN nodes WHERE 'Moment' IN labels(n)
                           AND ($branch_id = '' OR n.branch_id = $branch_id) | {
                 moment_id:        n.id,
                 version_id:       n.version_id,
                 asset_path:       n.asset_path,
                 branch_id:        n.branch_id,
                 moment_index:     n.moment_index,
                 start_sec:        n.start_sec,
                 end_sec:          n.end_sec,
                 asr_text:         n.asr_text,
                 lvlm_description: n.lvlm_description,
                 contextual_text:  n.contextual_text,
                 ocr_text:         n.ocr_text
             }] AS moments
        RETURN nodes, relationships, moments
        """
        async with self._driver.session() as session:
            result = await session.run(
                cypher, entity_id=entity_id, max_depth=max_depth,
                limit=limit, branch_id=branch_id,
            )
            record = await result.single()

        if not record:
            logger.warning(f"No subgraph for entity: {entity_id}")
            return {"nodes": [], "relationships": [], "moment_ids": []}

        nodes = [_serialize_node(n) for n in record["nodes"]]
        relationships = [_serialize_rel(r) for r in record["relationships"]]
        moments = record["moments"] or []

        logger.info(
            f"Subgraph for {entity_id}: {len(nodes)} nodes, "
            f"{len(relationships)} rels, {len(moments)} moments"
        )
        return {"nodes": nodes, "relationships": relationships, "moment_ids": moments}

    async def get_moment_subgraph(
        self, moment_id: str, max_depth: int = 1
    ) -> Dict[str, Any]:
        """從 Moment 節點出發，取周邊的 Source + Entity。"""
        # Neo4j 5.x 不允許 RETURN 同時混 raw column 跟 collect()，必須先用
        # WITH 把 aggregation 算完才能跟 [m] 拼。
        cypher = """
        MATCH (m:Moment {id: $moment_id})
        OPTIONAL MATCH (s:Source)-[:HAS_MOMENT]->(m)
        OPTIONAL MATCH (e:Entity)-[:MENTIONED_IN]->(s)
        WITH m,
             collect(DISTINCT s) AS sources,
             collect(DISTINCT e) AS entities
        RETURN [m] + sources + entities AS nodes,
               []                          AS relationships
        """
        async with self._driver.session() as session:
            result = await session.run(cypher, moment_id=moment_id)
            record = await result.single()

        if not record:
            return {"nodes": [], "relationships": []}

        nodes = [_serialize_node(n) for n in record["nodes"] if n]
        return {"nodes": nodes, "relationships": []}

    # ------------------------------------------------------------------
    # TKG — collect temporal facts for a list of entity_ids
    # ------------------------------------------------------------------

    async def collect_temporal_facts(
        self,
        entity_ids: List[str],
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """對給定 entity 列表拉所有 TemporalFact，套用時間 filter。"""
        cypher = """
        MATCH (e:Entity)-[:HAS_TEMPORAL_FACT]->(tf:TemporalFact)
        WHERE e.id IN $entity_ids
          AND ($time_start IS NULL OR tf.time_end IS NULL OR tf.time_end >= $time_start)
          AND ($time_end   IS NULL OR tf.time_start IS NULL OR tf.time_start <= $time_end)
        RETURN DISTINCT tf
        ORDER BY tf.time_start ASC
        """
        async with self._driver.session() as session:
            result = await session.run(
                cypher, entity_ids=entity_ids,
                time_start=time_start, time_end=time_end,
            )
            rows = await result.data()
        facts = [_to_json_safe(dict(r["tf"].items())) for r in rows]
        logger.info(f"collect_temporal_facts: {len(entity_ids)} entities → {len(facts)} facts")
        return facts

    # ------------------------------------------------------------------
    # Unified GraphRAG query
    # ------------------------------------------------------------------

    async def query_graph_rag(
        self,
        query_text: str,
        max_depth: int = 2,
        limit: int = 50,
        score_threshold: float = 0.5,
        strategy: str = "hybrid",
        branch_id: str = "",
    ) -> Dict[str, Any]:
        """
        完整 GraphRAG 查詢：
        1. entity_fulltext + moment_fulltext 搜尋
        2. 對 Entity 做子圖遍歷
        3. 對 Moment 取周邊 Source + Entity
        4. 合併去重 + 結構化 citations 回傳

        strategy: 'literal' / 'semantic' / 'hybrid' — 詳見 fulltext_search_moments
        """
        # 1. 多 index 搜尋
        entities = await self.fulltext_search_entities(
            query_text, threshold=score_threshold, branch_id=branch_id,
        )
        try:
            moments = await self.fulltext_search_moments(
                query_text,
                threshold=score_threshold * 0.6,
                strategy=strategy,
                branch_id=branch_id,
            )
        except Exception as exc:
            logger.warning(f"Moment fulltext search failed (index may be missing): {exc}")
            moments = []

        all_nodes: Dict[str, dict] = {}
        all_rels: Dict[tuple, dict] = {}

        # 2. Entity subgraph traversal（並行）
        all_moment_ids: Dict[str, dict] = {}  # moment_id → moment info
        if entities:
            sg_results = await asyncio.gather(*[
                self.get_subgraph(entity["id"], max_depth=max_depth, limit=limit, branch_id=branch_id)
                for entity in entities
            ], return_exceptions=True)
            for entity, sg in zip(entities, sg_results):
                if isinstance(sg, Exception):
                    logger.warning(f"Subgraph for {entity['id']} failed: {sg}")
                    continue
                for n in sg["nodes"]:
                    all_nodes[n["id"]] = n
                for r in sg["relationships"]:
                    key = (r["start_id"], r["type"], r["end_id"])
                    all_rels[key] = r
                for m in sg["moment_ids"]:
                    all_moment_ids[m["moment_id"]] = m

        # 3. Moment context（fulltext matched moments，並行取周邊子圖）
        if moments:
            moment_sg_results = await asyncio.gather(*[
                self.get_moment_subgraph(moment["id"]) for moment in moments
            ], return_exceptions=True)
            for moment, sg in zip(moments, moment_sg_results):
                if not isinstance(sg, Exception):
                    for n in sg["nodes"]:
                        all_nodes[n["id"]] = n
                # fulltext matched moment 本身也加入 moment_ids，帶 matched_via + snippets
                all_moment_ids[moment["id"]] = {
                "moment_id":    moment["id"],
                "version_id":   moment.get("version_id"),
                "asset_path":   moment.get("asset_path"),
                "branch_id":    moment.get("branch_id"),
                "moment_index": moment.get("moment_index"),
                "start_sec":    moment.get("start_sec"),
                "end_sec":      moment.get("end_sec"),
                "asr_text":          moment.get("asr_text"),
                "lvlm_description":  moment.get("lvlm_description"),
                "contextual_text":   moment.get("contextual_text"),
                "ocr_text":          moment.get("ocr_text"),
                "matched_via":  moment.get("matched_via", []),
                "snippets":     moment.get("snippets", {}),
                "score":        moment.get("score"),
            }

        # 4. 結構化 citations — 給 LLM agent 引用片段時用
        sorted_moments = sorted(
            all_moment_ids.values(),
            key=lambda m: (m.get("version_id") or "", m.get("start_sec") or 0),
        )
        citations = [
            {
                "label":        f"[{i + 1}]",
                "moment_id":    m["moment_id"],
                "version_id":   m.get("version_id"),
                "start_sec":    m.get("start_sec"),
                "end_sec":      m.get("end_sec"),
                "matched_via":  m.get("matched_via", []),
                "snippets":     m.get("snippets", {}),
            }
            for i, m in enumerate(sorted_moments)
        ]

        result = {
            "query":            query_text,
            "strategy":         strategy,
            "matched_entities": entities,
            "matched_moments":  moments,
            "nodes":            list(all_nodes.values()),
            "relationships":    list(all_rels.values()),
            # moment_ids：依 start_sec 排序，供 Clip Proxy 直接使用
            "moment_ids":       sorted_moments,
            # citations：LLM agent 用 [n] 格式引用，並可從 snippets 取證據文字
            "citations":        citations,
        }
        logger.info(
            f"GraphRAG '{query_text}' [strategy={strategy}]: "
            f"{len(entities)} entities, {len(moments)} moments, "
            f"{len(result['nodes'])} nodes, {len(citations)} citations"
        )
        return result


# ---------------------------------------------------------------------------
# Node / relationship serializers
# ---------------------------------------------------------------------------

def _serialize_node(node) -> Dict[str, Any]:
    """把 Neo4j Node 物件轉成乾淨的 dict，依 label 展開特定欄位。"""
    labels = list(node.labels)
    raw_props = dict(node.items())
    props = _to_json_safe(raw_props)   # neo4j.time.* → ISO string
    # Source nodes use version_id as primary key; others use id
    if "Source" in labels:
        node_id = props.get("version_id", str(node.element_id))
    else:
        node_id = props.get("id", str(node.element_id))

    base = {
        "id": node_id,
        "labels": labels,
        "properties": props,
    }

    # 依 label 補充語意欄位，方便 caller 直接使用
    if "Entity" in labels:
        base.update({
            "name": props.get("name", ""),
            "type": props.get("type", ""),
            "description": props.get("description", ""),
        })
    elif "Moment" in labels:
        base.update({
            "version_id":       props.get("version_id", ""),
            "asset_path":       props.get("asset_path", ""),
            "branch_id":        props.get("branch_id", ""),
            "moment_index":     props.get("moment_index"),
            "start_sec":        props.get("start_sec"),
            "end_sec":          props.get("end_sec"),
            "asr_text":         props.get("asr_text", ""),
            "lvlm_description": props.get("lvlm_description", ""),
            "contextual_text":  props.get("contextual_text", ""),
        })
    elif "TemporalFact" in labels:
        base.update({
            "entity": props.get("entity", ""),
            "relation": props.get("relation", ""),
            "value": props.get("value", ""),
            "time_start": str(props["time_start"]) if props.get("time_start") else None,
            "time_end": str(props["time_end"]) if props.get("time_end") else None,
            "confidence": props.get("confidence"),
            "entity_id": props.get("entity_id"),
            "moment_id": props.get("moment_id"),
            "source_document_id": props.get("source_document_id"),
        })
    elif "Source" in labels:
        base.update({
            "title": props.get("title", ""),
            "media_type": props.get("media_type", ""),
            "asset_path": props.get("asset_path", ""),
        })

    return base


def _serialize_rel(rel) -> Dict[str, Any]:
    props = dict(rel.items())
    start_props = dict(rel.start_node.items())
    end_props = dict(rel.end_node.items())
    return {
        "type": rel.type,
        "start_id": start_props.get("id", str(rel.start_node.element_id)),
        "end_id": end_props.get("id", str(rel.end_node.element_id)),
        "properties": _to_json_safe(props),
    }


def _to_json_safe(v: Any) -> Any:
    """
    Recursively convert Neo4j-only types (neo4j.time.DateTime / Date / Time /
    Duration / Point) into JSON-serialisable values so Pydantic can dump them.

    Pure-Python primitives pass through; everything else falls back to str().
    """
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, list):
        return [_to_json_safe(x) for x in v]
    if isinstance(v, tuple):
        return [_to_json_safe(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _to_json_safe(val) for k, val in v.items()}
    if hasattr(v, "isoformat") and callable(v.isoformat):
        try:
            return v.isoformat()
        except Exception:
            return str(v)
    return str(v)


# ---------------------------------------------------------------------------
# Citation helpers — figure out which Moment field matched the query
# and build short snippets for LLM provenance
# ---------------------------------------------------------------------------

_SNIPPET_RADIUS = 40  # characters around the match

def _detect_matches(query_text: str, moment_record: Dict[str, Any]) -> tuple:
    """
    Inspect each text field on a moment, return:
      matched_via: list of field names where any query token appears (case-insensitive)
      snippets:    {field_name: short context window around the match}

    Token strategy (multi-pass):
      1. Full query as one phrase (e.g. "原型體", "machine learning")
      2. Whitespace-split tokens (English words separated by space)
      3. CJK n-grams: bigrams (2 chars) for Chinese, falling back to single chars
         when no longer phrase fits. Catches '原型體' query matching ASR text
         that only contains '原型' or '多伊'.
    """
    if not query_text:
        return [], {}

    phrases: List[str] = [query_text]
    phrases.extend(t for t in query_text.split() if t and t != query_text)

    # CJK fallback: split into bigrams + unigrams from contiguous CJK runs
    cjk_run = "".join(c for c in query_text if _is_cjk(c))
    if cjk_run:
        # bigrams (overlapping)
        for i in range(len(cjk_run) - 1):
            phrases.append(cjk_run[i:i + 2])
        # unigrams as last resort
        for c in cjk_run:
            phrases.append(c)

    # de-dup, preserve order
    seen_phrases = set()
    unique_phrases = []
    for p in phrases:
        if p and p not in seen_phrases:
            seen_phrases.add(p)
            unique_phrases.append(p)

    fields = {
        "asr_text":         moment_record.get("asr_text") or "",
        "lvlm_description": moment_record.get("lvlm_description") or "",
        "contextual_text":  moment_record.get("contextual_text") or "",
    }

    matched_via: List[str] = []
    snippets: Dict[str, str] = {}

    for field_name, field_text in fields.items():
        if not field_text:
            continue
        lower_text = field_text.lower()
        for phrase in unique_phrases:
            idx = lower_text.find(phrase.lower())
            if idx >= 0:
                matched_via.append(field_name)
                start = max(0, idx - _SNIPPET_RADIUS)
                end   = min(len(field_text), idx + len(phrase) + _SNIPPET_RADIUS)
                snippet = field_text[start:end].strip()
                if start > 0:
                    snippet = "…" + snippet
                if end < len(field_text):
                    snippet = snippet + "…"
                snippets[field_name] = snippet
                break  # one match per field is enough for citation

    return matched_via, snippets


def _is_cjk(ch: str) -> bool:
    """True for Chinese / Japanese / Korean unified ideographs."""
    if not ch:
        return False
    cp = ord(ch)
    return (
        0x4E00 <= cp <= 0x9FFF   # CJK Unified Ideographs
        or 0x3400 <= cp <= 0x4DBF  # Extension A
        or 0xF900 <= cp <= 0xFAFF  # Compatibility
    )
