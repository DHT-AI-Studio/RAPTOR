# kafka/services/video_graph_service/neo4j_client.py
#
# Neo4j write client for video_graph_service.
#
# Schema:
#   (:Source)   -[:HAS_MOMENT]->  (:Moment)
#   (:Entity)   -[:MENTIONED_IN]-> (:Source)     # video-level
#   (:Entity)   -[:APPEARS_IN]->   (:Moment)     # moment-level (hybrid)
#   (:Entity)   -[:CO_OCCURS_WITH]-(:Entity)     # same-moment co-appearance
#   (:Entity)   -[:RELATION]->     (:Entity)     # LLM-extracted from summary

import logging
import os
from collections import defaultdict
from itertools import combinations
from typing import Any, Dict, List

from neo4j import AsyncGraphDatabase

logger = logging.getLogger(__name__)


class VideoNeo4jClient:

    def __init__(self):
        self._driver = None

    async def connect(self):
        uri      = os.getenv("NEO4J_URI",      "bolt://raptor-neo4j:7687")
        user     = os.getenv("NEO4J_USER",     "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "raptor0.3")
        self._driver = AsyncGraphDatabase.driver(
            uri, auth=(user, password), max_connection_pool_size=50
        )
        async with self._driver.session() as session:
            await (await session.run("RETURN 1")).consume()
        logger.info(f"Neo4j connected: {uri}")

    async def close(self):
        if self._driver:
            await self._driver.close()

    # ------------------------------------------------------------------
    # Source node
    # ------------------------------------------------------------------

    async def upsert_source(self, source_meta: Dict[str, Any], summary: str) -> None:
        """
        MERGE Source node.
        id = version_id (content hash, guaranteed unique per upload).
        """
        cypher = """
        MERGE (s:Source {version_id: $version_id})
        ON CREATE SET
            s.title        = $title,
            s.filename     = $filename,
            s.asset_path   = $asset_path,
            s.branch_id    = $branch_id,
            s.media_type   = 'video',
            s.summary      = $summary,
            s.moment_count = $moment_count,
            s.processed_at = datetime()
        ON MATCH SET
            s.title        = $title,
            s.filename     = $filename,
            s.asset_path   = $asset_path,
            s.branch_id    = $branch_id,
            s.summary      = $summary,
            s.moment_count = $moment_count,
            s.updated_at   = datetime()
        """
        async with self._driver.session() as session:
            await session.run(
                cypher,
                version_id    = source_meta["source_id"],
                title         = source_meta.get("filename", ""),
                filename      = source_meta.get("filename", ""),
                asset_path    = source_meta.get("asset_path", ""),
                branch_id     = source_meta.get("branch_id", ""),
                summary       = summary,
                moment_count  = source_meta.get("moment_count", 0),
            )
        logger.info(f"Source upserted: {source_meta['source_id']}")

    # ------------------------------------------------------------------
    # Moment nodes + HAS_MOMENT edges
    # ------------------------------------------------------------------

    async def upsert_moments(
        self, source_id: str, moments: List[Dict[str, Any]]
    ) -> None:
        """
        Batch MERGE Moment nodes and HAS_MOMENT edges.
        One Cypher call per moment to keep error isolation.
        """
        cypher = """
        MERGE (m:Moment {id: $moment_id})
        ON CREATE SET
            m.version_id       = $version_id,
            m.asset_path       = $asset_path,
            m.branch_id        = $branch_id,
            m.moment_index     = $moment_index,
            m.start_sec        = $start_sec,
            m.end_sec          = $end_sec,
            m.asr_text         = $asr_text,
            m.lvlm_description = $lvlm_description,
            m.contextual_text  = $contextual_text,
            m.ocr_text         = $ocr_text,
            m.speaker          = $speaker,
            m.qdrant_id        = $qdrant_id
        ON MATCH SET
            m.version_id       = $version_id,
            m.asset_path       = $asset_path,
            m.branch_id        = $branch_id,
            m.asr_text         = $asr_text,
            m.lvlm_description = $lvlm_description,
            m.contextual_text  = CASE
                WHEN $contextual_text <> ''
                THEN $contextual_text
                ELSE m.contextual_text
            END,
            m.ocr_text         = CASE
                WHEN $ocr_text <> ''
                THEN $ocr_text
                ELSE m.ocr_text
            END,
            m.qdrant_id        = $qdrant_id
        WITH m
        MATCH (s:Source {version_id: $version_id})
        MERGE (s)-[:HAS_MOMENT]->(m)
        """
        async with self._driver.session() as session:
            for m in moments:
                await session.run(
                    cypher,
                    moment_id        = m["moment_id"],
                    version_id       = m.get("version_id", source_id),
                    asset_path       = m.get("asset_path", ""),
                    branch_id        = m.get("branch_id", ""),
                    moment_index     = m.get("moment_index", 0),
                    start_sec        = m.get("start_sec", 0.0),
                    end_sec          = m.get("end_sec", 0.0),
                    asr_text         = m.get("asr_text", ""),
                    lvlm_description = m.get("lvlm_description", ""),
                    contextual_text  = m.get("contextual_text", ""),
                    ocr_text         = m.get("ocr_text", ""),
                    speaker          = m.get("speaker", ""),
                    qdrant_id        = m.get("qdrant_id"),
                )
        logger.info(f"Upserted {len(moments)} Moment nodes for source {source_id}")

    # ------------------------------------------------------------------
    # Entity nodes
    # ------------------------------------------------------------------

    async def upsert_entities(
        self, source_id: str, entities: List[Dict[str, Any]]
    ) -> None:
        cypher = """
        MERGE (e:Entity {id: $id})
        ON CREATE SET
            e.name          = $name,
            e.type          = $type,
            e.description   = $description,
            e.source_id     = $source_id,
            e.mention_count = 1,
            e.created_at    = datetime()
        ON MATCH SET
            e.name          = $name,
            e.type          = $type,
            e.description   = $description,
            e.mention_count = e.mention_count + 1,
            e.updated_at    = datetime()
        """
        async with self._driver.session() as session:
            for entity in entities:
                await session.run(
                    cypher,
                    id          = entity["id"],
                    name        = entity.get("name", ""),
                    type        = entity.get("type", "UNKNOWN"),
                    description = entity.get("description", ""),
                    source_id   = source_id,
                )
        logger.info(f"Upserted {len(entities)} entities for source {source_id}")

    # ------------------------------------------------------------------
    # MENTIONED_IN edges  (Entity → Source, video-level)
    # ------------------------------------------------------------------

    async def create_mentioned_in(
        self, source_id: str, entity_ids: List[str]
    ) -> None:
        cypher = """
        MATCH (s:Source {version_id: $source_id})
        WITH s
        UNWIND $entity_ids AS eid
        MATCH (e:Entity {id: eid})
        MERGE (e)-[:MENTIONED_IN]->(s)
        """
        async with self._driver.session() as session:
            await session.run(cypher, source_id=source_id, entity_ids=entity_ids)

    # ------------------------------------------------------------------
    # RELATION edges  (Entity → Entity, from LLM on summary)
    # ------------------------------------------------------------------

    async def upsert_relations(
        self, source_id: str, relations: List[Dict[str, Any]]
    ) -> int:
        cypher = """
        MATCH (a:Entity {id: $subject_id})
        MATCH (b:Entity {id: $object_id})
        MERGE (a)-[r:RELATION {predicate: $predicate, source_document_id: $source_id}]->(b)
        ON CREATE SET
            r.time_start = $time_start,
            r.time_end   = $time_end,
            r.confidence = $confidence,
            r.created_at = datetime()
        ON MATCH SET
            r.confidence = $confidence,
            r.updated_at = datetime()
        """
        count = 0
        async with self._driver.session() as session:
            for rel in relations:
                result = await session.run(
                    cypher,
                    subject_id = rel["subject_id"],
                    object_id  = rel["object_id"],
                    predicate  = rel.get("predicate", "RELATED_TO"),
                    source_id  = source_id,
                    time_start = rel.get("time_start"),
                    time_end   = rel.get("time_end"),
                    confidence = rel.get("confidence", 1.0),
                )
                summary = await result.consume()
                count += summary.counters.relationships_created
        logger.info(f"Upserted {count} RELATION edges for source {source_id}")
        return count

    # ------------------------------------------------------------------
    # APPEARS_IN edges  (Entity → Moment, from hybrid matching)
    # ------------------------------------------------------------------

    async def create_appears_in(
        self, matches: List[Dict[str, Any]]
    ) -> int:
        """
        MERGE Entity -[:APPEARS_IN]-> Moment for each hybrid match.
        Properties: confidence, match_type ("exact" | "fuzzy").
        """
        cypher = """
        MATCH (e:Entity {id: $entity_id})
        MATCH (m:Moment {id: $moment_id})
        MERGE (e)-[r:APPEARS_IN]->(m)
        ON CREATE SET
            r.confidence  = $confidence,
            r.match_type  = $match_type,
            r.created_at  = datetime()
        ON MATCH SET
            r.confidence  = $confidence
        """
        count = 0
        async with self._driver.session() as session:
            for match in matches:
                result = await session.run(
                    cypher,
                    entity_id   = match["entity_id"],
                    moment_id   = match["moment_id"],
                    confidence  = match.get("confidence", 1.0),
                    match_type  = match.get("match_type", "exact"),
                )
                summary = await result.consume()
                count += summary.counters.relationships_created
        logger.info(f"Created {count} APPEARS_IN edges")
        return count

    # ------------------------------------------------------------------
    # CO_OCCURS_WITH edges  (Entity pairs sharing the same Moment)
    # ------------------------------------------------------------------

    async def create_co_occurs_with(
        self, matches: List[Dict[str, Any]]
    ) -> int:
        """
        For every Moment that has ≥ 2 entities, create CO_OCCURS_WITH pairs.
        Weight increments on each re-occurrence (same video or across videos).
        """
        # Group entity_ids by moment_id
        moment_entities: Dict[str, List[str]] = defaultdict(list)
        for m in matches:
            moment_entities[m["moment_id"]].append(m["entity_id"])

        cypher = """
        MATCH (a:Entity {id: $id_a})
        MATCH (b:Entity {id: $id_b})
        MERGE (a)-[r:CO_OCCURS_WITH]-(b)
        ON CREATE SET r.weight = 1, r.created_at = datetime()
        ON MATCH SET  r.weight = r.weight + 1, r.updated_at = datetime()
        """
        pairs_seen = set()
        count = 0

        async with self._driver.session() as session:
            for ids in moment_entities.values():
                if len(ids) < 2:
                    continue
                for id_a, id_b in combinations(sorted(set(ids)), 2):
                    if (id_a, id_b) not in pairs_seen:
                        pairs_seen.add((id_a, id_b))
                        result = await session.run(cypher, id_a=id_a, id_b=id_b)
                        summary = await result.consume()
                        count += summary.counters.relationships_created

        logger.info(f"Created/updated {count} CO_OCCURS_WITH edges")
        return count
