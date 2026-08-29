# graph-services/raptor_api/pipeline.py
#
# Ingest pipeline as a callable function. Mirrors the steps in
# kafka/services/video_graph_service/test_ingest.py but takes records
# directly (instead of reading from a file path) and returns a dict.

import logging
import time
from typing import Any, Dict, List

from config import ENTITY_MATCH_THRESHOLD, LLM_MAX_ENTITIES, LLM_MAX_RELATIONS
from graph_builder import (
    extract_entities_and_relations,
    extract_moment_entities_batched,
    match_entities_to_moments,
    resolve_moment_entities_to_ids,
)
from moment_parser import extract_moment_fields
from neo4j_writer import VideoNeo4jClient
from tkg import write_relations_as_temporal_facts, write_moment_temporal_facts

logger = logging.getLogger("raptor_api.pipeline")


# ---------------------------------------------------------------------------
# Qdrant payload → moments structure
# ---------------------------------------------------------------------------

def parse_qdrant_records(
    records: List[Dict[str, Any]],
    branch_id: str = "",
) -> Dict[str, Any]:
    source_meta: Dict[str, Any] = {}
    summary = ""
    moments: List[Dict[str, Any]] = []
    version_id = ""
    asset_path = ""

    for rec in records:
        payload   = rec.get("payload", {})
        qdrant_id = rec.get("id")

        if payload.get("embedding_type") == "summary":
            version_id = payload.get("version_id", "")
            asset_path = payload.get("asset_path", "")
            source_meta = {
                "source_id":  version_id,
                "filename":   payload.get("filename", ""),
                "asset_path": asset_path,
                "media_type": "video",
                "branch_id":  branch_id,
            }
            summary = payload.get("summary", "")
        elif payload.get("embedding_type") == "text":
            fields = extract_moment_fields(payload)
            chunk_index = payload.get("chunk_index", 0)
            moments.append({
                "moment_id":        qdrant_id or f"{version_id}_{chunk_index}",
                "qdrant_id":        qdrant_id,
                "moment_index":     chunk_index,
                "start_sec":        float(payload.get("start_time", 0)),
                "end_sec":          float(payload.get("end_time", 0)),
                "asr_text":         fields["asr_text"],
                "lvlm_description": fields["lvlm_description"],
                "contextual_text":  fields["contextual_text"],
                "ocr_text":         fields["ocr_text"],
                "speaker":          payload.get("speaker", ""),
                "raw_text":         fields["raw_text"],
                "asset_path":       asset_path,
                "version_id":       version_id,
                "branch_id":        branch_id,
            })

    source_meta["moment_count"] = len(moments)
    moments.sort(key=lambda m: m["moment_index"])
    return {"source_meta": source_meta, "summary": summary, "moments": moments}


# ---------------------------------------------------------------------------
# Pipeline entry-point
# ---------------------------------------------------------------------------

async def run_ingest_pipeline(
    records: List[Dict[str, Any]],
    *,
    enable_per_moment_extraction: bool = True,
    enable_moment_temporal: bool = True,
    branch_id: str = "",
) -> Dict[str, Any]:
    """
    Full Raptor 0.3 ingest pipeline. Connects/closes its own Neo4j driver.
    Returns counters for Swagger consumption.
    """
    t0 = time.perf_counter()

    parsed      = parse_qdrant_records(records, branch_id=branch_id)
    source_meta = parsed["source_meta"]
    summary     = parsed["summary"]
    moments     = parsed["moments"]
    source_id   = source_meta.get("source_id", "")

    if not source_id:
        raise ValueError("No source_id found — make sure one record has "
                         "embedding_type='summary' with a version_id field.")

    neo4j = VideoNeo4jClient()
    await neo4j.connect()

    try:
        # 1) Source
        await neo4j.upsert_source(source_meta, summary)

        # 2) Moments + HAS_MOMENT
        await neo4j.upsert_moments(source_id, moments)

        # 3) LLM video-level entity / relation extraction
        entities: List[Dict[str, Any]] = []
        relations: List[Dict[str, Any]] = []
        if summary:
            extracted = await extract_entities_and_relations(
                text=summary,
                max_entities=LLM_MAX_ENTITIES,
                max_relations=LLM_MAX_RELATIONS,
            )
            entities  = extracted["entities"]
            relations = extracted["relations"]

        # 4) Entity nodes + MENTIONED_IN
        if entities:
            await neo4j.upsert_entities(source_id, entities)
            await neo4j.create_mentioned_in(source_id, [e["id"] for e in entities])

        # 5) RELATION edges
        relation_count = 0
        if relations:
            relation_count = await neo4j.upsert_relations(source_id, relations)

        # 6) APPEARS_IN — hybrid + LLM-contextual
        appears_in_count = 0
        co_occurs_count  = 0
        if entities and moments:
            all_matches: List[Dict[str, Any]] = []

            hybrid_matches = match_entities_to_moments(
                entities, moments, threshold=ENTITY_MATCH_THRESHOLD
            )
            all_matches.extend(hybrid_matches)
            logger.info(f"hybrid matches: {len(hybrid_matches)}")

            if enable_per_moment_extraction:
                moments_with_ctx = [m for m in moments
                                    if (m.get("contextual_text") or "").strip()]
                if moments_with_ctx:
                    moment_entity_names = await extract_moment_entities_batched(
                        moments_with_ctx, batch_size=8
                    )
                    llm_matches, moment_derived_entities = resolve_moment_entities_to_ids(
                        moment_entity_names, entities, moments
                    )
                    # Upsert entities discovered only in moments (missed by summary extraction)
                    if moment_derived_entities:
                        await neo4j.upsert_entities(source_id, moment_derived_entities)
                        await neo4j.create_mentioned_in(
                            source_id, [e["id"] for e in moment_derived_entities]
                        )
                        entities = entities + moment_derived_entities
                    seen = {(m["entity_id"], m["moment_id"]) for m in all_matches}
                    for m in llm_matches:
                        if (m["entity_id"], m["moment_id"]) not in seen:
                            all_matches.append(m)
                            seen.add((m["entity_id"], m["moment_id"]))
                    logger.info(
                        f"llm-contextual matches: {len(llm_matches)}, "
                        f"moment-derived entities: {len(moment_derived_entities)}"
                    )

            if all_matches:
                appears_in_count = await neo4j.create_appears_in(all_matches)
                co_occurs_count  = await neo4j.create_co_occurs_with(all_matches)

        # 7) Temporal facts — direct Neo4j write, no HTTP roundtrip
        driver = neo4j._driver
        temporal_count = await write_relations_as_temporal_facts(
            driver, source_id=source_id, entities=entities, relations=relations,
        )
        if enable_moment_temporal and moments:
            temporal_count += await write_moment_temporal_facts(
                driver, source_id=source_id, entities=entities, moments=moments,
            )

        elapsed = round(time.perf_counter() - t0, 3)
        logger.info(f"Ingest done for {source_id} in {elapsed}s")

        return {
            "source_id":          source_id,
            "moments_ingested":   len(moments),
            "entities_upserted":  len(entities),
            "relations_upserted": relation_count,
            "appears_in_created": appears_in_count,
            "co_occurs_created":  co_occurs_count,
            "temporal_facts":     temporal_count,
            "elapsed_sec":        elapsed,
        }

    finally:
        await neo4j.close()
