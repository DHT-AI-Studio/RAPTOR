# services/document_graph_service/neo4j_client.py

import logging
from typing import List, Dict, Any
from neo4j import AsyncGraphDatabase
from dotenv import load_dotenv
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(parent_dir, ".env"))

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://raptor-neo4j:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "raptor0.3")

logger = logging.getLogger(__name__)


class Neo4jClient:
    def __init__(self):
        self._driver = AsyncGraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
            max_connection_lifetime=3600,
        )

    async def close(self):
        await self._driver.close()

    async def upsert_entities_and_relations(
        self,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
        source_document_id: str,
    ) -> Dict[str, int]:
        """Write entity nodes and relationship edges extracted from a document."""
        entity_count = 0
        relation_count = 0

        async with self._driver.session() as session:
            for entity in entities:
                await session.run(
                    """
                    MERGE (e:Entity {id: $id})
                    SET e.name = $name,
                        e.type = $type,
                        e.source_document_id = $source_document_id,
                        e.updated_at = timestamp()
                    """,
                    id=entity.get("id", entity["name"].lower().replace(" ", "_")),
                    name=entity["name"],
                    type=entity.get("type", "UNKNOWN"),
                    source_document_id=source_document_id,
                )
                entity_count += 1

            for rel in relations:
                await session.run(
                    """
                    MATCH (s:Entity {name: $subject})
                    MATCH (o:Entity {name: $object})
                    MERGE (s)-[r:RELATION {type: $relation_type}]->(o)
                    SET r.time_start = $time_start,
                        r.time_end = $time_end,
                        r.source_document_id = $source_document_id,
                        r.updated_at = timestamp()
                    """,
                    subject=rel["subject"],
                    object=rel["object"],
                    relation_type=rel.get("relation", "RELATED_TO"),
                    time_start=rel.get("time_start"),
                    time_end=rel.get("time_end"),
                    source_document_id=source_document_id,
                )
                relation_count += 1

        logger.info(
            f"Neo4j upsert: {entity_count} entities, {relation_count} relations "
            f"for document {source_document_id}"
        )
        return {"entities": entity_count, "relations": relation_count}
