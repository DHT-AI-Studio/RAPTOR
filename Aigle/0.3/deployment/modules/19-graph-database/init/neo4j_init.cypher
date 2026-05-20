// Neo4j Raptor 0.3 Schema Initialization
// Run once after first startup:
//   docker exec aigle-neo4j cypher-shell -u neo4j -p <password> -f /init/neo4j_init.cypher

// Entity uniqueness constraint
CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE;

// Entity indexes
CREATE INDEX entity_name IF NOT EXISTS FOR (n:Entity) ON (n.name);
CREATE INDEX entity_type IF NOT EXISTS FOR (n:Entity) ON (n.type);

// Relation temporal indexes
CREATE INDEX relation_time_start IF NOT EXISTS FOR ()-[r:RELATION]-() ON (r.time_start);
CREATE INDEX relation_time_end   IF NOT EXISTS FOR ()-[r:RELATION]-() ON (r.time_end);

// Document source tracking
CREATE INDEX entity_source_doc IF NOT EXISTS FOR (n:Entity) ON (n.source_document_id);

// Fulltext indexes (required by graph-service /query/graph_rag and /query/entity|moment/fulltext)
CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS
FOR (n:Entity) ON EACH [n.name, n.description];

CREATE FULLTEXT INDEX moment_fulltext IF NOT EXISTS
FOR (n:Moment) ON EACH [n.asr_text, n.lvlm_description, n.contextual_text];
