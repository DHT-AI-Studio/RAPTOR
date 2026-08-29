// Migration 002: TKG (Neo4j-only) — TemporalFact properties + indexes
//
// Background:
//   v0.3 拿掉 PostgreSQL 雙寫，TemporalFact 直接存在 Neo4j。
//   temporal_model_service (8841) 寫入用的屬性名跟舊 schema 不同：
//     舊 (PG era):  subject / predicate / object / pg_fact_id
//     新 (Neo4j):   entity / entity_id / relation / value / moment_id
//   為相容，兩組屬性可共存，但新 index 必須補上才會走 index。
//
// Run:
//   docker exec -i aigle-neo4j cypher-shell -u neo4j -p <pwd> \
//       < migration_002_tkg_neo4j_only.cypher

// New indexes for the Neo4j-native field names ----------------------------

CREATE INDEX tf_entity IF NOT EXISTS
  FOR (n:TemporalFact) ON (n.entity);

CREATE INDEX tf_entity_id IF NOT EXISTS
  FOR (n:TemporalFact) ON (n.entity_id);

CREATE INDEX tf_relation IF NOT EXISTS
  FOR (n:TemporalFact) ON (n.relation);

CREATE INDEX tf_value IF NOT EXISTS
  FOR (n:TemporalFact) ON (n.value);

CREATE INDEX tf_moment_id IF NOT EXISTS
  FOR (n:TemporalFact) ON (n.moment_id);

// Composite covering index for the common time-window query
// (entity_id + time_start) — speeds up timeline / window endpoints
CREATE INDEX tf_entity_time IF NOT EXISTS
  FOR (n:TemporalFact) ON (n.entity_id, n.time_start);


// New edge type — TemporalFact 觀測自哪個 moment
// 用 OBSERVED_IN 而不是 MENTIONED_IN，語意更精確
// (TemporalFact)-[:OBSERVED_IN]->(Moment)


// Verify ------------------------------------------------------------------
SHOW INDEXES YIELD name, labelsOrTypes, properties
WHERE labelsOrTypes = ['TemporalFact']
RETURN name, properties;
