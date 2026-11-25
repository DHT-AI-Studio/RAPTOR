// Migration 001: Add contextual_text to Moment fulltext index
//
// Background:
//   Text Contextual RAG 加入後，Moment 多了 contextual_text 屬性。
//   既有 moment_fulltext index 沒包含此欄位 — 需要 drop + recreate。
//
// Run on existing Neo4j 5.20 deployment:
//   cypher-shell -u neo4j -p raptor0.3 -f migration_001_contextual_text.cypher
//
// Idempotent — 若 index 已含 contextual_text 會 error，可忽略。

// 1. 砍掉舊 index
DROP INDEX moment_fulltext IF EXISTS;

// 2. 用新欄位重建
CREATE FULLTEXT INDEX moment_fulltext IF NOT EXISTS
  FOR (n:Moment) ON EACH [n.asr_text, n.lvlm_description, n.contextual_text];

// 3. 驗證
SHOW FULLTEXT INDEXES YIELD name, properties
WHERE name = 'moment_fulltext'
RETURN name, properties;
