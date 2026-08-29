# Embedding Search (task=search) - DA-6

## Description

As a user, I want to upload a document and find semantically similar passages across previously indexed documents using the search task, so that DocAgent can power RAG retrieval over ad-hoc document uploads without a separate indexing pipeline.

## Acceptance Criteria

 - EmbeddingSearchTool embeds document text chunks via Module 07 (task: "text-embedding", model: "bge-m3", 1024-dim)

 - Upserts embeddings into Qdrant collection doc_agent_embeddings (cosine, 1024-dim); collection initialised idempotently in lifespan

 - EmbeddingSearchTool performs nearest-neighbour search on query string; returns ranked {id, score, text, source} list

 - POST /api/v1/docagent/process with task=search routes through EmbeddingSearchTool; document is embedded at upload time and searched in the same request

 - Integration test: upload a 3-page PDF about scope 3 emissions → query "scope 3 supplier emissions" → returns ≥ 1 result with score > 0.7

 - DA_QDRANT_DOC_COLLECTION=doc_agent_embeddings and DA_QDRANT_VECTOR_SIZE=1024 used from config

## Subtasks:

 - app/tools/embedding_search.py — EmbeddingSearchTool

 - Qdrant collection init in main.py lifespan

 - Integration test with real embedding call (or mocked Module 07)