#!/usr/bin/env python3
"""
Document Processing Worker Dispatcher
Reads WORKER_TYPE env var and routes to the appropriate service.

Valid WORKER_TYPE values:
  document_orchestrator  - Receives upload events, fans out to analysis/summary/save
  document_analysis      - OCR/VLM chunk extraction (GPU: InternVL)
  document_summary       - LLM summarization via Ollama
  document_indexer       - Ingests to hybrid search, emits graph/opensearch side-output topics
  document_graph         - Writes chunk entities to Neo4j knowledge graph
"""

import os
import sys
import asyncio
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SERVICE_MAP = {
    "document_orchestrator": "document_orchestrator_service",
    "document_analysis":     "document_analysis_service",
    "document_summary":      "document_summary_service",
    "document_indexer":      "document_indexer_service",
    "document_graph":        "document_graph_service",
}


def run():
    worker_type = os.environ.get("WORKER_TYPE", "").strip()

    if not worker_type:
        logger.error("WORKER_TYPE environment variable is not set")
        sys.exit(1)

    if worker_type not in SERVICE_MAP:
        logger.error(
            f"Unknown WORKER_TYPE: '{worker_type}'. "
            f"Valid values: {list(SERVICE_MAP.keys())}"
        )
        sys.exit(1)

    service_dir = os.path.join("/app/services", SERVICE_MAP[worker_type])
    if not os.path.isdir(service_dir):
        logger.error(f"Service directory not found: {service_dir}")
        sys.exit(1)

    # Prepend service directory so each service's flat imports (kafka_handler,
    # config, message_utils, …) resolve without any code changes.
    sys.path.insert(0, service_dir)
    logger.info(f"Starting worker '{worker_type}' from {service_dir}")

    from main import main as service_main  # noqa: PLC0415
    asyncio.run(service_main())


if __name__ == "__main__":
    run()
