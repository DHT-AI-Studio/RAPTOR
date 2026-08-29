# services/document_indexer_service/kafka_handler.py

import asyncio
import json
import logging
import os
# import aiohttp  -- only used by the retired ingest_to_search() below
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from typing import Dict, Any
import config
from message_utils import create_response_message
from personal_index_publisher import publish_personal_index, _load_entries
from dotenv import load_dotenv
import os
# 計算上層資料夾的路徑
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 構建 .env 檔案的完整路徑
dotenv_path = os.path.join(parent_dir, ".env")

# 載入上層資料夾的 .env 檔案
load_dotenv(dotenv_path)
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
# HYBRID_SEARCH_INGEST_URL = os.getenv("HYBRID_SEARCH_INGEST_URL")  -- module 17 retired
logger = logging.getLogger(__name__)

class DocumentIndexerKafkaHandler:
    def __init__(self):
        self.consumer = None
        self.producer = None
        
    async def start_consumer(self):
        """啟動 Kafka consumer"""
        self.consumer = AIOKafkaConsumer(
            config.KAFKA_TOPIC_SAVE_REQUEST,
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            group_id=config.KAFKA_GROUP_ID,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='latest',
            enable_auto_commit=True
        )
        
        self.producer = AIOKafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        
        await self.consumer.start()
        await self.producer.start()
        logger.info(f"Kafka consumer started, listening to {config.KAFKA_TOPIC_SAVE_REQUEST}")
        
        try:
            async for message in self.consumer:
                await self.process_message(message.value)
        finally:
            await self.consumer.stop()
            await self.producer.stop()
    
    async def process_message(self, message: Dict[str, Any]):
        """處理接收到的訊息"""
        payload = message.get("payload", {})
        parameters = payload.get("parameters", {})
        summary_result_path = parameters.get("summary_result_path")

        try:
            logger.info(f"Received message: {message.get('message_id')}")

            if not summary_result_path:
                raise ValueError("Missing summary_result_path in parameters")

            # Module 17/20 retired -- personal-db (25) is the only index now.
            # results kept as {} (not deleted from the response message) since
            # module 17's own response shape already fell back to {} via
            # result.get("results", {}) in the commented-out ingest_to_search()
            # below, so downstream consumers of this Kafka response already
            # tolerate an empty dict here.
            results = {}
            # results = await self.ingest_to_search(summary_result_path)

            # Side-outputs for downstream consumers (graph + opensearch)
            await self._publish_side_outputs(message, parameters, results)

            # 發送成功回應
            response = create_response_message(
                message,
                status="success",
                results=results
            )

            await self.send_response(response)
            logger.info(f"Successfully processed message: {message.get('message_id')}")

        except Exception as e:
            logger.error(f"Error processing message: {e}")

            response = create_response_message(
                message,
                status="failed",
                error=str(e)
            )

            await self.send_response(response)

        finally:
            # Clean up summary result file after ingest (chunks are now in Qdrant)
            if summary_result_path:
                try:
                    if os.path.exists(summary_result_path):
                        os.remove(summary_result_path)
                        logger.info(f"Cleaned up summary result file: {summary_result_path}")
                except Exception as cleanup_err:
                    logger.warning(f"Failed to cleanup summary result file: {cleanup_err}")
    
    async def _publish_side_outputs(
        self,
        original_message: Dict[str, Any],
        parameters: Dict[str, Any],
        qdrant_results: Dict[str, Any],
    ) -> None:
        """Publish document-graph-requests and opensearch-index-requests after successful Qdrant save."""
        import uuid
        from datetime import datetime, timezone

        # Graph side-output: deliberately left as-is (gated on parameters["chunks"]
        # alone, which document_orchestrator_service never populates today, so
        # this is currently always a no-op). Not in scope for this fix — graph
        # extraction for documents was never scoped (separate, known gap), and
        # this fix must not silently start feeding it.
        graph_chunks = parameters.get("chunks", [])

        # opensearch + personal-index: parameters["chunks"] is always empty for
        # the same reason, but unlike graph these two DO need to actually work —
        # fall back to reading summary_result_path directly (the same file
        # ingest_to_search() already posted to Qdrant) when it wasn't provided.
        # _load_entries() already exists for exactly this case (09/10's own
        # entries_path fallback) — reuse it rather than re-reading the file
        # with new logic.
        chunks = parameters.get("chunks") or _load_entries(None, parameters.get("summary_result_path"))
        if not chunks:
            logger.warning(
                "_publish_side_outputs: no chunks in parameters or summary_result_path=%s",
                parameters.get("summary_result_path"))

        base = {
            "message_id":     str(uuid.uuid4()),
            "correlation_id": original_message.get("correlation_id", ""),
            "timestamp":      datetime.now(timezone.utc).isoformat(),
            "source_service": config.SERVICE_NAME,
            "message_type":   "REQUEST",
            "priority":       "MEDIUM",
            "retry_count":    0,
            "ttl":            3600,
        }

        if graph_chunks:
            graph_msg = {**base, "payload": {"parameters": {"chunks": graph_chunks}}}
            try:
                await self.producer.send_and_wait(config.KAFKA_TOPIC_GRAPH_REQUEST, graph_msg)
                logger.info(f"Published {len(graph_chunks)} chunks to {config.KAFKA_TOPIC_GRAPH_REQUEST}")
            except Exception as e:
                logger.warning(f"Failed to publish graph side-output: {e}")

        if not chunks:
            return

        # opensearch-index-requests feeds module 17's OpenSearch index,
        # retired along with the direct ingest_to_search() call above.
        # if chunks:
        #     index_msg = {**base, "payload": {"parameters": {"chunks": chunks}}}
        #     try:
        #         await self.producer.send_and_wait(config.KAFKA_TOPIC_INDEX_REQUEST, index_msg)
        #         logger.info(f"Published {len(chunks)} chunks to {config.KAFKA_TOPIC_INDEX_REQUEST}")
        #     except Exception as e:
        #         logger.warning(f"Failed to publish opensearch side-output: {e}")

        # Personal DB side-output (VIE01-190) — third consumer of the same chunks,
        # alongside opensearch above. Fire-and-forget like its sibling.
        await publish_personal_index(
            self.producer,
            source_module="12-document",
            entries=chunks,
            branch_id=original_message.get("payload", {}).get("user_id", ""),
            asset_path=parameters.get("asset_path", ""),
            version_id=parameters.get("version_id", ""),
        )

    # Module 17 retired -- kept for rollback, not deleted.
    # async def ingest_to_search(self, file_path: str) -> Dict[str, Any]:
    #     """調用 Qdrant API 保存數據"""
    #     try:
    #         async with aiohttp.ClientSession() as session:
    #             with open(file_path, 'rb') as f:
    #                 data = aiohttp.FormData()
    #                 data.add_field('file',
    #                                f,
    #                                filename=file_path.split('/')[-1],
    #                                content_type='application/json')
    #
    #                 async with session.post(HYBRID_SEARCH_INGEST_URL, data=data) as resp:
    #                     if resp.status != 200:
    #                         raise Exception(f"Qdrant API error: {resp.status}")
    #
    #                     result = await resp.json()
    #                     logger.info(f"Qdrant API response: {result}")
    #                     return result.get("results", {})
    #
    #     except Exception as e:
    #         logger.error(f"Error calling Qdrant API: {e}")
    #         raise


    async def send_response(self, response: Dict[str, Any]):
        """發送回應到 Kafka"""
        try:
            await self.producer.send_and_wait(
                config.KAFKA_TOPIC_SAVE_RESULT,
                response
            )
            logger.info(f"Response sent: {response.get('message_id')}")
        except Exception as e:
            logger.error(f"Error sending response: {e}")
            raise
