"""GD-3 / VIE01-248: apply GD_KAFKA_RETENTION_MS to the processing-request
topics on startup, idempotently -- describe each topic's current
retention.ms first, ALTER only the ones that differ.

No AdminClient usage exists anywhere else in this codebase to follow; this
is a new pattern here. Best-effort: Kafka being unreachable (or this
deployment's broker version not supporting the Admin protocol) must never
block this service from starting, same tolerance
erasure_orchestrator/main.py's tombstone producer already has -- caught
broadly and logged, not raised.

aiokafka's AdminClient is a thin, mostly-unwrapped layer over the raw Kafka
protocol: describe_configs()/alter_configs() return protocol Response
objects whose nested resources/config_entries are plain positional tuples
(not named attributes) -- see aiokafka.protocol.admin's
DescribeConfigsResponse_v0 / AlterConfigsResponse_v0 schemas, which this
parses against directly.
"""
from __future__ import annotations

import logging

from aiokafka.admin import AIOKafkaAdminClient
from aiokafka.admin.config_resource import ConfigResource, ConfigResourceType
from aiokafka.errors import KafkaError

from app.core.config import settings

logger = logging.getLogger("privacy_compliance.kafka_retention")

# The topics that actually carry personal-data payloads end to end
# (uploaded file content/metadata, per modules 09-12's *_orchestrator_service
# config.py KAFKA_TOPIC_REQUEST constants) -- not their classifier/
# diarization/summary/indexer sub-topics or *-results/*-dlq topics, which
# carry derived/intermediate data rather than the original personal data.
PROCESSING_TOPICS = (
    "audio-processing-requests",
    "image-processing-requests",
    "video-processing-requests",
    "document-processing-requests",
)


async def apply_topic_retention() -> None:
    if not settings.kafka_retention_ms:
        logger.info("GD_KAFKA_RETENTION_MS not set -- skipping Kafka topic retention config")
        return

    target = str(settings.kafka_retention_ms)
    admin = AIOKafkaAdminClient(bootstrap_servers=settings.kafka_bootstrap_servers.split(","))
    try:
        await admin.start()

        # ConfigResource.configs must be a dict (or None for "all configs") --
        # aiokafka's own _convert_describe_config_resource_request calls
        # .keys() on it unconditionally when not falsy. A plain list (the
        # first version of this code) raises AttributeError; found live
        # against a real broker.
        described = await admin.describe_configs([
            ConfigResource(ConfigResourceType.TOPIC, topic, configs={"retention.ms": None})
            for topic in PROCESSING_TOPICS
        ])

        to_alter = []
        for response in described:
            for resource in response.resources:
                error_code, error_message, _resource_type, resource_name, config_entries = resource
                if error_code != 0:
                    logger.warning("describe_configs failed for topic=%s: %s", resource_name, error_message)
                    continue
                current_retention_ms = next(
                    (entry[1] for entry in config_entries if entry[0] == "retention.ms"), None,
                )
                if current_retention_ms != target:
                    to_alter.append(ConfigResource(ConfigResourceType.TOPIC, resource_name, configs={"retention.ms": target}))

        if not to_alter:
            logger.info("Kafka topic retention.ms already up to date (%s ms) on all processing topics", target)
            return

        altered = await admin.alter_configs(to_alter)
        for response in altered:
            for result in response.resources:
                error_code, error_message, _resource_type, resource_name = result
                if error_code != 0:
                    logger.error("alter_configs failed for topic=%s: %s", resource_name, error_message)
                else:
                    logger.info("retention.ms set to %s for topic=%s", target, resource_name)
    except Exception as exc:  # noqa: BLE001 -- Kafka admin protocol/version quirks span multiple error types; must never block service startup per module docstring, already logged
        logger.warning("Kafka topic retention config skipped (Kafka unreachable, or broker doesn't support the Admin API): %s", exc)
    finally:
        try:
            await admin.close()
        except KafkaError as e:
            logger.debug(f"Kafka admin client close failed: {e}")
