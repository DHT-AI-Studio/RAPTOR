"""Publish personal-index-requests events to Kafka (VIE01-190).

Dropped verbatim into each of Modules 09-12's indexer service — they are separate
containers with no shared package, so the file is duplicated rather than imported.
Keep the four copies identical.

Called from the indexer's success path, right after the global (Qdrant/OpenSearch)
ingest returns. Module 25 consumes the topic and builds the per-user ArcadeDB
index from the same entries that just went into the global stores.

Two properties the caller depends on:

  * **Idempotent.** `event_id = sha256(asset_path + version_id + branch_id)`, so a
    worker that retries its own send produces the identical id and Module 25
    recognises the replay instead of indexing the asset twice.
  * **Fire-and-forget.** Every failure here is caught and logged. Personal
    indexing is a downstream convenience; a Kafka hiccup must never fail or stall
    the global pipeline that already succeeded.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

PERSONAL_INDEX_TOPIC = os.getenv("KAFKA_TOPIC_PERSONAL_INDEX", "personal-index-requests")
SCHEMA_VERSION = "personal-index-v1"


def compute_event_id(asset_path: Optional[str], version_id: Optional[str], user_id: str) -> str:
    """Must stay byte-identical to Module 25's app/services/index_events.py.

    The 0x1f separator keeps the parts unambiguous: without it ("ab", "c") and
    ("a", "bc") hash the same, so two different assets could collide onto one
    event_id and the second would be silently dropped as a duplicate.
    """
    raw = "\x1f".join([asset_path or "", version_id or "", user_id or ""])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_entries(entries: Optional[List[Dict[str, Any]]],
                  entries_path: Optional[str]) -> List[Dict[str, Any]]:
    """Entries the caller already holds, else the JSON file it is about to ingest.

    Modules 09 and 10 only ever have the file (it is deleted in their `finally`),
    so this must be called before that cleanup runs.
    """
    if entries:
        return entries
    if not entries_path or not os.path.exists(entries_path):
        return []
    try:
        with open(entries_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as exc:
        logger.warning(f"[personal-index] could not read {entries_path}: {exc}")
        return []


def _first_payload_field(entries: List[Dict[str, Any]], field: str) -> str:
    """Pull a whole-asset field off whichever entry carries it.

    Not always the first entry: Module 09 prepends a summary item that may lack
    fields the per-chunk entries have, and vice versa.
    """
    for entry in entries:
        payload = entry.get("payload", entry)
        value = payload.get(field)
        if value:
            return str(value)
    return ""


async def publish_personal_index(
    producer,
    source_module: str,
    entries: Optional[List[Dict[str, Any]]] = None,
    entries_path: Optional[str] = None,
    moments: Optional[List[Dict[str, Any]]] = None,
    branch_id: str = "",
    asset_path: str = "",
    version_id: str = "",
) -> None:
    """Publish one index-request. Never raises.

    `producer` is the service's existing AIOKafkaProducer — no second connection.
    """
    try:
        chunk_entries = _load_entries(entries, entries_path)
        moment_entries = moments or []
        if not chunk_entries and not moment_entries:
            logger.info("[personal-index] nothing to publish — skipping")
            return

        all_entries = chunk_entries + moment_entries
        branch_id = branch_id or _first_payload_field(all_entries, "branch_id")
        if not branch_id:
            # Without an owner there is no database to route to. Module 25 would
            # skip every entry anyway, so do not put the message on the topic.
            logger.warning("[personal-index] no branch_id in payload — not publishing")
            return

        asset_path = asset_path or _first_payload_field(all_entries, "asset_path")
        version_id = version_id or _first_payload_field(all_entries, "version_id")

        envelope = {
            "event_id": compute_event_id(asset_path, version_id, branch_id),
            "schema_version": SCHEMA_VERSION,
            "source_module": source_module,
            "payload": {
                "branch_id": branch_id,
                "parameters": {
                    "version_id": version_id,
                    "asset_path": asset_path,
                    "chunks": chunk_entries,
                    "moments": moment_entries,
                    "entities": [],
                    "relationships": [],
                    "temporal_facts": [],
                },
            },
        }

        await producer.send_and_wait(PERSONAL_INDEX_TOPIC, envelope)
        logger.info(
            f"[personal-index] published {len(chunk_entries)} chunk(s) + "
            f"{len(moment_entries)} moment(s) for branch {branch_id} "
            f"(event_id={envelope['event_id'][:12]}…)"
        )

    except Exception as exc:
        logger.warning(f"[personal-index] publish failed (global indexing unaffected): {exc}")
